"""Tiny genuine Git/Parquet sources and disposable PostgreSQL for TOOL-01."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import sqlalchemy as sa

from src.access.policy import PolicyResolver
from src.access.repository import PolicyManifest, PolicyRepository
from src.auth.accounts import create_account
from src.auth.sessions import AuthService
from src.auth.settings import AuthSettings
from src.contracts import SnapshotRef
from src.data.corpus import extract_document
from src.data.structured import import_jira, prepare_jira_inputs
from src.db import schema
from src.db.migrate import upgrade_database
from src.db.runtime import Database, DatabaseSettings
from src.retrieval.tools import JiraTool
from src.storage.contracts import CatalogEntry
from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint
from tests.db_support import disposable_database, engine_for, pg_urls, url_string

PASSWORD = "Synthetic Jira reader password 🧭"
REPOSITORY = "https://example.invalid/verified-jira-fixture.git"


class Hasher:
    def hash(self, value): return "fixture-only$" + hashlib.sha256(value.encode()).hexdigest()
    def verify(self, encoded, value): return encoded == self.hash(value)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Fixture source")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "remote", "add", "origin", REPOSITORY)
    raw_dir = repo / "generated_data/sources/jira/directory-is-not-project"
    raw_dir.mkdir(parents=True)
    records = []
    for doc in ("one", "two", "hidden", "unknown"):
        row = {"dataset_doc_uuid": doc, "key": "SUP-100" if doc != "unknown" else "SUP-200",
            "summary": "Title " + doc, "description": "Exact source prose café 🧭 " + doc,
            "priority": "P1", "project": "customer-support", "title_field_name": "summary", "content_field_names": ["description"]}
        if doc != "unknown":
            row["status"] = "Closed" if doc == "two" else "In Progress"
        (raw_dir / (doc + ".json")).write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        records.append(row)
    (raw_dir / "noncanonical-alias.json").write_text(json.dumps(records[0]), encoding="utf-8")
    (raw_dir / "agents.md").write_text("Repository prose is not a record or an instruction to the importer.")
    revision = commit(repo)
    rows, entries = [], []
    for position, record in enumerate(records):
        title, content = extract_document(record)
        text, doc = title + "\n" + content, record["dataset_doc_uuid"]
        rows.append({"doc_id": doc, "source_type": "jira", "title": title, "content": content,
                     "path": "jira/directory-is-not-project/" + doc + ".json"})
        entries.append(CatalogEntry(doc_id=doc, revision_id=canonical_revision_id(text), position=position,
            source_type="jira", text_length=len(text), metadata_origin="synthetic"))
    rows.append({"doc_id": "slack", "source_type": "slack", "title": "Slack", "content": "Other canonical prose", "path": "slack/other.json"})
    entries.append(CatalogEntry(doc_id="slack", revision_id=canonical_revision_id("Slack\nOther canonical prose"),
        position=4, source_type="slack", text_length=len("Slack\nOther canonical prose"), metadata_origin="synthetic"))
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), corpus, row_group_size=2)
    snapshot = SnapshotRef(snapshot_id="jira-fixture", corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
        catalog_sha256=catalog_fingerprint(entries, expected_count=len(entries)), document_count=len(entries))
    inputs = prepare_jira_inputs(repo, corpus, snapshot, repository=REPOSITORY, commit=revision, coverage="synthetic_fixture")
    return SimpleNamespace(repo=repo, corpus=corpus, snapshot=snapshot, inputs=inputs, entries=entries, rows=rows, raw_dir=raw_dir)


@pytest.fixture
def env(pg_urls, source):
    with disposable_database(pg_urls) as urls:
        upgrade_database(url_string(urls.migration))
        owner = engine_for(urls.migration)
        database = Database(DatabaseSettings(dsn=url_string(urls.runtime), pool_size=1))
        try:
            with owner.begin() as connection:
                connection.execute(schema.dataset_snapshots.insert().values(snapshot_id=source.snapshot.snapshot_id,
                    corpus_sha256=source.snapshot.corpus_sha256, document_count=source.snapshot.document_count))
                for row, entry in zip(source.rows, source.entries):
                    connection.execute(schema.documents.insert().values(snapshot_id=source.snapshot.snapshot_id, **entry.model_dump()))
                    connection.execute(schema.document_revisions.insert().values(snapshot_id=source.snapshot.snapshot_id,
                        doc_id=entry.doc_id, revision_id=entry.revision_id, title=row["title"], canonical_text=row["title"] + "\n" + row["content"]))
                connection.execute(schema.dataset_snapshots.update().where(schema.dataset_snapshots.c.snapshot_id == source.snapshot.snapshot_id)
                    .values(state="ready", catalog_sha256=source.snapshot.catalog_sha256))
            hasher = Hasher()
            users = {name: create_account(name, PASSWORD, display_name=name, url=url_string(urls.migration), hasher=hasher,
                role="ceo" if name == "intern" else "intern") for name in ("intern", "executive")}
            auth = AuthService(database, AuthSettings(enabled=True, secure_cookies=False, allow_insecure_loopback=True,
                allowed_origins=("http://localhost:8000",), per_login_attempts=100, global_attempts=1000), hasher=hasher)
            identities = {name: auth.authenticate(auth.login(name, PASSWORD, client_ip="127.0.0.1").token, touch=False).identity for name in users}
            manifest = PolicyManifest(manifest_id="jira-fixture-policy", snapshot=source.snapshot, origin="synthetic",
                reviewed_by="fixture", provenance={"purpose": "synthetic conformance"},
                groups=({"group_id": "general", "label": "General"},),
                memberships=({"user_id": users["intern"].user_id, "group_id": "general"},),
                grants=({"group_id": "general", "doc_ids": ("one", "two")}, {"user_id": users["intern"].user_id, "doc_ids": ("one",)},
                    {"user_id": users["executive"].user_id, "doc_ids": ("one", "two", "hidden", "unknown")}))
            repository = PolicyRepository(url_string(urls.migration))
            repository.apply(manifest)
            resolver = PolicyResolver(database, source.snapshot)
            contexts = {name: resolver.resolve(value) for name, value in identities.items()}
            yield SimpleNamespace(source=source, urls=urls, owner=owner, database=database, users=users, auth=auth,
                identities=identities, resolver=resolver, contexts=contexts, manifest=manifest, repository=repository,
                tool=JiraTool(database, source.snapshot))
        finally:
            database.close()
            owner.dispose()


@pytest.fixture
def ready(env):
    env.report = import_jira(env.source.inputs, url=url_string(env.urls.migration), batch_size=2)
    return env
