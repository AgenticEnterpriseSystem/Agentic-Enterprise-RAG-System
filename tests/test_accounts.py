"""Reviewed account recovery: pure CLI checks and opt-in real PostgreSQL cases.

No service is started here. PostgreSQL cases use only disposable rag_test_*
databases supplied by the root test runner; the cheap hasher is test-only.
"""

from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

sa = pytest.importorskip("sqlalchemy")
pytest.importorskip("argon2")

import src.auth.accounts as accounts
from src.auth.contracts import CurrentUser
from src.db import schema
from tests.db_support import disposable_database, engine_for, pg_urls, url_string


PASSWORD = "synthetic old password only"
REPLACEMENT = "synthetic replacement password only"
REVIEW = {"reactivate": True, "reviewed_by": "Recovery operator",
          "policy_reconciliation": "review/fixture-42"}


class CheapHasher:
    def hash(self, password):
        return "test-only$" + hashlib.sha256(password.encode()).hexdigest()

    def verify(self, encoded, password):
        return encoded == self.hash(password)


def interactive(monkeypatch):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    answers = iter((REPLACEMENT, REPLACEMENT))
    monkeypatch.setattr(accounts.getpass, "getpass", lambda *_: next(answers))


@pytest.mark.parametrize("flags", [
    ["--reactivate"],
    ["--reactivate", "--reviewed-by", "Operator"],
    ["--reactivate", "--policy-reconciliation", "review/42"],
    ["--reactivate", "--reviewed-by", " ", "--policy-reconciliation", "review/42"],
    ["--reactivate", "--reviewed-by", "Operator", "--policy-reconciliation", "line\nbreak"],
    ["--reviewed-by", "Operator", "--policy-reconciliation", "review/42"],
])
def test_reactivation_attestations_are_required_before_password_prompt(flags, monkeypatch, capsys):
    monkeypatch.setattr(accounts.getpass, "getpass", lambda *_: pytest.fail("Password prompt preceded review validation"))
    monkeypatch.setattr(accounts, "reset_password", lambda *_a, **_k: pytest.fail("Invalid review reached database operation"))
    with pytest.raises(SystemExit) as failure:
        accounts.main(["reset-password", "fixture-reader", *flags])
    assert failure.value.code == 2
    output = capsys.readouterr()
    assert output.out == "" and REPLACEMENT not in output.err


def test_create_cli_prints_only_allowlisted_public_identifiers(monkeypatch, capsys):
    interactive(monkeypatch)
    user = CurrentUser(user_id="fixture-user", display_name="Fixture reader", role="private-role",
                       department="private-department", project_ids=("private-project",))
    monkeypatch.setattr(accounts, "create_account", lambda *_a, **_k: user)
    accounts.main(["create", "fixture-reader", "--display-name", user.display_name])
    output = capsys.readouterr()
    assert json.loads(output.out) == {"operation": "create", "user_id": user.user_id,
                                      "display_name": user.display_name}
    assert output.err == "" and REPLACEMENT not in output.out and "private-" not in output.out


def test_reactivation_cli_forwards_review_and_prints_only_receipt(monkeypatch, capsys):
    interactive(monkeypatch)
    calls = []
    receipt = {"user_id": "fixture-user", "version": 7, "reviewer": "Operator", "ref": "review/42"}

    def reset(login, password, **kwargs):
        calls.append((login, password, kwargs))
        return receipt

    monkeypatch.setattr(accounts, "reset_password", reset)
    accounts.main(["reset-password", "fixture-reader", "--reactivate", "--reviewed-by", "Operator",
                   "--policy-reconciliation", "review/42"])
    assert calls == [("fixture-reader", REPLACEMENT, {"reactivate": True, "reviewed_by": "Operator",
                                                   "policy_reconciliation": "review/42"})]
    output = capsys.readouterr()
    assert json.loads(output.out) == receipt and output.err == ""
    assert REPLACEMENT not in output.out and "hash" not in output.out


def test_normal_reset_cli_preserves_completion_and_no_review(monkeypatch, capsys):
    interactive(monkeypatch)
    calls = []
    monkeypatch.setattr(accounts, "reset_password", lambda *a, **k: calls.append((a, k)))
    accounts.main(["reset-password", "fixture-reader"])
    assert calls == [(("fixture-reader", REPLACEMENT), {"reactivate": False,
                       "reviewed_by": None, "policy_reconciliation": None})]
    assert capsys.readouterr().out == "Account operation completed.\n"


def test_reviewed_reset_still_requires_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(accounts.getpass, "getpass", lambda *_: pytest.fail("Non-TTY prompted"))
    with pytest.raises(SystemExit) as failure:
        accounts.main(["reset-password", "fixture-reader", "--reactivate", "--reviewed-by", "Operator",
                       "--policy-reconciliation", "review/42"])
    assert failure.value.code == 2


@pytest.mark.parametrize("review", [
    {"reactivate": 1, "reviewed_by": "Operator", "policy_reconciliation": "review/42"},
    {"reactivate": True},
    {"reactivate": True, "reviewed_by": "Operator", "policy_reconciliation": "\ud800"},
    {"reviewed_by": "Operator"},
])
def test_library_invalid_review_never_hashes_or_opens_admin(review, monkeypatch):
    monkeypatch.setattr(accounts, "_administrator", lambda *_: pytest.fail("Invalid review opened administrator"))

    class NeverHash:
        def hash(self, _):
            pytest.fail("Invalid review hashed password")

    with pytest.raises(accounts.AccountError):
        accounts.reset_password("fixture-reader", REPLACEMENT, hasher=NeverHash(), **review)


@pytest.fixture(scope="module")
def account_db(pg_urls):
    from src.db.migrate import upgrade_database
    with disposable_database(pg_urls) as urls:
        upgrade_database(url_string(urls.migration))
        owner = engine_for(urls.migration)
        try:
            yield SimpleNamespace(urls=urls, owner=owner, hasher=CheapHasher())
        finally:
            owner.dispose()


@pytest.fixture
def account(account_db):
    login = "account_" + uuid4().hex
    user = accounts.create_account(login, PASSWORD, display_name="Synthetic recovery fixture",
        url=url_string(account_db.urls.migration), hasher=account_db.hasher)
    return SimpleNamespace(db=account_db, login=login, user=user)


def account_state(account):
    with account.db.owner.connect() as connection:
        return dict(connection.execute(sa.select(schema.users).where(
            schema.users.c.user_id == account.user.user_id)).mappings().one())


def reset(account, **kwargs):
    return accounts.reset_password(account.login, REPLACEMENT,
        url=url_string(account.db.urls.migration), hasher=account.db.hasher, **kwargs)


def test_pg_reviewed_reset_reactivates_revokes_all_sessions_and_preserves_policy(account):
    from src.auth.sessions import AuthError, AuthService
    from src.auth.settings import AuthSettings
    from src.db.runtime import Database, DatabaseSettings
    db = account.db
    runtime = Database(DatabaseSettings(dsn=url_string(db.urls.runtime), pool_size=1))
    auth = AuthService(runtime, AuthSettings(enabled=True), hasher=db.hasher)
    try:
        grant = auth.login(account.login, PASSWORD, client_ip="127.0.0.1")
        accounts.disable_account(account.login, url=url_string(db.urls.migration))
        group_id, future_id = uuid4().hex, uuid4().hex
        with db.owner.begin() as connection:
            now = connection.execute(sa.select(sa.func.clock_timestamp())).scalar_one()
            connection.execute(schema.groups.insert().values(group_id=group_id, label="Reviewed fixture group"))
            connection.execute(schema.group_memberships.insert().values(user_id=account.user.user_id, group_id=group_id))
            connection.execute(schema.sessions.insert().values(session_id=future_id, user_id=account.user.user_id,
                token_hash=hashlib.sha256(future_id.encode()).hexdigest(), csrf_hash="a" * 64,
                created_at=now+timedelta(seconds=120), expires_at=now+timedelta(seconds=180)))
            epoch = connection.execute(sa.select(schema.policy_state.c.revision)).scalar_one()
        before = account_state(account)
        receipt = reset(account, **REVIEW)
        after = account_state(account)
        assert receipt == {"user_id": account.user.user_id, "version": before["session_version"]+1,
                           "reviewer": REVIEW["reviewed_by"], "ref": REVIEW["policy_reconciliation"]}
        assert after["is_active"] is True and after["account_kind"] == "human"
        assert db.hasher.verify(after["password_hash"], REPLACEMENT)
        assert not db.hasher.verify(after["password_hash"], PASSWORD)
        with db.owner.connect() as connection:
            sessions = connection.execute(sa.select(schema.sessions).where(
                schema.sessions.c.user_id == account.user.user_id)).mappings().all()
            assert all(row["revoked_at"] is not None and row["revoked_at"] >= row["created_at"] for row in sessions)
            assert connection.execute(sa.select(schema.group_memberships.c.group_id).where(
                schema.group_memberships.c.user_id == account.user.user_id)).scalars().all() == [group_id]
            assert connection.execute(sa.select(schema.policy_state.c.revision)).scalar_one() == epoch
        with pytest.raises(AuthError):
            auth.authenticate(grant.token)
        with pytest.raises(AuthError):
            auth.login(account.login, PASSWORD, client_ip="127.0.0.1")
        fresh = auth.login(account.login, REPLACEMENT, client_ip="127.0.0.1")
        assert auth.authenticate(fresh.token).identity.session_version == receipt["version"]
    finally:
        runtime.close()


@pytest.mark.parametrize("state", ["active-human", "disabled-benchmark", "missing"])
def test_pg_reactivation_rejects_ineligible_accounts_without_changes(account, state, tmp_path):
    if state == "disabled-benchmark":
        from src.access.repository import PolicyRepository
        from src.contracts import SnapshotRef
        from tests.db_support import seed_catalog

        with account.db.owner.begin() as connection:
            catalog = seed_catalog(connection, ready=True)
        principal = PolicyRepository(url_string(account.db.urls.migration)).provision_benchmark(
            "benchmark_" + uuid4().hex,
            snapshot=SnapshotRef(snapshot_id=catalog["snapshot_id"], corpus_sha256="a" * 64,
                                 catalog_sha256="b" * 64, document_count=1),
            credentials_file=tmp_path / "synthetic-benchmark.json",
            reviewed_by="fixture-operator", provenance={"purpose": "account-kind denial fixture"},
            all_documents=True, origin="synthetic", hasher=account.db.hasher)
        account.login = principal.login
        account.user = SimpleNamespace(user_id=principal.user_id)
        accounts.disable_account(account.login, url=url_string(account.db.urls.migration))
    before = account_state(account)
    if state == "missing":
        account.login = "missing_" + uuid4().hex
    with pytest.raises(accounts.AccountError):
        reset(account, **REVIEW)
    assert account_state(account) == before


def test_pg_reactivation_session_failure_rolls_back_password_activity_and_version(account, monkeypatch):
    accounts.disable_account(account.login, url=url_string(account.db.urls.migration))
    before = account_state(account)
    original = accounts._administrator

    @contextmanager
    def broken_admin(url):
        with original(url) as connection:
            def fail_revocation(_conn, _cursor, statement, _parameters, _context, _many):
                if statement.lstrip().startswith("UPDATE rag.sessions"):
                    raise sa.exc.SQLAlchemyError("synthetic revocation failure")
            sa.event.listen(connection, "before_cursor_execute", fail_revocation)
            try:
                yield connection
            finally:
                sa.event.remove(connection, "before_cursor_execute", fail_revocation)

    monkeypatch.setattr(accounts, "_administrator", broken_admin)
    with pytest.raises(sa.exc.SQLAlchemyError):
        reset(account, **REVIEW)
    assert account_state(account) == before


def test_pg_normal_reset_keeps_disabled_account_disabled_and_returns_none(account):
    accounts.disable_account(account.login, url=url_string(account.db.urls.migration))
    before = account_state(account)
    assert reset(account) is None
    after = account_state(account)
    assert after["is_active"] is False and after["session_version"] == before["session_version"] + 1
    assert account.db.hasher.verify(after["password_hash"], REPLACEMENT)


def test_pg_reactivation_requires_migrator_login(account):
    accounts.disable_account(account.login, url=url_string(account.db.urls.migration))
    before = account_state(account)
    with pytest.raises(accounts.AccountError):
        accounts.reset_password(account.login, REPLACEMENT, url=url_string(account.db.urls.runtime),
            hasher=account.db.hasher, **REVIEW)
    assert account_state(account) == before
