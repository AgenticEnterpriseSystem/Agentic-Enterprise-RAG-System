"""Operator-provisioned local accounts; passwords never enter CLI arguments."""

import argparse
from contextlib import contextmanager
import getpass
import json
import re
import sys
from uuid import uuid4

from argon2 import PasswordHasher as Argon2Hasher, extract_parameters
from argon2.exceptions import InvalidHashError, VerificationError
from argon2.profiles import RFC_9106_LOW_MEMORY
from pydantic import SecretStr
import sqlalchemy as sa
from sqlalchemy.pool import NullPool

from src.auth.contracts import CurrentUser
from src.db import schema
from src.db.clock import session_time
from src.db.runtime import database_url, normalize_url


class AccountError(ValueError):
    """Sanitized operator account failure."""


def normalize_login(value: str) -> str:
    if not isinstance(value, str):
        raise AccountError("Invalid account login")
    value = value.strip().lower()
    if not 1 <= len(value) <= 254 or not re.fullmatch(r"[a-z0-9][a-z0-9._@+-]*", value):
        raise AccountError("Account logins use ASCII letters, digits and ._@+-")
    return value


def password_value(value, *, provisioning=False):
    value = value.get_secret_value() if isinstance(value, SecretStr) else value
    if not isinstance(value, str) or not (12 if provisioning else 1) <= len(value) <= 1024 or "\0" in value:
        raise AccountError("Password must contain12 to1024 characters" if provisioning else "Invalid credentials")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise AccountError("Invalid password encoding") from None
    return value


class PasswordHasher:
    """Pinned Argon2id RFC9106 low-memory profile:64MiB, t=3, p=4.

    Verification rejects other encoded parameters before Argon2 runs; a changed
    database hash cannot silently increase the advertised verification budget.
    """

    def __init__(self):
        self._hasher = Argon2Hasher.from_parameters(RFC_9106_LOW_MEMORY)

    def hash(self, password) -> str:
        return self._hasher.hash(password_value(password, provisioning=True))

    def verify(self, encoded_hash, password) -> bool:
        try:
            if not isinstance(encoded_hash, str) or len(encoded_hash) > 512 or extract_parameters(encoded_hash) != RFC_9106_LOW_MEMORY:
                return False
            return self._hasher.verify(encoded_hash, password_value(password))
        except (InvalidHashError, VerificationError, ValueError, TypeError):
            return False


@contextmanager
def _administrator(url):
    engine = sa.create_engine(normalize_url(url) if url is not None else database_url(migration=True),
        poolclass=NullPool, hide_parameters=True,
        connect_args={"prepare_threshold": None, "connect_timeout": 5, "application_name": "rag-account-admin"})
    try:
        with engine.begin() as connection:
            current, session = connection.execute(sa.text("SELECT current_user, session_user")).one()
            if current != "rag_migrator" or session != current or connection.connection.driver_connection.info.user != current:
                raise AccountError("Account administration requires the dedicated migrator login")
            yield connection
    finally:
        engine.dispose()


def _public(row):
    return CurrentUser.model_validate({name: row[name] for name in CurrentUser.model_fields})


def create_account(login, password, *, display_name, url=None, user_id=None,
                   role=None, department=None, project_ids=(), hasher=None) -> CurrentUser:
    login, password = normalize_login(login), password_value(password, provisioning=True)
    public = CurrentUser(user_id=user_id or uuid4().hex, display_name=display_name,
                         role=role, department=department, project_ids=tuple(project_ids))
    encoded = (hasher or PasswordHasher()).hash(password)
    try:
        with _administrator(url) as connection:
            connection.execute(schema.users.insert().values(login=login, password_hash=encoded, account_kind="human", **public.model_dump()))
    except sa.exc.IntegrityError:
        raise AccountError("Account could not be provisioned") from None
    return public


def _review(reactivate, reviewed_by, policy_reconciliation):
    if type(reactivate) is not bool:
        raise AccountError("Invalid account operation")
    if not reactivate:
        if reviewed_by is not None or policy_reconciliation is not None:
            raise AccountError("Review attestations require explicit reactivation")
        return None
    values = []
    for value, limit in ((reviewed_by, 254), (policy_reconciliation, 1024)):
        if not isinstance(value, str):
            raise AccountError("Reactivation requires reviewer and policy reconciliation attestations")
        value = value.strip()
        if not value or len(value) > limit or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise AccountError("Invalid reactivation attestation")
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise AccountError("Invalid reactivation attestation") from None
        values.append(value)
    return tuple(values)


def reset_password(login, password, *, url=None, hasher=None, reactivate=False,
                   reviewed_by=None, policy_reconciliation=None) -> dict | None:
    """Reset a human password; optionally attest reviewed disabled-account recovery.

    Reactivation changes no document grants or memberships. Its returned receipt
    records caller attestations, not an automated policy review or database audit.
    The ordinary reset API continues to return None and leaves activity unchanged.
    """
    review = _review(reactivate, reviewed_by, policy_reconciliation)
    login, password = normalize_login(login), password_value(password, provisioning=True)
    encoded = (hasher or PasswordHasher()).hash(password)
    with _administrator(url) as connection:
        if reactivate:
            row = connection.execute(sa.select(schema.users.c.user_id,
                schema.users.c.session_version, schema.users.c.account_kind,
                schema.users.c.is_active).where(schema.users.c.login == login)
                .with_for_update()).mappings().one_or_none()
            if row is None or row["account_kind"] != "human" or row["is_active"] is not False:
                raise AccountError("Reviewed reactivation requires a disabled human account")
            version = connection.execute(schema.users.update().where(
                schema.users.c.user_id == row["user_id"], schema.users.c.account_kind == "human",
                schema.users.c.is_active.is_(False),
                schema.users.c.session_version == row["session_version"]).values(
                password_hash=encoded, is_active=True,
                session_version=schema.users.c.session_version + 1)
                .returning(schema.users.c.session_version)).scalar_one_or_none()
            if version is None:
                raise AccountError("Account is unavailable")
            now = session_time(connection)
            connection.execute(schema.sessions.update().where(
                schema.sessions.c.user_id == row["user_id"], schema.sessions.c.revoked_at.is_(None))
                .values(revoked_at=sa.func.greatest(now, schema.sessions.c.created_at)))
            receipt = {"user_id": row["user_id"], "version": version,
                       "reviewer": review[0], "ref": review[1]}
        else:
            receipt = None
            changed = connection.execute(schema.users.update().where(schema.users.c.login == login,
                schema.users.c.account_kind == "human").values(
                password_hash=encoded, session_version=schema.users.c.session_version + 1).returning(schema.users.c.user_id)).scalar_one_or_none()
            if changed is None:
                raise AccountError("Account is unavailable")
    return receipt


def disable_account(login, *, url=None) -> None:
    login = normalize_login(login)
    with _administrator(url) as connection:
        changed = connection.execute(schema.users.update().where(schema.users.c.login == login).values(
            is_active=False, session_version=schema.users.c.session_version + 1).returning(schema.users.c.user_id)).scalar_one_or_none()
        if changed is None:
            raise AccountError("Account is unavailable")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("login")
    create.add_argument("--display-name", required=True)
    create.add_argument("--role")
    create.add_argument("--department")
    create.add_argument("--project", action="append", default=[])
    reset = commands.add_parser("reset-password")
    reset.add_argument("login")
    reset.add_argument("--reactivate", action="store_true",
        help="Reset and reactivate a disabled human after operator policy review")
    reset.add_argument("--reviewed-by")
    reset.add_argument("--policy-reconciliation")
    disable = commands.add_parser("disable")
    disable.add_argument("login")
    args = parser.parse_args(argv)
    receipt = None
    try:
        if args.command == "disable":
            disable_account(args.login)
        else:
            if args.command == "reset-password":
                _review(args.reactivate, args.reviewed_by, args.policy_reconciliation)
            if sys.stdin is None or not sys.stdin.isatty():
                raise AccountError("Password entry requires an interactive terminal")
            password = getpass.getpass("Password: ")
            confirmation = getpass.getpass("Confirm password: ")
            if password != confirmation:
                raise AccountError("Passwords do not match")
            if args.command == "create":
                user = create_account(args.login, password, display_name=args.display_name,
                    role=args.role, department=args.department, project_ids=args.project)
                receipt = {"operation": "create", "user_id": user.user_id, "display_name": user.display_name}
            else:
                receipt = reset_password(args.login, password, reactivate=args.reactivate,
                    reviewed_by=args.reviewed_by, policy_reconciliation=args.policy_reconciliation)
    except (ValueError, sa.exc.SQLAlchemyError):
        parser.error("Account operation failed; check operator configuration and account input")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"))
          if receipt is not None else "Account operation completed.")


if __name__ == "__main__":
    main()
