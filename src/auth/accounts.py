"""Operator-provisioned local accounts; passwords never enter CLI arguments."""

import argparse
from contextlib import contextmanager
import getpass
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
            connection.execute(schema.users.insert().values(login=login, password_hash=encoded, **public.model_dump()))
    except sa.exc.IntegrityError:
        raise AccountError("Account could not be provisioned") from None
    return public


def reset_password(login, password, *, url=None, hasher=None) -> None:
    login, password = normalize_login(login), password_value(password, provisioning=True)
    encoded = (hasher or PasswordHasher()).hash(password)
    with _administrator(url) as connection:
        changed = connection.execute(schema.users.update().where(schema.users.c.login == login).values(
            password_hash=encoded, session_version=schema.users.c.session_version + 1).returning(schema.users.c.user_id)).scalar_one_or_none()
        if changed is None:
            raise AccountError("Account is unavailable")


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
    disable = commands.add_parser("disable")
    disable.add_argument("login")
    args = parser.parse_args(argv)
    try:
        if args.command == "disable":
            disable_account(args.login)
        else:
            if sys.stdin is None or not sys.stdin.isatty():
                raise AccountError("Password entry requires an interactive terminal")
            password = getpass.getpass("Password: ")
            confirmation = getpass.getpass("Confirm password: ")
            if password != confirmation:
                raise AccountError("Passwords do not match")
            if args.command == "create":
                create_account(args.login, password, display_name=args.display_name,
                    role=args.role, department=args.department, project_ids=args.project)
            else:
                reset_password(args.login, password)
    except (ValueError, sa.exc.SQLAlchemyError):
        parser.error("Account operation failed; check operator configuration and account input")
    print("Account operation completed.")


if __name__ == "__main__":
    main()
