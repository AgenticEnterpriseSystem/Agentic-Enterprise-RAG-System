"""One time authority for persisted sessions and PostgreSQL access checks."""

from datetime import datetime, timezone

import sqlalchemy as sa


def session_time(connection, *, clock=None) -> datetime:
    """Read fresh UTC time on the caller's existing transaction/connection.

    clock_timestamp advances during a transaction, unlike transaction-start
    timestamps. A trusted explicitly injected test clock remains supported;
    normal host wall-clock changes never alter persisted session deadlines.
    This helper opens no pool connection and retains no cached clock offset.
    """
    now = clock() if clock is not None else connection.execute(
        sa.select(sa.func.clock_timestamp())).scalar_one()
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Session clock must return an aware datetime")
    return now.astimezone(timezone.utc)
