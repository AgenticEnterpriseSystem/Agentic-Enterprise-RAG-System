"""Managed idle cleanup is explicit, bounded and separate from SQL limits."""
from contextlib import nullcontext
import socket
from types import SimpleNamespace

import pytest

from src.db.runtime import Database, DatabaseConfigurationError, DatabaseSettings


ENVIRONMENT = {'RAG_DATABASE_URL': 'postgresql://rag_runtime:SYNTHETIC-SECRET@invalid/rag'}
NAME = 'RAG_DB_IDLE_TRANSACTION_TIMEOUT_MS'


@pytest.fixture(autouse=True)
def no_services(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError('Idle setting tests never open a database or network')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def test_library_default_remains_5000_without_an_explicit_override():
    settings = DatabaseSettings.from_env(ENVIRONMENT)
    assert settings.idle_transaction_timeout_ms == 5000
    assert settings.statement_timeout_ms == 5000
    assert settings.connect_timeout_seconds == 5 and settings.pool_size == 2
    assert 'SYNTHETIC-SECRET' not in repr(settings)


@pytest.mark.parametrize('raw,expected', [('1', 1), ('15000', 15000), ('60000', 60000), ('015000', 15000)])
def test_explicit_decimal_environment_override_accepts_bounded_milliseconds(raw, expected):
    settings = DatabaseSettings.from_env({**ENVIRONMENT, NAME: raw})
    assert settings.idle_transaction_timeout_ms == expected
    assert settings.statement_timeout_ms == 5000


@pytest.mark.parametrize('raw', ['', '0', '-1', '60001', '1.5e4', '15000.0', ' 15000',
    '+15000', '１５０００', '١٥٠٠٠', True, 15000, None, 'SYNTHETIC-TIMEOUT-SECRET'])
def test_invalid_override_fails_with_a_fixed_nonsecret_configuration_error(raw):
    with pytest.raises(DatabaseConfigurationError) as caught:
        DatabaseSettings.from_env({**ENVIRONMENT, NAME: raw})
    assert str(caught.value) == NAME + ' must be a decimal integer from 1 through 60000.'
    assert 'SYNTHETIC' not in str(caught.value)


def test_process_environment_override_uses_the_same_validation(monkeypatch):
    monkeypatch.setenv('RAG_DATABASE_URL', ENVIRONMENT['RAG_DATABASE_URL'])
    monkeypatch.setenv(NAME, '15000')
    assert DatabaseSettings.from_env().idle_transaction_timeout_ms == 15000
    monkeypatch.setenv(NAME, '')
    with pytest.raises(DatabaseConfigurationError):
        DatabaseSettings.from_env()


def test_transaction_applies_idle_override_locally_without_widening_statement_or_lock_limits():
    calls = []
    connection = SimpleNamespace(begin=lambda: nullcontext(),
        execute=lambda sql, values: calls.append((str(sql), dict(values))))
    database = object.__new__(Database)
    database.settings = DatabaseSettings.from_env({**ENVIRONMENT, NAME: '15000'})
    database.engine = SimpleNamespace(connect=lambda: nullcontext(connection))
    with database.transaction() as issued:
        assert issued is connection
    values = {v['key']: v['value'] for _, v in calls}
    assert values['idle_in_transaction_session_timeout'] == '15000'
    assert values['statement_timeout'] == '5000' and values['lock_timeout'] == '2000'
    assert all(sql == 'SELECT set_config(:key, :value, true)' for sql, _ in calls)
    # The operational override never supplies or changes an authority identity.
    assert all(value == '' for key, value in values.items() if key.startswith('rag.'))
