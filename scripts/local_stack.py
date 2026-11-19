"""Ordered local Docker setup without seeded users or destructive volume cleanup.

The init command alone uses no Docker. Other commands are explicit operator
actions; no credential values, resolved Compose environments or SQL are printed.
The authoritative source database volume is never an allowed target.
"""

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_DATABASE_VOLUMES = frozenset({'rag-doc01-data-a283c181'})


class StackError(ValueError):
    pass


def initialize(path, *, artifacts_dir, models_dir, source_repo_dir, operator_dir, project='redwood', port=8000):
    path = Path(path)
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,47}', project) or type(port) is not int or not 1024 <= port <= 65535:
        raise StackError('Choose a simple project name and an unprivileged local port')
    directories = {}
    for name, value in (('ARTIFACTS_DIR', artifacts_dir), ('MODELS_DIR', models_dir),
                        ('SOURCE_REPO_DIR', source_repo_dir), ('OPERATOR_DIR', operator_dir)):
        directory = Path(value).resolve()
        if any(character in str(directory) for character in ("'", '\n', '\r', '\0')):
            raise StackError('Preparation paths with quotes/control characters need manual configuration')
        if name == 'OPERATOR_DIR':
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        elif not directory.is_dir():
            raise StackError('Prepared artifacts, pinned models and the pinned source checkout must already exist')
        directories[name] = "'" + str(directory) + "'"
    values = {**directories, 'COMPOSE_PROJECT_NAME': project, 'WEB_PORT': str(port),
        'LOCAL_UID': str(os.getuid()), 'LOCAL_GID': str(os.getgid()),
        **{name: secrets.token_urlsafe(36) for name in ('RAG_DB_BOOTSTRAP_PASSWORD',
            'RAG_DB_MIGRATOR_PASSWORD', 'RAG_DB_RUNTIME_PASSWORD')}}
    lines = []
    for line in (ROOT / '.env.example').read_text().splitlines():
        key, separator, _ = line.partition('=')
        lines.append(key + '=' + values[key] if separator and key in values else line)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return {'initialized': True, 'project': project, 'port': port,
            'accounts_seeded': False, 'credentials_printed': False}


class Stack:
    def __init__(self, env_file, *, compose_files=()):
        self.env_file = Path(env_file).resolve()
        descriptor = os.open(self.env_file, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
        with os.fdopen(descriptor, 'rb') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 256 * 1024:
                raise StackError('Stack configuration must be an ordinary private0600 env file')
        self.command = ['docker', 'compose', '--project-directory', str(ROOT), '--env-file', str(self.env_file)]
        for path in (ROOT / 'compose.yaml', *map(Path, compose_files)):
            self.command += ['-f', str(path.resolve())]
        self.configuration = self._json([*self.command, '--profile', 'setup', '--profile', 'operator',
                                         'config', '--format', 'json'])
        self.project = self.configuration.get('name')
        if not self.project:
            raise StackError('Compose did not resolve one project identity')
        environment = self.configuration['services']['db']['environment']
        passwords = [environment.get(name) for name in ('POSTGRES_PASSWORD',
            'RAG_DB_MIGRATOR_PASSWORD', 'RAG_DB_RUNTIME_PASSWORD')]
        if not all(isinstance(value, str) and value for value in passwords) or len(set(passwords)) != 3:
            raise StackError('Three different private database role secrets are required')
        # These jobs deliberately select their target/role through structured
        # settings. A direct DSN takes precedence in application/operator code
        # and could silently redirect a guarded job or bring owner secrets into
        # the serving container despite apparently safe structured variables.
        for service in self.configuration['services'].values():
            if {'RAG_DATABASE_URL', 'RAG_MIGRATION_DATABASE_URL'} & set(service.get('environment', {})):
                raise StackError('Managed containers require exclusive structured database settings, not direct DSNs')
        app = self.configuration['services']['app']['environment']
        if app.get('RAG_DB_USER') != 'rag_runtime' or app.get('RAG_DB_PASSWORD') != passwords[2]:
            raise StackError('The application must use only the runtime database login')
        if any(key in app for key in ('POSTGRES_PASSWORD', 'RAG_DB_BOOTSTRAP_PASSWORD', 'RAG_DB_MIGRATOR_PASSWORD')):
            raise StackError('Privileged database credentials cannot enter the serving container')
        required = {'/data/corpus', '/data/indexes', '/app/.cache/models', '/state'}
        managed = {item['target'] for item in self.configuration['services']['app'].get('volumes', ())
                   if item.get('type') == 'volume' and item.get('read_only') is True}
        if not required <= managed:
            raise StackError('Serving data must use read-only Linux-managed named volumes')

    @staticmethod
    def _json(command):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise StackError('Docker inspection/configuration failed; inspect local daemon/configuration privately')
        try:
            return json.loads(result.stdout)
        except ValueError:
            raise StackError('Docker returned invalid inspection metadata') from None

    def guard_database(self):
        # Every job is guarded, including root staging jobs which can write or
        # seal corpus/model/state volumes. The source DB must not be aliased as
        # any of those outputs, even when the db service itself is harmless.
        for service in self.configuration['services'].values():
            for item in service.get('volumes', ()):
                if item.get('type') == 'volume':
                    definition = self.configuration.get('volumes', {}).get(item.get('source'), {})
                    if definition.get('name', item.get('source')) in PROTECTED_DATABASE_VOLUMES:
                        raise StackError('The authoritative source database volume is protected; use a separate target')
        database = self.configuration['services']['db']
        if any(database.get(name) for name in ('tmpfs', 'volumes_from', 'configs', 'secrets')):
            raise StackError('Additional database storage mounts are outside the guarded stack')
        if database.get('environment', {}).get('PGDATA', '/var/lib/postgresql/data') != '/var/lib/postgresql/data':
            raise StackError('The guarded database uses only the pinned standard PGDATA path')
        mounts = database.get('volumes', ())
        data, bootstrap = [], []
        for item in mounts:
            if item.get('type') == 'volume':
                definition = self.configuration.get('volumes', {}).get(item.get('source'), {})
                resolved = definition.get('name', item.get('source'))
                if resolved in PROTECTED_DATABASE_VOLUMES:
                    raise StackError('The authoritative source database volume is protected; use a separate target')
                if (item.get('target') != '/var/lib/postgresql/data' or item.get('read_only', False)
                        or not isinstance(resolved, str) or not resolved
                        or definition.get('driver_opts') or definition.get('driver', 'local') != 'local'):
                    raise StackError('The database permits only its standard writable local named data volume')
                data.append(resolved)
            elif (item.get('type') == 'bind' and item.get('read_only') is True
                    and item.get('target') == '/docker-entrypoint-initdb.d/10-roles.sh'
                    and Path(item.get('source', '')).resolve() == (ROOT / 'docker/postgres/init.sh').resolve()):
                bootstrap.append(item)
            else:
                raise StackError('Only the exact read-only role bootstrap bind is allowed beside database data')
        if len(data) != 1 or len(bootstrap) != 1:
            raise StackError('The database requires exactly one data volume and its read-only role bootstrap')
        name = data[0]
        listing = subprocess.run(['docker', 'ps', '--all', '--quiet', '--filter', 'volume=' + name],
                                 capture_output=True, text=True, check=False)
        if listing.returncode:
            raise StackError('Cannot verify exclusive ownership of the target database volume')
        ids = listing.stdout.split()
        if ids:
            running = [item for item in self._json(['docker', 'inspect', *ids]) if item['State']['Running']]
            if len(running) > 1 or any(item.get('Config', {}).get('Labels', {}).get(
                    'com.docker.compose.project') != self.project or item.get('Config', {}).get('Labels', {}).get(
                    'com.docker.compose.service') != 'db' for item in running):
                raise StackError('Another running container owns the target database volume; no second server will start')
        return name

    def _run(self, *arguments, timeout=None):
        self.guard_database()
        try:
            result = subprocess.run([*self.command, *arguments], check=False,
                                    **({'timeout': timeout} if timeout is not None else {}))
        except subprocess.TimeoutExpired:
            raise StackError('Local stack startup exceeded its deadline; owned services may still be running. No volume was removed.') from None
        if result.returncode:
            raise StackError('Local stack step failed; no serving fallback or volume removal was attempted')

    def _job(self, service, *, command=None, database=None):
        args = ['--profile', 'setup', '--profile', 'operator', 'run', '--rm']
        if database is not None:
            if service != 'operator' or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,62}', database):
                raise StackError('A database override is only allowed for an explicit operator job')
            args += ['--env', 'RAG_DB_NAME=' + database]
        args.append(service)
        if command is not None:
            args += command
        self._run(*args)

    def build(self, services=('app', 'operator')):
        if shutil.disk_usage(self.env_file.parent).free < 12 * 1024**3:
            raise StackError('At least12GiB host disk headroom is required before building')
        self._run('--profile', 'operator', 'build', *services)

    def stage(self, *, adopt_existing=False):
        for service in ('stage-artifacts', 'stage-models'):
            command = self.configuration['services'][service]['command']
            self._job(service, command=[*command, '--adopt-existing'] if adopt_existing else None)

    def bootstrap(self, *, adopt_existing=False):
        self.build(('operator',))
        self.stage(adopt_existing=adopt_existing)
        self.build(('app',))
        for service in ('migrate', 'import-corpus', 'import-jira', 'prepare-state'):
            self._job(service)
        self.up()

    def up(self):
        # Re-run the dependency gate rather than reuse an old completed receipt
        # container; every application restart also performs its own live checks.
        self._run('up', '--detach', '--force-recreate', '--wait', 'verify', 'app', timeout=180)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--env-file', type=Path, default=ROOT / '.env')
    result.add_argument('--compose-file', type=Path, action='append', default=[])
    commands = result.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create new private configuration; no Docker or seeded accounts')
    init.add_argument('--artifacts-dir', type=Path, required=True)
    init.add_argument('--models-dir', type=Path, required=True)
    init.add_argument('--source-repo-dir', type=Path, required=True)
    init.add_argument('--operator-dir', type=Path, default=ROOT / '.cache/local-operator')
    init.add_argument('--project', default='redwood')
    init.add_argument('--port', type=int, default=8000)
    for name in ('stage', 'bootstrap'):
        commands.add_parser(name).add_argument('--adopt-existing', action='store_true')
    for name in ('build', 'up', 'stop', 'verify-data', 'migrate', 'import-corpus', 'import-jira', 'prepare-state', 'plan'):
        commands.add_parser(name)
    operator = commands.add_parser('operator')
    operator.add_argument('--database', help='Only this operator job uses the named target; saved/app configuration is unchanged')
    operator.add_argument('arguments', nargs=argparse.REMAINDER)
    return result


def main():
    arguments = parser().parse_args()
    try:
        if arguments.command == 'init':
            print(json.dumps(initialize(arguments.env_file, artifacts_dir=arguments.artifacts_dir,
                models_dir=arguments.models_dir, source_repo_dir=arguments.source_repo_dir,
                operator_dir=arguments.operator_dir, project=arguments.project, port=arguments.port)))
            return
        stack = Stack(arguments.env_file, compose_files=arguments.compose_file)
        if arguments.command == 'plan':
            volume = stack.guard_database()
            print(json.dumps({'project': stack.project, 'database_volume': volume,
                'authentication': 'local', 'accounts_seeded': False, 'privileged_app_credentials': False}))
        elif arguments.command in ('bootstrap', 'stage'):
            getattr(stack, arguments.command)(adopt_existing=arguments.adopt_existing)
        elif arguments.command in ('build', 'up'):
            getattr(stack, arguments.command)()
        elif arguments.command == 'stop':
            stack._run('stop', 'app', 'verify', 'db')
        elif arguments.command == 'verify-data':
            command = stack.configuration['services']['import-corpus']['command']
            stack._job('import-corpus', command=[*command, '--verify-only'])
            stack._job('verify')
        elif arguments.command == 'operator':
            command = arguments.arguments
            if command[:1] == ['--']:
                command = command[1:]
            if not command:
                raise StackError('Provide an explicit operator command; no account is created automatically')
            stack._job('operator', command=command, database=arguments.database)
        else:
            stack._job(arguments.command)
    except (StackError, OSError) as error:
        print(str(error) if isinstance(error, StackError) else 'Local stack operation failed safely', file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
