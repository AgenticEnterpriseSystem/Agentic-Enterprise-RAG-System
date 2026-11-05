"""Explicit three-arm fixed-run CLI; draft/help never load models or credentials.

Declared configuration is never permission or paid-call authorization. The actual
runner authenticates once, shares one bounded hybrid backend, and gives each arm
the same current identity/provider/router. Runtime capture remains authoritative.
"""
import argparse
from contextlib import contextmanager
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from uuid import uuid4


ARMS = ('bm25', 'dense', 'hybrid')
FORMAT = 'rag-fixed-cli-v1'
PLAN_FORMAT = 'rag-fixed-cli-plan-v1'
MAX_CONFIG_BYTES = 2 * 1024 * 1024
SOURCE_ROLES = {'questions', 'splits', 'corpus', 'catalog', 'source_repo'}
WEB_DEFAULTS = {'top_k': 5, 'max_context_chars': 24000, 'max_tokens': 16000,
    'max_seconds': 120, 'max_rounds': 2, 'question_budget_usd': .1, 'allowed_sources': []}
RETRIEVAL_KEYS = {'bm25_index', 'dense_index', 'device', 'fp16', 'dense_block_rows',
    'dense_file_cache_policy', 'bm25_weight', 'candidate_k', 'rerank_k', 'rrf_k'}
GENERATION_KEYS = {'provider', 'model', 'budget_usd', 'input_price_per_million',
    'output_price_per_million', 'max_output_tokens', 'timeout_seconds'}


class FixedCLIError(ValueError):
    """A bounded diagnostic containing no credential, question, or source text."""


def _need(value, message):
    if not value:
        raise FixedCLIError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')


def _finite(value):
    if isinstance(value, float):
        _need(math.isfinite(value), 'Configuration numbers must be finite')
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, 'Duplicate JSON fields are forbidden')
            result[key] = value
        return result
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
        | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(descriptor, 'rb') as handle:
        info = os.fstat(handle.fileno())
        _need(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_CONFIG_BYTES,
            'Expected a bounded ordinary JSON input file')
        raw = handle.read(MAX_CONFIG_BYTES + 1)
    _need(len(raw) <= MAX_CONFIG_BYTES, 'JSON input exceeds its size bound')
    value = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(FixedCLIError('Nonfinite JSON is forbidden')))
    _finite(value)
    return value


def _object(value, allowed, *, required=(), message='Unexpected configuration fields'):
    _need(type(value) is dict and set(required) <= value.keys() <= allowed, message)
    return value


def _path(value, base):
    _need(isinstance(value, str) and value.strip(), 'Configured paths must be nonblank strings')
    result = Path(value).expanduser()
    return str((result if result.is_absolute() else base / result).resolve())


def _number(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def load_config(filename, *, output_root=None):
    """Resolve a strict declaration without importing runtime or reading secrets."""
    filename = Path(filename).resolve()
    config = _read_json(filename)
    _object(config, {'format', 'sources', 'authentication', 'retrieval', 'web', 'generation',
        'selection', 'output_root', 'comparison', 'request_filters'},
        required={'format', 'sources', 'authentication', 'retrieval', 'generation', 'output_root'})
    _need(config['format'] == FORMAT, 'Unsupported fixed-run CLI configuration format')
    base = filename.parent
    sources = _object(config['sources'], SOURCE_ROLES, required=SOURCE_ROLES)
    sources = {name: _path(value, base) for name, value in sources.items()}
    auth = _object(config['authentication'], {'snapshot_file', 'credentials_file', 'principal_kind',
        'allowed_doc_ids_file'}, required={'snapshot_file', 'credentials_file'})
    auth = {**auth, 'principal_kind': auth.get('principal_kind', 'benchmark')}
    _need(auth['principal_kind'] in ('benchmark', 'session'), 'Unknown authenticated principal kind')
    for name in ('snapshot_file', 'credentials_file', 'allowed_doc_ids_file'):
        if name in ('snapshot_file', 'credentials_file') or auth.get(name) is not None:
            auth[name] = _path(auth[name], base)
    retrieval = dict(_object(config['retrieval'], RETRIEVAL_KEYS,
        required={'bm25_index', 'dense_index'}))
    for name in ('bm25_index', 'dense_index'):
        retrieval[name] = _path(retrieval[name], base)
    for name, upper in (('dense_block_rows', None), ('candidate_k', 100), ('rerank_k', 200)):
        if name in retrieval:
            _need(type(retrieval[name]) is int and retrieval[name] >= 1
                and (upper is None or retrieval[name] <= upper), 'Invalid bounded retrieval setting')
    _need(retrieval.get('dense_file_cache_policy', 'retain') in ('retain', 'discard'),
        'Unknown dense file-cache policy')
    _need(retrieval.get('fp16') is None or type(retrieval['fp16']) is bool, 'fp16 must be boolean or null')
    if 'device' in retrieval:
        _need(isinstance(retrieval['device'], str) and retrieval['device'].strip(), 'Device must be explicit text')
    _need(_number(retrieval.get('bm25_weight', .7)) and retrieval.get('bm25_weight', .7) <= 1
        and _number(retrieval.get('rrf_k', 60)), 'Invalid retrieval weights')
    web = {**WEB_DEFAULTS, **_object(config.get('web', {}), set(WEB_DEFAULTS))}
    for name, lower, upper in (('top_k', 1, 20), ('max_context_chars', 128, 48000),
            ('max_tokens', 256, 64000), ('max_rounds', 1, 5)):
        _need(type(web[name]) is int and lower <= web[name] <= upper, 'Invalid bounded web setting')
    _need(_number(web['max_seconds'], positive=True) and web['max_seconds'] <= 300
        and _number(web['question_budget_usd'], positive=True), 'Invalid execution budget')
    _need(type(web['allowed_sources']) is list and all(isinstance(item, str) and item.strip()
        for item in web['allowed_sources']) and len(set(web['allowed_sources'])) == len(web['allowed_sources']),
        'Allowed source narrowing must contain unique nonblank strings')
    generation = dict(_object(config['generation'], GENERATION_KEYS, required={'provider'}))
    _need(generation['provider'] in ('disabled', 'anthropic'), 'CLI supports disabled or explicit Anthropic generation')
    if generation['provider'] == 'disabled':
        _need(set(generation) == {'provider'}, 'Disabled generation cannot declare an active model or prices')
        generation = {'provider': 'disabled', 'model': 'retrieval-only', 'budget_usd': 0,
            'input_price_per_million': 0, 'output_price_per_million': 0,
            'max_output_tokens': 1024, 'timeout_seconds': min(30, web['max_seconds'])}
    else:
        _need(isinstance(generation.get('model'), str) and generation['model'].strip(),
            'Live generation requires an explicit model')
        _need(all(_number(generation.get(name), positive=True) for name in
            ('budget_usd', 'input_price_per_million', 'output_price_per_million')),
            'Live generation requires positive declared cap and token prices')
        generation.setdefault('max_output_tokens', 1024)
        generation.setdefault('timeout_seconds', min(30, web['max_seconds']))
        _need(type(generation['max_output_tokens']) is int and generation['max_output_tokens'] > 0
            and _number(generation['timeout_seconds'], positive=True), 'Invalid generation output/time limit')
        generation['timeout_seconds'] = min(generation['timeout_seconds'], web['max_seconds'])
    selection = {**{'split': 'development', 'repetition': 1, 'allow_final': False},
        **_object(config.get('selection', {}), {'split', 'question_ids', 'count', 'repetition',
            'allow_final', 'final_freeze'})}
    _need(selection['split'] in ('development', 'calibration', 'final') and type(selection['allow_final']) is bool,
        'Invalid split selection')
    _need(type(selection['repetition']) is int and selection['repetition'] > 0, 'Repetition must be positive')
    _need(not (selection.get('question_ids') is not None and selection.get('count') is not None),
        'Choose question IDs or count, not both')
    if selection.get('question_ids') is not None:
        ids = selection['question_ids']
        _need(type(ids) is list and ids and all(isinstance(item, str) and item.strip() for item in ids)
            and len(set(ids)) == len(ids), 'Question IDs must be unique nonblank strings')
    if selection.get('count') is not None:
        _need(type(selection['count']) is int and selection['count'] > 0, 'Selection count must be positive')
    if selection.get('final_freeze') is not None:
        selection['final_freeze'] = _path(selection['final_freeze'], base)
    if selection['split'] == 'final':
        _need(selection['allow_final'] and selection.get('final_freeze'), 'Final selection requires an explicit accepted freeze')
    request = dict(_object(config.get('request_filters', {}), {'source_types', 'after', 'before'}))
    requested = request.get('source_types', [])
    _need(type(requested) is list and all(isinstance(item, str) and item.strip() for item in requested)
        and len(set(requested)) == len(requested), 'Invalid request source narrowing')
    if web['allowed_sources']:
        _need(set(requested) <= set(web['allowed_sources']), 'Request sources exceed declared operator narrowing')
    dates = {}
    for name in ('after', 'before'):
        if request.get(name) is not None:
            _need(isinstance(request[name], str), 'Request dates must be ISO timestamp strings')
            dates[name] = datetime.fromisoformat(request[name])
            _need(dates[name].tzinfo is not None and dates[name].utcoffset() is not None,
                'Request dates must include a timezone')
    _need(not dates.get('after') or not dates.get('before') or dates['after'] <= dates['before'],
        'Request date bounds are reversed')
    comparison = dict(_object(config.get('comparison', {}), set(config.get('comparison', {}))))
    _need(not {'generation', 'web', 'request_filters'} & comparison.keys(),
        'Managed comparison fields cannot be overridden')
    comparison.update(generation=generation, web=web, request_filters=request)
    target = str(Path(output_root).expanduser().resolve()) if output_root else _path(config['output_root'], base)
    return {'format': FORMAT, 'sources': sources, 'authentication': auth, 'retrieval': retrieval,
        'web': web, 'generation': generation, 'selection': selection, 'output_root': target,
        'comparison': comparison, 'request_filters': request}


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(_canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def _authorization(config, command, *, allow_paid=False, max_total_cost_usd=0):
    _need(_number(max_total_cost_usd), 'Explicit total cost cap must be finite and nonnegative')
    if command not in ('run', 'resume'):
        return False
    if config['generation']['provider'] == 'disabled':
        _need(not allow_paid and max_total_cost_usd == 0, 'Disabled validation does not accept paid authorization')
        return False
    _need(allow_paid is True and max_total_cost_usd > 0,
        'Live execution requires --allow-paid and a positive --max-total-cost-usd')
    _need(max_total_cost_usd == config['generation']['budget_usd'],
        'Explicit total cap must equal the immutable planned generation budget')
    _need(bool(os.environ.get('ANTHROPIC_API_KEY')), 'Authorized live execution requires its API key in the environment')
    return True


def draft(config, output):
    """File-only selection. No provider, inference, authentication, or DB imports."""
    from src.evaluation.experiment_inputs import select_runtime_questions, question_set_sha256
    selection = config['selection']
    questions = select_runtime_questions(Path(config['sources']['questions']), Path(config['sources']['splits']),
        selection['split'], question_ids=selection.get('question_ids'), count=selection.get('count'),
        allow_final=selection['allow_final'])
    value = {'format': 'rag-fixed-cli-draft-v1', 'status': 'declared_selection_only',
        'runtime_authority_verified': False, 'scope': None, 'answers_generated': False,
        'paid_model_calls': 0, 'arms': list(ARMS), 'split': selection['split'],
        'question_ids': [item.question_id for item in questions], 'question_set_sha256': question_set_sha256(questions),
        'planned_generation': config['generation'], 'output_root': config['output_root'],
        'result_label': 'validation_only' if config['generation']['provider'] == 'disabled' else 'generated_baselines_pending_approved_budget'}
    _write_new(output, value)
    return {'passed': True, 'command': 'draft', 'scope_verified': False,
        'questions': len(questions), 'paid_model_calls': 0, 'result_label': value['result_label']}


@contextmanager
def _runtime_suite(config, *, paid):
    """One loaded hybrid pair/catalog/encoder/pool and one shared provider/router."""
    from src.config import RetrievalSettings
    from src.api.runtime import AuthenticatedRuntime, WebSettings
    from src.evaluation.scoped_retrieval import ScopedRetrievalRunner
    from src.storage.contracts import ReadFilters
    generation, auth = config['generation'], config['authentication']
    retrieval = RetrievalSettings(**config['retrieval'], retrieval='hybrid',
        corpus=Path(config['sources']['corpus']), snapshot_file=Path(auth['snapshot_file']))
    live = generation['provider'] == 'anthropic'
    web = WebSettings(**retrieval.model_dump(), **config['web'], auth_mode='local',
        mode='live' if live else 'retrieval', allow_paid=paid, model=generation['model'],
        budget_usd=generation['budget_usd'], input_price=generation['input_price_per_million'],
        output_price=generation['output_price_per_million'],
        allowed_doc_ids_file=Path(auth['allowed_doc_ids_file']) if auth.get('allowed_doc_ids_file') else None)
    provider, router = None, None
    if live:
        from src.generation.generator import AnthropicConfig, AnthropicProvider
        from src.agents.routing import ModelRouter
        provider = AnthropicProvider(AnthropicConfig(model=generation['model'], allow_paid=paid,
            budget_usd=generation['budget_usd'], input_usd_per_million=generation['input_price_per_million'],
            output_usd_per_million=generation['output_price_per_million'],
            max_output_tokens=generation['max_output_tokens'], timeout_seconds=generation['timeout_seconds']))
        router = ModelRouter(provider)  # Preserve actual live routing, including fixed runs.
    with ScopedRetrievalRunner.open(retrieval, snapshot_file=Path(auth['snapshot_file']),
            credentials_file=Path(auth['credentials_file']), principal_kind=auth['principal_kind'],
            doc_ids_file=Path(auth['allowed_doc_ids_file']) if auth.get('allowed_doc_ids_file') else None,
            filters=ReadFilters(source_types=tuple(config['web']['allowed_sources']))) as runner:
        wrappers = {}
        @contextmanager
        def open_runtime(arm):
            name = arm if isinstance(arm, str) else arm.name
            _need(name in ARMS, 'Unknown fixed-run arm')
            if name not in wrappers:
                wrappers[name] = AuthenticatedRuntime(web, runner.adapter(name), runner.resolver,
                    database=runner.database, provider=provider, router=router,
                    provider_max_output_tokens=generation['max_output_tokens'])
            # Resolve immediately before each case, not once for a whole arm.
            # The producer still compares the original frozen scope binding.
            yield wrappers[name], lambda: runner.context(touch=True).identity
        yield open_runtime


def _config_digest(config):
    return hashlib.sha256(_canonical(config)).hexdigest()


def _prepare(config, open_runtime):
    from src.evaluation.fixed_runs import prepare_fixed_arm
    target, selection = Path(config['output_root']), config['selection']
    _need(not (target / 'plan.json').exists(), 'A frozen plan already exists; resume it or choose a new output root')
    arms = []
    for name in ARMS:
        with open_runtime(name) as (actual, identity):
            arms.append(prepare_fixed_arm(name, actual, identity,
                sources={key: Path(value) for key, value in config['sources'].items()},
                prepared=target / 'prepared' / name, directory=target / 'runs' / name,
                comparison=config['comparison'], split=selection['split'],
                question_ids=selection.get('question_ids'), count=selection.get('count'),
                repetition=selection['repetition'], request_filters=config['request_filters'],
                allow_final=selection['allow_final'],
                final_freeze=Path(selection['final_freeze']) if selection.get('final_freeze') else None))
    _write_new(target / 'plan.json', {'format': PLAN_FORMAT, 'config_sha256': _config_digest(config),
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'principal_kind': config['authentication']['principal_kind'],
        'arms': [{'name': arm.name, 'prepared': str((target / 'prepared' / arm.name).resolve()),
            'directory': str(arm.directory.resolve())} for arm in arms]})
    return tuple(arms)


def _resume_arms(config):
    from src.evaluation.experiment_contracts import ExperimentManifest
    from src.evaluation.experiment_inputs import InputFiles, verify_inputs
    from src.evaluation.fixed_runs import FixedArm
    target = Path(config['output_root'])
    plan = _read_json(target / 'plan.json')
    _need(plan.get('format') == PLAN_FORMAT and plan.get('config_sha256') == _config_digest(config),
        'Resume configuration differs from the frozen CLI plan')
    _need(type(plan.get('arms')) is list and [item.get('name') for item in plan['arms']] == list(ARMS),
        'Frozen plan must contain the three exact ordered arms')
    arms = []
    roles = {item.name for item in fields(InputFiles)}
    for item in plan['arms']:
        name = item['name']
        prepared, directory = target / 'prepared' / name, target / 'runs' / name
        _need(item.get('prepared') == str(prepared.resolve()) and item.get('directory') == str(directory.resolve()),
            'Frozen arm paths differ from the original output root')
        manifest = ExperimentManifest.model_validate(_read_json(prepared / 'manifest.json'))
        mapping = _read_json(prepared / 'input-files.json')
        _need(type(mapping) is dict and set(mapping) == roles, 'Persisted input-file roles differ')
        _need(all(value is None or (isinstance(value, str) and Path(value).is_absolute()) for value in mapping.values()),
            'Persisted input paths must remain absolute')
        files = InputFiles(**{key: Path(value) if value is not None else None for key, value in mapping.items()})
        arms.append(FixedArm(name, manifest, verify_inputs(manifest, files), directory))
    return tuple(arms)


def execute(config, command, *, allow_paid=False, max_total_cost_usd=0):
    # This gate intentionally precedes every runtime/model/DB/SDK import.
    paid = _authorization(config, command, allow_paid=allow_paid, max_total_cost_usd=max_total_cost_usd)
    from src.evaluation.fixed_runs import preflight_fixed_runs, run_fixed_runs
    execution = None
    with _runtime_suite(config, paid=paid) as open_runtime:
        arms = _resume_arms(config) if command == 'resume' else _prepare(config, open_runtime)
        preflight_fixed_runs(arms, open_runtime=open_runtime)
        if command in ('run', 'resume'):
            execution = run_fixed_runs(arms, open_runtime=open_runtime, journal_dir=Path(config['output_root']) / 'attempts',
                resume=command == 'resume', max_total_cost_usd=max_total_cost_usd, allow_paid=paid)
    result = {'passed': True, 'command': command, 'arms': len(arms),
        'questions_per_arm': len(arms[0].manifest.question_ids), 'scope_verified': True,
        'result_label': 'validation_only' if config['generation']['provider'] == 'disabled'
            else 'generated_baselines_pending_approved_budget' if command == 'preflight' else 'generated_baseline_artifacts'}
    if command == 'preflight' or not paid:
        result['paid_model_calls'] = 0
    if execution is not None:
        # Failed model outcomes remain completed cases; expose their counts and
        # conservative reservations rather than hiding them behind a label.
        result['execution'] = execution
    _write_new(Path(config['output_root']) / 'commands' / (uuid4().hex + '.json'), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('draft', 'preflight', 'run', 'resume'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True, type=Path)
        command.add_argument('--output-root', type=Path,
            help='Explicit separate plan/output root; recorded in the frozen configuration')
        if name == 'draft':
            command.add_argument('--output', required=True, type=Path, help='New file-only unbound selection draft')
        if name in ('run', 'resume'):
            command.add_argument('--allow-paid', action='store_true', help='Explicit live-call authorization; never read from environment')
            command.add_argument('--max-total-cost-usd', type=float, default=0,
                help='Explicit whole-suite cap; must equal the declared live plan cap (default0)')
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        config = load_config(args.config, output_root=args.output_root)
        result = draft(config, args.output) if args.command == 'draft' else execute(config, args.command,
            allow_paid=getattr(args, 'allow_paid', False), max_total_cost_usd=getattr(args, 'max_total_cost_usd', 0))
    except Exception as error:
        # Driver/SDK/Pydantic exception representations may contain private values.
        message = str(error) if type(error) is FixedCLIError else 'Fixed-run command did not complete; inputs or runtime checks failed'
        print(json.dumps({'passed': False, 'command': args.command, 'error_kind': type(error).__name__, 'message': message}))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
