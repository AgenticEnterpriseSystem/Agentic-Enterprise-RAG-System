"""Scope-bound fixed-run production, with durable attempts and no implicit spend.

Runtime factories are context managers so callers can share validated indexes or
release one arm before opening another. Only runtime questions enter inference;
scoring references stay in their original input file. The journal is separate
from the deliberately strict immutable RunDirectory format.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import json
import math
import os
from pathlib import Path
import stat
from uuid import uuid4

from src.contracts import QueryRequest, ScopeBinding
from src.evaluation.experiment_contracts import (
    ExperimentManifest, runtime_case_digest, validate_resume_compatibility,
)
from src.evaluation.experiment_inputs import (
    ExperimentAdmission, InputFiles, VerifiedInputs, canonical_json_sha256,
    capture_code_provenance, file_sha256, question_set_sha256,
    select_runtime_questions, verify_inputs,
)
from src.evaluation.experiment_io import RunDirectory, _output_location

_METHODS = {'bm25': 'store-bm25', 'dense': 'store-dense', 'hybrid': 'store-hybrid'}
_PROFILE_FIELDS = ('generation_model', 'budgets', 'temperature', 'workflow', 'max_rounds',
    'generator_prompt_sha256', 'critic_prompt_sha256', 'router_prompt_sha256', 'retrieval')
_SHARED_FIELDS = ('schema_version', 'experiment_id', 'repetition', 'code', 'split',
    'split_manifest_sha256', 'question_set_sha256', 'scope', 'scope_fingerprint_sha256',
    'generation_model', 'generator_prompt_sha256', 'critic_prompt_sha256',
    'router_prompt_sha256', 'comparison_config_sha256', 'budgets', 'temperature', 'max_rounds')


class FixedRunError(ValueError):
    """Invalid, incompatible, or unsafe producer inputs."""


class UnresolvedAttempt(FixedRunError):
    """A prior attempt needs operator review; automatic execution is denied."""


class FixedBudgetError(FixedRunError):
    """The explicit cross-arm conservative reservation budget is unavailable."""


@dataclass(frozen=True)
class FixedArm:
    name: str
    manifest: ExperimentManifest
    inputs: VerifiedInputs
    directory: Path

    def __post_init__(self):
        if self.name not in _METHODS or type(self.inputs) is not VerifiedInputs:
            raise FixedRunError('A named fixed arm and verified inputs are required')
        manifest = ExperimentManifest.model_validate(self.manifest.model_dump(warnings=False))
        if manifest.workflow != 'fixed' or manifest.retrieval.retriever_id != _METHODS[self.name]:
            raise FixedRunError('Fixed arm name differs from its actual retrieval profile')
        object.__setattr__(self, 'manifest', manifest)
        object.__setattr__(self, 'directory', Path(self.directory).absolute())


def _bytes(value):
    # Exercise the same finite-JSON validation as scientific input hashing.
    canonical_json_sha256(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path, raw):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _sync(path.parent)


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FixedRunError('Duplicate JSON field')
            result[key] = value
        return result
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(descriptor, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
            raise FixedRunError('Expected a bounded regular producer JSON file')
        value = json.loads(handle.read(), object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(FixedRunError('Nonfinite JSON')))
    _bytes(value)
    return value


def _amount(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise FixedBudgetError('Cost caps must be finite nonnegative numbers')
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise FixedBudgetError('Invalid cost cap') from None
    if not result.is_finite() or result < 0 or not math.isfinite(float(result)):
        raise FixedBudgetError('Cost caps must be finite nonnegative numbers')
    return result


def _filters(value):
    if not isinstance(value, dict) or not set(value) <= {'source_types', 'after', 'before'}:
        raise FixedRunError('Only source/date request filters are supported')
    # QueryRequest performs the actual strict date/platform shape validation.
    return QueryRequest(question='Producer filter validation', **value).model_dump(
        mode='json', exclude={'question', 'mode'})


def _body(arm, question):
    config = _json(arm.inputs.files.retrieval_config)
    return QueryRequest(question=question.question, mode='baseline',
        **_filters(config.get('filters', {})))


def _resolved_identity(identity):
    # The real runtime still requires a genuine ResolvedIdentity; never coerce
    # client claims into one. Callbacks keep an active human session fresh.
    return identity() if callable(identity) else identity


def prepare_fixed_arm(name, actual, identity, *, sources, prepared, directory, comparison,
        split='development', question_ids=None, count=None, repetition=1,
        request_filters=None, allow_final=False, final_freeze=None):
    """Persist actual non-generating inputs, leaving source corpus/catalog in place."""
    if name not in _METHODS or not isinstance(sources, dict) or set(sources) != {
            'questions', 'splits', 'corpus', 'catalog', 'source_repo'}:
        raise FixedRunError('Five exact source roles and a supported arm are required')
    if any(not isinstance(path, Path) for path in sources.values()):
        raise FixedRunError('Source roles must be pathlib.Path values')
    if type(repetition) is not int or repetition < 1 or not isinstance(comparison, dict):
        raise FixedRunError('Explicit repetition and shared comparison settings are required')
    _bytes(comparison)
    questions = select_runtime_questions(sources['questions'], sources['splits'], split,
        question_ids=question_ids, count=count, allow_final=allow_final)
    if any(question.question != question.question.strip() for question in questions):
        raise FixedRunError('Frozen runtime questions must have no edge whitespace')
    filters = _filters(request_filters or {})
    profile = actual.experiment_profile(QueryRequest(
        question=questions[0].question, mode='baseline', **filters), _resolved_identity(identity))
    if profile['retrieval']['retriever_id'] != _METHODS[name] or profile['workflow'] != 'fixed':
        raise FixedRunError('Prepared arm differs from the actual runtime')
    prepared = _output_location(prepared, sources['source_repo'])
    _output_location(directory, sources['source_repo'])
    if prepared.is_symlink() or prepared.exists():
        raise FixedRunError('Prepared inputs require a fresh directory')
    prepared.mkdir(parents=True, mode=0o700)
    from src.generation.generator import SYSTEM_PROMPT
    from src.agents.routing import ROUTER_PROMPT
    scope = {'binding': profile['scope'], 'origin': 'operator',
             'purpose': 'scope-bound fixed-baseline producer'}
    values = {
        'index-manifest.json': profile['index_manifest'],
        'retrieval-config.json': profile['retrieval_config'],
        'comparison-config.json': comparison, 'scope.json': scope,
        'runtime-config.json': {key: profile[key] for key in (
            'generation_model', 'retrieval', 'budgets', 'temperature', 'workflow', 'max_rounds')},
    }
    for filename, value in values.items():
        # The index digest is of exact canonical JSON, with no extra newline.
        _write_new(prepared / filename, _bytes(value))
    _write_new(prepared / 'generator.txt', SYSTEM_PROMPT.encode('utf-8'))
    router = prepared / 'router.txt' if profile['router_prompt_sha256'] is not None else None
    if router is not None:
        _write_new(router, ROUTER_PROMPT.encode('utf-8'))
    code = capture_code_provenance(sources['source_repo'])
    experiment_id = comparison.get('experiment_id', 'fixed-' + canonical_json_sha256(comparison)[:20])
    manifest = ExperimentManifest(experiment_id=experiment_id,
        manifest_id=f'{experiment_id}-{name}-r{repetition}', repetition=repetition,
        created_at=datetime.now(timezone.utc), code=code, split=split,
        split_manifest_sha256=file_sha256(sources['splits']),
        question_set_sha256=question_set_sha256(questions), question_ids=tuple(q.question_id for q in questions),
        scope=ScopeBinding.model_validate(profile['scope']), scope_fingerprint_sha256=canonical_json_sha256(scope),
        comparison_config_sha256=canonical_json_sha256(comparison),
        **{key: profile[key] for key in _PROFILE_FIELDS})
    files = InputFiles(**sources, index_manifest=prepared/'index-manifest.json',
        retrieval_config=prepared/'retrieval-config.json', comparison_config=prepared/'comparison-config.json',
        scope=prepared/'scope.json', runtime_config=prepared/'runtime-config.json',
        generator_prompt=prepared/'generator.txt', router_prompt=router,
        final_freeze=Path(final_freeze) if final_freeze is not None else None)
    verified = verify_inputs(manifest, files)
    for relative, expected in profile['source_files'].items():
        if file_sha256(files.source_repo / relative) != expected:
            raise FixedRunError('Prepared source differs from the loaded runtime')
    _write_new(prepared/'manifest.json', _bytes(manifest.model_dump(mode='json')))
    _write_new(prepared/'input-files.json', _bytes({key: str(value) if value is not None else None
        for key, value in asdict(files).items()}))
    return FixedArm(name, manifest, verified, directory)


def _checked_arms(arms):
    arms = tuple(arms)
    if not arms or any(type(arm) is not FixedArm for arm in arms):
        raise FixedRunError('At least one typed fixed arm is required')
    for role in ('name', 'directory'):
        if len({getattr(arm, role) for arm in arms}) != len(arms):
            raise FixedRunError('Duplicate arm name or output directory')
    if len({arm.manifest.manifest_id for arm in arms}) != len(arms):
        raise FixedRunError('Every arm needs its own immutable manifest identity')
    for arm in arms:
        checked = arm.inputs.recheck(arm.manifest)
        if any(q.question != q.question.strip() for q in checked.questions):
            raise FixedRunError('All selected questions must match their exact runtime text')
        _output_location(arm.directory, arm.inputs.files.source_repo)
        if arm.directory.is_symlink():
            raise FixedRunError('Run output cannot be a symlink')
        for other in arms:
            if other is not arm and (arm.directory.resolve().is_relative_to(other.directory.resolve())
                    or other.directory.resolve().is_relative_to(arm.directory.resolve())):
                raise FixedRunError('Arm output directories cannot overlap')
        if arm.directory.exists():
            old = RunDirectory.load(arm.directory)
            validate_resume_compatibility(old.manifest, arm.manifest)
    first = arms[0].manifest
    for arm in arms[1:]:
        if (any(getattr(first, key) != getattr(arm.manifest, key) for key in _SHARED_FIELDS)
                or set(first.question_ids) != set(arm.manifest.question_ids)):
            raise FixedRunError('Fixed arms differ in shared scientific settings')
    return arms


def _check_generation(arm, actual, profile):
    declared = _json(arm.inputs.files.comparison_config).get('generation')
    if declared is None:
        if arm.manifest.generation_model.provider == 'anthropic':
            raise FixedRunError('Real providers require frozen managed generation prices and limits')
        return None
    expected_keys = {'provider', 'model', 'budget_usd', 'input_price_per_million',
        'output_price_per_million', 'max_output_tokens', 'timeout_seconds'}
    if not isinstance(declared, dict) or set(declared) != expected_keys:
        raise FixedRunError('Managed generation settings have an invalid shape')
    for key in ('budget_usd', 'input_price_per_million', 'output_price_per_million', 'timeout_seconds'):
        _amount(declared[key])
    if type(declared['max_output_tokens']) is not int or declared['max_output_tokens'] < 1:
        raise FixedRunError('Managed output limit must be a positive integer')
    model = profile['generation_model']
    if declared['provider'] != model['provider'] or declared['model'] != model['model_id']:
        raise FixedRunError('Managed model differs from the actual runtime')
    if declared['max_output_tokens'] != profile['budgets']['max_output_tokens']:
        raise FixedRunError('Managed output budget differs from the runtime')
    if model['provider'] == 'anthropic':
        config = actual.provider.config
        expected = {'budget_usd': config.budget_usd,
            'input_price_per_million': config.input_usd_per_million,
            'output_price_per_million': config.output_usd_per_million,
            'timeout_seconds': config.timeout_seconds}
        if any(declared[key] != value for key, value in expected.items()):
            raise FixedRunError('Managed provider prices or limits differ from the runtime')
    elif any(_amount(declared[key]) != 0 for key in (
            'budget_usd', 'input_price_per_million', 'output_price_per_million')):
        raise FixedRunError('Non-generating managed settings must declare zero spending')
    return declared


def _runtime_profile(arm, actual, identity):
    checked = arm.inputs.recheck(arm.manifest)
    profile = actual.experiment_profile(_body(arm, checked.questions[0]), _resolved_identity(identity))
    declared = arm.manifest.model_dump(mode='json')
    if profile['scope'] != declared['scope'] or any(profile[key] != declared[key] for key in _PROFILE_FIELDS):
        raise FixedRunError('Actual runtime scope/profile differs from the frozen arm')
    for relative, digest in profile['source_files'].items():
        if file_sha256(checked.files.source_repo / relative) != digest:
            raise FixedRunError('Actual runtime source differs from the frozen checkout')
    profile['managed_generation'] = _check_generation(arm, actual, profile)
    return profile


def _matched_profiles(profiles):
    values = tuple(profiles.values())
    first = values[0]
    for value in values[1:]:
        for key in ('source_files', 'managed_generation'):
            if first[key] != value[key]:
                raise FixedRunError('Arm runtime sources or provider limits differ')
        for key in ('router', 'filters'):
            if first['retrieval_config'][key] != value['retrieval_config'][key]:
                raise FixedRunError('Arm routing or admitted filters differ')
    if 'hybrid' in profiles and 'bm25' in profiles:
        if profiles['hybrid']['index_manifest']['bm25'] != profiles['bm25']['index_manifest']:
            raise FixedRunError('BM25 and hybrid use different lexical component manifests')
    if 'hybrid' in profiles and 'dense' in profiles:
        hybrid, dense = profiles['hybrid'], profiles['dense']
        if (hybrid['index_manifest']['dense'] != dense['index_manifest']
                or hybrid['retrieval']['embedding_model'] != dense['retrieval']['embedding_model']
                or hybrid['retrieval_config']['dense_execution'] != dense['retrieval_config']['dense_execution']):
            raise FixedRunError('Dense and hybrid execution/model/component settings differ')


def preflight_fixed_runs(arms, *, open_runtime):
    """Inspect every arm and exact input before any inference or run creation."""
    arms = _checked_arms(arms)
    profiles = {}
    for arm in arms:
        with open_runtime(arm) as (actual, identity):
            profiles[arm.name] = _runtime_profile(arm, actual, identity)
    _matched_profiles(profiles)
    return {'passed': True, 'arm_order': [arm.name for arm in arms],
        'matched_configuration': set(profiles) == set(_METHODS), 'matched_complete': False,
        'profiles': profiles, 'provider_calls': 0}


def _policy(arms, cap, allow_paid):
    if type(allow_paid) is not bool:
        raise FixedBudgetError('Paid execution requires an explicit boolean setting')
    for arm in arms:
        provider = arm.manifest.generation_model.provider
        maximum = _amount(arm.manifest.budgets.max_run_cost_usd)
        declared = _json(arm.inputs.files.comparison_config).get('generation')
        if declared is not None and _amount(declared['budget_usd']) != cap:
            raise FixedBudgetError('Execution cap differs from the frozen managed plan')
        if provider in {'disabled', 'fixture'}:
            continue
        elif provider != 'anthropic' or not allow_paid or maximum <= 0 or cap <= 0:
            raise FixedBudgetError('Real generation requires explicit paid approval and positive bounded caps')


def _reservation(arm):
    return (Decimal(0) if arm.manifest.generation_model.provider in {'fixture', 'disabled'}
            else _amount(arm.manifest.budgets.max_run_cost_usd))


def _plan(arms, cap, allow_paid):
    return {'cap_usd': format(cap.normalize(), 'f'), 'allow_paid': allow_paid, 'arms': [
        {'name': arm.name, 'manifest': arm.manifest.model_dump(mode='json'),
         'input_hashes': dict(arm.inputs.hashes),
         'input_files': {key: str(value) if value is not None else None
                        for key, value in asdict(arm.inputs.files).items()},
         'directory': str(arm.directory)} for arm in arms]}


class _Journal:
    def __init__(self, path, plan, *, resume):
        self.path, self.plan, self.resume = Path(path).absolute(), plan, resume
        self.fd = None

    def __enter__(self):
        if self.path.is_symlink():
            raise FixedRunError('Attempt journal cannot be a symlink')
        if self.resume:
            if not self.path.is_dir() or not (self.path/'state.json').is_file():
                raise UnresolvedAttempt('Resume requires the original attempt journal; never reset its spending state')
        else:
            self.path.mkdir(parents=True, mode=0o700)
        self.fd = os.open(self.path/'.journal.lock', os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(self.fd).st_mode):
                raise FixedRunError('Invalid journal lock')
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.resume:
                self.state = _json(self.path/'state.json')
                if set(self.state) != {'schema_version', 'plan', 'reserved_usd', 'attempts'} or (
                        self.state['schema_version'] != 1 or self.state['plan'] != self.plan):
                    raise FixedRunError('Attempt journal differs from the frozen plan or cap')
            else:
                self.state = {'schema_version': 1, 'plan': self.plan, 'reserved_usd': '0', 'attempts': []}
                _write_new(self.path/'state.json', _bytes(self.state))
            if any(p.name not in {'state.json', '.journal.lock'} for p in self.path.iterdir()):
                raise UnresolvedAttempt('Unexpected journal files require operator review')
            if not isinstance(self.state['attempts'], list):
                raise FixedRunError('Invalid attempt ledger')
            total = sum((_amount(item['reservation_usd']) for item in self.state['attempts']), Decimal(0))
            if total != _amount(self.state['reserved_usd']) or total > _amount(self.plan['cap_usd']):
                raise FixedBudgetError('Attempt reservations do not match the durable cap')
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_error):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _save(self):
        temporary = self.path/('.tmp-' + uuid4().hex)
        _write_new(temporary, _bytes(self.state))
        os.replace(temporary, self.path/'state.json')
        _sync(self.path)

    def reserve(self, arm, question_id):
        if any(item['arm'] == arm.name and item['question_id'] == question_id for item in self.state['attempts']):
            raise UnresolvedAttempt('Question already has a durable attempt; automatic rerun denied')
        reservation = _reservation(arm)
        total = _amount(self.state['reserved_usd']) + reservation
        if total > _amount(self.plan['cap_usd']):
            raise FixedBudgetError('Cross-arm/restart reservation cap reached before dispatch')
        item = {'arm': arm.name, 'question_id': question_id, 'state': 'intent',
                'reservation_usd': str(reservation)}
        self.state['reserved_usd'] = str(total)
        self.state['attempts'].append(item)
        self._save()
        return item

    def captured(self, item, run_id):
        item.update(state='captured', run_id=run_id)
        self._save()

    def committed(self, item, case):
        # Only the caller's normal return from export_capture authorizes this.
        item.update(state='committed', run_id=case.run_id, case_sha256=runtime_case_digest(case),
            api_calls=case.response.trace.usage.api_calls,
            reported_cost_usd=case.response.trace.usage.estimated_cost_usd)
        self._save()


def _review_journal(journal, arms, runs):
    cases = {(name, case.question_id): case for name, run in runs.items() for case in run.load_cases()}
    seen = set()
    specifications = {arm.name: arm for arm in arms}
    for item in journal.state['attempts']:
        key = (item.get('arm'), item.get('question_id'))
        if key in seen or key[0] not in specifications or key[1] not in specifications[key[0]].manifest.question_ids:
            raise FixedRunError('Invalid or repeated journal question identity')
        seen.add(key)
        if item.get('state') != 'committed':
            # A case can exist even though export's final freshness check failed.
            raise UnresolvedAttempt('Uncommitted attempt requires explicit operator review; no automatic retry')
        case = cases.get(key)
        if case is None or case.run_id != item.get('run_id') or runtime_case_digest(case) != item.get('case_sha256'):
            raise UnresolvedAttempt('Committed attempt differs from its immutable case')
        if _amount(item['reservation_usd']) != _reservation(specifications[key[0]]):
            raise FixedBudgetError('Attempt reservation differs from the frozen per-run cap')
        if item.get('api_calls') != case.response.trace.usage.api_calls or item.get(
                'reported_cost_usd') != case.response.trace.usage.estimated_cost_usd:
            raise FixedRunError('Attempt usage differs from its committed runtime case')
    if set(cases) != seen:
        raise UnresolvedAttempt('Cases without committed producer attempts require operator review')


def run_fixed_runs(arms, *, open_runtime, journal_dir, resume=False,
        max_total_cost_usd=0, allow_paid=False):
    """Produce pending fixed outcomes exactly once; uncertainty never authorizes retries."""
    if type(resume) is not bool:
        raise FixedRunError('Resume must be an explicit boolean')
    cap = _amount(max_total_cost_usd)
    arms = _checked_arms(arms)
    journal_exists = os.path.lexists(journal_dir)
    existing = {arm.name: arm.directory.exists() for arm in arms}
    if resume:
        if any(existing.values()) and not journal_exists:
            raise UnresolvedAttempt('Existing outputs require their original spending/attempt journal')
        canonical = []
        for arm in arms:
            if existing[arm.name]:
                old = RunDirectory.load(arm.directory)
                validate_resume_compatibility(old.manifest, arm.manifest)
                canonical.append(FixedArm(arm.name, old.manifest, arm.inputs, arm.directory))
            else:
                canonical.append(arm)
        arms = tuple(canonical)
    elif any(arm.directory.exists() for arm in arms):
        raise FixedRunError('Existing output requires explicit resume; never overwrite or restart it')
    _policy(arms, cap, allow_paid)
    journal_dir = _output_location(journal_dir, arms[0].inputs.files.source_repo)
    for arm in arms:
        if (journal_dir.resolve().is_relative_to(arm.directory.resolve())
                or arm.directory.resolve().is_relative_to(journal_dir.resolve())):
            raise FixedRunError('Attempt journal must be outside every strict runtime directory')
    def checked_open(arm):
        @contextmanager
        def opened():
            with open_runtime(arm) as (actual, identity):
                if arm.manifest.generation_model.provider == 'anthropic':
                    from src.generation.generator import AnthropicProvider
                    provider = actual.provider
                    if type(provider) is not AnthropicProvider or not provider.config.allow_paid:
                        raise FixedBudgetError('Paid execution requires the guarded SDK provider and explicit approval')
                yield actual, identity
        return opened()
    preflight = preflight_fixed_runs(arms, open_runtime=checked_open)
    # An untouched frozen preflight plan has neither a journal nor run headers;
    # its first execution may create both. Never reset a partially existing one.
    with _Journal(journal_dir, _plan(arms, cap, allow_paid), resume=resume and journal_exists) as journal, ExitStack() as stack:
        if resume and journal.state['attempts'] and not all(existing.values()):
            raise UnresolvedAttempt('Attempts exist but an arm directory is missing; operator review required')
        runs = {}
        # Acquire and validate EVERY arm before any runtime/provider operation.
        for arm in arms:
            run = (RunDirectory.resume(arm.directory, arm.manifest, arm.inputs) if resume and existing[arm.name]
                   else RunDirectory.create(arm.directory, arm.manifest, arm.inputs))
            runs[arm.name] = stack.enter_context(run)
        _review_journal(journal, arms, runs)
        for arm in arms:
            run = runs[arm.name]
            if not run.pending_question_ids:
                continue
            with checked_open(arm) as (actual, identity):
                _runtime_profile(arm, actual, identity)
                for question in arm.inputs.questions:
                    if question.question_id not in run.pending_question_ids:
                        continue
                    current_identity = _resolved_identity(identity)
                    item = journal.reserve(arm, question.question_id)
                    capture = actual.query_with_capture(_body(arm, question), current_identity,
                        experiment=ExperimentAdmission(run.manifest, arm.inputs, question.question_id))
                    journal.captured(item, capture.run_id)
                    case = actual.export_capture(capture, manifest=run.manifest, verified_inputs=arm.inputs,
                        question_id=question.question_id, directory=run)
                    if arm.manifest.generation_model.provider in {'fixture', 'disabled'} and (
                            case.response.trace.usage.api_calls != 0 or case.response.trace.usage.estimated_cost_usd not in (None, 0)):
                        raise FixedBudgetError('Scripted/retrieval execution reported external calls or cost')
                    journal.committed(item, case)
        _review_journal(journal, arms, runs)
        cases = {name: run.load_cases(require_complete=True) for name, run in runs.items()}
        attempts = journal.state['attempts']
        costs = [item['reported_cost_usd'] for item in attempts]
        free = all(arm.manifest.generation_model.provider in {'fixture', 'disabled'} for arm in arms)
        return {'passed': True, 'arm_order': preflight['arm_order'],
            'matched_complete': set(cases) == set(_METHODS),
            'arms': {name: {'directory': str(runs[name].path), 'cases': len(values),
                'errors': sum(case.error is not None for case in values)} for name, values in cases.items()},
            'journal_dir': str(journal.path), 'reserved_cost_usd': float(_amount(journal.state['reserved_usd'])),
            'reported_cost_usd': float(sum((_amount(value) for value in costs), Decimal(0)))
                if all(value is not None for value in costs) else None,
            'actual_cost_usd': 0.0 if free else None,
            'api_calls': sum(item['api_calls'] for item in attempts),
            'artifact_kind': ('retrieval-only-validation' if all(arm.manifest.generation_model.provider == 'disabled'
                for arm in arms) else 'scripted-conformance' if free else 'generated-baseline'),
            'quality_scored': False}
