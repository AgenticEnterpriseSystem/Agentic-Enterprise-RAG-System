"""FastAPI boundary; one worker owns expensive operations and the spending cap."""
import asyncio
import hmac
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from contextvars import copy_context
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.runtime import (AuthenticatedRuntime, WebSettings, RuntimeConfigurationError,
                             RuntimeErrorResponse, build_runtime)
from src.access.contracts import (AccessDenied, AccessError, EvidenceValidationError,
                                 PolicyUnavailable, SessionExpired, SnapshotMismatch, StaleContext)
from src.contracts import QueryRequest, QueryResponse
from src.observability import Recorder, bind_progress, current_observation, emit, span
from src.api.diagnostics import QueryObservationMiddleware, create_diagnostics_router

log = logging.getLogger(__name__)


def create_app(settings=None, runtime_factory=None, *, auth_settings=None, auth_service=None):
    settings = settings or WebSettings.from_env()
    runtime_factory = build_runtime if runtime_factory is None else runtime_factory
    auth_enabled = settings.auth_mode == 'local' or bool(auth_settings and auth_settings.enabled)
    # Explicit server-side settings override environment; the default app honors
    # either supported enabling flag and never silently falls back to the demo.
    if auth_settings is None and os.environ.get('RAG_AUTH_ENABLED', '') != '':
        flag = os.environ['RAG_AUTH_ENABLED'].lower()
        if flag not in {'true', 'false', '1', '0'}:
            raise RuntimeConfigurationError('RAG_AUTH_ENABLED must be a boolean setting.')
        if settings.auth_mode == 'local' and flag in {'false', '0'}:
            raise RuntimeConfigurationError('Local mode conflicts with disabled authentication.')
        auth_enabled = auth_enabled or flag in {'true', '1'}
    auth_worker, owned_auth_database = None, None
    # Two rounds with four hybrid text searches each need more than 128 stage
    # events; retain a fixed bound without losing the final generation stages.
    recorder = Recorder(max_events_per_request=256)
    if auth_service is not None and not auth_enabled:
        raise RuntimeConfigurationError('Injected authentication requires enabled local authentication settings.')
    if auth_enabled:
        from src.auth.settings import AuthSettings
        from src.auth.sessions import AuthService, AuthError
        from src.api.auth import AuthWorker, create_auth_router, _origin as validate_auth_origin
        from src.db.runtime import Database, DatabaseSettings
        auth_settings = (AuthSettings.model_validate(auth_settings.model_dump()) if auth_settings is not None
                         else AuthSettings.from_env(
                             allowed_origins=None if os.environ.get('RAG_AUTH_ALLOWED_ORIGINS') else settings.allowed_origins,
                             enabled=True))
        if not auth_settings.enabled:
            raise RuntimeConfigurationError('Local authentication must be explicitly enabled.')
        if auth_service is None:
            owned_auth_database = Database(DatabaseSettings.from_env())
            auth_service = AuthService(owned_auth_database, auth_settings)
        auth_worker = AuthWorker(auth_service, auth_settings)

    @asynccontextmanager
    async def lifespan(app):
        state = app.state
        state.runtime, state.error, state.busy = None, None, False
        state.closing, state.active_work = False, None
        state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rag')
        state.auth_service, state.auth_worker = auth_service, auth_worker
        loop = asyncio.get_running_loop()

        def submit_work(operation):
            # Retain the real future: cancelling an HTTP waiter cannot release
            # admission or close the shared pool while its thread still runs.
            observation = current_observation()
            def observed_operation():
                if observation is not None:
                    observation.worker_start()
                try:
                    result = operation()
                except BaseException as error:
                    if observation is not None:
                        observation.worker_finish(status='denied' if isinstance(error, (AccessDenied, SessionExpired))
                            else 'error', error=error)
                    raise
                if observation is not None:
                    failed = isinstance(result, QueryResponse) and result.answer.status == 'error'
                    observation.worker_finish(status='error' if failed else 'ok',
                        usage=result.trace.usage if isinstance(result, QueryResponse) else None)
                return result
            work = state.executor.submit(copy_context().run, observed_operation)
            state.active_work = work

            def finished(done):
                if not done.cancelled():
                    done.exception()
                def release():
                    if state.active_work is done:
                        state.busy = False
                with suppress(RuntimeError):
                    loop.call_soon_threadsafe(release)
            work.add_done_callback(finished)
            wrapped = asyncio.wrap_future(work, loop=loop)
            wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            return wrapped
        state.submit_work = submit_work

        async def initialize():
            try:
                def construct_runtime():
                    runtime = (build_runtime(settings, database=auth_service.database, authenticated=True)
                               if auth_enabled and runtime_factory is build_runtime
                               else runtime_factory(settings))
                    if auth_enabled:
                        if (not isinstance(runtime, AuthenticatedRuntime)
                                or runtime.database is not auth_service.database):
                            raise RuntimeConfigurationError('An authenticated document runtime is required.')
                        # One startup operation owns construction and its final
                        # metadata check. Transient auth admission cannot discard
                        # a built runtime; live readiness still uses AuthWorker.
                        runtime.assert_ready()
                    return runtime

                runtime = await asyncio.shield(submit_work(construct_runtime))
                if not state.closing:
                    state.runtime = runtime
            except Exception as exc:
                # No provider messages, credential values or source text in HTTP/logs.
                log.warning('Runtime initialization failed: %s', type(exc).__name__)
                if auth_enabled:
                    state.error = 'Document search is unavailable. Check the server configuration.'
                elif isinstance(exc, RuntimeConfigurationError):
                    state.error = str(exc)
                elif isinstance(exc, FileNotFoundError):
                    state.error = 'Configured corpus or index artifact is missing. Check RAG_CORPUS and index paths.'
                elif isinstance(exc, ValueError):
                    state.error = 'Runtime configuration or artifact compatibility check failed. Check demo access, live settings and matching index manifests.'
                else:
                    state.error = 'Runtime initialization failed. Check model availability, hardware and server configuration.'
        state.initialization = asyncio.create_task(initialize())
        try:
            yield
        finally:
            state.closing = True
            state.initialization.cancel()
            with suppress(asyncio.CancelledError):
                await state.initialization
            state.runtime = None
            state.executor.shutdown(wait=False, cancel_futures=True)
            if auth_worker is not None:
                await auth_worker.aclose()
            recorder.close()
            if owned_auth_database is not None:
                def close_when_idle():
                    # Auth has a separate executor, but uses the same pool.
                    # This callback remains useful after the ASGI loop closes.
                    while ((state.active_work is not None and not state.active_work.done())
                           or (auth_worker is not None and auth_worker.pending)):
                        time.sleep(.02)
                    owned_auth_database.close()
                    state.active_work = None
                if ((state.active_work is None or state.active_work.done())
                        and (auth_worker is None or not auth_worker.pending)):
                    close_when_idle()
                else:
                    state.database_cleanup = threading.Thread(target=close_when_idle,
                        name='rag-pool-cleanup', daemon=True)
                    state.database_cleanup.start()

    app = FastAPI(title='Redwood Knowledge Search', version='0.1.0', lifespan=lifespan)
    app.state.observability = recorder
    origins = auth_settings.allowed_origins if auth_enabled else settings.allowed_origins
    app.add_middleware(CORSMiddleware, allow_origins=list(origins),
                       allow_methods=['GET', 'POST'], allow_credentials=auth_enabled,
                       allow_headers=(['Content-Type', 'X-CSRF-Token', 'X-Query-Progress-ID'] if auth_enabled
                                      else ['Content-Type', 'X-Query-Progress-ID']),
                       expose_headers=['X-Request-ID'])
    app.add_middleware(QueryObservationMiddleware, recorder=recorder)
    app.include_router(create_diagnostics_router(settings, recorder, auth_service=auth_service,
        auth_worker=auth_worker, auth_settings=auth_settings))

    @app.get('/api/auth/profile')
    async def authentication_profile():
        return JSONResponse(content={'enabled': auth_enabled}, headers={'Cache-Control': 'no-store'})

    if auth_enabled:
        app.include_router(create_auth_router(auth_service, auth_settings, worker=auth_worker))

        @app.exception_handler(AuthError)
        async def authentication_error(_request, error):
            emit('auth.authenticate', 'error', status='denied' if error.status in (401, 403)
                else 'error', error=error, error_code=error.code)
            return JSONResponse(status_code=error.status,
                content={'detail': {'code': error.code, 'message': error.message}},
                headers={'Cache-Control': 'no-store'})

    @app.get('/api/health')
    async def health():
        return JSONResponse(content={'status': 'alive', 'ready': app.state.runtime is not None},
                            headers={'Cache-Control': 'no-store'} if auth_enabled else None)

    @app.get('/api/ready')
    async def ready():
        if auth_enabled:
            runtime = app.state.runtime
            if runtime is None or app.state.closing:
                return JSONResponse(status_code=503, content={'status': 'not_ready',
                    'message': app.state.error or 'Document search is starting. Please wait.'},
                    headers={'Cache-Control': 'no-store'})
            try:
                await auth_worker.call(runtime.assert_ready)
                if app.state.closing or app.state.runtime is not runtime:
                    raise PolicyUnavailable('Document readiness changed during its check')
            except Exception:
                # Missing metadata, lost connectivity and bounded-worker
                # admission/timeouts are all not-ready, without public inventory
                # or underlying SQL/credential details. A later poll can recover.
                return JSONResponse(status_code=503, content={'status': 'not_ready',
                    'message': 'Document search is temporarily unavailable.'},
                    headers={'Cache-Control': 'no-store'})
            # Public readiness must not reveal corpus/source counts, permissions
            # or other users' shared spending state.
            safe = runtime.info()
            return JSONResponse(content={'status': 'ready', 'busy': app.state.busy,
                **{name: safe[name] for name in ('mode', 'retrieval', 'generation_enabled') if name in safe}},
                headers={'Cache-Control': 'no-store'})
        if app.state.runtime is None:
            return JSONResponse(status_code=503, content={'status': 'not_ready',
                'mode': settings.mode, 'retrieval': settings.retrieval,
                'message': app.state.error or 'Loading the configured retriever. Please wait.'})
        return {'status': 'ready', 'busy': app.state.busy, **app.state.runtime.info()}

    @app.post('/api/query', response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request):
        def failure(status, code, message, headers=None):
            emit('http', 'error', status='denied' if status in (401, 403) else
                'timeout' if status == 504 else 'error', error_code=code)
            values = dict(headers or {})
            if auth_enabled:
                values['Cache-Control'] = 'no-store'
            return HTTPException(status, detail={'code': code, 'message': message}, headers=values or None)

        origin = request.headers.get('origin')
        if origin and origin not in origins:
            raise failure(403, 'origin_denied', 'This browser origin is not allowed.')
        owner = None
        if auth_enabled:
            token = request.cookies.get('rag_session')
            if not token:
                raise failure(401, 'authentication_required', 'Authentication is required.')
            with span('auth.authenticate'):
                session = await auth_worker.call(auth_service.authenticate, token, touch=False)
            validate_auth_origin(request, auth_settings, required=True)
            supplied_csrf = request.headers.get('X-CSRF-Token', '')
            if (not origin or origin not in origins or not hmac.compare_digest(
                    session.response.csrf_token.encode('ascii'), supplied_csrf.encode('utf-8'))):
                raise failure(403, 'csrf_failed', 'Request verification failed.')
            owner = session.identity.user_id
        # Bound before admission checks so a busy/not-ready rejection is also visible.
        progress_id = request.headers.get('X-Query-Progress-ID')
        if progress_id:
            bind_progress(progress_id, owner)
        if app.state.runtime is None or app.state.closing:
            raise failure(503, 'not_ready', app.state.error or 'Retriever is still loading.')
        if app.state.busy:
            raise failure(429, 'busy', 'A query is still running. Wait before trying again.', {'Retry-After': '2'})
        app.state.busy = True
        runtime = app.state.runtime

        def operation():
            if auth_enabled:
                # Only an admitted real query counts as activity. Resolution
                # and all synchronous policy/retrieval work stay off ASGI.
                with span('auth.authenticate'):
                    admitted = auth_service.authenticate(token, touch=True)
                return runtime.query(body, admitted.identity)
            return runtime.query(body)
        try:
            future = app.state.submit_work(operation)
        except RuntimeError:
            app.state.busy = False
            raise failure(503, 'not_ready', 'Document search is unavailable.') from None
        try:
            # Permission resolution and run admission happen before the agent's
            # own clock starts. Give that work a separate configured deadline.
            response = await asyncio.wait_for(asyncio.shield(future),
                timeout=settings.effective_request_timeout_seconds)
        except asyncio.TimeoutError:
            raise failure(504, 'timeout', 'Query timed out. Its worker may still be finishing; further queries remain blocked until it exits.') from None
        except RuntimeErrorResponse as exc:
            raise failure(exc.status, exc.code, exc.message) from None
        except SessionExpired:
            raise failure(401, 'session_expired', 'This session is unavailable. Sign in again.') from None
        except PolicyUnavailable:
            raise failure(503, 'policy_unavailable', 'Document access is temporarily unavailable.') from None
        except StaleContext:
            raise failure(409, 'stale_context', 'Access changed. Please retry the question.') from None
        except AccessDenied:
            raise failure(403, 'scope_denied', 'The requested information is unavailable in this scope.') from None
        except SnapshotMismatch:
            raise failure(503, 'not_ready', 'Document search is temporarily unavailable.') from None
        except EvidenceValidationError:
            raise failure(502, 'query_failed', 'Query failed safely. No answer is available.') from None
        except AccessError:
            raise failure(403, 'scope_denied', 'The requested information is unavailable in this scope.') from None
        except Exception as exc:
            if auth_enabled and isinstance(exc, AuthError):
                raise
            log.warning('Query failed: %s', type(exc).__name__)
            raise failure(502, 'query_failed', 'Query failed safely. No answer is available.') from None
        status = 502 if response.answer.status == 'error' else (429 if response.trace.stop_reason == 'budget' else 200)
        return JSONResponse(status_code=status, content=response.model_dump(mode='json'),
            headers={'Cache-Control': 'no-store'} if auth_enabled else None)

    @app.get('/api/query/progress/{progress_id}')
    async def query_progress(progress_id: str, request: Request):
        """Content-free stage events; deliberately outside the busy flag and query executor."""
        headers = {'Cache-Control': 'no-store'}

        def failure(status, code, message):
            return HTTPException(status, detail={'code': code, 'message': message}, headers=headers)

        origin = request.headers.get('origin')
        if origin and origin not in origins:
            raise failure(403, 'origin_denied', 'This browser origin is not allowed.')
        owner = None
        if auth_enabled:
            token = request.cookies.get('rag_session')
            if not token:
                raise failure(401, 'authentication_required', 'Authentication is required.')
            session = await auth_worker.call(auth_service.authenticate, token, touch=False)
            owner = session.identity.user_id
        result = recorder.progress(progress_id, owner)
        if result is None:
            raise failure(404, 'not_found', 'Progress is unavailable or has expired.')
        return JSONResponse(content=result, headers=headers)

    if auth_enabled:
        @app.get('/api/runs/{run_id}')
        async def run_metadata(run_id: str, request: Request):
            """One current-owner receipt; no question, answer or trace history."""
            def failure(status, code, message):
                return HTTPException(status, detail={'code': code, 'message': message},
                                     headers={'Cache-Control': 'no-store'})

            origin = request.headers.get('origin')
            if origin and origin not in origins:
                raise failure(403, 'origin_denied', 'This browser origin is not allowed.')
            token = request.cookies.get('rag_session')
            if not token:
                raise failure(401, 'authentication_required', 'Authentication is required.')
            # Reading a receipt does not keep an idle session alive.
            await auth_worker.call(auth_service.authenticate, token, touch=False)
            if app.state.runtime is None or app.state.closing:
                raise failure(503, 'not_ready', 'Document search is unavailable.')
            if app.state.busy:
                raise failure(429, 'busy', 'A query is still running. Wait before trying again.')
            app.state.busy = True
            runtime = app.state.runtime

            def operation():
                current = auth_service.authenticate(token, touch=False)
                return runtime.get_run_metadata(run_id, current.identity)

            try:
                future = app.state.submit_work(operation)
            except RuntimeError:
                app.state.busy = False
                raise failure(503, 'not_ready', 'Document search is unavailable.') from None
            try:
                metadata = await asyncio.wait_for(asyncio.shield(future), timeout=settings.max_seconds)
            except asyncio.TimeoutError:
                raise failure(504, 'timeout', 'Run metadata is temporarily unavailable.') from None
            except SessionExpired:
                raise failure(401, 'session_expired', 'This session is unavailable. Sign in again.') from None
            except StaleContext:
                raise failure(409, 'stale_context', 'Access changed. Please retry.') from None
            except PolicyUnavailable:
                raise failure(503, 'policy_unavailable', 'Document access is temporarily unavailable.') from None
            except AccessError:
                raise failure(403, 'scope_denied', 'Run is unavailable in this current scope.') from None
            except AuthError:
                raise
            except Exception as exc:
                log.warning('Run metadata failed: %s', type(exc).__name__)
                raise failure(502, 'query_failed', 'Run metadata is unavailable.') from None
            return JSONResponse(content=metadata.model_dump(mode='json'), headers={'Cache-Control': 'no-store'})

    if (settings.frontend_dist / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=settings.frontend_dist / 'assets'), name='assets')

    @app.get('/', include_in_schema=False)
    @app.get('/observability', include_in_schema=False)
    async def index():
        file = settings.frontend_dist / 'index.html'
        if not file.is_file():
            raise HTTPException(503, detail='Frontend is not built. Run npm ci and npm run build in frontend/.')
        return FileResponse(file)

    return app


app = create_app()
