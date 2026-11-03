"""FastAPI boundary; one worker owns expensive operations and the spending cap."""
import asyncio
import hmac
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.runtime import WebSettings, RuntimeConfigurationError, RuntimeErrorResponse, build_runtime
from src.contracts import QueryRequest, QueryResponse

log = logging.getLogger(__name__)


def create_app(settings=None, runtime_factory=build_runtime, *, auth_settings=None, auth_service=None):
    settings = settings or WebSettings.from_env()
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
    if auth_service is not None and not auth_enabled:
        raise RuntimeConfigurationError('Injected authentication requires enabled local authentication settings.')
    if auth_enabled:
        from src.auth.settings import AuthSettings
        from src.auth.sessions import AuthService, AuthError
        from src.api.auth import AuthWorker, create_auth_router
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
        state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rag')
        state.auth_service, state.auth_worker = auth_service, auth_worker

        async def initialize():
            if auth_enabled and runtime_factory is build_runtime:
                state.error = 'Document access is not configured yet.'
                return
            try:
                state.runtime = await asyncio.get_running_loop().run_in_executor(
                    state.executor, runtime_factory, settings)
            except Exception as exc:
                # No provider messages, credential values or source text in HTTP/logs.
                log.warning('Runtime initialization failed: %s', type(exc).__name__)
                if isinstance(exc, RuntimeConfigurationError):
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
            # Running work retains its resources until it actually exits.
            try:
                await state.initialization
            finally:
                state.executor.shutdown(wait=False, cancel_futures=True)
                if auth_worker is not None:
                    await auth_worker.aclose()
                if owned_auth_database is not None:
                    owned_auth_database.close()

    app = FastAPI(title='Redwood Knowledge Search', version='0.1.0', lifespan=lifespan)
    origins = auth_settings.allowed_origins if auth_enabled else settings.allowed_origins
    app.add_middleware(CORSMiddleware, allow_origins=list(origins),
                       allow_methods=['GET', 'POST'], allow_credentials=auth_enabled,
                       allow_headers=['Content-Type', 'X-CSRF-Token'] if auth_enabled else ['Content-Type'])
    if auth_enabled:
        app.include_router(create_auth_router(auth_service, auth_settings, worker=auth_worker))

        @app.exception_handler(AuthError)
        async def authentication_error(_request, error):
            return JSONResponse(status_code=error.status,
                content={'detail': {'code': error.code, 'message': error.message}},
                headers={'Cache-Control': 'no-store'})

    @app.get('/api/health')
    async def health():
        return {'status': 'alive', 'ready': not auth_enabled and app.state.runtime is not None}

    @app.get('/api/ready')
    async def ready():
        if auth_enabled:
            return JSONResponse(status_code=503, content={'status': 'not_ready',
                'message': 'Document access is not configured yet.'}, headers={'Cache-Control': 'no-store'})
        if app.state.runtime is None:
            return JSONResponse(status_code=503, content={'status': 'not_ready',
                'mode': settings.mode, 'retrieval': settings.retrieval,
                'message': app.state.error or 'Loading the configured retriever. Please wait.'})
        return {'status': 'ready', 'busy': app.state.busy, **app.state.runtime.info()}

    @app.post('/api/query', response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request):
        origin = request.headers.get('origin')
        if origin and origin not in origins:
            raise HTTPException(403, detail={'code': 'origin_denied', 'message': 'This browser origin is not allowed.'})
        if auth_enabled:
            token = request.cookies.get('rag_session')
            if not token:
                raise HTTPException(401, detail={'code': 'authentication_required', 'message': 'Authentication is required.'})
            session = await auth_worker.call(auth_service.authenticate, token, touch=False)
            supplied_csrf = request.headers.get('X-CSRF-Token', '')
            if (not origin or origin not in origins or not hmac.compare_digest(
                    session.response.csrf_token.encode('ascii'), supplied_csrf.encode('utf-8'))):
                raise HTTPException(403, detail={'code': 'csrf_failed', 'message': 'Request verification failed.'})
            raise HTTPException(403, detail={'code': 'scope_denied', 'message': 'Document access is not configured yet.'})
        if app.state.runtime is None:
            raise HTTPException(503, detail={'code': 'not_ready', 'message': app.state.error or 'Retriever is still loading.'})
        if app.state.busy:
            raise HTTPException(429, detail={'code': 'busy', 'message': 'A query is still running. Wait before trying again.'}, headers={'Retry-After': '2'})
        app.state.busy = True
        future = asyncio.get_running_loop().run_in_executor(app.state.executor, app.state.runtime.query, body)
        def finished(work):
            app.state.busy = False
            # Consume late failures after an HTTP timeout/disconnect as well.
            if not work.cancelled():
                work.exception()
        future.add_done_callback(finished)
        try:
            response = await asyncio.wait_for(asyncio.shield(future), timeout=settings.max_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(504, detail={'code': 'timeout', 'message': 'Query timed out. Its worker may still be finishing; further queries remain blocked until it exits.'}) from None
        except RuntimeErrorResponse as exc:
            raise HTTPException(exc.status, detail={'code': exc.code, 'message': exc.message}) from None
        except Exception as exc:
            log.warning('Query failed: %s', type(exc).__name__)
            raise HTTPException(502, detail={'code': 'query_failed', 'message': 'Query failed safely. No answer is available.'}) from None
        status = 502 if response.answer.status == 'error' else (429 if response.trace.stop_reason == 'budget' else 200)
        return JSONResponse(status_code=status, content=response.model_dump(mode='json'))

    if (settings.frontend_dist / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=settings.frontend_dist / 'assets'), name='assets')

    @app.get('/', include_in_schema=False)
    async def index():
        file = settings.frontend_dist / 'index.html'
        if not file.is_file():
            raise HTTPException(503, detail='Frontend is not built. Run npm ci and npm run build in frontend/.')
        return FileResponse(file)

    return app


app = create_app()
