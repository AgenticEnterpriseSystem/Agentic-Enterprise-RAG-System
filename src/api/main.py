"""FastAPI boundary; one worker owns expensive operations and the spending cap."""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.runtime import WebSettings, RuntimeConfigurationError, RuntimeErrorResponse, build_runtime
from src.contracts import QueryRequest, QueryResponse

log = logging.getLogger(__name__)


def create_app(settings=None, runtime_factory=build_runtime):
    settings = settings or WebSettings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        state = app.state
        state.runtime, state.error, state.busy = None, None, False
        state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rag')

        async def initialize():
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
        yield
        # Running synchronous work cannot be killed safely; it retains its slot
        # and reservations until it finishes, even after an HTTP timeout.
        await state.initialization
        state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title='Redwood Knowledge Search', version='0.1.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

    @app.get('/api/health')
    async def health():
        return {'status': 'alive', 'ready': app.state.runtime is not None}

    @app.get('/api/ready')
    async def ready():
        if app.state.runtime is None:
            return JSONResponse(status_code=503, content={'status': 'not_ready',
                'mode': settings.mode, 'retrieval': settings.retrieval,
                'message': app.state.error or 'Loading the configured retriever. Please wait.'})
        return {'status': 'ready', 'busy': app.state.busy, **app.state.runtime.info()}

    @app.post('/api/query', response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request):
        origin = request.headers.get('origin')
        if origin and origin not in settings.allowed_origins:
            raise HTTPException(403, detail={'code': 'origin_denied', 'message': 'This browser origin is not allowed.'})
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
