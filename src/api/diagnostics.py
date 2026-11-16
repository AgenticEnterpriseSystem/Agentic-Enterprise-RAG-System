"""Content-free query observations and explicit authenticated admin reads."""
import asyncio
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from src.observability import observation_context


class QueryObservationMiddleware:
    """Start before body validation/auth; finish HTTP separately from its worker."""

    def __init__(self, app, *, recorder):
        self.app, self.recorder = app, recorder

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path') != '/api/query':
            return await self.app(scope, receive, send)
        try:
            observation = self.recorder.start_request()
        except Exception:
            # An unavailable recorder must never affect the query path.
            return await self.app(scope, receive, send)
        status_code, finished, started = 500, False, False

        async def observed_send(message):
            nonlocal status_code, finished, started
            if message['type'] == 'http.response.start':
                started = True
                status_code = message['status']
                message = {**message, 'headers': [
                    (key, value) for key, value in message.get('headers', ())
                    if key.lower() != b'x-request-id'
                ] + [(b'x-request-id', observation.request_id.encode('ascii'))]}
            await send(message)
            if message['type'] == 'http.response.body' and not message.get('more_body', False):
                finished = True
                observation.http_finish(status_code,
                    status='timeout' if status_code == 504 else None,
                    error_code='validation_error' if status_code == 422 else None)

        with observation_context(observation):
            try:
                await self.app(scope, receive, observed_send)
            except asyncio.CancelledError as error:
                observation.http_finish(status_code if started else 499, status='cancelled', error=error)
                raise
            except Exception as error:
                observation.emit('http', 'error', status='error', error=error, error_code='internal_error')
                if not started:
                    await JSONResponse(status_code=500, content={'detail': {
                        'code': 'internal_error', 'message': 'Request failed safely.'}},
                        headers={'Cache-Control': 'no-store'})(scope, receive, observed_send)
                else:
                    observation.http_finish(status_code, status='error', error=error)
                    raise
            finally:
                if not finished:
                    observation.http_finish(status_code if started else 499, status='cancelled')


def create_diagnostics_router(settings, recorder, *, auth_service=None, auth_worker=None,
                              auth_settings=None):
    router = APIRouter(prefix='/api/observability')

    async def require_admin(request):
        def failure(status, code):
            return HTTPException(status, detail={'code': code,
                'message': 'Observability access is unavailable.'}, headers={'Cache-Control': 'no-store'})
        token = request.cookies.get('rag_session')
        if auth_service is None or not token:
            raise failure(401, 'authentication_required')
        from src.api.auth import _origin
        _origin(request, auth_settings, required=False)
        session = await auth_worker.call(auth_service.authenticate, token, touch=False)
        if session.identity.user_id not in settings.observability_admin_user_ids:
            raise failure(403, 'forbidden')

    def response(content):
        return JSONResponse(content=content, headers={'Cache-Control': 'no-store', 'Pragma': 'no-cache'})

    @router.get('/access')
    async def access(request: Request):
        await require_admin(request)
        return response({'allowed': True})

    @router.get('/requests')
    async def requests(request: Request, limit: int = Query(50, ge=1, le=50),
                       offset: int = Query(0, ge=0, le=500)):
        await require_admin(request)
        return response({**recorder.list(limit=limit, offset=offset),
            'retention_seconds': recorder.retention_seconds, 'max_requests': recorder.max_requests,
            'history_scope': 'current_process'})

    @router.get('/requests/{request_id}')
    async def detail(request_id: str, request: Request):
        await require_admin(request)
        try:
            key = str(UUID(request_id))
        except ValueError:
            key = None
        result = recorder.get(key) if key else None
        if result is None:
            raise HTTPException(404, detail={'code': 'not_found', 'message': 'Request history expired or is unavailable.'},
                                headers={'Cache-Control': 'no-store'})
        return response(result)

    return router
