"""Bounded authentication HTTP work, independent of the expensive RAG worker."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from src.auth.contracts import CSRF_HEADER_NAME, SESSION_COOKIE_NAME
from src.auth.sessions import AuthError, AuthService
from src.auth.settings import AuthSettings, canonical_origin

BOOTSTRAP_COOKIE_NAME = "rag_login_csrf"
MAX_AUTH_BODY_BYTES = 16 * 1024


class AuthWorker:
    """One real worker; running+queued admission stays bounded across timeouts.

    Cancelling/timing out an HTTP waiter never releases its slot while the
    underlying synchronous task remains running. The executor's unbounded queue
    is reachable only through this admission check. No RAG executor is used.
    """

    def __init__(self, service: AuthService, settings: AuthSettings):
        self.service = service
        self.settings = AuthSettings.model_validate(settings.model_dump(warnings=False))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-auth")
        self._lock, self._pending, self._closed = threading.Lock(), 0, False

    @property
    def pending(self):
        with self._lock:
            return self._pending

    async def call(self, function, *args, **kwargs):
        with self._lock:
            if self._closed or self._pending >= self.settings.max_waiting + 1:
                raise AuthError("busy", 429, "Authentication is busy. Try again shortly.", retry_after=2)
            self._pending += 1
            try:
                future = self._executor.submit(function, *args, **kwargs)
            except BaseException:
                self._pending -= 1
                raise

        def finished(_future):
            with self._lock:
                self._pending -= 1

        future.add_done_callback(finished)
        wrapped = asyncio.wrap_future(future)
        # Consume a late exception even after a waiter timeout/disconnect.
        def consume(done):
            if not done.cancelled():
                done.exception()
        wrapped.add_done_callback(consume)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), self.settings.worker_timeout_seconds)
        except asyncio.TimeoutError:
            raise AuthError("busy", 429, "Authentication is still finishing. Try again shortly.", retry_after=2) from None

    async def aclose(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


def _origin(request: Request, settings: AuthSettings, *, required=False):
    supplied = request.headers.get("origin")
    if supplied is None:
        if required:
            raise AuthError("csrf_failed", 403, "This browser origin is not allowed.")
        supplied = str(request.base_url).rstrip("/")
    try:
        origin = canonical_origin(supplied)
    except ValueError:
        raise AuthError("csrf_failed", 403, "This browser origin is not allowed.") from None
    if origin not in settings.allowed_origins or request.headers.get("sec-fetch-site") == "cross-site":
        raise AuthError("csrf_failed", 403, "This browser origin is not allowed.")


def _no_cache(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return response


def _failure(error):
    headers = {"Retry-After": str(error.retry_after)} if error.retry_after is not None else None
    return _no_cache(JSONResponse(status_code=error.status,
        content={"detail": {"code": error.code, "message": error.message}}, headers=headers))


def _invalid_body():
    return AuthError("invalid_request", 422, "Invalid sign-in request.")


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError()
        value[key] = item
    return value


async def _credentials(request):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            count = int(content_length)
        except ValueError:
            raise _invalid_body() from None
        if count < 0:
            raise _invalid_body()
        if count > MAX_AUTH_BODY_BYTES:
            raise AuthError("invalid_request", 413, "Sign-in request is too large.")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise _invalid_body()
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_AUTH_BODY_BYTES:
            raise AuthError("invalid_request", 413, "Sign-in request is too large.")
        chunks.append(chunk)
    try:
        value = json.loads(b"".join(chunks), object_pairs_hook=_unique)
        if (not isinstance(value, dict) or set(value) != {"login", "password"}
                or not isinstance(value["login"], str) or not 1 <= len(value["login"]) <= 254
                or not isinstance(value["password"], str) or not 1 <= len(value["password"]) <= 1024):
            raise ValueError()
    except (ValueError, UnicodeError):
        raise _invalid_body() from None
    return value["login"], value["password"]


def create_auth_router(service: AuthService, settings: AuthSettings, *, worker: AuthWorker | None = None):
    settings = AuthSettings.model_validate(settings.model_dump(warnings=False))
    worker = worker or AuthWorker(service, settings)
    router = APIRouter(prefix="/api/auth", tags=["authentication"])

    async def safely(function):
        try:
            return await function()
        except AuthError as error:
            return _failure(error)
        except Exception:
            return _failure(AuthError("policy_unavailable", 503, "Authentication is temporarily unavailable."))

    @router.get("/csrf")
    async def csrf(request: Request):
        async def operation():
            _origin(request, settings)
            grant = await worker.call(service.issue_bootstrap)
            response = _no_cache(JSONResponse(content=grant.response.model_dump(mode="json")))
            response.set_cookie(BOOTSTRAP_COOKIE_NAME, grant.token.get_secret_value(),
                max_age=settings.bootstrap_seconds, secure=settings.secure_cookies, httponly=True,
                samesite="strict", path="/api/auth")
            return response
        return await safely(operation)

    @router.post("/login")
    async def login(request: Request):
        async def operation():
            _origin(request, settings, required=True)
            login_value, password = await _credentials(request)
            def sign_in():
                service.validate_bootstrap(request.cookies.get(BOOTSTRAP_COOKIE_NAME), request.headers.get(CSRF_HEADER_NAME))
                return service.login(login_value, password,
                    client_ip=request.client.host if request.client else "unknown",
                    previous_token=request.cookies.get(SESSION_COOKIE_NAME))
            grant = await worker.call(sign_in)
            response = _no_cache(JSONResponse(content=grant.response.model_dump(mode="json")))
            response.set_cookie(SESSION_COOKIE_NAME, grant.token.get_secret_value(), max_age=settings.absolute_seconds,
                secure=settings.secure_cookies, httponly=True, samesite="strict", path="/")
            response.delete_cookie(BOOTSTRAP_COOKIE_NAME, path="/api/auth", secure=settings.secure_cookies,
                                   httponly=True, samesite="strict")
            return response
        return await safely(operation)

    @router.get("/me")
    async def me(request: Request):
        async def operation():
            _origin(request, settings)
            authenticated = await worker.call(service.authenticate, request.cookies.get(SESSION_COOKIE_NAME), touch=False)
            return _no_cache(JSONResponse(content=authenticated.response.model_dump(mode="json")))
        return await safely(operation)

    @router.post("/logout")
    async def logout(request: Request):
        async def operation():
            _origin(request, settings, required=True)
            await worker.call(service.logout, request.cookies.get(SESSION_COOKIE_NAME), request.headers.get(CSRF_HEADER_NAME))
            response = _no_cache(Response(status_code=204))
            response.delete_cookie(SESSION_COOKIE_NAME, path="/", secure=settings.secure_cookies,
                                   httponly=True, samesite="strict")
            response.delete_cookie(BOOTSTRAP_COOKIE_NAME, path="/api/auth", secure=settings.secure_cookies,
                                   httponly=True, samesite="strict")
            return response
        return await safely(operation)

    return router
