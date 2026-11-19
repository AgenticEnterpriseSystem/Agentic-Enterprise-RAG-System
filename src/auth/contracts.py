"""Public auth DTOs for CONTRACT-01; these do not implement authentication."""

from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, SecretStr

from src.contracts import Contract, Identifier

SESSION_COOKIE_NAME = "rag_session"
CSRF_HEADER_NAME = "X-CSRF-Token"


class LoginRequest(Contract):
    # AUTH-01 defines a single account-normalization policy before provisioning.
    login: Annotated[str, Field(min_length=1, max_length=254, pattern=r"^\S+$")]
    password: Annotated[SecretStr, Field(min_length=1, max_length=1024)]


class CurrentUser(Contract):
    user_id: Identifier
    display_name: Identifier
    role: str | None = None
    department: str | None = None
    project_ids: tuple[Identifier, ...] = ()


class CsrfResponse(Contract):
    # CSRF values intentionally reach JS memory; session tokens never do.
    csrf_token: Annotated[str, Field(min_length=1, max_length=512)]


class SessionResponse(CsrfResponse):
    user: CurrentUser
    expires_at: AwareDatetime


class AccessErrorDetail(Contract):
    code: Literal[
        "authentication_required", "invalid_credentials", "session_expired",
        "scope_denied", "stale_context", "csrf_failed", "policy_unavailable",
    ]
    message: Identifier


class AccessErrorResponse(Contract):
    detail: AccessErrorDetail


# Later endpoints return these codes with generic messages, never existence or
# provider/source details. A stale policy is retriable with a fresh request;
# an expired/revoked login requires a new session.
ACCESS_ERROR_STATUS = MappingProxyType({
    "authentication_required": 401,
    "invalid_credentials": 401,
    "session_expired": 401,
    "scope_denied": 403,
    "stale_context": 409,
    "csrf_failed": 403,
    "policy_unavailable": 503,
})
