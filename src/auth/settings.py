"""Explicit local-account transport, expiry and bounded-admission settings."""

import ipaddress
import os
import re
from typing import Annotated, Mapping
from urllib.parse import urlsplit

from pydantic import Field, StrictBool, model_validator

from src.contracts import Contract


def canonical_origin(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or any(character.isspace() for character in value)):
            raise ValueError()
        host, port = parsed.hostname.lower(), parsed.port
        if ":" in host:
            ipaddress.ip_address(host)
            host = "[" + host + "]"
        elif not re.fullmatch(r"[a-z0-9.-]+", host):
            raise ValueError()
        suffix = "" if port is None or (parsed.scheme, port) in {("http", 80), ("https", 443)} else ":" + str(port)
        return parsed.scheme + "://" + host + suffix
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Authentication origins must be explicit HTTP(S) origins") from None


def _loopback(origin):
    host = urlsplit(origin).hostname
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class AuthSettings(Contract):
    enabled: StrictBool = False
    allowed_origins: tuple[str, ...] = ("https://localhost:8000",)
    secure_cookies: StrictBool = True
    allow_insecure_loopback: StrictBool = False
    idle_seconds: Annotated[int, Field(strict=True, ge=1, le=86400)] = 1800
    absolute_seconds: Annotated[int, Field(strict=True, ge=1, le=604800)] = 28800
    bootstrap_seconds: Annotated[int, Field(strict=True, ge=1, le=900)] = 300
    max_waiting: Annotated[int, Field(strict=True, ge=0, le=32)] = 4
    worker_timeout_seconds: Annotated[float, Field(gt=0, le=60)] = 10
    throttle_window_seconds: Annotated[int, Field(strict=True, ge=1, le=3600)] = 60
    per_login_attempts: Annotated[int, Field(strict=True, ge=1, le=10000)] = 5
    per_ip_attempts: Annotated[int, Field(strict=True, ge=1, le=10000)] = 20
    global_attempts: Annotated[int, Field(strict=True, ge=1, le=10000)] = 120
    throttle_capacity: Annotated[int, Field(strict=True, ge=1, le=100000)] = 1024

    @model_validator(mode="after")
    def explicit_profile(self):
        origins = tuple(canonical_origin(origin) for origin in self.allowed_origins)
        if not origins or len(origins) != len(set(origins)):
            raise ValueError("Authentication requires unique explicit origins")
        if self.idle_seconds > self.absolute_seconds:
            raise ValueError("Idle lifetime cannot exceed absolute lifetime")
        if self.enabled:
            if self.secure_cookies:
                if any(not origin.startswith("https://") for origin in origins):
                    raise ValueError("Secure authentication requires HTTPS origins")
            elif (not self.allow_insecure_loopback or any(not _loopback(origin) for origin in origins)):
                raise ValueError("Insecure authentication requires the explicit loopback-only profile")
        object.__setattr__(self, "allowed_origins", origins)
        return self

    @classmethod
    def from_env(cls, *, allowed_origins=None, enabled=None, environ: Mapping[str, str] | None = None):
        values, result = os.environ if environ is None else environ, {}
        boolean = {"enabled", "secure_cookies", "allow_insecure_loopback"}
        for name in cls.model_fields:
            raw = values.get("RAG_AUTH_" + name.upper())
            if raw is None or raw == "":
                continue
            try:
                if name == "allowed_origins":
                    result[name] = tuple(item.strip() for item in raw.split(",") if item.strip())
                elif name in boolean:
                    if raw.lower() not in {"true", "false", "1", "0"}:
                        raise ValueError()
                    result[name] = raw.lower() in {"true", "1"}
                elif name == "worker_timeout_seconds":
                    result[name] = float(raw)
                else:
                    result[name] = int(raw)
            except (ValueError, TypeError):
                raise ValueError("Invalid authentication environment setting") from None
        if allowed_origins is not None:
            result["allowed_origins"] = tuple(allowed_origins)
        if enabled is not None:
            result["enabled"] = enabled
        return cls.model_validate(result)
