"""
Cursor authentication + endpoint constants for ``cursor_core``.

Holds the static service coordinates (host / port / RPC paths / client version)
and the credential plumbing: loading the on-disk Cursor access token, building
the HTTP/2 and Connect-unary header sets, and a (currently stubbed) token
refresh entry point.

The access token is read fresh from known Cursor auth files on every request so
an external refresh (the Cursor desktop app or ``agent login``) is picked up
without restarting the process.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

# ── Service endpoints ─────────────────────────────────────────────────────────

AGENT_HOST = "agentn.global.api5.cursor.sh"
AGENT_PORT = 443
RUN_PATH = "/agent.v1.AgentService/Run"
CLIENT_VERSION = "cli-2026.06.03-0bbb28e"

API2_BASE = "https://api2.cursor.sh"
TRANSCRIBE_PATH = "/aiserver.v1.AiService/TranscribeAudio"
AVAILABLE_MODELS_PATH = "/aiserver.v1.AiService/AvailableModels"
GET_USABLE_MODELS_PATH = "/aiserver.v1.AiService/GetUsableModels"
GET_DEFAULT_MODEL_PATH = "/aiserver.v1.AiService/GetDefaultModelForCli"

# api2.cursor.sh OAuth token endpoint (used by the not-yet-implemented refresh).
OAUTH_TOKEN_PATH = "/oauth/token"


class AuthError(RuntimeError):
    """Cursor auth token is missing, malformed, or expired (surfaced as HTTP 401)."""


_CURSOR_DIR = ".cursor"


def _auth_candidate_paths() -> list[Path]:
    """Ordered paths where Cursor may store an access token."""
    home = Path.home()
    return [
        home / ".config" / "cursor" / "auth.json",
        home / _CURSOR_DIR / "auth.json",
        home / _CURSOR_DIR / "cli-config.json",
    ]


def _token_from_mapping(data: dict[str, Any]) -> str | None:
    """Pull an access token from a JSON object (top-level or authInfo)."""
    for key in ("accessToken", "access_token"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    auth_info = data.get("authInfo")
    if isinstance(auth_info, dict):
        for key in ("accessToken", "access_token"):
            val = auth_info.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _jwt_payload(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        seg = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(seg))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _jwt_exp(token: str) -> float | None:
    """Best-effort 'exp' (unix seconds) from a JWT access token; None if absent."""
    payload = _jwt_payload(token)
    if not payload:
        return None
    exp = payload.get("exp")
    try:
        return float(exp) if exp is not None else None
    except (TypeError, ValueError):
        return None


def read_cli_auth_info() -> dict[str, Any]:
    """Non-secret account metadata from ``~/.cursor/cli-config.json`` (if any)."""
    path = Path.home() / _CURSOR_DIR / "cli-config.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    info = data.get("authInfo")
    return dict(info) if isinstance(info, dict) else {}


def _account_fields(data: dict[str, Any]) -> tuple[str | None, str | None]:
    nested = data.get("authInfo") if isinstance(data.get("authInfo"), dict) else {}
    email = nested.get("email") if isinstance(nested.get("email"), str) else None
    display = nested.get("displayName") if isinstance(nested.get("displayName"), str) else None
    return email, display


def _auth_status(
    *,
    authenticated: bool,
    email: str | None,
    display_name: str | None,
    expires_at: float | None,
    auth_path: str | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "authenticated": authenticated,
        "email": email,
        "display_name": display_name,
        "expires_at": expires_at,
        "auth_path": auth_path,
        "error": error,
    }


def peek_auth() -> dict[str, Any]:
    """Inspect on-disk Cursor auth without raising.

    Returns keys: authenticated, email, display_name, expires_at, auth_path, error.
    Never includes the raw access token.
    """
    info = read_cli_auth_info()
    email = info.get("email") if isinstance(info.get("email"), str) else None
    display_name = (
        info.get("displayName") if isinstance(info.get("displayName"), str) else None
    )

    last_error: str | None = None
    for path in _auth_candidate_paths():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            last_error = f"{path} unreadable: {e}"
            continue
        if not isinstance(data, dict):
            continue
        nested_email, nested_display = _account_fields(data)
        email = email or nested_email
        display_name = display_name or nested_display

        token = _token_from_mapping(data)
        if not token:
            continue
        exp = _jwt_exp(token)
        if exp is not None and exp <= time.time():
            return _auth_status(
                authenticated=False,
                email=email,
                display_name=display_name,
                expires_at=exp,
                auth_path=str(path),
                error="access token expired",
            )
        return _auth_status(
            authenticated=True,
            email=email,
            display_name=display_name,
            expires_at=exp,
            auth_path=str(path),
            error=None,
        )

    if email or display_name:
        return _auth_status(
            authenticated=False,
            email=email,
            display_name=display_name,
            expires_at=None,
            auth_path=None,
            error=last_error or "no access token found",
        )
    return _auth_status(
        authenticated=False,
        email=None,
        display_name=None,
        expires_at=None,
        auth_path=None,
        error=last_error or "Cursor auth not found",
    )


def peek_token(
    token: str,
    *,
    email: str | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Inspect a stored access token without raising. Never echoes the token."""
    raw = (token or "").strip()
    if not raw:
        return {
            "authenticated": False,
            "email": email,
            "display_name": display_name,
            "expires_at": None,
            "auth_path": None,
            "error": "no access token",
        }
    exp = _jwt_exp(raw)
    payload = _jwt_payload(raw) or {}
    if email is None and isinstance(payload.get("email"), str):
        email = payload["email"]
    if exp is not None and exp <= time.time():
        return {
            "authenticated": False,
            "email": email,
            "display_name": display_name,
            "expires_at": exp,
            "auth_path": None,
            "error": "access token expired",
        }
    return {
        "authenticated": True,
        "email": email,
        "display_name": display_name,
        "expires_at": exp,
        "auth_path": None,
        "error": None,
    }


def resolve_access_token(token: str | None = None) -> str:
    """Use *token* when provided; otherwise load the host CLI credential."""
    raw = (token or "").strip()
    if raw:
        peek = peek_token(raw)
        if not peek.get("authenticated"):
            raise AuthError(peek.get("error") or "Cursor access token expired")
        return raw
    return load_auth()


def load_auth() -> str:
    """Return a non-expired Cursor access token from a known on-disk location."""
    tried: list[str] = []
    for path in _auth_candidate_paths():
        if not path.exists():
            continue
        tried.append(str(path))
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise AuthError(f"Cursor auth file {path} is unreadable: {e}") from e
        if not isinstance(data, dict):
            continue
        token = _token_from_mapping(data)
        if not token:
            continue
        # Re-read on every request so external refresh/login is picked up live.
        exp = _jwt_exp(token)
        if exp is not None and exp <= time.time():
            raise AuthError(
                "Cursor access token expired. Re-login with the Cursor app / "
                f"`agent login` to refresh {path}."
            )
        return token

    if tried:
        raise AuthError(
            "No accessToken in Cursor auth files "
            f"({', '.join(tried)}). Re-login with `agent login`."
        )
    paths = ", ".join(str(p) for p in _auth_candidate_paths())
    raise FileNotFoundError(
        f"Cursor auth not found (tried {paths}). "
        "Log in with the Cursor desktop app / `agent login` first."
    )


async def refresh_token(refresh_tok: str, machine_id: str) -> str:
    """Exchange a refresh token for a fresh access token.

    Not yet implemented. A later phase will perform a
    ``POST {API2_BASE}{OAUTH_TOKEN_PATH}`` (api2.cursor.sh/oauth/token) refresh
    flow and return the new access token.
    """
    raise NotImplementedError(
        "token refresh is not implemented yet; re-login with the Cursor app / "
        "`agent login` to refresh on-disk Cursor credentials"
    )


# ── Header builders ───────────────────────────────────────────────────────────

def _aiservice_headers(token: str) -> dict:
    return {
        "authorization": f"Bearer {token}",
        "connect-protocol-version": "1",
        "content-type": "application/proto",
        "user-agent": "connect-es/1.6.1",
        "x-cursor-client-type": "cli",
        "x-cursor-client-version": CLIENT_VERSION,
        "x-ghost-mode": "true",
    }


def _build_headers(token: str, request_id: str) -> list[tuple[str, str]]:
    return [
        (":method", "POST"),
        (":scheme", "https"),
        (":authority", AGENT_HOST),
        (":path", RUN_PATH),
        ("te", "trailers"),
        ("authorization", f"Bearer {token}"),
        ("connect-accept-encoding", "identity"),
        ("connect-protocol-version", "1"),
        ("content-type", "application/connect+proto"),
        ("user-agent", "connect-es/1.6.1"),
        ("x-cursor-client-type", "cli"),
        ("x-cursor-client-version", CLIENT_VERSION),
        ("x-ghost-mode", "true"),
        ("x-original-request-id", request_id),
        ("x-request-id", request_id),
    ]
