"""Offline tests for cursor_core.auth."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import pytest

from cursor_core.auth import (
    AuthError,
    _aiservice_headers,
    _auth_candidate_paths,
    _build_headers,
    _jwt_exp,
    _jwt_payload,
    _token_from_mapping,
    load_auth,
    peek_auth,
    peek_token,
    refresh_token,
    resolve_access_token,
)


def _jwt(*, exp: float | None = None, email: str | None = None) -> str:
    payload: dict = {}
    if exp is not None:
        payload["exp"] = exp
    if email is not None:
        payload["email"] = email
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"eyJhbGciOiJub25lIn0.{body}.sig"


def test_token_from_mapping_top_level_and_nested():
    assert _token_from_mapping({"accessToken": " abc "}) == "abc"
    assert _token_from_mapping({"authInfo": {"access_token": "xyz"}}) == "xyz"
    assert _token_from_mapping({}) is None
    assert _token_from_mapping({"accessToken": 1}) is None


def test_jwt_payload_and_exp():
    token = _jwt(exp=1_700_000_000, email="a@b.c")
    payload = _jwt_payload(token)
    assert payload is not None
    assert payload["email"] == "a@b.c"
    assert _jwt_exp(token) == 1_700_000_000
    assert _jwt_payload("not-a-jwt") is None
    assert _jwt_exp("a.b.c") is None


def test_peek_token_empty_expired_valid():
    empty = peek_token("  ")
    assert empty["authenticated"] is False
    assert empty["error"] == "no access token"

    expired = peek_token(_jwt(exp=time.time() - 10, email="x@y.z"))
    assert expired["authenticated"] is False
    assert expired["email"] == "x@y.z"
    assert expired["error"] == "access token expired"

    ok = peek_token(_jwt(exp=time.time() + 3600))
    assert ok["authenticated"] is True
    assert ok["error"] is None


def test_resolve_access_token_rejects_expired():
    with pytest.raises(AuthError, match="expired"):
        resolve_access_token(_jwt(exp=time.time() - 5))


def test_load_auth_from_disk(tmp_path: Path, monkeypatch):
    token = _jwt(exp=time.time() + 3600)
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"accessToken": token}))
    monkeypatch.setattr(
        "cursor_core.auth._auth_candidate_paths",
        lambda: [auth_file],
    )
    assert load_auth() == token
    assert resolve_access_token() == token
    peek = peek_auth()
    assert peek["authenticated"] is True
    assert peek["auth_path"] == str(auth_file)


def test_load_auth_expired_and_missing(tmp_path: Path, monkeypatch):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"accessToken": _jwt(exp=time.time() - 1)}))
    monkeypatch.setattr(
        "cursor_core.auth._auth_candidate_paths",
        lambda: [auth_file],
    )
    with pytest.raises(AuthError, match="expired"):
        load_auth()

    missing = tmp_path / "gone.json"
    monkeypatch.setattr(
        "cursor_core.auth._auth_candidate_paths",
        lambda: [missing],
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        load_auth()


def test_peek_auth_unreadable(tmp_path: Path, monkeypatch):
    bad = tmp_path / "auth.json"
    bad.write_text("{not json")
    monkeypatch.setattr(
        "cursor_core.auth._auth_candidate_paths",
        lambda: [bad],
    )
    monkeypatch.setattr("cursor_core.auth.read_cli_auth_info", lambda: {})
    peek = peek_auth()
    assert peek["authenticated"] is False
    assert "unreadable" in (peek["error"] or "")


def test_headers():
    headers = _aiservice_headers("tok")
    assert headers["authorization"] == "Bearer tok"
    pairs = dict(_build_headers("tok", "req-1"))
    assert pairs[":path"] == "/agent.v1.AgentService/Run"
    assert pairs["x-request-id"] == "req-1"


@pytest.mark.asyncio
async def test_refresh_token_not_implemented():
    with pytest.raises(NotImplementedError):
        await refresh_token("r", "m")


def test_auth_candidate_paths_use_home(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    paths = _auth_candidate_paths()
    assert paths[0] == tmp_path / ".config" / "cursor" / "auth.json"
    assert all(p.parent.name in {"cursor", ".cursor"} or ".cursor" in str(p) for p in paths)


def test_jwt_exp_rejects_non_numeric():
    body = base64.urlsafe_b64encode(json.dumps({"exp": {"n": 1}}).encode()).rstrip(b"=").decode()
    assert _jwt_exp(f"eyJhbGciOiJub25lIn0.{body}.sig") is None
    assert _jwt_payload("a.not-base64.sig") is None
    assert _jwt_payload("a." + base64.urlsafe_b64encode(b"[]").rstrip(b"=").decode() + ".sig") is None


def test_peek_auth_account_without_token(tmp_path: Path, monkeypatch):
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    (cursor_dir / "cli-config.json").write_text(
        json.dumps({"authInfo": {"email": "a@b.c", "displayName": "A"}})
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        "cursor_core.auth._auth_candidate_paths",
        lambda: [tmp_path / "missing.json"],
    )
    peek = peek_auth()
    assert peek["authenticated"] is False
    assert peek["email"] == "a@b.c"
    assert "token" in (peek["error"] or "")

    monkeypatch.setattr("cursor_core.auth.read_cli_auth_info", lambda: {})
    peek2 = peek_auth()
    assert peek2["authenticated"] is False
    assert "not found" in (peek2["error"] or "")

    (cursor_dir / "cli-config.json").write_text("{nope")
    from cursor_core.auth import read_cli_auth_info as read_info

    assert read_info() == {}


def test_load_auth_skips_non_dict_and_empty(tmp_path: Path, monkeypatch):
    listed = tmp_path / "listed.json"
    listed.write_text("[]")
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    monkeypatch.setattr(
        "cursor_core.auth._auth_candidate_paths",
        lambda: [listed, empty],
    )
    with pytest.raises(AuthError, match="No accessToken"):
        load_auth()

    unreadable = tmp_path / "bad.json"
    unreadable.write_text("{nope")
    monkeypatch.setattr(
        "cursor_core.auth._auth_candidate_paths",
        lambda: [unreadable],
    )
    with pytest.raises(AuthError, match="unreadable"):
        load_auth()


def test_peek_auth_skips_bad_files(tmp_path: Path, monkeypatch):
    listed = tmp_path / "listed.json"
    listed.write_text("[]")
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    expired = tmp_path / "expired.json"
    expired.write_text(json.dumps({"accessToken": _jwt(exp=time.time() - 10)}))
    monkeypatch.setattr("cursor_core.auth.read_cli_auth_info", lambda: {})
    monkeypatch.setattr(
        "cursor_core.auth._auth_candidate_paths",
        lambda: [listed, empty, expired],
    )
    peek = peek_auth()
    assert peek["authenticated"] is False
    assert peek["error"] == "access token expired"
