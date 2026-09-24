"""Offline tests for CursorClient / AgentSession."""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest

from cursor_core import Checkpoint, CursorClient, RunConfig
from cursor_core.client import AgentSession


@pytest.mark.asyncio
async def test_client_session_and_token(monkeypatch):
    client = CursorClient(access_token="tok")
    monkeypatch.setattr(
        "cursor_core.client.resolve_access_token",
        lambda token: token or "tok",
    )
    assert client.token() == "tok"
    session = client.session()
    assert isinstance(session, AgentSession)
    assert session.checkpoint.conversation_id == ""


@pytest.mark.asyncio
async def test_session_run_persists_checkpoint():
    client = CursorClient(access_token="tok")
    session = AgentSession(client, checkpoint=Checkpoint(blobs={b"a": b"1"}))
    blob = base64.b64encode(b"STATE").decode()

    async def fake_run_turn(*_a, **_k):
        yield "text", "hi"
        yield "checkpoint", blob

    with patch("cursor_core.client.run_turn", fake_run_turn):
        events = []
        async for kind, payload in session.run([{"role": "user", "content": "x"}]):
            events.append((kind, payload))

    assert events[0] == ("text", "hi")
    assert session.checkpoint.state == b"STATE"


@pytest.mark.asyncio
async def test_session_default_kv_round_trip():
    session = AgentSession(CursorClient(access_token="tok"))
    kv = session._wrap_kv(None)
    assert await kv("set", b"k", b"v") is None
    assert await kv("get", b"k", None) == b"v"
    assert await kv("get", b"missing", None) is None
    assert await kv("noop", None, None) is None


@pytest.mark.asyncio
async def test_session_custom_kv_passthrough():
    session = AgentSession(CursorClient(access_token="tok"))
    inner = AsyncMock(return_value=b"from-host")
    kv = session._wrap_kv(inner)
    result = await kv("get", b"k", None)
    assert result == b"from-host"
    inner.assert_awaited_once()


def test_seed_run_config_copies_checkpoint():
    ckpt = Checkpoint(
        state=b"s",
        conversation_id="cid",
        group_id="gid",
        blobs={b"b": b"1"},
    )
    session = AgentSession(CursorClient(access_token="tok"), checkpoint=ckpt)
    cfg = session._seed_run_config(None)
    assert cfg.conversation_id == "cid"
    assert cfg.conversation_group_id == "gid"
    assert cfg.conversation_state == b"s"
    assert cfg.initial_blobs == {b"b": b"1"}
    assert cfg.access_token == "tok"
    assert cfg.session_out == {}


@pytest.mark.asyncio
async def test_list_models_shapes_catalog(monkeypatch):
    from cursor_core.models import CatalogModel

    async def fake_catalog(**_k):
        return [CatalogModel("gpt-5.5", "GPT", True, True)], "gpt-5.5"

    monkeypatch.setattr("cursor_core.client.get_catalog", fake_catalog)
    monkeypatch.setattr(
        "cursor_core.client.resolve_access_token",
        lambda token: "tok",
    )
    client = CursorClient(access_token="tok")
    listing = await client.list_models()
    assert listing["default_id"] == "gpt-5.5"
    assert listing["models"] == [{"id": "gpt-5.5", "name": "GPT"}]


@pytest.mark.asyncio
async def test_session_custom_kv_set_updates_checkpoint():
    session = AgentSession(CursorClient(access_token="tok"))
    inner = AsyncMock(return_value=None)
    kv = session._wrap_kv(inner)
    await kv("set", b"k", b"v")
    assert session.checkpoint.blobs[b"k"] == b"v"
    await kv("other", None, None)


def test_remember_checkpoint_and_seed_ids():
    session = AgentSession(CursorClient(access_token="tok"))
    cfg = RunConfig(conversation_id="cid", conversation_group_id="gid")
    session._remember_checkpoint("checkpoint", "!!!not-b64!!!", cfg)
    session._remember_checkpoint("text", "x", cfg)
    assert session.checkpoint.conversation_id == "cid"
    assert session.checkpoint.group_id == "gid"


@pytest.mark.asyncio
async def test_client_transcribe_and_load_auth(monkeypatch):
    client = CursorClient(access_token="tok")
    monkeypatch.setattr("cursor_core.client.load_auth", lambda: "tok")
    assert client.load_auth() == "tok"

    async def fake_transcribe(*_a, **_k):
        return "hello"

    monkeypatch.setattr("cursor_core.client._transcribe", fake_transcribe)
    assert await client.transcribe(b"wav") == "hello"
