"""Offline tests for engine helpers that do not open an H2 stream."""
from __future__ import annotations

import base64
import ipaddress
from pathlib import Path
from unittest.mock import patch

import pytest

from cursor_core.engine import (
    RunConfig,
    _NoRedirect,
    _decode_image_url,
    _execute_local_grep,
    _extract_documents,
    _extract_images,
    _extract_system_prompt,
    _fetch_ssrf_guard,
    _flatten_content,
    _is_blocked_ip,
    _make_default_exec_handler,
    _make_default_kv_handler,
    _parse_available_models,
    _parse_one_model,
    _parse_transcribe_response,
    _parse_usable_models,
    _reject_exec_frames,
    _render_prompt,
    _resolve_workspace_path,
    build_interaction_approval,
    build_requested_model,
    build_run_request,
    complete,
)
from cursor_core.framing import pb_int, pb_msg, pb_str
from cursor_core.wire import ServerMsg


def test_blocked_ip_classes():
    assert _is_blocked_ip(ipaddress.ip_address("10.1.2.3"))
    assert _is_blocked_ip(ipaddress.ip_address("192.168.0.9"))
    assert _is_blocked_ip(ipaddress.ip_address("127.0.0.1"))
    assert _is_blocked_ip(ipaddress.ip_address("169.254.1.1"))
    assert _is_blocked_ip(ipaddress.ip_address("100.64.0.1"))
    assert _is_blocked_ip(ipaddress.ip_address("::1"))
    assert not _is_blocked_ip(ipaddress.ip_address("8.8.8.8"))
    assert not _is_blocked_ip(ipaddress.ip_address("2001:4860:4860::8888"))


def test_ssrf_guard_rejects_http_and_private(monkeypatch):
    with pytest.raises(PermissionError, match="scheme"):
        _fetch_ssrf_guard("http://example.com/x")
    with pytest.raises(ValueError, match="no host"):
        _fetch_ssrf_guard("https://")

    monkeypatch.setattr(
        "cursor_core.engine.socket.getaddrinfo",
        lambda *_a, **_k: [(0, 0, 0, 0, ("10.0.0.1", 443))],
    )
    with pytest.raises(PermissionError, match="SSRF"):
        _fetch_ssrf_guard("https://evil.example")

    monkeypatch.setattr(
        "cursor_core.engine.socket.getaddrinfo",
        lambda *_a, **_k: [(0, 0, 0, 0, ("8.8.8.8", 443))],
    )
    _fetch_ssrf_guard("https://example.com/ok")


def test_flatten_and_prompts():
    assert _flatten_content(None) == ""
    assert _flatten_content("hi") == "hi"
    assert _flatten_content([{"type": "text", "text": "a"}, "b"]) == "ab"
    assert _flatten_content(12) == "12"

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "there"},
    ]
    assert _extract_system_prompt(messages) == "sys"
    body = _render_prompt(messages)
    assert "User: hello" in body
    assert "sys" in body
    no_sys = _render_prompt(messages, include_system=False)
    assert "sys" not in no_sys


def test_decode_image_and_extract_parts():
    raw = b"PNGDATA"
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    assert _decode_image_url(url) == ("image/png", raw)
    assert _decode_image_url("http://example.com/x.png") is None
    assert _decode_image_url("not-a-url") is None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {
                    "type": "file",
                    "file": {
                        "filename": "a.pdf",
                        "file_data": "data:application/pdf;base64,"
                        + base64.b64encode(b"%PDF").decode(),
                    },
                },
            ],
        }
    ]
    images = _extract_images(messages)
    assert images == [("image/png", raw)]
    docs = _extract_documents(messages)
    assert docs[0][1] == "a.pdf"
    assert docs[0][2] == b"%PDF"


def test_build_run_request_and_model():
    body = build_run_request(
        [{"role": "user", "content": "hello"}],
        "auto",
        run_config=RunConfig(chat_only=True, conversation_id="cid"),
    )
    assert b"hello" in body
    assert b"cid" in body
    model = build_requested_model("gpt-5.5", [("reasoning", "medium")], max_mode=True)
    assert b"gpt-5.5" in model
    assert b"reasoning" in model

    with pytest.raises(ValueError, match="user message"):
        build_run_request([], "auto")


def test_interaction_approval_and_no_redirect():
    search = build_interaction_approval(3, "search")
    fetch = build_interaction_approval(4, "fetch")
    assert search != fetch
    assert _NoRedirect().redirect_request() is None


def test_parse_model_and_transcribe_bytes():
    one = (
        pb_str(1, "gpt-5.5")
        + pb_int(2, 1)
        + pb_int(5, 1)
        + pb_int(10, 1)
        + pb_str(17, "GPT")
    )
    parsed = _parse_one_model(one)
    assert parsed["name"] == "gpt-5.5"
    assert parsed["supports_agent"] is True
    assert parsed["display"] == "GPT"

    available = _parse_available_models(pb_msg(2, one))
    assert available[0]["name"] == "gpt-5.5"
    usable = _parse_usable_models(pb_msg(1, one))
    assert usable[0]["name"] == "gpt-5.5"

    assert _parse_transcribe_response(pb_str(1, "hello world")) == "hello world"
    assert _parse_transcribe_response(b"") == ""


def test_local_grep_and_workspace(tmp_path: Path):
    (tmp_path / "a.py").write_text("hello_grep")
    status, matches, err = _execute_local_grep(".", "hello_grep", str(tmp_path))
    assert status == "success"
    assert any(str(tmp_path / "a.py") == m or m.endswith("a.py") for m in matches)
    assert err is None

    status, _, err = _execute_local_grep(".", "tool_abcdef12", str(tmp_path))
    assert status == "error"
    assert "tool call id" in (err or "")

    status, matches, err = _execute_local_grep(".", "*.py", str(tmp_path))
    assert status == "success"
    assert any(m.endswith("a.py") for m in matches)

    status, _, err = _execute_local_grep("missing-dir", "x", str(tmp_path))
    assert status == "error"


@pytest.mark.asyncio
async def test_default_handlers():
    kv, store = _make_default_kv_handler({b"seed": b"1"})
    assert await kv("get", b"seed", None) == b"1"
    await kv("set", b"k", b"v")
    assert store[b"k"] == b"v"

    handler = _make_default_exec_handler("/tmp")
    smsg = ServerMsg()
    smsg.exec_id = 1
    smsg.exec_id_str = "x"
    smsg.wants_read = True
    smsg.read_path = "nope.txt"
    frames = await handler("read", smsg)
    assert frames

    unknown = ServerMsg()
    unknown.exec_id = 2
    unknown.exec_id_str = "u"
    unknown.shell_command = "echo"
    unknown.write_path = "a"
    unknown.edit_path = "a"
    unknown.fetch_url = "https://example.com"
    unknown.grep_search_path = "."
    unknown.read_path = "a"
    for kind in ("read", "shell", "write", "edit", "fetch", "grep", "delete", "mcp"):
        frames = _reject_exec_frames(kind, unknown)
        assert isinstance(frames, list)


@pytest.mark.asyncio
async def test_complete_collects_text():
    async def fake_run_turn(*_a, **_k):
        yield "reasoning", "think"
        yield "text", "ans"
        yield "text", "wer"

    with patch("cursor_core.engine.run_turn", fake_run_turn):
        assert await complete([{"role": "user", "content": "q"}]) == "answer"


def test_trailer_error():
    from cursor_core.engine import _trailer_error

    assert _trailer_error(b"not-json") is None
    assert _trailer_error(b'{"error":{"message":"boom"}}') == "boom"
    assert _trailer_error(b'{"error":{"code":7}}') == '{"code": 7}'
    assert _trailer_error(b'{"error":"plain"}') == "plain"
    assert _trailer_error(b'{"ok":true}') is None


def test_ssrf_dns_and_unparseable(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("nxdomain")

    monkeypatch.setattr("cursor_core.engine.socket.getaddrinfo", boom)
    with pytest.raises(PermissionError, match="DNS"):
        _fetch_ssrf_guard("https://missing.example")

    monkeypatch.setattr(
        "cursor_core.engine.socket.getaddrinfo",
        lambda *_a, **_k: [(0, 0, 0, 0, ("not-an-ip", 443))],
    )
    with pytest.raises(PermissionError, match="unparseable"):
        _fetch_ssrf_guard("https://evil.example")


def test_decode_image_percent_and_https(monkeypatch):
    assert _decode_image_url("data:image/png,hello%20world") == (
        "image/png",
        b"hello world",
    )
    with patch("cursor_core.engine.base64.b64decode", side_effect=ValueError("bad")):
        assert _decode_image_url("data:image/png;base64,xxxx") is None

    class _Resp:
        headers = {"Content-Type": "image/jpeg; charset=binary"}

        def read(self):
            return b"JPEG"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr("cursor_core.engine._fetch_ssrf_guard", lambda _url: None)
    monkeypatch.setattr(
        "cursor_core.engine.urllib.request.urlopen",
        lambda *_a, **_k: _Resp(),
    )
    assert _decode_image_url("https://cdn.example/a.jpg") == ("image/jpeg", b"JPEG")

    monkeypatch.setattr(
        "cursor_core.engine.urllib.request.urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("fail")),
    )
    assert _decode_image_url("https://cdn.example/a.jpg") is None


def test_extract_empty_and_partial_parts():
    assert _extract_images([]) == []
    assert _extract_images([{"role": "user", "content": "plain"}]) == []
    assert _extract_documents([]) == []
    assert _extract_documents([{"role": "user", "content": "plain"}]) == []
    assert _extract_documents(
        [{"role": "user", "content": [{"type": "file", "file": "nope"}]}]
    ) == []


def test_build_run_request_with_parts_and_chat_only():
    raw = b"PNG"
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see"},
                {"type": "image_url", "image_url": {"url": url}},
                {
                    "type": "file",
                    "file": {
                        "filename": "a.pdf",
                        "file_data": "data:application/pdf;base64,"
                        + base64.b64encode(b"%PDF").decode(),
                    },
                },
            ],
        },
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "result"},
    ]
    cfg = RunConfig(chat_only=True, tools=[{"name": "x"}])
    body = build_run_request(messages, "auto", run_config=cfg)
    assert b"see" in body
    assert cfg.system_prompt and "chat-only" in cfg.system_prompt.lower()

    cfg2 = RunConfig(bridge_mode="off")
    body2 = build_run_request([{"role": "user", "content": "q"}], "gpt-5.5", run_config=cfg2)
    assert b"q" in body2


def test_kv_workspace_and_fetch_builders():
    from cursor_core.engine import (
        build_exec_fetch_error,
        build_exec_fetch_response,
        build_kv_response,
        build_steer_message,
        build_workspace_context_response,
    )

    assert build_kv_response(1, "set")
    assert build_kv_response(0, "get", None)
    assert build_kv_response(3, "get", b"blob")
    assert build_steer_message("go", "mid")
    ctx = build_workspace_context_response(
        4, "e", system_prompt="sys", mcp_tools_body=b"m", cursor_rules=[b"r"]
    )
    assert ctx
    assert build_exec_fetch_response(5, "f", "https://ex", "ok", 200)
    assert build_exec_fetch_error(5, "f", "https://ex", "nope")
    assert build_exec_fetch_response(0, "", "https://ex", "ok")
    assert build_exec_fetch_error(0, "", "https://ex", "nope")


def test_decode_frame_and_turn_clock():
    from cursor_core.engine import (
        _decode_frame,
        _looks_like_message,
        _turn_expired,
        _wire_log,
    )
    from cursor_core.framing import pb_bytes, pb_int, pb_msg, pb_str

    inner = pb_str(1, "hello") + pb_int(2, 7)
    nested = pb_msg(3, inner)
    text = _decode_frame(nested + pb_bytes(4, b"\xff\xfe"))
    assert "hello" in text
    assert "4#B" in text or "#B" in text
    assert _looks_like_message(b"") is False
    assert _looks_like_message(inner) is True
    assert _looks_like_message(b"\xff") is False
    assert _turn_expired(0, 0, 10, turn_sec=0) is False
    assert _turn_expired(0, 0, 100, turn_sec=1) is True
    _wire_log("x", 1, 0, b"ab")
    _wire_log("x", 1, 0, 3)
    deep = _decode_frame(inner, depth=7)
    assert "deep" in deep


def test_tool_tracker_bind_and_complete_empty():
    from cursor_core.engine import ToolRunTracker, _exec_tool_call_id

    t = ToolRunTracker()
    assert t.complete("missing") == ""
    rid = t.start("grep", "call-1")
    assert t.bind_call("") == rid
    assert t.bind_call("call-1") == rid
    assert t.bind_call("call-1") == rid
    other = t.start("read", "call-2")
    assert t.bind_call("call-1") == rid
    assert t.complete("grep", "call-1") == rid
    assert t.active_run_id == other
    t.complete("read", "call-2")
    rid2 = t.start("grep")
    assert t.bind_call(rid2) == rid2

    smsg = ServerMsg()
    smsg.read_tool_call_id = "tc-9"
    assert _exec_tool_call_id(smsg) == "tc-9"


def test_exec_type_remaining_flags():
    from cursor_core.engine import _exec_type_of

    flags = [
        ("wants_read", "read"),
        ("wants_shell", "shell"),
        ("wants_write", "write"),
        ("wants_grep", "grep"),
        ("wants_edit", "edit"),
        ("wants_delete", "delete"),
        ("wants_fetch", "fetch"),
        ("wants_mcp", "mcp"),
        ("wants_list_mcp_resources", "list_mcp_resources"),
        ("wants_get_mcp_tools", "get_mcp_tools"),
        ("wants_mcp_state", "mcp_state"),
        ("wants_subagent", "subagent"),
        ("wants_force_background_subagent", "force_background_subagent"),
        ("wants_subagent_await", "subagent_await"),
        ("wants_wait", "wait"),
    ]
    for attr, expected in flags:
        msg = ServerMsg()
        setattr(msg, attr, True)
        assert _exec_type_of(msg) == expected
    assert _exec_type_of(ServerMsg()) is None


@pytest.mark.asyncio
async def test_default_exec_grep_and_wait(tmp_path: Path):
    from cursor_core.engine import _default_approval_handler, _emit, execute_wait

    (tmp_path / "a.py").write_text("needle")
    handler = _make_default_exec_handler(str(tmp_path))
    smsg = ServerMsg()
    smsg.exec_id = 1
    smsg.exec_id_str = "g"
    smsg.grep_search_path = "."
    smsg.grep_pattern = "needle"
    frames = await handler("grep", smsg)
    assert frames
    smsg.grep_pattern = "tool_abcdef12"
    err_frames = await handler("grep", smsg)
    assert err_frames
    wait_msg = ServerMsg()
    wait_msg.exec_id = 2
    wait_msg.exec_id_str = "w"
    wait_msg.wait_ms = 0
    waited = await handler("wait", wait_msg)
    assert waited
    mcp_state = ServerMsg()
    mcp_state.exec_id = 9
    mcp_state.exec_id_str = "m"
    mcp_state.unknown_exec_field = 36
    assert _reject_exec_frames("mcp_state", mcp_state)
    assert await _default_approval_handler("search", "1") is True
    assert await _default_approval_handler("shell", "1") is False
    q: list = []

    class _Q:
        async def put(self, item):
            q.append(item)

    assert await _emit(_Q(), "text", "hi") == ("text", "hi")
    assert q == [("text", "hi")]
    assert await _emit(None, "text", "x") == ("text", "x")

    waiter_called = {}

    async def waiter(session_id, pattern, ms):
        waiter_called["v"] = (session_id, pattern, ms)

    wait_msg.wait_session_id = "s"
    wait_msg.wait_pattern = "done"
    wait_msg.wait_ms = 5
    await execute_wait(wait_msg, waiter=waiter)
    assert waiter_called["v"][0] == "s"


@pytest.mark.asyncio
async def test_run_turn_queue_and_errors():
    from cursor_core.engine import run_turn

    async def fake_ok(*_a, **_k):
        q = _k["output_queue"]
        await q.put(("text", "hi"))
        await q.put(("_done", ""))

    with patch("cursor_core.engine._h2_run_loop", fake_ok):
        cfg = RunConfig(session_out={})
        events = [item async for item in run_turn([{"role": "user", "content": "q"}], run_config=cfg)]
    assert events == [("text", "hi")]
    assert "blob_store" in cfg.session_out

    async def fake_raise(*_a, **_k):
        raise RuntimeError("boom")

    with patch("cursor_core.engine._h2_run_loop", fake_raise):
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in run_turn([{"role": "user", "content": "q"}]):
                pass

    async def fake_empty(*_a, **_k):
        return None

    with patch("cursor_core.engine._h2_run_loop", fake_empty):
        events = [item async for item in run_turn([{"role": "user", "content": "q"}])]
    assert events == []
