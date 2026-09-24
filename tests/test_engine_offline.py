"""Offline contract tests for cursor_core.engine.

These tests replay recorded-like server message sequences against the engine
WITHOUT hitting the network. They patch _h2_run_loop to inject synthetic
events and verify that:
  - The correct events are yielded by run_turn
  - exec_handler is invoked with properly-parsed ServerMsg structs
  - kv_handler correctly stores and retrieves blobs
  - The wire layer correctly parses synthetic protobuf bytes
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cursor_core import run_turn
from cursor_core.engine import ToolRunTracker, _exec_type_of
from cursor_core.framing import connect_frame, pb_int, pb_msg, pb_str
from cursor_core.wire import (
    ServerMsg,
    parse_frames,
    parse_server_message,
)


# ── Synthetic protobuf builders ───────────────────────────────────────────────

def _make_text_delta_bytes(text: str) -> bytes:
    """AgentServerMessage{f1: InteractionUpdate{f1: TextDelta{f1: text}}}."""
    text_delta = pb_str(1, text)           # TextDelta { f1: text }
    interaction_update = pb_msg(1, text_delta)  # InteractionUpdate { f1: text_delta }
    return pb_msg(1, interaction_update)   # AgentServerMessage { f1: interaction_update }


def _make_turn_ended_bytes() -> bytes:
    """AgentServerMessage{f1: InteractionUpdate{f14: TurnEnded{}}}."""
    turn_ended = pb_msg(14, b"")           # TurnEnded {}
    interaction_update = turn_ended        # InteractionUpdate { f14: turn_ended }
    return pb_msg(1, interaction_update)


def _make_request_context_bytes(exec_id: int = 1, exec_id_str: str = "ctx-1") -> bytes:
    """AgentServerMessage{f2: ExecServerMessage{f1: exec_id, f10: RequestContextArgs{}}}."""
    exec_msg = (
        pb_int(1, exec_id)
        + pb_str(15, exec_id_str)
        + pb_msg(10, b"")                  # f10 = request_context_args (empty)
    )
    return pb_msg(2, exec_msg)


def _make_read_args_bytes(
    path: str,
    exec_id: int = 2,
    exec_id_str: str = "read-1",
) -> bytes:
    """AgentServerMessage{f2: ExecServerMessage{f1: id, f7: ReadArgs{f1: path}}}."""
    read_args = pb_str(1, path)            # ReadArgs { f1: path }
    exec_msg = (
        pb_int(1, exec_id)
        + pb_str(15, exec_id_str)
        + pb_msg(7, read_args)             # f7 = read_args
    )
    return pb_msg(2, exec_msg)


def _make_delete_args_bytes(
    path: str,
    exec_id: int = 3,
    exec_id_str: str = "delete-1",
) -> bytes:
    """AgentServerMessage{f2: ExecServerMessage{f1: id, f4: DeleteArgs{f1: path}}}."""
    delete_args = pb_str(1, path)
    exec_msg = (
        pb_int(1, exec_id)
        + pb_str(15, exec_id_str)
        + pb_msg(4, delete_args)           # f4 = delete_args
    )
    return pb_msg(2, exec_msg)


def _make_kv_set_bytes(
    kv_id: int,
    blob_id: bytes,
    blob_data: bytes,
) -> bytes:
    """AgentServerMessage{f4: KvServerMessage{f1: id, f3: SetBlobArgs{...}}}."""
    set_args = (
        pb_msg(1, blob_id)                 # SetBlobArgs { f1: blob_id }
        + pb_msg(2, blob_data)             # SetBlobArgs { f2: blob_data }
    )
    kv_msg = pb_int(1, kv_id) + pb_msg(3, set_args)
    return pb_msg(4, kv_msg)              # AgentServerMessage { f4: kv_msg }


def _make_kv_get_bytes(kv_id: int, blob_id: bytes) -> bytes:
    """AgentServerMessage{f4: KvServerMessage{f1: id, f2: GetBlobArgs{f1: blob_id}}}."""
    get_args = pb_msg(1, blob_id)          # GetBlobArgs { f1: blob_id }
    kv_msg = pb_int(1, kv_id) + pb_msg(2, get_args)
    return pb_msg(4, kv_msg)


# ── Wire-layer parsing tests ──────────────────────────────────────────────────

class TestWireParsing:
    """Verify parse_server_message correctly decodes synthetic protobuf bytes."""

    def test_text_delta(self):
        raw = _make_text_delta_bytes("hello world")
        msg = parse_server_message(raw)
        assert msg.text_delta == "hello world"
        assert not msg.turn_ended
        assert not msg.wants_context

    def test_turn_ended(self):
        raw = _make_turn_ended_bytes()
        msg = parse_server_message(raw)
        assert msg.turn_ended
        assert msg.text_delta is None

    def test_request_context(self):
        raw = _make_request_context_bytes(exec_id=42, exec_id_str="ctx-xyz")
        msg = parse_server_message(raw)
        assert msg.wants_context
        assert msg.exec_id == 42
        assert msg.exec_id_str == "ctx-xyz"

    def test_read_args(self):
        raw = _make_read_args_bytes("README.md", exec_id=7, exec_id_str="r-1")
        msg = parse_server_message(raw)
        assert msg.wants_read
        assert msg.read_path == "README.md"
        assert msg.exec_id == 7
        assert msg.exec_id_str == "r-1"

    def test_delete_args(self):
        raw = _make_delete_args_bytes("tmp/staging.md", exec_id=8, exec_id_str="d-1")
        msg = parse_server_message(raw)
        assert msg.wants_delete
        assert not msg.wants_read
        assert msg.read_path == "tmp/staging.md"
        assert msg.exec_id == 8
        assert msg.exec_id_str == "d-1"
        assert _exec_type_of(msg) == "delete"

    def test_subagent_args(self):
        """Native Task tool arrives as ExecServerMessage.subagent_args (f28)."""
        args = (
            pb_str(1, "tc-9")
            + pb_str(2, "explore")
            + pb_str(4, "Find skill MCP tools")
            + pb_int(7, 1)
        )
        exec_msg = pb_int(1, 19) + pb_str(15, "sub-exec") + pb_msg(28, args)
        raw = pb_msg(2, exec_msg)
        msg = parse_server_message(raw)
        assert msg.wants_subagent
        assert msg.subagent_prompt == "Find skill MCP tools"
        assert msg.subagent_run_in_background is True
        assert msg.subagent_tool_call_id == "tc-9"
        assert _exec_type_of(msg) == "subagent"

    def test_kv_set(self):
        raw = _make_kv_set_bytes(kv_id=3, blob_id=b"\x01\x02", blob_data=b"some data")
        msg = parse_server_message(raw)
        assert msg.kv_op == "set"
        assert msg.kv_id == 3
        assert msg.kv_blob_id == b"\x01\x02"
        assert msg.kv_blob_data == b"some data"

    def test_kv_get(self):
        raw = _make_kv_get_bytes(kv_id=4, blob_id=b"\xaa\xbb")
        msg = parse_server_message(raw)
        assert msg.kv_op == "get"
        assert msg.kv_id == 4
        assert msg.kv_blob_id == b"\xaa\xbb"

    def test_parse_frames_single(self):
        data = _make_text_delta_bytes("hi")
        frame = connect_frame(data)
        frames, leftover = parse_frames(frame)
        assert len(frames) == 1
        assert leftover == b""
        flag, body = frames[0]
        assert flag == 0x00
        assert body == data

    def test_parse_frames_multiple(self):
        data1 = _make_text_delta_bytes("a")
        data2 = _make_turn_ended_bytes()
        combined = connect_frame(data1) + connect_frame(data2)
        frames, leftover = parse_frames(combined)
        assert len(frames) == 2
        assert leftover == b""

    def test_tool_call_started_preserves_call_id(self):
        tool_call = pb_msg(5, b"")  # grep
        started = pb_str(1, "call-xyz") + pb_msg(2, tool_call)
        raw = pb_msg(1, pb_msg(2, started))
        msg = parse_server_message(raw)
        assert msg.tool_event == ("started", "grep")
        assert msg.tool_event_call_id == "call-xyz"

    def test_tool_call_completed_preserves_call_id(self):
        tool_call = pb_msg(5, b"")
        completed = pb_str(1, "call-xyz") + pb_msg(2, tool_call)
        raw = pb_msg(1, pb_msg(3, completed))
        msg = parse_server_message(raw)
        assert msg.tool_event == ("completed", "grep")
        assert msg.tool_event_call_id == "call-xyz"

    def test_get_mcp_tools_exec_field_44(self):
        args = pb_str(3, "email_(get|search|list)") + pb_str(4, "call-44")
        exec_msg = pb_int(1, 9) + pb_str(15, "gmt-1") + pb_msg(44, args)
        msg = parse_server_message(pb_msg(2, exec_msg))
        assert msg.wants_get_mcp_tools
        assert msg.get_mcp_tools_pattern == "email_(get|search|list)"
        assert msg.get_mcp_tools_call_id == "call-44"
        assert msg.get_mcp_tools_result_field == 44
        assert msg.exec_id == 9
        assert _exec_type_of(msg) == "get_mcp_tools"

    def test_get_mcp_tools_wrapped_args_unused_field(self):
        args = pb_msg(1, pb_str(3, "web_search|web_fetch") + pb_str(4, "cid"))
        exec_msg = pb_int(1, 10) + pb_msg(56, args)
        msg = parse_server_message(pb_msg(2, exec_msg))
        assert msg.wants_get_mcp_tools
        assert msg.get_mcp_tools_result_field == 56
        assert msg.get_mcp_tools_pattern == "web_search|web_fetch"

    def test_unknown_asm_field_is_recorded(self):
        raw = pb_msg(9, pb_int(1, 42))
        msg = parse_server_message(raw)
        assert msg.unknown_top_fields
        assert msg.unknown_top_fields[0][0] == 9
        assert not msg.wants_get_mcp_tools

    def test_git_diff_shaped_field_44_is_not_get_mcp_tools(self):
        blob = pb_str(1, "/tmp/workspace") + pb_int(4, 1)
        exec_msg = pb_int(1, 3) + pb_msg(44, blob)
        msg = parse_server_message(pb_msg(2, exec_msg))
        assert not msg.wants_get_mcp_tools
        assert msg.unknown_exec_field == 44

    def test_mcp_state_exec_field_36(self):
        args = pb_str(1, "demo")
        exec_msg = pb_int(1, 11) + pb_str(15, "st-1") + pb_msg(36, args)
        msg = parse_server_message(pb_msg(2, exec_msg))
        assert msg.wants_mcp_state
        assert msg.mcp_state_servers == ["demo"]
        assert not msg.wants_get_mcp_tools
        assert _exec_type_of(msg) == "mcp_state"


class TestToolRunTracker:
    def test_parallel_complete_uses_matching_id_not_latest_start(self):
        t = ToolRunTracker()
        a = t.start("mcp:set_chat_title", "call-a")
        b = t.start("grep", "call-b")
        assert t.complete("mcp:set_chat_title", "call-a") == a
        assert t.active_run_id == b
        assert t.complete("grep", "call-b") == b
        assert not t

    def test_complete_without_call_id_matches_label(self):
        t = ToolRunTracker()
        grep_id = t.start("grep")
        t.start("read")
        assert t.complete("grep") == grep_id
        assert t.active_label == "read"

    def test_leftovers_close_open_cards(self):
        t = ToolRunTracker()
        rid = t.start("task", "tc-1")
        left = t.leftovers()
        assert left == [(rid, "task")]
        assert not t

    def test_tool_started_without_dispatch_skips_tool_stall(self):
        t = ToolRunTracker()
        t.start("get_mcp_tools", "call-44")
        assert t
        assert not t.uses_tool_stall
        t.mark_dispatched()
        assert t.uses_tool_stall
        t.complete("get_mcp_tools", "call-44")
        assert not t
        assert not t.uses_tool_stall


# ── run_turn contract tests (mocked H2 loop) ─────────────────────────────────

@pytest.mark.asyncio
async def test_chat_only_turn():
    """Scenario 1: a pure Q&A turn.

    The mocked loop emits:
      RequestContextArgs (context handshake) → text_delta → turn_ended
    We verify run_turn yields the correct text event.
    """

    async def fake_h2_loop(
        messages, model, conversation_id, cfg,
        *,
        output_queue,
        exec_handler,
        kv_handler,
        approval_handler,
    ):
        # Simulate the context handshake (wants_context is handled inline in
        # _h2_run_loop; we only need to emit the visible outputs for this test).
        await output_queue.put(("text", "The answer is 4."))
        await output_queue.put(("_done", ""))

    with patch("cursor_core.engine._h2_run_loop", side_effect=fake_h2_loop):
        events = []
        async for kind, payload in run_turn(
            [{"role": "user", "content": "What is 2+2?"}]
        ):
            events.append((kind, payload))

    assert any(k == "text" for k, _ in events), "expected at least one text event"
    text = "".join(p for k, p in events if k == "text")
    assert "4" in text


@pytest.mark.asyncio
async def test_read_tool_turn():
    """Scenario 2: a read-tool turn.

    The mocked loop calls exec_handler with a read request for README.md.
    We verify:
      - exec_handler is invoked with exec_type='read' and the correct path
      - exec_handler returns a non-empty list of bytes (a valid read response)
      - run_turn surfaces the text event emitted after the exec completes
    """
    captured_exec_calls: list[tuple[str, ServerMsg]] = []

    async def stub_exec(exec_type: str, smsg: ServerMsg) -> list[bytes]:
        captured_exec_calls.append((exec_type, smsg))
        return [b"ok"]

    async def fake_h2_loop(
        messages, model, conversation_id, cfg,
        *,
        output_queue,
        exec_handler,
        kv_handler,
        approval_handler,
    ):
        smsg = ServerMsg()
        smsg.wants_read = True
        smsg.read_path = "README.md"
        smsg.exec_id = 1
        smsg.exec_id_str = "read-1"

        frames = await exec_handler("read", smsg)
        assert len(frames) == 1
        assert isinstance(frames[0], bytes)
        assert len(frames[0]) > 0

        await output_queue.put(("text", "I read the file"))
        await output_queue.put(("_done", ""))

    with patch("cursor_core.engine._h2_run_loop", side_effect=fake_h2_loop):
        events = []
        async for kind, payload in run_turn(
            [{"role": "user", "content": "read README.md"}],
            exec_handler=stub_exec,
        ):
            events.append((kind, payload))

    assert captured_exec_calls, "exec_handler never called"
    assert captured_exec_calls[0][0] == "read"
    assert captured_exec_calls[0][1].read_path == "README.md"
    assert any(k == "text" for k, _ in events)


@pytest.mark.asyncio
async def test_get_mcp_tools_followup_frame_invokes_handler():
    """Unknown-style exec follow-up (field 44) is parsed and answered."""
    captured: list[tuple[str, ServerMsg, list[bytes]]] = []

    async def stub_exec(exec_type: str, smsg: ServerMsg) -> list[bytes]:
        frames = [b"tools-ok"]
        captured.append((exec_type, smsg, frames))
        return frames

    async def fake_h2_loop(
        messages, model, conversation_id, cfg,
        *,
        output_queue,
        exec_handler,
        kv_handler,
        approval_handler,
    ):
        args = pb_str(3, "email_(get|search|list)") + pb_str(4, "call-44")
        raw = pb_msg(2, pb_int(1, 9) + pb_str(15, "gmt-1") + pb_msg(44, args))
        smsg = parse_server_message(raw)
        exec_type = _exec_type_of(smsg)
        frames = await exec_handler(exec_type, smsg)
        captured.append((exec_type, smsg, frames))
        await output_queue.put(("text", "found email_get"))
        await output_queue.put(("_done", ""))

    with patch("cursor_core.engine._h2_run_loop", side_effect=fake_h2_loop):
        events = []
        async for kind, payload in run_turn(
            [{"role": "user", "content": "find email tools"}],
            exec_handler=stub_exec,
        ):
            events.append((kind, payload))

    assert captured, "exec_handler never called"
    exec_type, smsg, frames = captured[0]
    assert exec_type == "get_mcp_tools"
    assert smsg.get_mcp_tools_result_field == 44
    assert len(frames) == 1 and frames[0]
    assert any(k == "text" for k, _ in events)


@pytest.mark.asyncio
async def test_kv_blob_round_trip():
    """Scenario 3: KV blob set → get round-trip.

    The mocked loop exercises the kv_handler passed into run_turn:
      1. set_blob with key=b'k1', data=b'my value'
      2. get_blob for the same key → data must be returned
      3. get_blob for unknown key → None
    """
    recorded_kv: list[tuple[str, bytes | None, bytes | None, bytes | None]] = []

    async def fake_h2_loop(
        messages, model, conversation_id, cfg,
        *,
        output_queue,
        exec_handler,
        kv_handler,
        approval_handler,
    ):
        # Step 1: set blob.
        result = await kv_handler("set", b"k1", b"my value")
        recorded_kv.append(("set", b"k1", b"my value", result))

        # Step 2: get back the blob we just set.
        data = await kv_handler("get", b"k1", None)
        recorded_kv.append(("get", b"k1", None, data))

        # Step 3: get an unknown key → None.
        missing = await kv_handler("get", b"no-such-key", None)
        recorded_kv.append(("get", b"no-such-key", None, missing))

        await output_queue.put(("text", "kv ok"))
        await output_queue.put(("_done", ""))

    with patch("cursor_core.engine._h2_run_loop", side_effect=fake_h2_loop):
        events = []
        async for kind, payload in run_turn(
            [{"role": "user", "content": "kv test"}]
        ):
            events.append((kind, payload))

    # Assertions on KV operations.
    assert len(recorded_kv) == 3, "expected exactly 3 KV ops"

    op_set = recorded_kv[0]
    assert op_set[0] == "set"
    assert op_set[3] is None  # set returns None

    op_get_found = recorded_kv[1]
    assert op_get_found[0] == "get"
    assert op_get_found[3] == b"my value"  # should be found

    op_get_missing = recorded_kv[2]
    assert op_get_missing[0] == "get"
    assert op_get_missing[3] is None  # not found

    # run_turn still emitted the text event.
    assert any(k == "text" for k, _ in events)


# ── Native checkpoint / wait / suspend clock ─────────────────────────────────

def _frame_fixtures() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        cand = p / "tests" / "fixtures" / "cursor_frames"
        if cand.is_dir():
            return cand
        cand2 = p / "fixtures" / "cursor_frames"
        if cand2.is_dir():
            return cand2
    raise FileNotFoundError("cursor_frames fixtures")


def test_asm_f3_fixture_parses_as_checkpoint():
    raw = (_frame_fixtures() / "asm_f3_checkpoint.bin").read_bytes()
    msg = parse_server_message(raw)
    assert msg.checkpoint
    assert b"msg_REDACTED" in msg.checkpoint
    assert not msg.unknown_top_fields


def test_asm_f5_fixture_parses_as_exec_control():
    raw = (_frame_fixtures() / "asm_f5_exec_control.bin").read_bytes()
    msg = parse_server_message(raw)
    assert msg.exec_control
    assert not msg.unknown_top_fields


def test_await_toolcall_fixture():
    raw = (_frame_fixtures() / "toolcall_await.bin").read_bytes()
    msg = parse_server_message(raw)
    assert msg.tool_event == ("started", "await")
    assert "5000ms" in (msg.tool_detail or "")


def test_exec_wait_fixture():
    raw = (_frame_fixtures() / "exec_wait.bin").read_bytes()
    msg = parse_server_message(raw)
    assert msg.wants_wait
    assert msg.wait_ms == 5000
    assert msg.wait_pattern == "done"
    assert msg.wait_session_id == "sess-redacted"
    assert _exec_type_of(msg) == "wait"


def test_build_steer_message_is_conversation_action():
    from cursor_core.engine import build_steer_message
    from cursor_core.wire import _read_field

    body = build_steer_message("look at the other host", "mid-1")
    top = _read_field(body, 0)
    assert top is not None
    field, wire, val, pos = top
    assert field == 4 and wire == 2
    assert _read_field(body, pos) is None
    action = _read_field(val, 0)
    assert action is not None and action[0] == 1
    uma = _read_field(action[2], 0)
    assert uma is not None and uma[0] == 1
    assert b"look at the other host" in uma[2]
    assert b"mid-1" in uma[2]


def test_user_message_appended_parses_message_id():
    from cursor_core.framing import pb_msg, pb_str
    from cursor_core.wire import parse_server_message

    user = pb_str(1, "look at the other host") + pb_str(2, "mid-1")
    appended = pb_msg(1, user)
    raw = pb_msg(1, pb_msg(6, appended))
    msg = parse_server_message(raw)
    assert msg.user_message_appended_id == "mid-1"


def test_build_run_request_with_checkpoint_is_delta():
    from cursor_core.engine import RunConfig, build_run_request

    state = b"CKPT\x00BLOB"
    body = build_run_request(
        [{"role": "user", "content": "what was the codeword?"}],
        "auto",
        conversation_id="cid-1",
        run_config=RunConfig(conversation_state=state, conversation_id="cid-1"),
    )
    assert b"CKPT\x00BLOB" in body
    assert b"what was the codeword?" in body
    assert b"earlier secret" not in body


def test_turn_expired_pauses_for_suspend():
    from cursor_core.engine import _turn_expired

    assert not _turn_expired(0.0, 50.0, 100.0, 60.0)
    assert _turn_expired(0.0, 0.0, 100.0, 60.0)


@pytest.mark.asyncio
async def test_execute_wait_sleeps_and_acks():
    from cursor_core.engine import execute_wait
    from cursor_core.wire import ServerMsg

    smsg = ServerMsg()
    smsg.exec_id = 3
    smsg.exec_id_str = "wait-1"
    smsg.wait_ms = 10
    smsg.wait_result_field = 40
    frames = await execute_wait(smsg)
    assert frames
    assert smsg.ui_output.startswith("waited")


@pytest.mark.asyncio
async def test_default_wait_handler_does_not_reject():
    from cursor_core.engine import _make_default_exec_handler
    from cursor_core.wire import ServerMsg

    smsg = ServerMsg()
    smsg.exec_id = 1
    smsg.exec_id_str = "w"
    smsg.wants_wait = True
    smsg.wait_ms = 1
    smsg.wait_result_field = 40
    handler = _make_default_exec_handler("/tmp")
    frames = await handler("wait", smsg)
    assert frames

