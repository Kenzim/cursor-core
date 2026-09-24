"""
Cursor ``agent.v1`` wire layer: server-message parsing + exec arg/result codec.

This module sits between the raw framing primitives (``cursor_core.framing``) and
the streaming engine (``cursor_core.engine``). It is responsible for two things:

  1. **Decoding** what Cursor sends us mid-stream: ``AgentServerMessage`` and its
     nested oneofs (interaction updates, exec requests, KV blob I/O, web
     search/fetch approval queries). The decoded form is the flat ``ServerMsg``
     struct the engine reacts to.
  2. **Encoding** the per-tool ``ExecClientMessage`` results we send back when the
     server asks the client to run a tool (read / shell / write / delete / grep /
     edit / fetch). These are the ``build_*`` response builders plus the exec-arg
     parsers and their dataclasses.

It has no I/O and no Cursor account knowledge — pure (de)serialization. MCP tool
parsing is intentionally omitted here (it lives in a higher layer in later
phases); the ``ServerMsg.wants_mcp`` / ``mcp_parsed`` slots are preserved for
forward compatibility but are not populated by this module.
"""
import gzip
import json
import re
import struct
from dataclasses import dataclass

from .framing import decode_value, pb_int, pb_msg, pb_str

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_HEX_ID_RE = re.compile(r"^[0-9a-fA-F]{16,}$")
_DETAIL_MAX = 8192
_OUTPUT_MAX = 16384
_TRUNCATED_SUFFIX = "\n… truncated"
_MCP_SKIP_KEYS = {"tool_call_id", "provider_identifier", "provider"}

# Diagnostic: log every inbound ServerMessage / exec field (incl. unhandled) to
# stderr. Used to reverse-engineer the MCP-server handshake Cursor expects.
import os  # noqa: E402 — kept next to its DEBUG_WIRE use for clarity

DEBUG_WIRE = os.getenv("CURSOR_DEBUG_WIRE", "") == "1"

# agent.v1.ToolCall.tool oneof field numbers -> canonical tool names
TOOL_ONEOF_FIELD: dict[int, str] = {
    1: "shell",
    2: "await",
    3: "delete",
    4: "glob",
    5: "grep",
    8: "read",
    9: "update_todos",
    10: "read_todos",
    12: "edit",
    13: "ls",
    14: "read_lints",
    15: "mcp",
    16: "sem_search",
    17: "create_plan",
    18: "web_search",
    19: "task",
    20: "list_mcp_resources",
    21: "read_mcp_resource",
    22: "apply_agent_diff",
    23: "ask_question",
    24: "web_fetch",
    25: "switch_mode",
    26: "exa_search",
    27: "exa_fetch",
    28: "generate_image",
    29: "record_screen",
    30: "computer_use",
    31: "write_shell_stdin",
    32: "reflect",
    33: "setup_vm_environment",
    34: "truncated",
    35: "start_grind_execution",
    36: "start_grind_planning",
    44: "get_mcp_tools",
}

# AgentServerMessage fields we already decode (others are logged + probed).
_ASM_HANDLED_FIELDS = frozenset({1, 2, 3, 4, 5, 7})

# ExecServerMessage fields with dedicated parsers. Unused length-delimited
# fields are probed as GetMcpToolsArgs (same-field args/result convention).
_EXEC_HANDLED_FIELDS = frozenset({
    1, 2, 3, 4, 5, 7, 10, 11, 12, 14, 15, 17, 19, 20, 28, 31, 36, 37,
})


def _wire_log(label: str, field: int, wire: int, val) -> None:
    if not DEBUG_WIRE:
        return
    import sys
    prev = val.hex()[:220] if isinstance(val, (bytes, bytearray)) else val
    print(f"[wire] {label} field={field} wire={wire} val={prev}", file=sys.stderr, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Connect-RPC frame parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_frames(buf: bytes) -> tuple[list[tuple[int, bytes]], bytes]:
    """Pull complete Connect frames out of buf. Returns (frames, leftover)."""
    frames: list[tuple[int, bytes]] = []
    pos = 0
    while pos + 5 <= len(buf):
        flag = buf[pos]
        msg_len = struct.unpack(">I", buf[pos + 1:pos + 5])[0]
        if pos + 5 + msg_len > len(buf):
            break
        frames.append((flag, buf[pos + 5:pos + 5 + msg_len]))
        pos += 5 + msg_len
    return frames, buf[pos:]


def _maybe_gunzip(flag: int, data: bytes) -> bytes:
    return gzip.decompress(data) if (flag & 0x01) else data


# ──────────────────────────────────────────────────────────────────────────────
# Minimal protobuf field reader (server messages)
# ──────────────────────────────────────────────────────────────────────────────

def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
    raise ValueError("truncated varint")


def _read_field(data: bytes, pos: int):
    """Returns (field_num, wire_type, value, new_pos) or None."""
    if pos >= len(data):
        return None
    try:
        tag, pos = _decode_varint(data, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, pos = _decode_varint(data, pos)
            return field, 0, val, pos
        if wire == 2:
            length, pos = _decode_varint(data, pos)
            return field, 2, data[pos:pos + length], pos + length
        if wire == 1:
            return field, 1, struct.unpack("<q", data[pos:pos + 8])[0], pos + 8
        if wire == 5:
            return field, 5, struct.unpack("<i", data[pos:pos + 4])[0], pos + 4
    except Exception:
        return None
    return None


class ServerMsg:
    __slots__ = ("text_delta", "reasoning_delta", "turn_ended", "wants_context",
                 "exec_id", "exec_id_str", "error",
                 "kv_id", "kv_op", "kv_blob_id", "kv_blob_data",
                 "query_id", "query_kind",
                 "wants_fetch", "fetch_url", "fetch_tool_call_id",
                 "wants_shell", "shell_is_stream", "shell_command",
                 "shell_working_directory", "shell_tool_call_id",
                 "wants_read", "read_path", "read_tool_call_id",
                 "wants_write", "write_path", "write_file_text", "write_tool_call_id",
                 "wants_grep", "grep_search_path", "grep_pattern", "grep_mode",
                 "grep_tool_call_id",
                 "wants_edit", "edit_path", "edit_old_string", "edit_new_string",
                 "edit_tool_call_id",
                 "wants_delete",
                 "wants_mcp", "mcp_parsed",
                 "wants_list_mcp_resources", "list_mcp_resources_exec_id_str",
                 "wants_get_mcp_tools", "get_mcp_tools_server",
                 "get_mcp_tools_tool_name", "get_mcp_tools_pattern",
                 "get_mcp_tools_call_id", "get_mcp_tools_result_field",
                 "get_mcp_tools_via_asm",
                 "wants_mcp_state", "mcp_state_servers", "mcp_state_kick_only",
                 "unknown_top_fields", "unknown_exec_fields", "unknown_exec_field",
                 # Native Task / subagent exec (ExecServerMessage f28/f31/f37)
                 "wants_subagent", "subagent_tool_call_id", "subagent_type",
                 "subagent_model_id", "subagent_prompt", "subagent_readonly",
                 "subagent_resume_agent_id", "subagent_run_in_background",
                 "wants_force_background_subagent",
                 "force_background_subagent_tool_call_id",
                 "wants_subagent_await", "subagent_await_agent_id",
                 "subagent_await_timeout_ms",
                 "heartbeat", "checkpoint", "exec_control",
                 "user_message_appended_id",
                 "wants_wait", "wait_ms", "wait_pattern", "wait_session_id",
                 "wait_result_field",
                 "tool_event", "tool_event_call_id", "tool_detail",
                 "cursor_tool_label", "tool_run_id", "ui_output")

    def __init__(self):
        self.text_delta: str | None = None
        self.reasoning_delta: str | None = None   # thinking_delta text
        self.tool_event: tuple[str, str] | None = None  # (started|completed, label)
        self.tool_event_call_id: str = ""  # Cursor ToolCallStarted/Completed call_id
        self.tool_detail: str = ""  # short arg detail (query/path/command)
        self.turn_ended = False
        self.wants_context = False
        self.exec_id = 0
        self.exec_id_str = ""
        self.error: str | None = None
        self.kv_id = 0
        self.kv_op: str | None = None        # 'get' | 'set' | None
        self.kv_blob_id: bytes | None = None
        self.kv_blob_data: bytes | None = None
        self.query_id = 0                    # interaction_query id (uint32)
        self.query_kind: str | None = None   # 'search' | 'fetch' | None
        self.wants_fetch = False
        self.fetch_url = ""
        self.fetch_tool_call_id = ""
        self.wants_shell = False
        self.shell_is_stream = False
        self.shell_command = ""
        self.shell_working_directory = ""
        self.shell_tool_call_id = ""
        self.wants_read = False
        self.read_path = ""
        self.read_tool_call_id = ""
        self.wants_write = False
        self.write_path = ""
        self.write_file_text = ""
        self.write_tool_call_id = ""
        self.wants_grep = False
        self.grep_search_path = ""
        self.grep_pattern = ""
        self.grep_mode = ""
        self.grep_tool_call_id = ""
        self.wants_edit = False
        self.edit_path = ""
        self.edit_old_string = ""
        self.edit_new_string = ""
        self.edit_tool_call_id = ""
        self.wants_delete = False
        self.wants_mcp = False
        self.mcp_parsed = None
        self.wants_list_mcp_resources = False
        self.list_mcp_resources_exec_id_str = ""
        self.wants_get_mcp_tools = False
        self.get_mcp_tools_server = ""
        self.get_mcp_tools_tool_name = ""
        self.get_mcp_tools_pattern = ""
        self.get_mcp_tools_call_id = ""
        self.get_mcp_tools_result_field = 0
        self.get_mcp_tools_via_asm = False
        self.wants_mcp_state = False
        self.mcp_state_servers: list[str] = []
        self.mcp_state_kick_only = False
        self.unknown_top_fields: list[tuple[int, bytes]] = []
        self.unknown_exec_fields: list[tuple[int, bytes]] = []
        self.unknown_exec_field = 0
        self.wants_subagent = False
        self.subagent_tool_call_id = ""
        self.subagent_type = ""
        self.subagent_model_id = ""
        self.subagent_prompt = ""
        self.subagent_readonly = False
        self.subagent_resume_agent_id = ""
        self.subagent_run_in_background = False
        self.wants_force_background_subagent = False
        self.force_background_subagent_tool_call_id = ""
        self.wants_subagent_await = False
        self.subagent_await_agent_id = ""
        self.subagent_await_timeout_ms = 0
        self.heartbeat = False
        self.checkpoint: bytes = b""
        self.exec_control: bytes = b""
        self.user_message_appended_id = ""
        self.wants_wait = False
        self.wait_ms = 0
        self.wait_pattern = ""
        self.wait_session_id = ""
        self.wait_result_field = 0
        self.cursor_tool_label = ""
        self.tool_run_id = ""
        self.ui_output = ""

    def __repr__(self):
        p = []
        if self.text_delta is not None:
            p.append(f"text={self.text_delta!r}")
        if self.reasoning_delta is not None:
            p.append(f"think={self.reasoning_delta!r}")
        if self.wants_context:
            p.append(f"wants_context(id={self.exec_id},{self.exec_id_str!r})")
        if self.kv_op:
            p.append(f"kv_{self.kv_op}(id={self.kv_id})")
        if self.query_kind:
            p.append(f"web_{self.query_kind}(id={self.query_id})")
        if self.wants_fetch:
            p.append(f"fetch({self.fetch_url!r})")
        if self.wants_shell:
            p.append(f"shell({self.shell_command!r})")
        if self.wants_read:
            p.append(f"read({self.read_path!r}, id={self.read_tool_call_id!r})")
        if self.wants_grep:
            p.append(f"grep({self.grep_pattern!r} in {self.grep_search_path!r})")
        if self.wants_edit:
            p.append(f"edit({self.edit_path!r})")
        if self.wants_delete:
            p.append(f"delete({self.read_path!r}, id={self.read_tool_call_id!r})")
        if self.wants_get_mcp_tools:
            p.append(
                f"get_mcp_tools(pattern={self.get_mcp_tools_pattern!r}, "
                f"name={self.get_mcp_tools_tool_name!r}, field={self.get_mcp_tools_result_field})"
            )
        if self.wants_mcp_state:
            p.append(f"mcp_state({self.mcp_state_servers!r})")
        if self.wants_subagent:
            p.append(
                f"subagent(prompt={self.subagent_prompt[:60]!r}, "
                f"bg={self.subagent_run_in_background})"
            )
        if self.wants_subagent_await:
            p.append(f"subagent_await({self.subagent_await_agent_id!r})")
        if self.heartbeat:
            p.append("heartbeat")
        if self.checkpoint:
            p.append(f"checkpoint({len(self.checkpoint)}B)")
        if self.wants_wait:
            p.append(f"wait({self.wait_ms}ms)")
        if self.wants_force_background_subagent:
            p.append(
                f"force_bg_subagent({self.force_background_subagent_tool_call_id!r})"
            )
        if self.turn_ended:
            p.append("TURN_END")
        if self.error:
            p.append(f"error={self.error!r}")
        return f"ServerMsg({', '.join(p) or 'heartbeat'})"


def parse_server_message(raw: bytes) -> ServerMsg:
    msg = ServerMsg()
    pos = 0
    while (r := _read_field(raw, pos)) is not None:
        field, wire, val, pos = r
        _wire_log("server_msg", field, wire, val)
        if field == 1 and wire == 2:        # interaction_update
            _parse_interaction_update(val, msg)
        elif field == 2 and wire == 2:      # exec_server_message
            _parse_exec_server_message(val, msg)
        elif field == 3 and wire == 2 and isinstance(val, bytes):
            # Conversation checkpoint (KV blob hashes + latest message).
            msg.checkpoint = val
        elif field == 4 and wire == 2:      # kv_server_message
            _parse_kv_server_message(val, msg)
        elif field == 5 and wire == 2 and isinstance(val, bytes):
            # Exec control / stream-control message (opaque; surfaced for capture).
            msg.exec_control = val
        elif field == 7 and wire == 2:      # interaction_query (web search/fetch approval)
            _parse_interaction_query(val, msg)
        elif field not in _ASM_HANDLED_FIELDS:
            blob = val if isinstance(val, bytes) else b""
            msg.unknown_top_fields.append((field, blob))
            if wire == 2 and blob:
                _try_parse_get_mcp_tools_container(
                    blob, msg, result_field=field, via_asm=True,
                )
    return msg


def _parse_interaction_query(data: bytes, msg: ServerMsg):
    # InteractionQuery{f1:id, f2:web_search_request_query, f9:web_fetch_request_query, ...}
    pos = 0
    qid, kind = 0, None
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 0:
            qid = val
        elif field == 2:                    # web_search_request_query
            kind = "search"
        elif field == 9:                    # web_fetch_request_query
            kind = "fetch"
    if kind:
        msg.query_id = qid
        msg.query_kind = kind


def _parse_kv_server_message(data: bytes, msg: ServerMsg):
    # KvServerMessage{f1:id, f2:get_blob_args, f3:set_blob_args, f4:span_context}
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 0:
            msg.kv_id = val
        elif field == 2 and wire == 2:      # get_blob_args{f1: blob_id}
            msg.kv_op = "get"
            inner = _read_field(val, 0)
            if inner and inner[0] == 1:
                msg.kv_blob_id = inner[2]
        elif field == 3 and wire == 2:      # set_blob_args{f1: blob_id, f2: blob_data}
            msg.kv_op = "set"
            p2 = 0
            while (r2 := _read_field(val, p2)) is not None:
                f2, w2, v2, p2 = r2
                if f2 == 1:
                    msg.kv_blob_id = v2
                elif f2 == 2:
                    msg.kv_blob_data = v2


def _parse_mcp_tool_name(mcp_bytes: bytes) -> str | None:
    """McpToolCall{f1: McpArgs} -> tool_name (f5) or name (f1)."""
    pos = 0
    while (r := _read_field(mcp_bytes, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:
            ipos = 0
            fallback = ""
            while (r2 := _read_field(val, ipos)) is not None:
                ff, ww, vv, ipos = r2
                if ff == 5 and ww == 2 and isinstance(vv, bytes):
                    name = vv.decode("utf-8", "replace")
                    if name:
                        return name
                elif ff == 1 and ww == 2 and isinstance(vv, bytes):
                    fallback = vv.decode("utf-8", "replace")
            if fallback:
                return fallback
    return None


def cap_tool_text(s: str, *, limit: int = _OUTPUT_MAX) -> str:
    """Cap stored tool-card text. Trailing marker when truncated."""
    if not s:
        return ""
    if len(s) <= limit:
        return s
    keep = max(0, limit - len(_TRUNCATED_SUFFIX))
    return s[:keep] + _TRUNCATED_SUFFIX


def stamp_ui_output(smsg: object, text: str | None) -> None:
    """Attach a capped result preview for the chat tool card."""
    try:
        smsg.ui_output = cap_tool_text(text or "")  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return


def _truncate_detail(s: str) -> str:
    return cap_tool_text((s or "").strip(), limit=_DETAIL_MAX)


def _format_args_obj(obj: dict) -> str:
    """Pretty-print tool args for the card Input pane."""
    cleaned = {k: v for k, v in obj.items() if v not in ("", None, [], {})}
    if not cleaned:
        return ""
    try:
        rendered = json.dumps(cleaned, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(cleaned)
    return _truncate_detail(rendered)


def _is_useful_detail(s: str) -> bool:
    """Filter out ids / empty noise from candidate tool-arg strings."""
    if len(s) < 2 or not s.isprintable():
        return False
    if _UUID_RE.fullmatch(s) or _HEX_ID_RE.fullmatch(s):
        return False
    return True


def _is_nested_pb_message(data: bytes) -> bool:
    """True when *data* is a complete protobuf message (not a plain string).

    Plain UTF-8 paths/queries usually fail this check (invalid wire types),
    while XxxToolCall.args wrappers parse cleanly as nested messages.
    """
    if not data:
        return False
    pos = 0
    fields = 0
    while pos < len(data):
        r = _read_field(data, pos)
        if r is None:
            return False
        field, wire, _val, new_pos = r
        if new_pos <= pos or field < 1 or field > 512 or wire not in (0, 1, 2, 5):
            return False
        pos = new_pos
        fields += 1
    return fields >= 1 and pos == len(data)


def _unwrap_tool_args(variant_bytes: bytes) -> bytes:
    """XxxToolCall is typically ``{ f1: Args }`` — return Args when present."""
    pos = 0
    while (r := _read_field(variant_bytes, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2 and isinstance(val, bytes) and val:
            return val
    return variant_bytes


def _pick_best_detail(candidates: list[str], *, exclude: set[str] | None = None) -> str:
    if not candidates:
        return ""
    skip = exclude or set()
    usable = [c for c in candidates if c not in skip and _is_useful_detail(c)]
    if not usable:
        return ""

    def score(s: str) -> tuple[int, int]:
        # Prefer queries/paths/commands/JSON over short tokens (tool names, modes).
        rich = (
            1
            if ("/" in s or " " in s or "://" in s or s.startswith("{") or s.startswith("-"))
            else 0
        )
        return (rich, len(s))

    return _truncate_detail(max(usable, key=score))


def _extract_tool_detail(variant_bytes: bytes, *, exclude: set[str] | None = None) -> str:
    """Best-effort: pull a short human-readable detail from a ToolCall variant.

    Walks nested protobuf messages (``XxxToolCall.args { … }``) and picks the
    most informative printable string (query / path / command / pattern).
    Earlier versions decoded nested args as UTF-8, which dropped most details
    and occasionally corrupted paths with length-prefix bytes (e.g. ``D/root…``).
    """
    candidates: list[str] = []

    def walk(data: bytes, depth: int) -> None:
        if depth > 8 or not data:
            return
        pos = 0
        while (r := _read_field(data, pos)) is not None:
            _field, wire, val, pos = r
            if wire != 2 or not isinstance(val, (bytes, bytearray)) or not val:
                continue
            raw = bytes(val)
            if _is_nested_pb_message(raw):
                walk(raw, depth + 1)
                continue
            try:
                s = raw.decode("utf-8")
            except UnicodeDecodeError:
                walk(raw, depth + 1)
                continue
            s = s.strip()
            if _is_useful_detail(s):
                candidates.append(s)

    walk(variant_bytes, 0)
    return _pick_best_detail(candidates, exclude=exclude)


def _keep_mcp_arg(key: str, val: object, tool_name: str | None) -> bool:
    if key in _MCP_SKIP_KEYS:
        return False
    if tool_name and key in {"name", "tool_name"} and val == tool_name:
        return False
    if val in (None, "", [], {}):
        return False
    return True


def _parse_mcp_tool_detail(mcp_bytes: bytes, tool_name: str | None) -> str:
    """Pretty-print McpToolCall / McpArgs for the chat UI."""
    args_msg = _unwrap_tool_args(mcp_bytes)
    pairs: list[tuple[str, object]] = []
    pos = 0
    while (r := _read_field(args_msg, pos)) is not None:
        field, wire, val, pos = r
        if field != 2 or wire != 2 or not isinstance(val, bytes):
            continue
        key, val_bytes = "", b""
        kpos = 0
        while (r2 := _read_field(val, kpos)) is not None:
            ff, ww, vv, kpos = r2
            if ff == 1 and ww == 2 and isinstance(vv, bytes):
                key = vv.decode("utf-8", "replace")
            elif ff == 2 and ww == 2 and isinstance(vv, bytes):
                val_bytes = vv
        if not key or not val_bytes:
            continue
        try:
            decoded = decode_value(val_bytes)
        except Exception:
            decoded = val_bytes.decode("utf-8", "replace")
        if decoded is None or decoded == "":
            continue
        pairs.append((key, decoded))

    if pairs:
        by_key = {k: v for k, v in pairs if _keep_mcp_arg(k, v, tool_name)}
        if by_key:
            return _format_args_obj(by_key)

    exclude = {tool_name} if tool_name else set()
    # Also exclude common provider/name fields gathered as bare strings.
    exclude.update({"mcp", "forge"})
    return _extract_tool_detail(mcp_bytes, exclude=exclude)


def _parse_task_tool_call_detail(task_tool_call_bytes: bytes) -> str:
    """Prefer TaskArgs.description, fall back to prompt."""
    pos = 0
    while (r := _read_field(task_tool_call_bytes, pos)) is not None:
        field, wire, val, pos = r
        if field != 1 or wire != 2 or not isinstance(val, bytes):
            continue  # TaskToolCall.args
        description, prompt = "", ""
        apos = 0
        while (ar := _read_field(val, apos)) is not None:
            af, aw, av, apos = ar
            if aw == 2 and isinstance(av, (bytes, bytearray)):
                try:
                    s = bytes(av).decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if af == 1:
                    description = s
                elif af == 2:
                    prompt = s
        detail = description or prompt
        return _truncate_detail(detail) if detail else ""
    return ""


def _grep_detail(pattern: str, path: str, mode: str = "") -> str:
    obj: dict[str, str] = {}
    if pattern:
        obj["pattern"] = pattern
    if path:
        obj["path"] = path
    if mode:
        obj["mode"] = mode
    return _format_args_obj(obj)


def _edit_detail(path: str, old: str = "", new: str = "") -> str:
    obj: dict[str, str] = {}
    if path:
        obj["path"] = path
    if old:
        obj["old"] = old
    if new:
        obj["new"] = new
    return _format_args_obj(obj)


def _detail_from_known_args(name: str, variant_bytes: bytes) -> str:
    """Use structured arg parsers when the tool shape is known."""
    args = _unwrap_tool_args(variant_bytes)
    if name == "shell":
        parsed = parse_shell_args(args)
        if parsed and parsed.command:
            return _truncate_detail(parsed.command)
    elif name in ("read", "delete"):
        parsed = parse_read_args(args)
        if parsed and parsed.path:
            return _truncate_detail(parsed.path)
    elif name in ("grep", "glob"):
        parsed = parse_grep_args(args)
        if parsed:
            return _grep_detail(parsed.pattern, parsed.search_path, parsed.mode)
    elif name == "edit":
        parsed = parse_edit_args(args)
        if parsed and parsed.path:
            return _edit_detail(parsed.path, parsed.old_string, parsed.new_string)
    elif name in ("web_fetch", "fetch"):
        parsed = parse_fetch_args(args)
        if parsed and parsed.url:
            return _truncate_detail(parsed.url)
    elif name == "write":
        parsed = parse_write_args(args)
        if parsed and parsed.path:
            return _truncate_detail(parsed.path)
    elif name == "get_mcp_tools":
        parsed = parse_get_mcp_tools_args(args)
        if parsed:
            return _truncate_detail(
                parsed.pattern or parsed.tool_name or parsed.server
            )
    return ""


def _tool_call_variant(tool_call_bytes: bytes) -> tuple[str, str]:
    """First populated field in agent.v1.ToolCall -> (name, detail)."""
    unknown: tuple[int, bytes] | None = None
    pos = 0
    while (r := _read_field(tool_call_bytes, pos)) is not None:
        field, wire, val, pos = r
        if wire != 2 or not isinstance(val, bytes):
            continue
        if field not in TOOL_ONEOF_FIELD:
            if unknown is None and field >= 1:
                unknown = (field, val)
            continue
        name = TOOL_ONEOF_FIELD[field]
        if field == 15:
            mcp_name = _parse_mcp_tool_name(val)
            label = f"mcp:{mcp_name}" if mcp_name else "mcp"
            return label, _parse_mcp_tool_detail(val, mcp_name)
        if field == 19:
            return "task", _parse_task_tool_call_detail(val)
        if field == 2:
            wait = parse_wait_args(val)
            if wait is not None:
                return "await", f"{wait.duration_ms}ms"
        detail = _detail_from_known_args(name, val) or _extract_tool_detail(val)
        return name, detail
    if unknown is not None:
        field, val = unknown
        return f"tool#{field}", _extract_tool_detail(val)
    return "tool", ""


def detail_from_exec_server_msg(smsg: "ServerMsg") -> str:
    """Human-readable arg summary from a parsed exec request (shell/read/MCP/…).

    Used to refresh the UI when ToolCallStarted had empty/partial detail but the
    subsequent ExecServerMessage carries the real command/path/args.
    """
    if smsg.wants_shell and smsg.shell_command:
        return _truncate_detail(smsg.shell_command)
    if smsg.wants_read and smsg.read_path:
        return _truncate_detail(smsg.read_path)
    if smsg.wants_delete and smsg.read_path:
        return _truncate_detail(smsg.read_path)
    if smsg.wants_write and smsg.write_path:
        return _truncate_detail(smsg.write_path)
    if smsg.wants_edit and smsg.edit_path:
        return _edit_detail(smsg.edit_path, smsg.edit_old_string, smsg.edit_new_string)
    if smsg.wants_grep:
        return _grep_detail(smsg.grep_pattern, smsg.grep_search_path, smsg.grep_mode)
    if smsg.wants_fetch and smsg.fetch_url:
        return _truncate_detail(smsg.fetch_url)
    if smsg.wants_mcp and smsg.mcp_parsed:
        mcp_name = _parse_mcp_tool_name(pb_msg(1, smsg.mcp_parsed)) or ""
        # mcp_parsed is raw McpArgs (not wrapped in McpToolCall).
        return _parse_mcp_tool_detail(pb_msg(1, smsg.mcp_parsed), mcp_name)
    if smsg.wants_get_mcp_tools:
        return _truncate_detail(
            smsg.get_mcp_tools_pattern
            or smsg.get_mcp_tools_tool_name
            or smsg.get_mcp_tools_server
        )
    if smsg.wants_mcp_state and smsg.mcp_state_servers:
        return _truncate_detail(", ".join(smsg.mcp_state_servers))
    if smsg.wants_subagent and smsg.subagent_prompt:
        return _truncate_detail(smsg.subagent_prompt)
    return ""


def _parse_tool_call_update(data: bytes) -> tuple[str, str, str] | None:
    """Parse ToolCallStarted/CompletedUpdate -> (call_id, tool_name, detail)."""
    call_id = ""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2 and isinstance(val, bytes):
            call_id = val.decode("utf-8", "replace")
        elif field == 2 and wire == 2 and isinstance(val, bytes):
            name, detail = _tool_call_variant(val)
            return call_id, name, detail
    return None


def _parse_interaction_update(data: bytes, msg: ServerMsg):
    # InteractionUpdate oneof: f1=text_delta, f2=tool_call_started, f3=tool_call_completed, ...
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:        # text_delta { f1: text }
            inner = _read_field(val, 0)
            if inner and inner[0] == 1 and isinstance(inner[2], bytes):
                try:
                    msg.text_delta = inner[2].decode("utf-8")
                except UnicodeDecodeError:
                    pass
        elif field == 2 and wire == 2:      # tool_call_started
            parsed = _parse_tool_call_update(val)
            if parsed:
                msg.tool_event = ("started", parsed[1])
                msg.tool_event_call_id = parsed[0] or ""
                msg.tool_detail = parsed[2]
        elif field == 3 and wire == 2:      # tool_call_completed
            parsed = _parse_tool_call_update(val)
            if parsed:
                msg.tool_event = ("completed", parsed[1])
                msg.tool_event_call_id = parsed[0] or ""
                msg.tool_detail = parsed[2]
        elif field == 4 and wire == 2:      # thinking_delta { f1: text }
            inner = _read_field(val, 0)
            if inner and inner[0] == 1 and isinstance(inner[2], bytes):
                try:
                    msg.reasoning_delta = inner[2].decode("utf-8")
                except UnicodeDecodeError:
                    pass
        elif field == 6 and wire == 2 and isinstance(val, bytes):
            # UserMessageAppendedUpdate{f1: UserMessage{f2: message_id}}
            msg.user_message_appended_id = _appended_user_message_id(val)
        elif field == 13:                   # keepalive heartbeat
            msg.heartbeat = True
        elif field == 14:                   # turn_ended
            msg.turn_ended = True


def _appended_user_message_id(data: bytes) -> str:
    """Pull UserMessage.message_id out of UserMessageAppendedUpdate."""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2 and isinstance(val, bytes):
            inner = 0
            while (r2 := _read_field(val, inner)) is not None:
                f2, w2, v2, inner = r2
                if f2 == 2 and w2 == 2 and isinstance(v2, bytes):
                    return v2.decode("utf-8", "replace")
    return ""


def _parse_exec_server_message(data: bytes, msg: ServerMsg):
    pos = 0
    exec_id, exec_id_str = 0, ""
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        _wire_log("exec", field, wire, val)
        if field == 1 and wire == 0:
            exec_id = val
        elif field == 15 and wire == 2 and isinstance(val, bytes):
            exec_id_str = val.decode("utf-8", "replace")
        elif field == 10:
            msg.wants_context = True
        elif field == 20 and wire == 2:
            parsed = parse_fetch_args(val)
            if parsed:
                msg.wants_fetch = True
                msg.fetch_url = parsed.url
                msg.fetch_tool_call_id = parsed.tool_call_id
        elif field in (2, 14) and wire == 2:  # shell_args | shell_stream_args
            parsed = parse_shell_args(val)
            if parsed:
                msg.wants_shell = True
                msg.shell_is_stream = field == 14
                msg.shell_command = parsed.command
                msg.shell_working_directory = parsed.working_directory
                msg.shell_tool_call_id = parsed.tool_call_id
        elif field == 3 and wire == 2:
            parsed = parse_write_args(val)
            if parsed:
                msg.wants_write = True
                msg.write_path = parsed.path
                msg.write_file_text = parsed.file_text
                msg.write_tool_call_id = parsed.tool_call_id
        elif field == 4 and wire == 2:
            parsed = parse_delete_args(val)
            if parsed:
                msg.wants_delete = True
                msg.read_path = parsed.path
                msg.read_tool_call_id = parsed.tool_call_id
        elif field == 7 and wire == 2:
            parsed = parse_read_args(val)
            if parsed:
                msg.wants_read = True
                msg.read_path = parsed.path
                msg.read_tool_call_id = parsed.tool_call_id
        elif field == 5 and wire == 2:
            parsed = parse_grep_args(val)
            if parsed:
                msg.wants_grep = True
                msg.grep_search_path = parsed.search_path
                msg.grep_pattern = parsed.pattern
                msg.grep_mode = parsed.mode
                msg.grep_tool_call_id = parsed.tool_call_id
        elif field == 12 and wire == 2:
            parsed = parse_edit_args(val)
            if parsed:
                msg.wants_edit = True
                msg.edit_path = parsed.path
                msg.edit_old_string = parsed.old_string
                msg.edit_new_string = parsed.new_string
                msg.edit_tool_call_id = parsed.tool_call_id
        elif field == 11 and wire == 2 and isinstance(val, bytes):
            # mcp_args: store raw bytes for higher-layer parsing (McpManager).
            msg.wants_mcp = True
            msg.mcp_parsed = val
        elif field == 17 and wire == 2:
            # list_mcp_resources exec request.
            msg.wants_list_mcp_resources = True
        elif field == 36 and wire == 2 and isinstance(val, bytes):
            servers, kick = parse_mcp_state_args(val)
            msg.wants_mcp_state = True
            msg.mcp_state_servers = servers
            msg.mcp_state_kick_only = kick
        elif field == 28 and wire == 2 and isinstance(val, bytes):
            parsed = parse_subagent_args(val)
            if parsed:
                msg.wants_subagent = True
                msg.subagent_tool_call_id = parsed.tool_call_id
                msg.subagent_type = parsed.subagent_type
                msg.subagent_model_id = parsed.model_id
                msg.subagent_prompt = parsed.prompt
                msg.subagent_readonly = parsed.readonly
                msg.subagent_resume_agent_id = parsed.resume_agent_id
                msg.subagent_run_in_background = parsed.run_in_background
        elif field == 31 and wire == 2 and isinstance(val, bytes):
            parsed = parse_force_background_subagent_args(val)
            if parsed is not None:
                msg.wants_force_background_subagent = True
                msg.force_background_subagent_tool_call_id = parsed
        elif field == 37 and wire == 2 and isinstance(val, bytes):
            parsed = parse_subagent_await_args(val)
            if parsed:
                msg.wants_subagent_await = True
                msg.subagent_await_agent_id = parsed.agent_id
                msg.subagent_await_timeout_ms = parsed.timeout_ms
        elif wire == 2 and isinstance(val, bytes) and field not in _EXEC_HANDLED_FIELDS:
            wait_parsed = parse_wait_args(val)
            if wait_parsed is not None:
                _apply_wait(msg, wait_parsed, field)
            else:
                parsed = parse_get_mcp_tools_args(val)
                if parsed and not msg.wants_get_mcp_tools:
                    _apply_get_mcp_tools(msg, parsed, field, via_asm=False)
                else:
                    msg.unknown_exec_fields.append((field, val))
                    if not msg.unknown_exec_field:
                        msg.unknown_exec_field = field
    if exec_id or exec_id_str:
        msg.exec_id = exec_id
        msg.exec_id_str = exec_id_str


# ──────────────────────────────────────────────────────────────────────────────
# Exec-arg dataclasses + parsers
# ──────────────────────────────────────────────────────────────────────────────

def _exec_client_envelope(
    exec_id: int, exec_id_str: str, result_field: int, result_body: bytes
) -> bytes:
    exec_client = b""
    if exec_id:
        exec_client += pb_int(1, exec_id)
    if exec_id_str:
        exec_client += pb_str(15, exec_id_str)
    exec_client += pb_msg(result_field, result_body)
    return pb_msg(2, exec_client)


def build_unknown_exec_error_response(
    exec_id: int, exec_id_str: str, result_field: int, reason: str
) -> bytes:
    """Error on an unused ExecClientMessage oneof field (same number as the args)."""
    return _exec_client_envelope(
        exec_id, exec_id_str, result_field, pb_msg(2, pb_str(1, reason)),
    )


@dataclass
class GetMcpToolsArgs:
    """agent.v1.GetMcpToolsArgs (ToolCall 44 / matching exec oneof)."""

    server: str = ""
    tool_name: str = ""
    pattern: str = ""
    tool_call_id: str = ""


def parse_mcp_state_args(data: bytes) -> tuple[list[str], bool]:
    """McpStateExecArgs { repeated string serverIdentifiers = 1, bool kickOnly = 2 }."""
    servers: list[str] = []
    kick = False
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2 and isinstance(val, bytes):
            servers.append(val.decode("utf-8", "replace"))
        elif field == 2 and wire == 0:
            kick = bool(val)
    return servers, kick


def parse_get_mcp_tools_args(data: bytes) -> GetMcpToolsArgs | None:
    """Parse GetMcpToolsArgs, including the XxxToolCall { f1: args } wrapper.

    Rejects gitDiff-shaped payloads (field 4 varint/bool). Requires a pattern,
    a tool name, or (server + tool_call_id) so unrelated 3-string messages
    are not treated as catalog search.
    """
    if not data:
        return None
    inner = _read_field(data, 0)
    if (
        inner
        and inner[0] == 1
        and inner[1] == 2
        and isinstance(inner[2], bytes)
        and _read_field(data, inner[3]) is None
        and _is_nested_pb_message(inner[2])
    ):
        wrapped = _parse_get_mcp_tools_flat(inner[2])
        if wrapped is not None:
            return wrapped
    return _parse_get_mcp_tools_flat(data)


def _parse_get_mcp_tools_flat(data: bytes) -> GetMcpToolsArgs | None:
    server = tool_name = pattern = tool_call_id = ""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 4 and wire == 0:
            return None
        if wire != 2 or not isinstance(val, bytes):
            continue
        if field == 1 and _is_nested_pb_message(val):
            continue
        try:
            text = val.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if field == 1:
            server = text
        elif field == 2:
            tool_name = text
        elif field == 3:
            pattern = text
        elif field == 4:
            tool_call_id = text
    if pattern or tool_name or (server and tool_call_id):
        return GetMcpToolsArgs(server, tool_name, pattern, tool_call_id)
    return None


def _apply_get_mcp_tools(
    msg: ServerMsg,
    parsed: GetMcpToolsArgs,
    result_field: int,
    *,
    via_asm: bool,
) -> None:
    msg.wants_get_mcp_tools = True
    msg.get_mcp_tools_server = parsed.server
    msg.get_mcp_tools_tool_name = parsed.tool_name
    msg.get_mcp_tools_pattern = parsed.pattern
    msg.get_mcp_tools_call_id = parsed.tool_call_id
    msg.get_mcp_tools_result_field = result_field
    msg.get_mcp_tools_via_asm = via_asm


def _try_parse_get_mcp_tools_container(
    data: bytes,
    msg: ServerMsg,
    *,
    result_field: int,
    via_asm: bool,
) -> bool:
    """Parse GetMcpToolsArgs, or an ExecServerMessage-shaped wrapper around it."""
    parsed = parse_get_mcp_tools_args(data)
    if parsed is not None:
        _apply_get_mcp_tools(msg, parsed, result_field, via_asm=via_asm)
        return True
    pos = 0
    exec_id, exec_id_str = 0, ""
    candidates: list[tuple[int, bytes]] = []
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 0:
            exec_id = val
        elif field == 15 and wire == 2 and isinstance(val, bytes):
            exec_id_str = val.decode("utf-8", "replace")
        elif wire == 2 and isinstance(val, bytes) and field not in (15, 19):
            candidates.append((field, val))
    for field, blob in candidates:
        parsed = parse_get_mcp_tools_args(blob)
        if parsed is None:
            continue
        if exec_id or exec_id_str:
            msg.exec_id = exec_id
            msg.exec_id_str = exec_id_str
        reply_asm = via_asm and not (exec_id or exec_id_str)
        _apply_get_mcp_tools(
            msg, parsed, result_field if reply_asm else field, via_asm=reply_asm,
        )
        return True
    return False


@dataclass
class ReadArgs:
    path: str
    tool_call_id: str = ""


@dataclass
class ShellArgs:
    command: str
    working_directory: str = ""
    tool_call_id: str = ""


@dataclass
class WriteArgs:
    path: str
    file_text: str
    tool_call_id: str = ""


@dataclass
class FetchArgs:
    url: str
    tool_call_id: str = ""


@dataclass
class GrepArgs:
    search_path: str
    pattern: str
    mode: str = ""
    tool_call_id: str = ""


@dataclass
class EditArgs:
    path: str
    old_string: str = ""
    new_string: str = ""
    tool_call_id: str = ""


@dataclass
class SubagentArgs:
    """agent.v1.SubagentArgs — native Task tool exec payload (f28)."""
    tool_call_id: str = ""
    subagent_type: str = ""
    model_id: str = ""
    prompt: str = ""
    readonly: bool = False
    resume_agent_id: str = ""
    run_in_background: bool = False


@dataclass
class SubagentAwaitArgs:
    agent_id: str = ""
    timeout_ms: int = 0


# SubagentBackgroundReason
SUBAGENT_BG_UNSPECIFIED = 0
SUBAGENT_BG_AGENT_REQUEST = 1
SUBAGENT_BG_USER_REQUEST = 2
SUBAGENT_BG_QUEUED_FOLLOW_UP = 3

# ForceBackgroundSubagentStatus
FORCE_BG_SUBAGENT_UNSPECIFIED = 0
FORCE_BG_SUBAGENT_ACCEPTED = 1
FORCE_BG_SUBAGENT_NOT_FOUND = 2


def _decode_str_field(val: bytes) -> str:
    return val.decode("utf-8", "replace") if isinstance(val, bytes) else ""


def parse_read_args(data: bytes) -> ReadArgs | None:
    path, tool_call_id = "", ""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:
            path = _decode_str_field(val)
        elif field == 2 and wire == 2:
            tool_call_id = _decode_str_field(val)
    return ReadArgs(path, tool_call_id) if path else None


def parse_delete_args(data: bytes) -> ReadArgs | None:
    """DeleteArgs uses the same wire shape as ReadArgs (path + tool_call_id)."""
    return parse_read_args(data)


def parse_shell_args(data: bytes) -> ShellArgs | None:
    command, working_directory, tool_call_id = "", "", ""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:
            command = _decode_str_field(val)
        elif field == 2 and wire == 2:
            working_directory = _decode_str_field(val)
        elif field == 4 and wire == 2:
            tool_call_id = _decode_str_field(val)
    return ShellArgs(command, working_directory, tool_call_id) if command else None


def parse_write_args(data: bytes) -> WriteArgs | None:
    path, file_text, tool_call_id = "", "", ""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:
            path = _decode_str_field(val)
        elif field == 2 and wire == 2:
            file_text = _decode_str_field(val)
        elif field == 3 and wire == 2:
            tool_call_id = _decode_str_field(val)
    return WriteArgs(path, file_text, tool_call_id) if path else None


def parse_fetch_args(data: bytes) -> FetchArgs | None:
    url, tool_call_id = "", ""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:
            url = _decode_str_field(val)
        elif field == 2 and wire == 2:
            tool_call_id = _decode_str_field(val)
    return FetchArgs(url, tool_call_id) if url else None


def parse_grep_args(data: bytes) -> GrepArgs | None:
    """grep_args / glob — both use exec field 5."""
    search_path, pattern, mode, tool_call_id = "", "", "", ""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 2 and wire == 2:
            search_path = _decode_str_field(val)
        elif field == 3 and wire == 2:
            pattern = _decode_str_field(val)
        elif field == 4 and wire == 2:
            mode = _decode_str_field(val)
        elif field == 14 and wire == 2:
            tool_call_id = _decode_str_field(val)
    if search_path or pattern:
        return GrepArgs(search_path, pattern, mode, tool_call_id)
    return None


def parse_edit_args(data: bytes) -> EditArgs | None:
    path, old_string, new_string, tool_call_id = "", "", "", ""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:
            path = _decode_str_field(val)
        elif field == 2 and wire == 2:
            old_string = _decode_str_field(val)
        elif field == 3 and wire == 2:
            new_string = _decode_str_field(val)
        elif field in (4, 14) and wire == 2:
            tool_call_id = _decode_str_field(val)
    if path and (old_string or new_string):
        return EditArgs(path, old_string, new_string, tool_call_id)
    return None


def parse_subagent_args(data: bytes) -> SubagentArgs | None:
    """Parse agent.v1.SubagentArgs (ExecServerMessage.subagent_args)."""
    args = SubagentArgs()
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:
            args.tool_call_id = _decode_str_field(val)
        elif field == 2 and wire == 2:
            args.subagent_type = _decode_str_field(val)
        elif field == 3 and wire == 2:
            args.model_id = _decode_str_field(val)
        elif field == 4 and wire == 2:
            args.prompt = _decode_str_field(val)
        elif field == 5 and wire == 0:
            args.readonly = bool(val)
        elif field == 6 and wire == 2:
            args.resume_agent_id = _decode_str_field(val)
        elif field == 7 and wire == 0:
            args.run_in_background = bool(val)
    if not args.prompt and not args.resume_agent_id:
        return None
    return args


def parse_force_background_subagent_args(data: bytes) -> str | None:
    """Return tool_call_id from ForceBackgroundSubagentArgs, or empty string."""
    tool_call_id = ""
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:
            tool_call_id = _decode_str_field(val)
    return tool_call_id  # may be "" — still a valid (parsed) request


def parse_subagent_await_args(data: bytes) -> SubagentAwaitArgs | None:
    args = SubagentAwaitArgs()
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 2:
            args.agent_id = _decode_str_field(val)
        elif field == 2 and wire == 0:
            args.timeout_ms = int(val)
    return args if args.agent_id else None


@dataclass
class WaitArgs:
    """Cursor await/wait tool: sleep, optionally until a session pattern matches."""

    duration_ms: int = 0
    pattern: str = ""
    session_id: str = ""
    tool_call_id: str = ""


def parse_wait_args(data: bytes) -> WaitArgs | None:
    """Parse AwaitToolCall / wait exec args.

    Accepts a duration as field 1 (varint). Values below 1000 are treated as
    seconds; 1000 and above as milliseconds. Nested ``{ f1: args }`` wrappers
    are unwrapped. Returns None when no duration is present.
    """
    if not data:
        return None
    inner = _read_field(data, 0)
    if (
        inner
        and inner[0] == 1
        and inner[1] == 2
        and isinstance(inner[2], bytes)
        and _read_field(data, inner[3]) is None
        and _is_nested_pb_message(inner[2])
    ):
        nested = parse_wait_args(inner[2])
        if nested is not None:
            return nested
    duration = 0
    pattern = ""
    session_id = ""
    tool_call_id = ""
    saw_duration = False
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field, wire, val, pos = r
        if field == 1 and wire == 0:
            duration = int(val)
            saw_duration = True
        elif field == 2 and wire == 2 and isinstance(val, bytes):
            pattern = _decode_str_field(val)
        elif field == 3 and wire == 2 and isinstance(val, bytes):
            session_id = _decode_str_field(val)
        elif field == 15 and wire == 2 and isinstance(val, bytes):
            tool_call_id = _decode_str_field(val)
    if not saw_duration or duration < 0:
        return None
    if duration > 7_200_000:
        return None
    duration_ms = duration if duration >= 1000 else duration * 1000
    return WaitArgs(
        duration_ms=duration_ms,
        pattern=pattern,
        session_id=session_id,
        tool_call_id=tool_call_id,
    )


def _apply_wait(msg: ServerMsg, parsed: WaitArgs, field: int) -> None:
    msg.wants_wait = True
    msg.wait_ms = parsed.duration_ms
    msg.wait_pattern = parsed.pattern
    msg.wait_session_id = parsed.session_id
    msg.wait_result_field = field


def build_wait_response(
    exec_id: int, exec_id_str: str, result_field: int, duration_ms: int
) -> bytes:
    """Success envelope on the same exec oneof field the server used for wait."""
    body = pb_int(1, duration_ms)
    return _exec_client_envelope(exec_id, exec_id_str, result_field, pb_msg(1, body))


# ──────────────────────────────────────────────────────────────────────────────
# Result builders
# ──────────────────────────────────────────────────────────────────────────────

def build_read_success_response(
    exec_id: int,
    exec_id_str: str,
    path: str,
    content: str,
    *,
    truncated: bool = False,
) -> bytes:
    total_lines = content.count("\n") + (1 if content else 0)
    file_size = len(content.encode("utf-8"))
    success = (
        pb_str(1, path)
        + pb_str(2, content)
        + pb_int(3, total_lines)
        + pb_int(4, file_size)
        + pb_int(6, 1 if truncated else 0)
    )
    return _exec_client_envelope(exec_id, exec_id_str, 7, pb_msg(1, success))


def build_read_file_not_found_response(exec_id: int, exec_id_str: str, path: str) -> bytes:
    return _exec_client_envelope(exec_id, exec_id_str, 7, pb_msg(4, pb_str(1, path)))


def build_read_error_response(exec_id: int, exec_id_str: str, path: str, error: str) -> bytes:
    err = pb_str(1, path) + pb_str(2, error)
    return _exec_client_envelope(exec_id, exec_id_str, 7, pb_msg(2, err))


def build_exec_stream_close(exec_id: int) -> bytes:
    """AgentClientMessage{f5: ExecClientControlMessage{f1: stream_close{id}}}."""
    stream_close = pb_int(1, exec_id)
    control = pb_msg(1, stream_close)
    return pb_msg(5, control)


def build_shell_stream_start(exec_id: int, exec_id_str: str) -> bytes:
    """ShellStream { start } — required before stdout/exit for shell_stream_args."""
    stream = pb_msg(4, b"")
    return _exec_client_envelope(exec_id, exec_id_str, 14, stream)


def build_shell_stream_stdout(exec_id: int, exec_id_str: str, stdout: str) -> bytes:
    """ShellStream { stdout: ShellStreamStdout { data } } on ExecClientMessage f14."""
    stream = pb_msg(1, pb_str(1, stdout))
    return _exec_client_envelope(exec_id, exec_id_str, 14, stream)


def build_shell_stream_exit(
    exec_id: int,
    exec_id_str: str,
    exit_code: int = 0,
    cwd: str = "",
) -> bytes:
    """ShellStream { exit: ShellStreamExit { code, cwd } } on ExecClientMessage f14."""
    exit_body = pb_int(1, exit_code) + pb_str(2, cwd)
    stream = pb_msg(3, exit_body)
    return _exec_client_envelope(exec_id, exec_id_str, 14, stream)


def build_shell_success_response(
    exec_id: int,
    exec_id_str: str,
    command: str,
    stdout: str,
    stderr: str = "",
    exit_code: int = 0,
    working_directory: str = "",
    execution_time_ms: int = 0,
) -> bytes:
    success = (
        pb_str(1, command)
        + pb_str(2, working_directory)
        + pb_int(3, exit_code)
        + pb_str(5, stdout)
        + pb_str(6, stderr)
        + pb_int(7, execution_time_ms)
    )
    return _exec_client_envelope(exec_id, exec_id_str, 2, pb_msg(1, success))


def build_shell_response_frames(
    exec_id: int,
    exec_id_str: str,
    command: str,
    stdout: str,
    stderr: str = "",
    exit_code: int = 0,
    working_directory: str = "",
    *,
    is_stream: bool = False,
) -> list[bytes]:
    if is_stream:
        out = stdout + (f"\n{stderr}" if stderr else "")
        frames = [build_shell_stream_start(exec_id, exec_id_str)]
        if out:
            frames.append(build_shell_stream_stdout(exec_id, exec_id_str, out))
        frames.append(build_shell_stream_exit(exec_id, exec_id_str, exit_code, working_directory))
        frames.append(build_exec_stream_close(exec_id))
        return frames
    return [
        build_shell_success_response(
            exec_id,
            exec_id_str,
            command,
            stdout,
            stderr=stderr,
            exit_code=exit_code,
            working_directory=working_directory,
        )
    ]


def build_shell_error_response(
    exec_id: int,
    exec_id_str: str,
    command: str,
    error: str,
) -> bytes:
    err = pb_str(1, command) + pb_str(2, error)
    return _exec_client_envelope(exec_id, exec_id_str, 2, pb_msg(2, err))


def build_shell_rejected_response(
    exec_id: int,
    exec_id_str: str,
    command: str,
    reason: str,
) -> bytes:
    rejected = pb_str(1, command or "unknown") + pb_str(3, reason)
    return _exec_client_envelope(exec_id, exec_id_str, 2, pb_msg(4, rejected))


def build_write_success_response(
    exec_id: int,
    exec_id_str: str,
    path: str,
    *,
    lines_created: int = 0,
    file_size: int = 0,
    file_content_after_write: str = "",
) -> bytes:
    success = pb_str(1, path) + pb_int(2, lines_created) + pb_int(3, file_size)
    if file_content_after_write:
        success += pb_str(4, file_content_after_write)
    return _exec_client_envelope(exec_id, exec_id_str, 3, pb_msg(1, success))


def build_write_error_response(
    exec_id: int,
    exec_id_str: str,
    path: str,
    error: str,
) -> bytes:
    err = pb_str(1, path) + pb_str(2, error)
    return _exec_client_envelope(exec_id, exec_id_str, 3, pb_msg(5, err))


def build_delete_success_response(
    exec_id: int,
    exec_id_str: str,
    path: str,
    *,
    file_size: int = 0,
    prev_content: str = "",
) -> bytes:
    deleted_file = path.rstrip("/").rsplit("/", 1)[-1] if path else ""
    success = pb_str(1, path) + pb_str(2, deleted_file) + pb_int(3, file_size)
    if prev_content:
        success += pb_str(4, prev_content)
    return _exec_client_envelope(exec_id, exec_id_str, 4, pb_msg(1, success))


def build_delete_error_response(
    exec_id: int,
    exec_id_str: str,
    path: str,
    error: str,
) -> bytes:
    err = pb_str(1, path) + pb_str(2, error)
    return _exec_client_envelope(exec_id, exec_id_str, 4, pb_msg(7, err))


def build_delete_rejected_response(
    exec_id: int,
    exec_id_str: str,
    path: str,
    reason: str,
) -> bytes:
    rejected = pb_str(1, path) + pb_str(2, reason)
    return _exec_client_envelope(exec_id, exec_id_str, 4, pb_msg(6, rejected))


def build_fetch_success_response(
    exec_id: int,
    exec_id_str: str,
    url: str,
    content: str,
    status_code: int = 200,
    content_type: str = "text/plain",
) -> bytes:
    success = (
        pb_str(1, url)
        + pb_str(2, content)
        + pb_int(3, status_code)
        + pb_str(4, content_type)
    )
    return _exec_client_envelope(exec_id, exec_id_str, 20, pb_msg(1, success))


def build_fetch_error_response(
    exec_id: int,
    exec_id_str: str,
    url: str,
    error: str,
) -> bytes:
    err = pb_str(1, url) + pb_str(2, error)
    return _exec_client_envelope(exec_id, exec_id_str, 20, pb_msg(2, err))


def build_grep_success_response(
    exec_id: int,
    exec_id_str: str,
    search_path: str,
    pattern: str,
    matched_paths: list[str],
    workspace_key: str = "",
) -> bytes:
    """agent.v1.GrepResult{1: GrepSuccess}.

    GrepSuccess{1:pattern, 2:path, 3:output_mode, 4:workspace_results map}.
    workspace_results is map<string, GrepUnionResult>; for a file listing the
    value is GrepUnionResult{2: GrepFilesResult{1: files[], 2: total_files}}.
    Our local executor only ever produces a list of matching files, so we always
    emit the ``files`` union and report output_mode=files_with_matches (the union
    case, not this string, drives Cursor's rendering). An empty match set emits an
    empty map, which Cursor renders as "No results found.".
    """
    success = (
        pb_str(1, pattern)
        + pb_str(2, search_path)
        + pb_str(3, "files_with_matches")
    )
    if matched_paths:
        files_result = b"".join(pb_str(1, m) for m in matched_paths)
        files_result += pb_int(2, len(matched_paths))
        union = pb_msg(2, files_result)              # GrepUnionResult.files
        entry = pb_str(1, workspace_key) + pb_msg(2, union)
        success += pb_msg(4, entry)                  # workspace_results[key]
    return _exec_client_envelope(exec_id, exec_id_str, 5, pb_msg(1, success))


def build_grep_error_response(
    exec_id: int,
    exec_id_str: str,
    search_path: str,
    error: str,
) -> bytes:
    """agent.v1.GrepResult{2: GrepError{1: error}}."""
    err = pb_str(1, error)
    return _exec_client_envelope(exec_id, exec_id_str, 5, pb_msg(2, err))


def build_edit_success_response(
    exec_id: int,
    exec_id_str: str,
    path: str,
    *,
    file_content_after_edit: str = "",
) -> bytes:
    success = pb_str(1, path)
    if file_content_after_edit:
        success += pb_str(4, file_content_after_edit)
    return _exec_client_envelope(exec_id, exec_id_str, 12, pb_msg(1, success))


def build_edit_error_response(
    exec_id: int,
    exec_id_str: str,
    path: str,
    error: str,
) -> bytes:
    err = pb_str(1, path) + pb_str(2, error)
    return _exec_client_envelope(exec_id, exec_id_str, 12, pb_msg(2, err))


def build_subagent_success_response(
    exec_id: int,
    exec_id_str: str,
    *,
    agent_id: str,
    final_message: str = "",
    tool_call_count: int = 0,
    background_reason: int = SUBAGENT_BG_UNSPECIFIED,
    transcript_path: str = "",
) -> bytes:
    """ExecClientMessage.subagent_result { success: SubagentSuccess }."""
    success = pb_str(1, agent_id)
    if final_message:
        success += pb_str(2, final_message)
    if tool_call_count:
        success += pb_int(3, tool_call_count)
    if background_reason:
        success += pb_int(4, background_reason)
    if transcript_path:
        success += pb_str(5, transcript_path)
    return _exec_client_envelope(exec_id, exec_id_str, 28, pb_msg(1, success))


def build_subagent_error_response(
    exec_id: int,
    exec_id_str: str,
    error: str,
    *,
    agent_id: str = "",
) -> bytes:
    err = b""
    if agent_id:
        err += pb_str(1, agent_id)
    err += pb_str(2, error)
    return _exec_client_envelope(exec_id, exec_id_str, 28, pb_msg(2, err))


def build_force_background_subagent_response(
    exec_id: int,
    exec_id_str: str,
    status: int = FORCE_BG_SUBAGENT_ACCEPTED,
) -> bytes:
    body = pb_int(1, status)
    return _exec_client_envelope(exec_id, exec_id_str, 31, body)


def build_subagent_await_complete_response(
    exec_id: int,
    exec_id_str: str,
    *,
    agent_id: str,
    final_message: str = "",
    tool_call_count: int = 0,
    transcript_path: str = "",
) -> bytes:
    complete = pb_str(1, agent_id)
    if transcript_path:
        complete += pb_str(2, transcript_path)
    if tool_call_count:
        complete += pb_int(3, tool_call_count)
    if final_message:
        complete += pb_str(4, final_message)
    return _exec_client_envelope(exec_id, exec_id_str, 37, pb_msg(1, complete))


def build_subagent_await_still_running_response(
    exec_id: int,
    exec_id_str: str,
    *,
    agent_id: str,
    transcript_path: str = "",
) -> bytes:
    body = pb_str(1, agent_id)
    if transcript_path:
        body += pb_str(2, transcript_path)
    return _exec_client_envelope(exec_id, exec_id_str, 37, pb_msg(2, body))


def build_subagent_await_not_found_response(
    exec_id: int,
    exec_id_str: str,
    *,
    agent_id: str,
) -> bytes:
    return _exec_client_envelope(exec_id, exec_id_str, 37, pb_msg(3, pb_str(1, agent_id)))


def build_subagent_await_error_response(
    exec_id: int,
    exec_id_str: str,
    error: str,
    *,
    agent_id: str = "",
) -> bytes:
    body = b""
    if agent_id:
        body += pb_str(1, agent_id)
    body += pb_str(2, error)
    return _exec_client_envelope(exec_id, exec_id_str, 37, pb_msg(4, body))
