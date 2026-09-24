"""
Cursor AgentService/Run bidirectional streaming engine.

Cursor's chat endpoint speaks Connect-RPC (bidi-streaming flavour) over HTTP/2.
The exchange is genuinely full-duplex, which `httpx` cannot do (it drains the
whole request body before reading the response). We therefore drive an HTTP/2
connection directly with the `h2` library.

Protocol flow on /agent.v1.AgentService/Run:
  1. Client -> Server : AgentClientMessage{f1: AgentRunRequest{...chat request...}}
  2. Server -> Client : AgentServerMessage{...} heartbeats / acks / checkpoints
  3. Server -> Client : AgentServerMessage{f2: ExecServerMessage{f10: RequestContextArgs}}
  4. Client -> Server : AgentClientMessage{f2: ExecClientMessage{...request_context...}}
  5. Server -> Client : AgentServerMessage{f1: ...TextDelta...} streamed tokens
  6. Mid-stream exec / KV / wait answered inline on the same H2 stream
  7. Server -> Client : AgentServerMessage{f1: ...TurnEnded...}
  8. Connect end-of-stream frame (flag bit 0x02) carrying JSON trailers

Host-side tool and wait work pauses the stall and turn clocks (``suspended_total``).
A stream that closes before ``turn_ended`` is reported as ``end_reason=dropped``.
"""
import asyncio
import base64
import fnmatch
import json
import os
import re
import ssl
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable
from urllib.parse import unquote_to_bytes

import certifi
import h2.config
import h2.connection
import h2.events
import h2.exceptions
import httpx

from .auth import (
    AGENT_HOST,
    AGENT_PORT,
    API2_BASE,
    AVAILABLE_MODELS_PATH,
    GET_DEFAULT_MODEL_PATH,
    GET_USABLE_MODELS_PATH,
    TRANSCRIBE_PATH,
    _aiservice_headers,
    _build_headers,
    load_auth,
    resolve_access_token,
)
from .framing import connect_frame, pb_bytes, pb_int, pb_msg, pb_str
from .wire import (
    ServerMsg,
    _maybe_gunzip,
    _read_field,
    build_edit_error_response,
    build_delete_rejected_response,
    build_fetch_error_response,
    build_grep_error_response,
    build_grep_success_response,
    build_read_error_response,
    build_shell_rejected_response,
    build_write_error_response,
    detail_from_exec_server_msg,
    parse_frames,
    parse_server_message,
    stamp_ui_output,
    build_unknown_exec_error_response,
    build_wait_response,
)

PROXY_SYSTEM_RULE_PATH = ".cursor/rules/proxy-system.mdc"
WORKSPACE_ROOT = os.getenv("CURSOR_WORKSPACE_ROOT", ".")
TOOL_BRIDGE_ENV = os.getenv("CURSOR_TOOL_BRIDGE", "off")
READ_MAX_BYTES = int(os.getenv("CURSOR_READ_MAX_BYTES", "512000"))
CHAT_ONLY_ENV = os.getenv("CURSOR_CHAT_ONLY", "0") == "1"
# Hard ceiling on a single Cursor turn (seconds). Guards against a silent backend
# that never sends turn_ended/EOS, which would otherwise hang a request forever.
# This is a *backstop* against a wedged stream, NOT a cap on legitimate work: a
# long research turn (e.g. 20+ web searches) can easily exceed several minutes of
# active time while making steady progress. The stall guards below (STALL_SEC /
# TOOL_STALL_SEC / REASONING_STALL_SEC) are what actually catch hung turns, so this
# is set generously to avoid killing healthy long-running turns. 0 = off.
TURN_TIMEOUT_SEC = float(os.getenv("CURSOR_TURN_TIMEOUT_SEC", "1200"))
# Keepalive-stall guard. Cursor's `auto` backend intermittently stalls in two
# distinct ways, both of which would otherwise hang until TURN_TIMEOUT_SEC:
#   (a) Total silence: it stops sending anything but empty keepalive heartbeats
#       (no content, reasoning, tool, or handshake), never delivering turn_ended.
#   (b) Reasoning loop: it streams *thinking* tokens indefinitely but never
#       produces visible output (when reasoning is not forwarded to the client).
# We therefore track two timers: silence of ALL upstream frames (catches (a)) and
# silence of CLIENT-VISIBLE output (catches (b)). Heartbeat frames (IU f13) count
# as upstream liveness so a thinking model that keeps the stream alive is not
# killed. 0 disables.
STALL_SEC = float(os.getenv("CURSOR_STALL_SEC", "60"))          # total-silence cap
# Cap on client-visible silence while upstream is still emitting (e.g. reasoning
# that isn't forwarded). Larger than STALL_SEC so a model that legitimately thinks
# for a while before answering isn't cut off, but a stuck think-loop still ends.
REASONING_STALL_SEC = float(os.getenv("CURSOR_REASONING_STALL_SEC", "60"))
# A genuine Cursor-side tool (web_search/fetch) emits no frames while it runs, so
# raise the total-silence cap while one is active to avoid killing a slow tool.
TOOL_STALL_SEC = float(os.getenv("CURSOR_TOOL_STALL_SEC", "90"))
STALL_MAX_RETRIES = int(os.getenv("CURSOR_STALL_RETRIES", "1"))
# Cap for native await / wait exec (milliseconds). Longer waits belong on the
# host scheduler, not on this H2 stream.
WAIT_CAP_MS = int(os.getenv("CURSOR_WAIT_CAP_MS", str(30 * 60 * 1000)))


@dataclass
class Timeouts:
    """Per-run stall / turn clocks. Env vars supply defaults."""

    turn_sec: float = field(default_factory=lambda: TURN_TIMEOUT_SEC)
    stall_sec: float = field(default_factory=lambda: STALL_SEC)
    reasoning_stall_sec: float = field(default_factory=lambda: REASONING_STALL_SEC)
    tool_stall_sec: float = field(default_factory=lambda: TOOL_STALL_SEC)
    stall_retries: int = field(default_factory=lambda: STALL_MAX_RETRIES)


@dataclass
class Checkpoint:
    """Opaque Cursor conversation state to send on the next Run."""

    state: bytes = b""
    conversation_id: str = ""
    group_id: str = ""
    blobs: dict[bytes, bytes] = field(default_factory=dict)

CHAT_ONLY_SYSTEM_RULE = (
    "You are in chat-only Q&A mode (like ChatGPT in a browser). "
    "Answer from general knowledge in well-formatted markdown. "
    "You MAY use web_search and web_fetch for current events, news, prices, docs, "
    "or anything that benefits from up-to-date information, and cite sources inline "
    "as [title](https://full-url). "
    "Do NOT read, grep, glob, edit, write, run shell commands, or otherwise explore "
    "any local filesystem, codebase, or project unless the user explicitly asks you "
    "to inspect their local files or repository. "
    "For troubleshooting questions, explain causes and fixes generally — do not "
    "investigate the user's machine or project."
)

RESEARCH_PROXY_HINT = (
    "For live web/news/product research use web_search and web_fetch only; "
    "do not run shell or curl. "
    "Cite sources with markdown links: [Site or article title](https://full-url) "
    "inline after claims. Prefer primary sources (manufacturer, reviews, news)."
)


# Injected callback signatures. ``exec_handler`` receives the canonical exec type
# (``read``/``shell``/``write``/``grep``/``edit``/``fetch``/``mcp``) and the parsed
# :class:`~cursor_core.wire.ServerMsg` (which carries the exec id + parsed args)
# and returns the protobuf bodies to frame back. ``kv_handler`` services the blob
# store (op, blob_id, blob_data) -> blob bytes | None. ``approval_handler`` gates
# web search/fetch (kind, query_id) -> approve?.
ExecHandler = Callable[[str, ServerMsg], Awaitable[list[bytes]]]
KvHandler = Callable[[str, "bytes | None", "bytes | None"], Awaitable["bytes | None"]]
ApprovalHandler = Callable[[str, str], Awaitable[bool]]


class ToolRunTracker:
    """Map Cursor tool-call start/complete onto stable UI run ids.

    Cursor often starts several tools before any of them complete. A single
    "active" id then stamps the first completion with the *latest* start's id,
    which leaves the original card spinning. Track each open call separately.
    """

    def __init__(self) -> None:
        self._by_run: dict[str, str] = {}
        self._by_call: dict[str, str] = {}
        self.active_label: str = ""
        self.active_run_id: str = ""
        # True once an exec was dispatched or a web query was approved.
        # ToolCallStarted alone must not stretch the silence cap to TOOL_STALL_SEC.
        self._dispatched = False

    def __bool__(self) -> bool:
        return bool(self._by_run)

    @property
    def uses_tool_stall(self) -> bool:
        return self._dispatched and bool(self._by_run)

    def mark_dispatched(self) -> None:
        self._dispatched = True

    def start(self, label: str, call_id: str = "") -> str:
        cid = (call_id or "").strip()
        run_id = cid or str(uuid.uuid4())
        self._by_run[run_id] = label
        if cid:
            self._by_call[cid] = run_id
        self.active_label = label
        self.active_run_id = run_id
        return run_id

    def complete(self, label: str, call_id: str = "") -> str:
        cid = (call_id or "").strip()
        run_id = ""
        if cid:
            run_id = self._by_call.get(cid) or (cid if cid in self._by_run else "")
        if not run_id and label:
            matches = [rid for rid, lab in self._by_run.items() if lab == label]
            if matches:
                run_id = matches[-1]
        if not run_id:
            return ""
        self._pop(run_id)
        return run_id

    def bind_call(self, call_id: str) -> str:
        cid = (call_id or "").strip()
        if not cid:
            return self.active_run_id
        if cid in self._by_call:
            rid = self._by_call[cid]
        elif cid in self._by_run:
            rid = cid
        else:
            return self.active_run_id
        self.active_run_id = rid
        self.active_label = self._by_run.get(rid, self.active_label)
        return rid

    def leftovers(self) -> list[tuple[str, str]]:
        items = list(self._by_run.items())
        self.clear()
        return items

    def clear(self) -> None:
        self._by_run.clear()
        self._by_call.clear()
        self.active_label = ""
        self.active_run_id = ""
        self._dispatched = False

    def _pop(self, run_id: str) -> None:
        self._by_run.pop(run_id, None)
        for key, val in list(self._by_call.items()):
            if val == run_id:
                self._by_call.pop(key, None)
        if not self._by_run:
            self._dispatched = False
            self.active_run_id = ""
            self.active_label = ""
            return
        if self.active_run_id != run_id:
            return
        rid, lab = next(reversed(list(self._by_run.items())))
        self.active_run_id = rid
        self.active_label = lab


def _exec_tool_call_id(smsg) -> str:
    """Best-effort Cursor tool_call_id from a parsed exec / tool-event frame."""
    return (
        getattr(smsg, "tool_event_call_id", "")
        or getattr(smsg, "shell_tool_call_id", "")
        or getattr(smsg, "read_tool_call_id", "")
        or getattr(smsg, "write_tool_call_id", "")
        or getattr(smsg, "grep_tool_call_id", "")
        or getattr(smsg, "edit_tool_call_id", "")
        or getattr(smsg, "fetch_tool_call_id", "")
        or getattr(smsg, "subagent_tool_call_id", "")
        or getattr(smsg, "get_mcp_tools_call_id", "")
    )


class _StallRetry(Exception):
    """Upstream went silent before any output was streamed; retry the run."""


def _turn_expired(
    turn_start: float,
    suspended_total: float,
    now: float,
    turn_sec: float | None = None,
) -> bool:
    """True when active (non-suspended) turn time has exceeded the turn cap."""
    cap = TURN_TIMEOUT_SEC if turn_sec is None else turn_sec
    if cap <= 0:
        return False
    return (now - turn_start - suspended_total) > cap


def _wire_log(label: str, field_num: int, wire: int, val) -> None:
    from .wire import DEBUG_WIRE
    if not DEBUG_WIRE:
        return
    import sys
    prev = val.hex()[:220] if isinstance(val, (bytes, bytearray)) else val
    print(f"[wire] {label} field={field_num} wire={wire} val={prev}", file=sys.stderr, flush=True)


# Full-frame recursive protobuf decoder for stall diagnostics. Gated behind
# CURSOR_DEBUG_FRAMES so it costs nothing in normal operation. Decodes every
# field of every frame (recursing into length-delimited submessages) so we can
# see exactly what Cursor sends in the dead zone after a tool batch.
DEBUG_FRAMES = os.getenv("CURSOR_DEBUG_FRAMES", "0") == "1"


def _looks_like_message(data: bytes) -> bool:
    """Heuristic: can `data` be fully consumed as a sequence of protobuf fields?"""
    from .wire import _read_field
    if not data:
        return False
    pos = 0
    seen = 0
    while pos < len(data):
        r = _read_field(data, pos)
        if r is None:
            return False
        field, wire, _val, newpos = r
        if newpos <= pos or field == 0 or wire not in (0, 1, 2, 5):
            return False
        pos = newpos
        seen += 1
    return pos == len(data) and seen > 0


def _decode_frame(data: bytes, depth: int = 0, max_depth: int = 6) -> str:
    """Recursively render a protobuf blob as `field#W=value` tokens."""
    from .wire import _read_field
    if depth > max_depth:
        return f"<{len(data)}B deep>"
    parts: list[str] = []
    pos = 0
    while pos < len(data):
        r = _read_field(data, pos)
        if r is None:
            parts.append(f"<unparsed:{data[pos:].hex()[:40]}>")
            break
        field, wire, val, pos = r
        if wire == 0:
            parts.append(f"{field}#V={val}")
        elif wire in (1, 5):
            parts.append(f"{field}#F={val}")
        elif wire == 2 and isinstance(val, (bytes, bytearray)):
            if _looks_like_message(bytes(val)):
                parts.append(f"{field}{{{_decode_frame(bytes(val), depth + 1, max_depth)}}}")
            else:
                try:
                    s = val.decode("utf-8")
                    if s.isprintable() or "\n" in s:
                        disp = s if len(s) <= 80 else s[:80] + f"…(+{len(s)-80})"
                        parts.append(f"{field}#S={disp!r}")
                    else:
                        raise UnicodeDecodeError("x", b"", 0, 1, "x")
                except (UnicodeDecodeError, ValueError):
                    h = val.hex()
                    disp = h if len(h) <= 60 else h[:60] + f"…(+{len(val)-30}B)"
                    parts.append(f"{field}#B[{len(val)}]={disp}")
        else:
            parts.append(f"{field}#?{wire}")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Per-turn configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    """Per-turn Cursor AgentService/Run options."""

    conversation_id: str | None = None
    conversation_group_id: str | None = None
    conversation_state: bytes = b""
    model_id: str = "default"
    model_parameters: list[tuple[str, str]] = field(default_factory=list)
    max_mode: bool = False
    cancel_event: asyncio.Event | None = None
    initial_blobs: dict[bytes, bytes] = field(default_factory=dict)
    session_out: dict | None = None  # populated with blob_store after the turn
    system_prompt: str | None = None
    use_system_in_context: bool = field(
        default_factory=lambda: os.getenv("CURSOR_SYSTEM_IN_CONTEXT", "1") == "1"
    )
    tools: list[dict] | None = None
    bridge_mode: str = field(default_factory=lambda: TOOL_BRIDGE_ENV)
    workspace_root: str = field(default_factory=lambda: WORKSPACE_ROOT)
    thread_key: str | None = None
    chat_only: bool = False
    # Whether the model's reasoning_delta is forwarded to the client for this
    # turn (drives the client-visible stall timer: reasoning only counts as
    # visible progress when it is actually streamed out). Defaults True.
    stream_reasoning: bool = True
    # Pre-encoded CursorRule bytes (each is pb_msg(2, rule_bytes)) injected into
    # RequestContext. Populated by the capabilities layer (rules + skills).
    cursor_rules: list[bytes] = field(default_factory=list)
    # Pre-encoded MCP tools body for RequestContext (fields 7,14,23,36,44).
    # Populated by McpManager.to_request_context_tools().
    mcp_tools: bytes | None = None
    # Optional Cursor access token. When unset, the host CLI credential is used.
    access_token: str | None = None
    timeouts: Timeouts = field(default_factory=Timeouts)
    # Mid-run follow-ups. Items are (message_id, text). Drained onto the open
    # H2 stream as AgentClientMessage.conversation_action before the next
    # outbound frame (tool result or the 1s keepalive pump).
    steer_queue: asyncio.Queue | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Protobuf message builders
# ──────────────────────────────────────────────────────────────────────────────

def build_model_parameter(param_id: str, value: str) -> bytes:
    """RequestedModel.ModelParameterValue{f1: id, f2: value}."""
    return pb_str(1, param_id) + pb_str(2, value)


def build_requested_model(
    model_id: str,
    parameters: list[tuple[str, str]] | None = None,
    *,
    max_mode: bool = False,
) -> bytes:
    """agent.v1.RequestedModel."""
    body = pb_str(1, model_id)
    if max_mode:
        body += pb_int(2, 1)
    for pid, pval in parameters or []:
        body += pb_msg(3, build_model_parameter(pid, pval))
    return body


def build_run_request(
    messages: list[dict],
    model: str = "auto",
    conversation_id: str | None = None,
    *,
    run_config: RunConfig | None = None,
) -> bytes:
    """AgentClientMessage{f1: AgentRunRequest}."""
    cfg = run_config or RunConfig()
    if conversation_id is None:
        conversation_id = cfg.conversation_id or str(uuid.uuid4())
    if not cfg.conversation_id:
        cfg.conversation_id = conversation_id
    group_id = cfg.conversation_group_id or conversation_id
    if not cfg.conversation_group_id:
        cfg.conversation_group_id = group_id

    # "auto" and "" are UI aliases for Cursor's internal "default" router name.
    _raw_model = cfg.model_id if run_config else model
    model_name = "default" if _raw_model in ("auto", "default", "") else _raw_model

    msg_id = str(uuid.uuid4())

    if cfg.system_prompt is None:
        extracted = _extract_system_prompt(messages)
        if run_config and extracted:
            cfg.system_prompt = extracted

    if run_config and cfg.chat_only:
        if cfg.system_prompt:
            if CHAT_ONLY_SYSTEM_RULE not in cfg.system_prompt:
                cfg.system_prompt = f"{cfg.system_prompt}\n\n{CHAT_ONLY_SYSTEM_RULE}"
        else:
            cfg.system_prompt = CHAT_ONLY_SYSTEM_RULE

    omit_system = bool(
        run_config and cfg.use_system_in_context and cfg.system_prompt
    )
    prompt_text = _render_prompt(
        messages,
        include_system=not omit_system,
        omit_tool_messages=bool(cfg.tools),
    )
    images = _extract_images(messages)
    documents = _extract_documents(messages)
    if not prompt_text and not images and not documents:
        raise ValueError("At least one user message is required")

    if not cfg.tools and cfg.bridge_mode == "off" and not cfg.chat_only:
        if RESEARCH_PROXY_HINT not in prompt_text:
            prompt_text = f"{RESEARCH_PROXY_HINT}\n\n{prompt_text}"

    # UserMessage { f1:text, f2:message_id, f3:selected_context, f4:mode(1=NORMAL) }
    user_message = pb_str(1, prompt_text) + pb_str(2, msg_id)
    if images or documents:
        # SelectedContext { f1: selected_images, f25: selected_documents }
        selected_context = b"".join(
            pb_msg(1, _encode_selected_image(mime, data)) for mime, data in images
        )
        selected_context += b"".join(
            pb_msg(25, _encode_selected_document(mime, filename, data))
            for mime, filename, data in documents
        )
        user_message += pb_msg(3, selected_context)
    user_message += pb_int(4, 1)
    # ConversationAction.f1 = UserMessageAction.f1 = UserMessage
    action = pb_msg(1, pb_msg(1, user_message))

    requested_model = build_requested_model(
        model_name, cfg.model_parameters, max_mode=cfg.max_mode
    )
    state = cfg.conversation_state if run_config else b""
    run_request = (
        pb_bytes(1, state)           # f1 = conversation_state
        + pb_msg(2, action)          # f2 = action
        + pb_str(5, conversation_id) # f5 = conversation_id
        + pb_msg(9, requested_model) # f9 = requested_model
        + pb_int(12, 0)              # f12 = exclude_workspace_context = false
        + pb_msg(14, requested_model)# f14 = selected_subagent_models[0]
        + pb_str(16, group_id)        # f16 = conversation_group_id
    )
    return pb_msg(1, run_request)


def build_steer_message(text: str, message_id: str | None = None) -> bytes:
    """AgentClientMessage{f4: ConversationAction{f1: UserMessageAction{f1: UserMessage}}}.

    Mid-run steer. Cursor holds it and applies it at the next tool boundary
    instead of ending the turn.
    """
    msg_id = message_id or str(uuid.uuid4())
    user_message = pb_str(1, text) + pb_str(2, msg_id) + pb_int(4, 1)
    action = pb_msg(1, pb_msg(1, user_message))
    return pb_msg(4, action)


def _extract_system_prompt(messages: list[dict]) -> str:
    """Join OpenAI system messages into one string."""
    parts: list[str] = []
    for m in messages:
        if m.get("role") == "system":
            content = _flatten_content(m.get("content"))
            if content:
                parts.append(content)
    return "\n\n".join(parts)


def _render_prompt(
    messages: list[dict],
    *,
    include_system: bool = True,
    omit_tool_messages: bool = False,
) -> str:
    """
    Flatten an OpenAI message list into a single prompt string.

    When include_system is False the caller supplies system instructions via
    RequestContext.rules (Option A) instead of prepending them here.
    """
    system_parts: list[str] = []
    convo: list[tuple[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        if omit_tool_messages and role == "tool":
            continue
        if omit_tool_messages and role == "assistant" and m.get("tool_calls"):
            continue
        content = _flatten_content(m.get("content"))
        if role == "system":
            if include_system and content:
                system_parts.append(content)
        elif content:
            convo.append((role, content))

    system_prompt = "\n\n".join(system_parts)
    if not convo:
        return system_prompt

    label = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
    if len(convo) == 1 and convo[0][0] == "user":
        body = convo[0][1]
    else:
        lines = [f"{label.get(r, r.title())}: {c}" for r, c in convo]
        body = "\n\n".join(lines)

    if system_prompt:
        return f"{system_prompt}\n\n{body}"
    return body


def _build_cursor_rule(content: str, full_path: str = PROXY_SYSTEM_RULE_PATH) -> bytes:
    """agent.v1.CursorRule { f1: path, f2: content, f3: type{global}, f4: source }."""
    rule_type = pb_msg(1, b"")  # CursorRuleType.global (empty message)
    return (
        pb_str(1, full_path)
        + pb_str(2, content)
        + pb_msg(3, rule_type)
        + pb_int(4, 1)
    )


def _build_request_context(
    *,
    system_prompt: str | None = None,
    mcp_tools_body: bytes = b"",
    cursor_rules: "list[bytes] | tuple[bytes, ...] | None" = None,
) -> bytes:
    """agent.v1.RequestContext — synthetic rules + MCP tool definitions."""
    body = b""
    if system_prompt:
        body += pb_msg(2, _build_cursor_rule(system_prompt))
    for rule_bytes in cursor_rules or []:
        body += rule_bytes
    body += mcp_tools_body
    return body


def build_kv_response(kv_id: int, op: str, blob_data: bytes | None = None) -> bytes:
    """
    AgentClientMessage{f3: KvClientMessage{f1:id, f2:get_blob_result | f3:set_blob_result}}.

    The server uses the client as a conversation blob store. We must ack every
    set_blob and answer every get_blob, otherwise the server never finalises the
    turn (no turn_ended is emitted).
    """
    if op == "set":
        # SetBlobResult{} (no f1=error -> success)
        kv = pb_msg(3, b"")
    else:  # get
        if blob_data is not None:
            result = pb_bytes(1, blob_data)   # GetBlobResult{f1: blob_data}
        else:
            result = b""                      # blob_data unset -> not found
        kv = pb_msg(2, result)
    body = (pb_int(1, kv_id) if kv_id else b"") + kv
    return pb_msg(3, body)


def build_interaction_approval(query_id: int, kind: str) -> bytes:
    """
    AgentClientMessage{f6: InteractionResponse{f1:id,
        f2:web_search_request_response | f9:web_fetch_request_response
        = {f1: approved{}}}}.

    Web search and web fetch are executed server-side but gated by an approval
    interaction. We auto-approve them (they're safe, read-only) so the tool runs
    and its results stream back into the model's answer. Without this reply the
    turn never completes.
    """
    approved = pb_msg(1, b"")                 # {Web*RequestResponse}{f1: approved (empty)}
    field_num = 2 if kind == "search" else 9  # f2 = web_search, f9 = web_fetch
    resp = pb_int(1, query_id) + pb_msg(field_num, approved)
    return pb_msg(6, resp)


def build_workspace_context_response(
    exec_id: int = 0,
    exec_id_str: str = "",
    *,
    system_prompt: str | None = None,
    mcp_tools_body: bytes = b"",
    cursor_rules: "list[bytes] | tuple[bytes, ...] | None" = None,
) -> bytes:
    """
    AgentClientMessage{f2: ExecClientMessage{
        f1:id, f15:exec_id,
        f10: RequestContextResult{f1: success{f1: request_context{}}}}}.

    When system_prompt is set we attach it as RequestContext.rules[] (Option A)
    instead of folding it into the user message. An empty context is still valid.
    """
    request_context = _build_request_context(
        system_prompt=system_prompt,
        mcp_tools_body=mcp_tools_body,
        cursor_rules=cursor_rules,
    )
    success = pb_msg(1, request_context)         # RequestContextSuccess{f1: request_context}
    result = pb_msg(1, success)                  # RequestContextResult{f1: success}

    exec_client_msg = b""
    if exec_id:
        exec_client_msg += pb_int(1, exec_id)    # f1 = id
    if exec_id_str:
        exec_client_msg += pb_str(15, exec_id_str)  # f15 = exec_id
    exec_client_msg += pb_msg(10, result)        # f10 = request_context_result

    return pb_msg(2, exec_client_msg)


def build_exec_fetch_response(
    exec_id: int,
    exec_id_str: str,
    url: str,
    content: str,
    status_code: int = 200,
    content_type: str = "text/plain",
) -> bytes:
    """ExecClientMessage{f20: FetchResult{f1: FetchSuccess{url, content, ...}}}."""
    success = (
        pb_str(1, url)
        + pb_str(2, content)
        + pb_int(3, status_code)
        + pb_str(4, content_type)
    )
    fetch_result = pb_msg(1, success)
    exec_client = b""
    if exec_id:
        exec_client += pb_int(1, exec_id)
    if exec_id_str:
        exec_client += pb_str(15, exec_id_str)
    exec_client += pb_msg(20, fetch_result)
    return pb_msg(2, exec_client)


def build_exec_fetch_error(
    exec_id: int,
    exec_id_str: str,
    url: str,
    error: str,
) -> bytes:
    """ExecClientMessage{f20: FetchResult{f2: FetchError{url, error}}}."""
    err = pb_str(1, url) + pb_str(2, error)
    fetch_result = pb_msg(2, err)
    exec_client = b""
    if exec_id:
        exec_client += pb_int(1, exec_id)
    if exec_id_str:
        exec_client += pb_str(15, exec_id_str)
    exec_client += pb_msg(20, fetch_result)
    return pb_msg(2, exec_client)


def _fetch_ssrf_guard(url: str) -> None:
    """Reject non-HTTP(S) schemes and hosts resolving to private/link-local IPs.

    Inline so cursor_core stays app-agnostic. Hosts with their own egress
    policy should inject ``exec_handler`` instead of using this default.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise PermissionError(f"scheme not allowed: {url!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"no host in URL: {url!r}")
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    blocked = (
        ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"), ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"), ipaddress.ip_network("100.64.0.0/10"),
        ipaddress.ip_network("0.0.0.0/8"), ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"), ipaddress.ip_network("fe80::/10"),
    )
    try:
        addrs = [str(i[4][0]) for i in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)]
    except OSError as exc:
        raise PermissionError(f"DNS lookup failed for {host!r}") from exc
    for a in addrs:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            raise PermissionError(f"unparseable address for {host!r}") from None
        if any(ip in net for net in blocked):
            raise PermissionError(f"SSRF blocked: {host!r} -> {a}")


def _http_fetch(url: str, max_bytes: int = 512_000, timeout: int = 30) -> tuple[str, int, str]:
    """Fetch a URL for ExecServerMessage fetch_args (client-side web fetch path)."""
    _fetch_ssrf_guard(url)
    # Disable redirect following so a public URL cannot bounce to a private one.
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(url, timeout=timeout) as resp:  # noqa: S310
        body = resp.read(max_bytes)
        ctype = (resp.headers.get("Content-Type") or "text/plain").split(";")[0]
        return body.decode("utf-8", "replace"), resp.status, ctype


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401, ANN002, ANN003
        return None  # do not follow redirects


def _flatten_content(content) -> str:
    """OpenAI content may be None, a string, or a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return str(content)


def _encode_selected_image(mime: str, data: bytes) -> bytes:
    """SelectedImage{f2: uuid, f7: mime_type, f8: data} (inline image bytes)."""
    img = pb_str(2, str(uuid.uuid4()))
    if mime:
        img += pb_str(7, mime)
    img += pb_bytes(8, data)          # f8 = data (oneof data_or_blob_id)
    return img


def _encode_selected_document(mime: str, filename: str, data: bytes) -> bytes:
    """agent.v1.SelectedDocument{f2:uuid, f3:filename, f4:mime_type, f8:data}.

    Mirrors SelectedImage but for documents (e.g. PDFs). `data` is the inline
    file bytes (oneof data_or_blob_id → field 8). Verified against Cursor's
    agent.v1 schema (selected_documents is SelectedContext field 25).
    """
    doc = pb_str(2, str(uuid.uuid4()))
    if filename:
        doc += pb_str(3, filename)
    if mime:
        doc += pb_str(4, mime)
    doc += pb_bytes(8, data)          # f8 = data (oneof data_or_blob_id)
    return doc


def _decode_image_url(url: str) -> tuple[str, bytes] | None:
    """Resolve an OpenAI image_url (data: URL or http(s) URL) to (mime, bytes)."""
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        meta = header[5:]
        mime = meta.split(";")[0] or "image/png"
        try:
            if ";base64" in meta:
                return mime, base64.b64decode(payload)
            return mime, unquote_to_bytes(payload)
        except Exception:
            return None
    if url.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:  # noqa: S310
                data = resp.read()
                mime = (resp.headers.get("Content-Type") or "image/png").split(";")[0]
                return mime, data
        except Exception:
            return None
    return None


def _extract_images(messages: list[dict]) -> list[tuple[str, bytes]]:
    """Pull image parts from the latest user message (the current turn)."""
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not last_user:
        return []
    content = last_user.get("content")
    if not isinstance(content, list):
        return []
    images: list[tuple[str, bytes]] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        iu = part.get("image_url")
        url = iu.get("url") if isinstance(iu, dict) else iu
        if isinstance(url, str):
            decoded = _decode_image_url(url)
            if decoded:
                images.append(decoded)
    return images


def _extract_documents(messages: list[dict]) -> list[tuple[str, str, bytes]]:
    """Pull document (e.g. PDF) parts from the latest user message.

    Recognises OpenAI-style file parts:
        {"type": "file", "file": {"filename": "x.pdf",
                                  "file_data": "data:application/pdf;base64,..."}}
    Returns a list of (mime, filename, bytes).
    """
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not last_user:
        return []
    content = last_user.get("content")
    if not isinstance(content, list):
        return []
    docs: list[tuple[str, str, bytes]] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "file":
            continue
        f = part.get("file")
        if not isinstance(f, dict):
            continue
        url = f.get("file_data") or f.get("url")
        filename = f.get("filename") or "document"
        if isinstance(url, str):
            decoded = _decode_image_url(url)  # generic data:/http(s): decoder
            if decoded:
                mime, data = decoded
                docs.append((mime or "application/pdf", filename, data))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Default inline handlers (grep/glob local, in-memory KV, auto-approve web)
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_workspace_path(path: str, workspace_root: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(workspace_root) / p
    return p.resolve()


_TOOL_CALL_ID_RE = re.compile(r"^tool_[0-9a-fA-F-]{8,}$")


def _execute_local_grep(
    search_path: str,
    pattern: str,
    workspace_root: str,
) -> tuple[str, list[str], str | None]:
    if _TOOL_CALL_ID_RE.match((pattern or "").strip()):
        return (
            "error",
            [],
            "that looks like a tool call id, not a search. Call the tool by name "
            "(browser_navigate, browser_solve_captcha). Do not grep the codebase "
            "to learn how browser tools work.",
        )
    root = _resolve_workspace_path(search_path or ".", workspace_root)
    if not root.exists():
        return "error", [], f"{search_path} not found"
    try:
        if "*" in pattern or "?" in pattern or "**" in pattern:
            glob_pattern = pattern.removeprefix("**/")
            matches: list[str] = []
            if root.is_dir():
                for p in root.rglob(glob_pattern):
                    if p.is_file():
                        matches.append(str(p))
            elif fnmatch.fnmatch(root.name, glob_pattern):
                matches.append(str(root))
            return "success", sorted(matches), None

        proc = subprocess.run(
            ["rg", "-l", "--fixed-strings", pattern, str(root)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode not in (0, 1):
            return "error", [], proc.stderr.strip() or f"rg exit {proc.returncode}"
        matches = [line for line in proc.stdout.splitlines() if line.strip()]
        return "success", matches, None
    except subprocess.TimeoutExpired:
        return "error", [], "grep timed out after 30s"
    except FileNotFoundError:
        try:
            proc = subprocess.run(
                ["grep", "-RIl", pattern, str(root)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            matches = [line for line in proc.stdout.splitlines() if line.strip()]
            return "success", matches, None
        except subprocess.TimeoutExpired:
            return "error", [], "grep timed out after 30s"
        except OSError as e:
            return "error", [], str(e)
    except OSError as e:
        return "error", [], str(e)


def _exec_type_of(smsg: ServerMsg) -> str | None:
    """Canonical exec type for a parsed ExecServerMessage tool request, or None."""
    if smsg.wants_read:
        return "read"
    if smsg.wants_shell:
        return "shell"
    if smsg.wants_write:
        return "write"
    if smsg.wants_grep:
        return "grep"
    if smsg.wants_edit:
        return "edit"
    if smsg.wants_delete:
        return "delete"
    if smsg.wants_fetch:
        return "fetch"
    if smsg.wants_mcp:
        return "mcp"
    if smsg.wants_list_mcp_resources:
        return "list_mcp_resources"
    if smsg.wants_get_mcp_tools:
        return "get_mcp_tools"
    if smsg.wants_mcp_state:
        return "mcp_state"
    if smsg.wants_subagent:
        return "subagent"
    if smsg.wants_force_background_subagent:
        return "force_background_subagent"
    if smsg.wants_subagent_await:
        return "subagent_await"
    if smsg.wants_wait:
        return "wait"
    return None


async def execute_wait(
    smsg: ServerMsg,
    *,
    waiter: Callable[[str, str, int], Awaitable[None]] | None = None,
) -> list[bytes]:
    """Sleep (or wait for a session pattern) then ack the await exec.

    Duration is capped at ``WAIT_CAP_MS``. ``waiter``, when set, receives
    ``(session_id, pattern, duration_ms)`` so the host can watch a PTY.
    """
    ms = max(0, int(getattr(smsg, "wait_ms", 0) or 0))
    ms = min(ms, WAIT_CAP_MS)
    session_id = getattr(smsg, "wait_session_id", "") or ""
    pattern = getattr(smsg, "wait_pattern", "") or ""
    if waiter is not None and (session_id or pattern):
        await waiter(session_id, pattern, ms)
    elif ms:
        await asyncio.sleep(ms / 1000.0)
    stamp_ui_output(smsg, f"waited {ms}ms")
    field = getattr(smsg, "wait_result_field", 0) or 2
    return [build_wait_response(smsg.exec_id, smsg.exec_id_str, field, ms)]


def _reject_exec_frames(
    exec_type: str,
    smsg: ServerMsg,
    reason: str = "tool not available (cursor_core default handler)",
) -> list[bytes]:
    """A type-appropriate error frame so the server can finalise the turn."""
    if exec_type == "read":
        return [build_read_error_response(smsg.exec_id, smsg.exec_id_str, smsg.read_path, reason)]
    if exec_type == "shell":
        return [
            build_shell_rejected_response(
                smsg.exec_id, smsg.exec_id_str, smsg.shell_command, reason
            )
        ]
    if exec_type == "write":
        return [build_write_error_response(smsg.exec_id, smsg.exec_id_str, smsg.write_path, reason)]
    if exec_type == "edit":
        return [build_edit_error_response(smsg.exec_id, smsg.exec_id_str, smsg.edit_path, reason)]
    if exec_type == "fetch":
        return [build_fetch_error_response(smsg.exec_id, smsg.exec_id_str, smsg.fetch_url, reason)]
    if exec_type == "grep":
        return [
            build_grep_error_response(
                smsg.exec_id, smsg.exec_id_str, smsg.grep_search_path, reason
            )
        ]
    if exec_type == "delete":
        return [
            build_delete_rejected_response(
                smsg.exec_id, smsg.exec_id_str, smsg.read_path, reason
            )
        ]
    result_field = (
        getattr(smsg, "get_mcp_tools_result_field", 0)
        or getattr(smsg, "wait_result_field", 0)
        or getattr(smsg, "unknown_exec_field", 0)
    )
    if exec_type == "mcp_state":
        result_field = result_field or 36
    if result_field and (smsg.exec_id or smsg.exec_id_str):
        return [build_unknown_exec_error_response(
            smsg.exec_id, smsg.exec_id_str, result_field, reason,
        )]
    # mcp / unknown with no result field: cannot build an envelope.
    return []


def _make_default_exec_handler(workspace_root: str) -> ExecHandler:
    """Default exec_handler: serve grep/glob locally, reject everything else."""

    async def handler(exec_type: str, smsg: ServerMsg) -> list[bytes]:
        if exec_type == "grep":
            status, matches, err = await asyncio.to_thread(
                _execute_local_grep, smsg.grep_search_path, smsg.grep_pattern, workspace_root
            )
            if status == "success":
                return [
                    build_grep_success_response(
                        smsg.exec_id, smsg.exec_id_str,
                        smsg.grep_search_path, smsg.grep_pattern, matches,
                    )
                ]
            return [
                build_grep_error_response(
                    smsg.exec_id, smsg.exec_id_str,
                    smsg.grep_search_path, err or "grep failed",
                )
            ]
        if exec_type == "wait":
            return await execute_wait(smsg)
        return _reject_exec_frames(exec_type, smsg)

    return handler


def _make_default_kv_handler(
    initial_blobs: dict[bytes, bytes],
) -> tuple[KvHandler, dict[bytes, bytes]]:
    """Default kv_handler: an in-memory blob store seeded with ``initial_blobs``."""
    store: dict[bytes, bytes] = dict(initial_blobs)

    async def handler(op: str, blob_id: bytes | None, blob_data: bytes | None) -> bytes | None:
        if op == "set":
            if blob_id is not None:
                store[blob_id] = blob_data or b""
            return None
        return store.get(blob_id) if blob_id else None

    return handler, store


async def _default_approval_handler(kind: str, query_id: str) -> bool:
    """Auto-approve web_search / web_fetch (read-only, safe)."""
    return kind in ("search", "fetch")


# ──────────────────────────────────────────────────────────────────────────────
# HTTP/2 full-duplex transport
# ──────────────────────────────────────────────────────────────────────────────

async def _emit(queue: asyncio.Queue | None, kind: str, payload: str):
    if queue is not None:
        await queue.put((kind, payload))
    return kind, payload


async def _h2_run_loop(
    messages: list[dict],
    model: str,
    conversation_id: str | None,
    cfg: RunConfig,
    *,
    output_queue: asyncio.Queue | None,
    exec_handler: ExecHandler,
    kv_handler: KvHandler,
    approval_handler: ApprovalHandler,
) -> None:
    """Drive one AgentService/Run H2 session; emit events to the output queue.

    Wrapped in a keepalive-stall retry loop: if the upstream goes silent before
    any output is streamed, the H2 session is torn down and retried (see
    STALL_SEC / STALL_MAX_RETRIES). `produced_output` persists across attempts so
    once anything has reached the client we never retry (avoids dupes / re-running
    tool side effects) and instead finalize on the next stall.

    Every mid-stream exec / KV / wait / approval request is answered **inline**
    on this same stream. Stall and turn clocks pause while a local handler runs.
    """
    token = resolve_access_token(cfg.access_token)
    _t0 = time.monotonic()

    def _tlog(label: str) -> None:
        print(f"[timing] +{time.monotonic()-_t0:6.3f}s  {label}", flush=True)

    async def put(kind: str, payload: str):
        if output_queue is not None:
            await output_queue.put((kind, payload))

    produced_output = False
    tracker = ToolRunTracker()
    timeouts = cfg.timeouts if cfg.timeouts is not None else Timeouts()
    last_checkpoint = cfg.conversation_state or b""

    async def finish(reason: str = "clean", *, cause: str = "") -> None:
        """Emit a stable end_reason then the internal _done terminator."""
        if last_checkpoint:
            await put("checkpoint", base64.b64encode(last_checkpoint).decode("ascii"))
            if cfg.session_out is not None:
                cfg.session_out["checkpoint"] = last_checkpoint
                cfg.session_out["conversation_id"] = cfg.conversation_id or ""
                cfg.session_out["conversation_group_id"] = cfg.conversation_group_id or ""
        _tlog(f"finish reason={reason} cause={cause or reason}")
        for run_id, label in tracker.leftovers():
            await put("tool", json.dumps({
                "phase": "completed", "name": label, "id": run_id,
            }))
        await put("end_reason", reason)
        await put("_done", "")

    _tlog("start")
    for attempt in range(timeouts.stall_retries + 1):
        request_id = str(uuid.uuid4())
        run_request = build_run_request(messages, model, conversation_id, run_config=cfg)
        _tlog("run_request built")

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        ssl_ctx.set_alpn_protocols(["h2"])
        reader, writer = await asyncio.open_connection(
            AGENT_HOST, AGENT_PORT, ssl=ssl_ctx, server_hostname=AGENT_HOST
        )
        _tlog("tcp+tls connected")
        _retry = False
        try:
            conn = h2.connection.H2Connection(
                config=h2.config.H2Configuration(client_side=True)
            )
            conn.initiate_connection()
            stream_id = conn.get_next_available_stream_id()
            conn.send_headers(stream_id, _build_headers(token, request_id), end_stream=False)
            sender = _Sender(conn, writer, stream_id)
            _raw_pump = sender.pump

            async def _pump_with_steers() -> None:
                # Prepend steers so they leave before a tool result queued
                # by the caller. Cursor applies them at the next tool boundary.
                q = cfg.steer_queue
                if q is not None:
                    prefix = bytearray()
                    while True:
                        try:
                            msg_id, text = q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if text:
                            prefix.extend(connect_frame(build_steer_message(text, msg_id)))
                            _tlog(f"steer {str(msg_id)[:8]}")
                    if prefix:
                        sender.pending = prefix + sender.pending
                await _raw_pump()

            sender.pump = _pump_with_steers  # type: ignore[method-assign]
            sender.queue(connect_frame(run_request))
            await sender.pump()
            _tlog("run_request sent")

            buf = b""
            context_sent = False
            turn_finished = False
            h2_done = False
            mcp_tools_body = cfg.mcp_tools or b""
            turn_start = time.monotonic()
            # Two stall timers (see STALL_SEC / REASONING_STALL_SEC notes above):
            #   last_upstream: any non-keepalive frame from Cursor (detects a dead
            #                  stream / total silence).
            #   last_visible:  output the client will actually surface (detects a
            #                  reasoning loop that never produces a visible answer).
            last_upstream = turn_start
            last_visible = turn_start
            suspended_total = 0.0

            def _receive_events(chunk: bytes) -> list:
                if h2_done or not chunk:
                    return []
                try:
                    return conn.receive_data(chunk)
                except (h2.exceptions.ProtocolError, h2.exceptions.StreamClosedError) as e:
                    if "CLOSED" in str(e) or "closed" in str(e).lower():
                        return []
                    raise

            while True:
                if cfg.cancel_event and cfg.cancel_event.is_set():
                    await _close(conn, writer, stream_id)
                    await finish("cancel", cause="cancel_event")
                    return
                _now = time.monotonic()
                if not turn_finished and _turn_expired(
                    turn_start, suspended_total, _now, timeouts.turn_sec
                ):
                    await put(
                        "error",
                        f"Cursor turn timed out after {timeouts.turn_sec:.0f}s",
                    )
                    await _close(conn, writer, stream_id)
                    await finish("timeout", cause="turn_timeout")
                    return
                # Keepalive-stall guard. Heartbeats (IU f13) count as upstream
                # liveness so a thinking model that keeps the stream alive is
                # not killed. Visible-silence still catches think-loops with
                # no forwarded output.
                if not turn_finished and timeouts.stall_sec > 0:
                    upstream_silence = _now - last_upstream
                    visible_silence = _now - last_visible
                    silence_cap = (
                        timeouts.tool_stall_sec
                        if tracker.uses_tool_stall
                        else timeouts.stall_sec
                    )
                    total_silent = upstream_silence > silence_cap
                    visible_silent = (
                        not tracker
                        and timeouts.reasoning_stall_sec > 0
                        and visible_silence > timeouts.reasoning_stall_sec
                    )
                    if total_silent or visible_silent:
                        reason = (
                            f"upstream silent {upstream_silence:.0f}s"
                            if total_silent
                            else f"no client-visible output {visible_silence:.0f}s"
                        )
                        await _close(conn, writer, stream_id)
                        if not produced_output and attempt < timeouts.stall_retries:
                            print(
                                f"[stall] {reason}, no visible output yet; retrying run "
                                f"(attempt {attempt + 2}/{timeouts.stall_retries + 1})",
                                flush=True,
                            )
                            raise _StallRetry()
                        if not produced_output:
                            print(
                                f"[stall] {reason}, no visible output after "
                                f"{timeouts.stall_retries + 1} attempt(s); surfacing error",
                                flush=True,
                            )
                            await put("error", "Cursor produced no output (stalled)")
                            await finish("stall", cause=reason)
                            return
                        print(
                            f"[stall] {reason}; finalizing turn with partial output",
                            flush=True,
                        )
                        await finish("stall", cause=reason)
                        return
                try:
                    data = await asyncio.wait_for(reader.read(65536), timeout=1.0)
                except asyncio.TimeoutError:
                    await sender.pump()
                    if turn_finished:
                        break
                    continue
                if not data:
                    _tlog("read returned empty (EOF)")
                    break
                for event in _receive_events(data):
                    if isinstance(event, h2.events.DataReceived):
                        conn.acknowledge_received_data(
                            event.flow_controlled_length, event.stream_id
                        )
                        buf += event.data
                        frames, buf = parse_frames(buf)
                        for flag, fd in frames:
                            if flag & 0x02:
                                err = _trailer_error(fd)
                                if err:
                                    _tlog(f"trailer error: {err}")
                                    if cfg.conversation_state and not produced_output:
                                        await put("native_fallback", err)
                                        await finish(
                                            "error",
                                            cause=f"native_state_rejected:{err}",
                                        )
                                        return
                                    raise RuntimeError(f"Cursor stream error: {err}")
                                h2_done = True
                                await _close(conn, writer, stream_id)
                                if turn_finished:
                                    await finish("clean", cause="eos_trailer")
                                else:
                                    await finish("dropped", cause="eos_trailer_before_turn_ended")
                                return
                            fd = _maybe_gunzip(flag, fd)
                            if DEBUG_FRAMES:
                                _tlog(f"FRAME[{len(fd)}B] {_decode_frame(fd)}")
                            smsg = parse_server_message(fd)
                            # Any real (non-keepalive) upstream frame proves the
                            # stream is alive -> resets the total-silence timer.
                            _upstream_active = (
                                smsg.text_delta or smsg.reasoning_delta or smsg.turn_ended
                                or smsg.wants_context or smsg.kv_op or smsg.query_kind
                                or smsg.tool_event or smsg.wants_fetch
                                or smsg.wants_get_mcp_tools
                                or smsg.heartbeat
                                or bool(smsg.checkpoint)
                                or smsg.wants_wait
                                or bool(smsg.user_message_appended_id)
                            )
                            # Log all non-keepalive upstream frames for diagnostics
                            if _upstream_active:
                                last_upstream = time.monotonic()
                            # Protocol-drift signal: unused ASM/exec fields with a decode.
                            _exec_t = _exec_type_of(smsg)
                            unknown_parts: list[str] = []
                            for kind, num, blob in (
                                [("asm", n, b) for n, b in smsg.unknown_top_fields]
                                + [("exec", n, b) for n, b in smsg.unknown_exec_fields]
                            ):
                                decoded = _decode_frame(blob)
                                if len(decoded) > 400:
                                    decoded = decoded[:400] + "…"
                                unknown_parts.append(
                                    f"{kind} f{num}({len(blob)}B): {decoded}"
                                )
                            if unknown_parts and not smsg.wants_get_mcp_tools:
                                _tlog("unknown fields: " + "; ".join(unknown_parts))
                            elif not _upstream_active and not _exec_t and fd:
                                decoded = _decode_frame(fd)
                                if len(decoded) > 400:
                                    decoded = decoded[:400] + "…"
                                _tlog(
                                    f"frame: no recognized fields (raw {len(fd)}B) {decoded}"
                                )
                            # Client-visible progress also resets the visible-silence
                            # timer. Reasoning counts only when it's forwarded; the
                            # handshake/tool/fetch frames below set it where they fire.
                            if smsg.text_delta or smsg.turn_ended or (
                                smsg.reasoning_delta and cfg.stream_reasoning
                            ):
                                last_visible = time.monotonic()

                            if smsg.user_message_appended_id:
                                _tlog(f"steer ack {smsg.user_message_appended_id[:8]}")
                                await put("steer_ack", smsg.user_message_appended_id)

                            if smsg.checkpoint:
                                last_checkpoint = smsg.checkpoint
                                last_upstream = time.monotonic()
                                _tlog(f"checkpoint {len(smsg.checkpoint)}B")
                                await put(
                                    "checkpoint",
                                    base64.b64encode(smsg.checkpoint).decode("ascii"),
                                )

                            if smsg.wants_context:
                                if not context_sent:
                                    # Startup handshake: the turn is actively setting up,
                                    # give the model the full visible-silence window from here.
                                    last_visible = time.monotonic()
                                    _tlog(f"context req #1 (exec_id={smsg.exec_id})")
                                else:
                                    _tlog(f"context req #2+ (exec_id={smsg.exec_id})")
                                context_sent = True
                                # Respond to EVERY context request — the server sends one
                                # per tool invocation (each with a distinct exec_id) and
                                # will not dispatch the exec until we reply.
                                ctx_prompt = (
                                    cfg.system_prompt if cfg.use_system_in_context else None
                                )
                                sender.queue(connect_frame(
                                    build_workspace_context_response(
                                        smsg.exec_id,
                                        smsg.exec_id_str,
                                        system_prompt=ctx_prompt,
                                        mcp_tools_body=mcp_tools_body,
                                        cursor_rules=cfg.cursor_rules or None,
                                    )
                                ))
                                await sender.pump()
                                _tlog(f"context resp sent (exec_id={smsg.exec_id})")

                            if smsg.kv_op:
                                # Blob I/O is active protocol work, not a stuck loop.
                                last_visible = time.monotonic()
                                _tlog(f"kv {smsg.kv_op} id={smsg.kv_id}")
                                if smsg.kv_op == "set":
                                    await kv_handler("set", smsg.kv_blob_id, smsg.kv_blob_data)
                                    resp = build_kv_response(smsg.kv_id, "set")
                                else:
                                    data_blob = await kv_handler("get", smsg.kv_blob_id, None)
                                    resp = build_kv_response(smsg.kv_id, "get", data_blob)
                                sender.queue(connect_frame(resp))
                                await sender.pump()
                                _tlog(f"kv {smsg.kv_op} done")

                            if smsg.query_kind:
                                # Web-search / fetch interaction approval.
                                # Treat an approved web query like an active Cursor tool:
                                # it runs server-side with no upstream frames until the
                                # results land, so use TOOL_STALL_SEC for the silence cap.
                                last_visible = time.monotonic()
                                _tlog(f"web query kind={smsg.query_kind} id={smsg.query_id}")
                                approved = await approval_handler(
                                    smsg.query_kind, str(smsg.query_id)
                                )
                                if approved:
                                    if not tracker:
                                        tracker.start(f"web_{smsg.query_kind}")
                                    tracker.mark_dispatched()
                                    sender.queue(connect_frame(
                                        build_interaction_approval(
                                            smsg.query_id, smsg.query_kind
                                        )
                                    ))
                                    await sender.pump()
                                    _tlog("web approval sent")

                            if smsg.tool_event:
                                phase, label = smsg.tool_event
                                detail = smsg.tool_detail or ""
                                call_id = getattr(smsg, "tool_event_call_id", "") or ""
                                _tlog(f"tool_event {phase}:{label} {detail!r}")
                                run_id = ""
                                if phase == "started":
                                    run_id = tracker.start(label, call_id)
                                    smsg.tool_run_id = run_id
                                elif phase == "completed":
                                    run_id = tracker.complete(label, call_id)
                                # JSON payload so the UI can show name + arg detail.
                                # Falls back gracefully: old "phase:name" still parses.
                                tool_payload = {
                                    "phase": phase, "name": label, "detail": detail,
                                }
                                if run_id:
                                    tool_payload["id"] = run_id
                                await put("tool", json.dumps(tool_payload))
                                produced_output = True
                                last_visible = time.monotonic()

                            smsg.cursor_tool_label = tracker.active_label

                            # Inline exec: server asked the client to run a tool.
                            exec_type = _exec_type_of(smsg)
                            if exec_type:
                                tracker.mark_dispatched()
                                exec_cid = _exec_tool_call_id(smsg)
                                if exec_cid:
                                    smsg.tool_run_id = tracker.bind_call(exec_cid)
                                if not getattr(smsg, "tool_run_id", ""):
                                    if not tracker.active_run_id:
                                        tracker.start(exec_type, exec_cid)
                                    smsg.tool_run_id = tracker.active_run_id
                                # Refresh UI detail from the exec payload when
                                # ToolCallStarted omitted args (common for MCP/shell).
                                exec_detail = detail_from_exec_server_msg(smsg)
                                if exec_detail:
                                    label = tracker.active_label or exec_type
                                    updated = {
                                        "phase": "updated",
                                        "name": label,
                                        "detail": exec_detail,
                                    }
                                    if getattr(smsg, "tool_run_id", ""):
                                        updated["id"] = smsg.tool_run_id
                                    await put("tool", json.dumps(updated))
                                last_visible = time.monotonic()
                                _tlog(f"exec start type={exec_type}")
                                _exec_t0 = time.monotonic()
                                try:
                                    resp_frames = await exec_handler(exec_type, smsg)
                                except Exception as exc:
                                    _tlog(f"exec_handler {exec_type} failed: {exc}")
                                    stamp_ui_output(
                                        smsg, f"{type(exc).__name__}: {exc}",
                                    )
                                    resp_frames = _reject_exec_frames(
                                        exec_type, smsg, f"{type(exc).__name__}: {exc}"[:300]
                                    )
                                finally:
                                    suspended_total += time.monotonic() - _exec_t0
                                _tlog(f"exec done  type={exec_type}")
                                ui_output = getattr(smsg, "ui_output", "") or ""
                                if ui_output:
                                    result_update = {
                                        "phase": "updated",
                                        "name": tracker.active_label or exec_type,
                                        "output": ui_output,
                                    }
                                    if exec_detail:
                                        result_update["detail"] = exec_detail
                                    if getattr(smsg, "tool_run_id", ""):
                                        result_update["id"] = smsg.tool_run_id
                                    await put("tool", json.dumps(result_update))
                                    produced_output = True
                                    last_visible = time.monotonic()
                                # A tool (incl. slow MCP/browser calls) runs with the
                                # read loop blocked and emits no upstream frames. Reset
                                # the total-silence timer so a legitimately slow tool
                                # isn't mistaken for a dead stream.
                                last_upstream = time.monotonic()
                                if DEBUG_FRAMES:
                                    _tlog(
                                        f"exec req eid={smsg.exec_id} "
                                        f"eid_str={smsg.exec_id_str!r} "
                                        f"-> {len(resp_frames)} resp frame(s)"
                                    )
                                    for _rf in resp_frames:
                                        _tlog(f"  RESP[{len(_rf)}B] {_decode_frame(_rf)}")
                                if not resp_frames and (smsg.exec_id or smsg.exec_id_str):
                                    field = (
                                        getattr(smsg, "get_mcp_tools_result_field", 0)
                                        or getattr(smsg, "unknown_exec_field", 0)
                                    )
                                    if field:
                                        _tlog(
                                            f"empty exec reply type={exec_type}; "
                                            f"error on field {field}"
                                        )
                                        resp_frames = [build_unknown_exec_error_response(
                                            smsg.exec_id, smsg.exec_id_str, field,
                                            f"unhandled exec type {exec_type}",
                                        )]
                                if resp_frames:
                                    for frame in resp_frames:
                                        sender.queue(connect_frame(frame))
                                    await sender.pump()
                                    produced_output = True
                                    last_visible = time.monotonic()
                            elif (
                                (smsg.exec_id or smsg.exec_id_str)
                                and getattr(smsg, "unknown_exec_field", 0)
                            ):
                                tracker.mark_dispatched()
                                field = smsg.unknown_exec_field
                                blob = (
                                    smsg.unknown_exec_fields[0][1]
                                    if smsg.unknown_exec_fields else b""
                                )
                                decoded = _decode_frame(blob)[:400] if blob else ""
                                _tlog(f"unhandled exec field={field} {decoded}")
                                resp_frames = [build_unknown_exec_error_response(
                                    smsg.exec_id, smsg.exec_id_str, field,
                                    f"unsupported exec field {field}",
                                )]
                                for frame in resp_frames:
                                    sender.queue(connect_frame(frame))
                                await sender.pump()
                                produced_output = True
                                last_visible = time.monotonic()
                                last_upstream = time.monotonic()

                            if smsg.reasoning_delta:
                                if not produced_output:
                                    _tlog("first reasoning_delta")
                                await put("reasoning", smsg.reasoning_delta)
                                # Reasoning is client-visible progress only when it's
                                # actually forwarded. When dropped it must NOT mark the
                                # turn as having produced output (last_visible gated
                                # above) -- otherwise a silent reasoning loop looks
                                # productive and never retries.
                                if cfg.stream_reasoning:
                                    produced_output = True

                            if smsg.text_delta:
                                if not produced_output:
                                    _tlog("first text_delta")
                                await put("text", smsg.text_delta)
                                produced_output = True

                            if smsg.turn_ended:
                                _tlog("turn_ended received")
                                turn_finished = True
                    elif isinstance(event, h2.events.WindowUpdated):
                        await sender.pump()
                    elif isinstance(event, h2.events.StreamEnded):
                        h2_done = True
                        extra = getattr(event, "additional_data", None)
                        await _close(conn, writer, stream_id)
                        if turn_finished:
                            await finish("clean", cause="stream_ended")
                        else:
                            await finish(
                                "dropped",
                                cause=f"stream_ended extra={extra!r}",
                            )
                        return
                    elif isinstance(event, h2.events.ConnectionTerminated):
                        h2_done = True
                        err_code = getattr(event, "error_code", None)
                        extra = getattr(event, "additional_data", None)
                        if turn_finished:
                            await finish("clean", cause="connection_terminated")
                        else:
                            await finish(
                                "dropped",
                                cause=(
                                    f"connection_terminated error_code={err_code} "
                                    f"additional_data={extra!r}"
                                ),
                            )
                        return
                    elif isinstance(event, h2.events.StreamReset):
                        err = f"HTTP/2 stream reset: {event.error_code}"
                        if cfg.conversation_state and not produced_output:
                            await put("native_fallback", err)
                            await finish("error", cause=f"native_state_rejected:{err}")
                            return
                        raise RuntimeError(err)
                await _flush(conn, writer)
                if turn_finished and h2_done:
                    await finish("clean", cause="turn_ended_and_h2_done")
                    return
            # Fell out of the read loop via break (turn_ended without a prompt EOS,
            # or upstream EOF). Always emit the terminator so the consumer finalizes.
            if turn_finished:
                await finish("clean", cause="read_loop_exit")
            else:
                await finish("dropped", cause="eof_before_turn_ended")
        except _StallRetry:
            _retry = True
            tracker.clear()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        if _retry:
            continue
        return

    # All attempts exhausted (only reached if the final attempt itself raised
    # _StallRetry, which it shouldn't since the last attempt finalizes instead).
    await finish("stall", cause="stall_retries_exhausted")


async def run_turn(
    messages: list[dict],
    model: str = "auto",
    *,
    run_config: RunConfig | None = None,
    exec_handler: ExecHandler | None = None,
    kv_handler: KvHandler | None = None,
    approval_handler: ApprovalHandler | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """
    Yield (kind, payload) chunks from Cursor's AgentService/Run RPC, answering
    every mid-stream exec / KV / approval request inline on the same H2 stream.

    Kinds: text, reasoning, tool, error, end_reason, checkpoint, native_fallback
    (end_reason is stall | timeout | clean | cancel | error | dropped).

    Injected handlers (all optional; sensible defaults if omitted):
      - exec_handler(exec_type, server_msg) -> list[frame_bytes]
            Default: serve grep/glob locally against ``run_config.workspace_root``,
            reject every other tool with a type-appropriate error frame.
      - kv_handler(op, blob_id, blob_data) -> blob_bytes | None
            Default: an in-memory blob store seeded with ``run_config.initial_blobs``;
            on completion it is written to ``run_config.session_out['blob_store']``.
      - approval_handler(kind, query_id) -> bool
            Default: auto-approve web_search / web_fetch.
    """
    cfg = run_config or RunConfig()

    default_store: dict[bytes, bytes] | None = None
    if exec_handler is None:
        exec_handler = _make_default_exec_handler(cfg.workspace_root)
    if kv_handler is None:
        kv_handler, default_store = _make_default_kv_handler(cfg.initial_blobs)
    if approval_handler is None:
        approval_handler = _default_approval_handler

    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(
        _h2_run_loop(
            messages, model, cfg.conversation_id, cfg,
            output_queue=queue,
            exec_handler=exec_handler,
            kv_handler=kv_handler,
            approval_handler=approval_handler,
        )
    )
    try:
        while True:
            # Poll with a short timeout so we detect task failure even if it
            # raises before putting ("_done", "") in the queue.
            get_coro = asyncio.ensure_future(queue.get())
            try:
                done, _ = await asyncio.wait(
                    [get_coro, task], return_when=asyncio.FIRST_COMPLETED
                )
            except Exception:
                get_coro.cancel()
                raise
            if task in done and get_coro not in done:
                # Task ended (exception or clean return) before putting _done.
                get_coro.cancel()
                exc = task.exception()
                if exc:
                    raise exc
                break
            kind, payload = await get_coro
            if kind == "_done":
                break
            yield (kind, payload)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Persist the default in-memory blob store for the next turn (parity with
        # the proxy's session_out hand-off). Injected kv_handlers own their state.
        if default_store is not None and cfg.session_out is not None:
            cfg.session_out["blob_store"] = default_store


class _Sender:
    """Flow-control-aware writer for one HTTP/2 stream.

    Large payloads (e.g. inline images) exceed the peer's 64KB flow-control
    window, so we buffer and drain only as much as the window allows, resuming
    whenever the server grants more via WINDOW_UPDATE.
    """

    def __init__(self, conn, writer, stream_id):
        self.conn = conn
        self.writer = writer
        self.stream_id = stream_id
        self.pending = bytearray()

    def queue(self, data: bytes):
        self.pending.extend(data)

    async def pump(self):
        while self.pending:
            window = self.conn.local_flow_control_window(self.stream_id)
            if window <= 0:
                break
            n = min(len(self.pending), window, self.conn.max_outbound_frame_size)
            self.conn.send_data(self.stream_id, bytes(self.pending[:n]), end_stream=False)
            del self.pending[:n]
        await _flush(self.conn, self.writer)


async def _flush(conn, writer):
    data = conn.data_to_send()
    if data:
        writer.write(data)
        await writer.drain()


async def _close(conn, writer, stream_id):
    try:
        conn.end_stream(stream_id)
        await _flush(conn, writer)
    except Exception:
        pass


def _trailer_error(fd: bytes) -> str | None:
    """End-of-stream frame body is JSON; surface any error message."""
    try:
        obj = json.loads(fd.decode("utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict) and obj.get("error"):
        err = obj["error"]
        if isinstance(err, dict):
            return err.get("message") or json.dumps(err)
        return str(err)
    return None


async def complete(
    messages: list[dict],
    model: str = "auto",
    conversation_id: str | None = None,
) -> str:
    """Collect just the assistant answer text (reasoning chunks are dropped)."""
    parts = []
    cfg = RunConfig(conversation_id=conversation_id) if conversation_id else None
    async for kind, text in run_turn(messages, model, run_config=cfg):
        if kind == "text":
            parts.append(text)
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Speech-to-text (voice input)
# ──────────────────────────────────────────────────────────────────────────────

async def transcribe(
    audio: bytes,
    mime_type: str = "audio/wav",
    language: str | None = None,
) -> str:
    """
    Transcribe audio via Cursor's aiserver.v1.AiService/TranscribeAudio (Connect
    unary). This is the same backend the IDE's mic button uses.

    TranscribeAudioRequest { f1: audio (bytes), f2: mime_type, f3: language? }
    TranscribeAudioResponse { f1: text, f2: transcription_time_ms }
    """
    token = load_auth()
    body = pb_bytes(1, audio) + pb_str(2, mime_type)
    if language:
        body += pb_str(3, language)

    async with httpx.AsyncClient(http2=True, timeout=120, verify=certifi.where()) as client:
        resp = await client.post(
            API2_BASE + TRANSCRIBE_PATH, content=body, headers=_aiservice_headers(token)
        )

    if resp.status_code != 200:
        # Connect unary errors come back as HTTP status + JSON {code, message}
        detail = resp.text[:300]
        try:
            obj = resp.json()
            detail = obj.get("message", detail) if isinstance(obj, dict) else detail
        except Exception:
            pass
        raise RuntimeError(f"TranscribeAudio failed ({resp.status_code}): {detail}")

    return _parse_transcribe_response(resp.content)


def _parse_transcribe_response(raw: bytes) -> str:
    """Extract f1 (text) from a serialized TranscribeAudioResponse."""
    pos = 0
    while (r := _read_field(raw, pos)) is not None:
        field_num, wire, val, pos = r
        if field_num == 1 and wire == 2 and isinstance(val, bytes):
            return val.decode("utf-8", "replace")
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Model catalogue (live)
# ──────────────────────────────────────────────────────────────────────────────

async def list_models() -> list[dict]:
    """
    Fetch the live usable-model catalogue from Cursor's
    aiserver.v1.AiService/AvailableModels (Connect unary) — the same list the
    IDE/CLI model picker shows.

    Returns dicts shaped like:
        {"name", "display", "default_on", "supports_agent", "supports_images"}
    """
    token = load_auth()
    # AvailableModelsRequest { f2, f5, f7 } — variants + parameter definitions (CLI)
    body = pb_int(2, 1) + pb_int(5, 1) + pb_int(7, 1)
    async with httpx.AsyncClient(http2=True, timeout=30, verify=certifi.where()) as client:
        resp = await client.post(
            API2_BASE + AVAILABLE_MODELS_PATH, content=body, headers=_aiservice_headers(token)
        )
    if resp.status_code != 200:
        detail = resp.text[:300]
        try:
            obj = resp.json()
            detail = obj.get("message", detail) if isinstance(obj, dict) else detail
        except Exception:
            pass
        raise RuntimeError(f"AvailableModels failed ({resp.status_code}): {detail}")
    return _parse_available_models(resp.content)


async def get_usable_models() -> list[dict]:
    """aiserver.v1.AiService/GetUsableModels — picker list (ModelDetails)."""
    token = load_auth()
    async with httpx.AsyncClient(http2=True, timeout=30, verify=certifi.where()) as client:
        resp = await client.post(
            API2_BASE + GET_USABLE_MODELS_PATH, content=b"", headers=_aiservice_headers(token)
        )
    if resp.status_code != 200:
        raise RuntimeError(f"GetUsableModels failed ({resp.status_code})")
    return _parse_usable_models(resp.content)


async def get_default_model_for_cli() -> dict | None:
    """aiserver.v1.AiService/GetDefaultModelForCli."""
    token = load_auth()
    async with httpx.AsyncClient(http2=True, timeout=30, verify=certifi.where()) as client:
        resp = await client.post(
            API2_BASE + GET_DEFAULT_MODEL_PATH, content=b"", headers=_aiservice_headers(token)
        )
    if resp.status_code != 200:
        return None
    pos = 0
    while (r := _read_field(resp.content, pos)) is not None:
        f, w, v, pos = r
        if f == 1 and w == 2:
            m = _parse_one_model(v)
            if m.get("name"):
                return m
    return None


def _parse_usable_models(raw: bytes) -> list[dict]:
    """GetUsableModelsResponse { f1: repeated ModelDetails }."""
    models: list[dict] = []
    pos = 0
    while (r := _read_field(raw, pos)) is not None:
        f, w, v, pos = r
        if f == 1 and w == 2:
            m = _parse_one_model(v)
            if m.get("name"):
                models.append(m)
    return models


def _parse_available_models(raw: bytes) -> list[dict]:
    """AvailableModelsResponse { f2: repeated AvailableModel } (f1 is deprecated)."""
    models: list[dict] = []
    pos = 0
    while (r := _read_field(raw, pos)) is not None:
        field_num, wire, val, pos = r
        if field_num == 2 and wire == 2 and isinstance(val, bytes):
            m = _parse_one_model(val)
            if m.get("name"):
                models.append(m)
    return models


def _parse_one_model(data: bytes) -> dict:
    """
    AvailableModelsResponse.AvailableModel:
      f1 name, f2 default_on, f5 supports_agent, f10 supports_images,
      f17 client_display_name.
    """
    m: dict = {}
    pos = 0
    while (r := _read_field(data, pos)) is not None:
        field_num, wire, val, pos = r
        if field_num == 1 and wire == 2 and isinstance(val, bytes):
            m["name"] = val.decode("utf-8", "replace")
        elif field_num == 2 and wire == 0:
            m["default_on"] = bool(val)
        elif field_num == 5 and wire == 0:
            m["supports_agent"] = bool(val)
        elif field_num == 10 and wire == 0:
            m["supports_images"] = bool(val)
        elif field_num == 17 and wire == 2 and isinstance(val, bytes):
            m["display"] = val.decode("utf-8", "replace")
    return m
