"""ToolCall detail extraction for the chat tool cards."""
from __future__ import annotations

from cursor_core.framing import json_to_value, pb_msg, pb_str
from cursor_core.wire import (
    ServerMsg,
    _DETAIL_MAX,
    _OUTPUT_MAX,
    _extract_tool_detail,
    _tool_call_variant,
    cap_tool_text,
    detail_from_exec_server_msg,
)


def test_nested_web_search_query():
    args = pb_str(1, "WireGuard Android embedding API")
    variant = pb_msg(1, args) + pb_str(2, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    name, detail = _tool_call_variant(pb_msg(18, variant))
    assert name == "web_search"
    assert detail == "WireGuard Android embedding API"


def test_nested_read_path_not_corrupted_by_length_prefix():
    path = "/opt/src/workspace/examples/bitline-azure-secure-config-checklist.md"
    assert len(path) == 0x44  # previously became a leading 'D'
    variant = pb_msg(1, pb_str(1, path))
    name, detail = _tool_call_variant(pb_msg(8, variant))
    assert name == "read"
    assert detail == path
    assert not detail.startswith("D/")


def test_shell_prefers_command_over_description():
    # description (f3) first in encode order; command still preferred via parser
    args = pb_str(3, "Count Win10 editions") + pb_str(1, "python3 -c 'print(1)'")
    name, detail = _tool_call_variant(pb_msg(1, pb_msg(1, args)))
    assert name == "shell"
    assert detail == "python3 -c 'print(1)'"


def test_grep_pattern_and_path():
    args = pb_str(2, "workspace") + pb_str(3, "WireGuard")
    name, detail = _tool_call_variant(pb_msg(5, pb_msg(1, args)))
    assert name == "grep"
    assert "WireGuard" in detail
    assert "workspace" in detail
    assert "pattern" in detail
    assert "path" in detail


def test_edit_includes_old_and_new():
    args = pb_str(1, "a.txt") + pb_str(2, "old text") + pb_str(3, "new text")
    name, detail = _tool_call_variant(pb_msg(12, pb_msg(1, args)))
    assert name == "edit"
    assert "a.txt" in detail
    assert "old text" in detail
    assert "new text" in detail


def test_mcp_args_json_includes_title():
    entry = pb_str(1, "title") + pb_msg(2, json_to_value("WireGuard on Android"))
    mcp_args = (
        pb_str(1, "set_chat_title")
        + pb_msg(2, entry)
        + pb_str(5, "set_chat_title")
    )
    name, detail = _tool_call_variant(pb_msg(15, pb_msg(1, mcp_args)))
    assert name == "mcp:set_chat_title"
    assert "title" in detail
    assert "WireGuard on Android" in detail


def test_mcp_args_json_includes_url():
    entry = pb_str(1, "url") + pb_msg(2, json_to_value("https://example.com/docs"))
    mcp_args = pb_str(5, "browser_navigate") + pb_msg(2, entry)
    name, detail = _tool_call_variant(pb_msg(15, pb_msg(1, mcp_args)))
    assert name == "mcp:browser_navigate"
    assert "url" in detail
    assert "https://example.com/docs" in detail


def test_mcp_args_json_includes_all_keys():
    q = pb_str(1, "query") + pb_msg(2, json_to_value("clause 10.2"))
    space = pb_str(1, "space_id") + pb_msg(2, json_to_value("iso-space"))
    mcp_args = pb_str(5, "docs_search") + pb_msg(2, q) + pb_msg(2, space)
    name, detail = _tool_call_variant(pb_msg(15, pb_msg(1, mcp_args)))
    assert name == "mcp:docs_search"
    assert "clause 10.2" in detail
    assert "space_id" in detail
    assert "iso-space" in detail


def test_unknown_tool_field_labeled_with_number():
    name, detail = _tool_call_variant(
        pb_msg(40, pb_msg(1, pb_str(1, "some future query")))
    )
    assert name == "tool#40"
    assert detail == "some future query"


def test_get_mcp_tools_field_44_name_and_query():
    args = pb_str(3, "email_(get|search|list)") + pb_str(4, "call-44")
    name, detail = _tool_call_variant(pb_msg(44, pb_msg(1, args)))
    assert name == "get_mcp_tools"
    assert "email_(get|search|list)" in detail


def test_extract_skips_uuid_only_payload():
    assert _extract_tool_detail(pb_str(1, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")) == ""


def test_detail_from_exec_shell_and_mcp():
    shell = ServerMsg()
    shell.wants_shell = True
    shell.shell_command = "ls -la src/"
    assert detail_from_exec_server_msg(shell) == "ls -la src/"

    entry = pb_str(1, "query") + pb_msg(2, json_to_value("memory key"))
    mcp_args = pb_str(5, "memory_get") + pb_msg(2, entry)
    mcp = ServerMsg()
    mcp.wants_mcp = True
    mcp.mcp_parsed = mcp_args
    detail = detail_from_exec_server_msg(mcp)
    assert "query" in detail
    assert "memory key" in detail


def test_cap_tool_text_marks_truncation():
    assert cap_tool_text("short") == "short"
    blob = "x" * (_OUTPUT_MAX + 50)
    out = cap_tool_text(blob)
    assert out.endswith("… truncated")
    assert len(out) <= _OUTPUT_MAX
    assert cap_tool_text("y" * 100, limit=_DETAIL_MAX) == "y" * 100
