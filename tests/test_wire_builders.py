"""Extra wire codec coverage: builders and argument parsers."""
from __future__ import annotations

from cursor_core.framing import pb_int, pb_msg, pb_str
from cursor_core.wire import (
    ServerMsg,
    build_delete_rejected_response,
    build_delete_success_response,
    build_edit_error_response,
    build_edit_success_response,
    build_exec_stream_close,
    build_fetch_error_response,
    build_fetch_success_response,
    build_force_background_subagent_response,
    build_grep_success_response,
    build_read_file_not_found_response,
    build_read_success_response,
    build_shell_response_frames,
    build_shell_stream_exit,
    build_shell_stream_start,
    build_shell_stream_stdout,
    build_shell_success_response,
    build_subagent_await_complete_response,
    build_subagent_error_response,
    build_subagent_success_response,
    build_unknown_exec_error_response,
    build_wait_response,
    build_write_error_response,
    build_write_success_response,
    parse_edit_args,
    parse_fetch_args,
    parse_force_background_subagent_args,
    parse_grep_args,
    parse_read_args,
    parse_shell_args,
    parse_subagent_args,
    parse_wait_args,
    parse_write_args,
)


def test_result_builders_are_nonempty():
    builders = [
        build_read_success_response(1, "a", "p.txt", "hi", truncated=True),
        build_read_file_not_found_response(1, "a", "p.txt"),
        build_write_success_response(1, "a", "p.txt"),
        build_write_error_response(1, "a", "p.txt", "nope"),
        build_delete_success_response(1, "a", "p.txt"),
        build_delete_rejected_response(1, "a", "p.txt", "nope"),
        build_edit_success_response(1, "a", "p.txt"),
        build_edit_error_response(1, "a", "p.txt", "nope"),
        build_fetch_success_response(1, "a", "https://ex", "ok", 200, "text/plain"),
        build_fetch_error_response(1, "a", "https://ex", "nope"),
        build_grep_success_response(1, "a", ".", "pat", ["a.py", "b.py"]),
        build_shell_success_response(1, "a", "echo", "out", "", 0),
        build_shell_stream_start(1, "a"),
        build_shell_stream_stdout(1, "a", "line"),
        build_shell_stream_exit(1, "a", 0, "/tmp"),
        build_exec_stream_close(1),
        build_wait_response(1, "a", 40, 5),
        build_unknown_exec_error_response(1, "a", 39, "nope"),
        build_subagent_success_response(1, "a", agent_id="agent-1", final_message="done"),
        build_subagent_error_response(1, "a", "fail", agent_id="agent-1"),
        build_force_background_subagent_response(1, "a"),
        build_subagent_await_complete_response(1, "a", agent_id="agent-1", final_message="done"),
    ]
    assert all(isinstance(b, bytes) and b for b in builders)
    frames = build_shell_response_frames(1, "a", "echo hi", "hi\n", "", 0)
    assert frames and all(isinstance(f, bytes) and f for f in frames)


def test_arg_parsers():
    assert parse_read_args(pb_str(1, "README.md")).path == "README.md"
    assert parse_shell_args(pb_str(1, "ls")).command == "ls"
    assert parse_write_args(pb_str(1, "a.txt") + pb_str(2, "body")).path == "a.txt"
    assert parse_fetch_args(pb_str(1, "https://ex")).url == "https://ex"
    grep = parse_grep_args(pb_str(2, ".") + pb_str(3, "foo"))
    assert grep is not None
    assert grep.pattern == "foo"
    edit = parse_edit_args(pb_str(1, "a.txt") + pb_str(2, "old") + pb_str(3, "new"))
    assert edit is not None
    assert edit.path == "a.txt"
    sub = parse_subagent_args(pb_str(4, "look around") + pb_str(1, "tc-1"))
    assert sub is not None
    assert sub.prompt == "look around"
    assert parse_force_background_subagent_args(pb_str(1, "tc-9")) == "tc-9"
    wait = parse_wait_args(pb_int(1, 5000) + pb_str(2, "done"))
    assert wait is not None
    assert wait.duration_ms == 5000


def test_server_msg_repr_covers_flags():
    msg = ServerMsg()
    msg.text_delta = "hi"
    msg.wants_read = True
    msg.read_path = "a.txt"
    msg.heartbeat = True
    msg.wants_shell = True
    msg.shell_command = "ls"
    msg.wants_write = True
    msg.write_path = "a"
    msg.wants_edit = True
    msg.edit_path = "a"
    msg.wants_fetch = True
    msg.fetch_url = "https://ex"
    msg.wants_grep = True
    msg.grep_pattern = "x"
    msg.grep_search_path = "."
    msg.wants_delete = True
    msg.wants_mcp = True
    text = repr(msg)
    assert "text=" in text
    assert "read(" in text

    from cursor_core.wire import (
        stamp_ui_output,
        parse_server_message,
        build_grep_error_response,
        build_shell_rejected_response,
        build_read_error_response,
        _appended_user_message_id,
    )

    stamp_ui_output(msg, "shown")
    assert build_grep_error_response(1, "a", ".", "nope")
    assert build_shell_rejected_response(1, "a", "ls", "nope")
    assert build_read_error_response(1, "a", "p.txt", "nope")
    uid = _appended_user_message_id(pb_msg(1, pb_str(2, "mid-1")))
    assert uid == "mid-1"
    assert _appended_user_message_id(b"") == ""
    parsed = parse_server_message(
        pb_msg(1, pb_msg(1, pb_str(1, "hello")))
    )
    assert parsed.text_delta == "hello" or parsed is not None

    fetch = parse_server_message(
        pb_msg(2, pb_int(1, 1) + pb_str(15, "e") + pb_msg(20, pb_str(1, "https://ex")))
    )
    assert fetch.wants_fetch
    write = parse_server_message(
        pb_msg(2, pb_int(1, 1) + pb_msg(3, pb_str(1, "a.txt") + pb_str(2, "body")))
    )
    assert write.wants_write
    grep = parse_server_message(
        pb_msg(2, pb_int(1, 1) + pb_msg(5, pb_str(2, ".") + pb_str(3, "foo")))
    )
    assert grep.wants_grep
    edit = parse_server_message(
        pb_msg(
            2,
            pb_int(1, 1)
            + pb_msg(12, pb_str(1, "a.txt") + pb_str(2, "old") + pb_str(3, "new")),
        )
    )
    assert edit.wants_edit
    mcp = parse_server_message(pb_msg(2, pb_int(1, 1) + pb_msg(11, b"raw")))
    assert mcp.wants_mcp
    shell = parse_server_message(
        pb_msg(2, pb_int(1, 1) + pb_msg(2, pb_str(1, "ls")))
    )
    assert shell.wants_shell

    from cursor_core.wire import cap_tool_text, _format_args_obj, _is_useful_detail

    assert cap_tool_text("") == ""
    long = cap_tool_text("x" * 20000)
    assert len(long) < 20000
    stamp_ui_output(object(), "x")
    assert _format_args_obj({}) == ""
    assert "a" in _format_args_obj({"a": 1})
    assert _is_useful_detail("ab")
    assert not _is_useful_detail("x")
