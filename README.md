# cursor-core

Unofficial Python client for Cursor’s **`agent.v1.AgentService/Run`**
endpoint. Import name is `cursor_core`.

The IDE chat path is Connect-RPC bidirectional streaming over HTTP/2.
`httpx` cannot do true full-duplex (it finishes the request body before
reading the response), so this library drives the H2 connection with
`h2` and answers mid-stream exec, KV, wait, and checkpoints on the same
stream.

Not affiliated with Cursor. The protocol is undocumented and can change
when the desktop app or CLI updates. MIT licensed.

## Install

Python 3.11+. Runtime: `h2`, `httpx`, `certifi`, `protobuf>=5.29,<6`.

```bash
pip install -e ".[dev]"
```

There is no PyPI release yet. Point pip at this repo (or a git URL).

## Authentication

`CursorClient()` with no token uses `load_auth()`:

1. First file that has `accessToken` / `access_token` (top-level or
   `authInfo`): `~/.config/cursor/auth.json`, `~/.cursor/auth.json`,
   `~/.cursor/cli-config.json`
2. else pass `CursorClient(access_token=...)` or `RunConfig.access_token`

The token is re-read from disk on each request so an `agent login` or
desktop refresh is picked up without restarting. Expired JWTs raise
`AuthError`. This library does not implement OAuth refresh.

Never commit tokens or `auth.json`.

## How it works

Layers, bottom to top:

| Module | Role |
|--------|------|
| `cursor_core.framing` | Connect-RPC 5-byte frames + schema-free protobuf |
| `cursor_core.wire` | `AgentServerMessage` parse; exec arg/result codecs |
| `cursor_core.auth` | endpoints + credential load |
| `cursor_core.models` | live catalogue + `gpt-5.5/medium/272k`-style ids |
| `cursor_core.engine` | `run_turn` — the H2 loop |
| `cursor_core.client` | `CursorClient` / `AgentSession` |

A Run looks like this:

1. Client sends `AgentClientMessage{f1: AgentRunRequest}`
2. Server heartbeats / acks; may ask for **request context**
3. Server asks the client to run a tool (`ExecServerMessage`)
4. Client replies on the **same** stream with `ExecClientMessage`
5. Text / reasoning deltas; more exec / KV / wait as needed
6. `TurnEnded`, then a Connect trailer (flag `0x02`)

If you persist `Checkpoint.state` (`conversation_state` blob) plus KV
blobs, the next Run can send only the new user message. If Cursor
rejects the checkpoint, retry that turn with the full history.

A stream that closes before `turn_ended` is `end_reason=dropped`. Stall
and turn clocks **pause** while a local tool or `wait` is in flight.

## Quickstart

```python
import asyncio
from cursor_core import CursorClient, RunConfig

async def main():
    client = CursorClient()
    models = await client.list_models()
    print(models["default_id"])

    session = client.session()
    async for kind, payload in session.run(
        [{"role": "user", "content": "What is 2+2?"}],
        model="auto",
        run_config=RunConfig(chat_only=True),
    ):
        if kind == "text":
            print(payload, end="", flush=True)

asyncio.run(main())
```

`chat_only=True` tells the model not to poke the local filesystem.
Without it, Cursor will request `read` / `grep` / `shell` / … and the
engine needs an `exec_handler` (or you accept the built-in default,
which serves grep/glob against `workspace_root` and rejects the rest).

## Hosting a turn

Prefer `AgentSession` if you want conversation continuity. Prefer
`run_turn` if you already own IDs, rules, and MCP blobs.

```python
from cursor_core import AgentSession, Checkpoint, CursorClient, RunConfig, run_turn

client = CursorClient()
session = AgentSession(client, checkpoint=Checkpoint())

async def exec_handler(exec_type, server_msg) -> list[bytes]:
    # exec_type is "read" / "shell" / "write" / "grep" / "edit" / "mcp" / …
    # Return protobuf bodies for ExecClientMessage (see cursor_core.wire).
    from cursor_core.wire import build_read_error_response
    return [build_read_error_response(
        server_msg.exec_id, server_msg.exec_id_str,
        getattr(server_msg, "read_path", ""), "not implemented",
    )]

async for kind, payload in session.run(
    [{"role": "user", "content": "open README.md and summarise it"}],
    exec_handler=exec_handler,
):
    print(kind, payload[:80])

# session.checkpoint.state / .conversation_id / .blobs → persist, reuse
```

Yielded `(kind, payload)`:

| kind | payload |
|------|---------|
| `text` | token |
| `reasoning` | thinking token (if `stream_reasoning`) |
| `tool` | UI label for a tool card |
| `checkpoint` | base64 `conversation_state` blob |
| `end_reason` | `clean` / `stall` / `timeout` / `cancel` / `error` / `dropped` |
| `error` | message |
| `native_fallback` | Cursor rejected the checkpoint; retry with history |

Injected callbacks (all optional):

- **`exec_handler(exec_type, server_msg) -> list[bytes]`** — tool
  results, framed as protobuf bodies. Default: local grep/glob, reject
  everything else.
- **`kv_handler(op, blob_id, blob_data) -> bytes | None`** — Cursor’s
  blob store (`set` / `get`). Default: in-memory, copied to
  `RunConfig.session_out["blob_store"]` at the end.
- **`approval_handler(kind, query_id) -> bool`** — web search/fetch
  gate. Default: approve.

`RunConfig.steer_queue` is `(message_id, text)` follow-ups applied at
the next tool boundary (`AgentClientMessage` field 4).

## Timeouts

Env vars set defaults; `RunConfig.timeouts` overrides per turn.

| Var | Default | Meaning |
|-----|---------|---------|
| `CURSOR_TURN_TIMEOUT_SEC` | 1200 | hard ceiling; 0 = off |
| `CURSOR_STALL_SEC` | 60 | silence including heartbeats |
| `CURSOR_REASONING_STALL_SEC` | 60 | no *visible* output while thinking |
| `CURSOR_TOOL_STALL_SEC` | 90 | silence cap while a Cursor-side tool runs |
| `CURSOR_STALL_RETRIES` | 1 | reopen after a stall |
| `CURSOR_WAIT_CAP_MS` | 30 min | native await / wait exec cap |
| `CURSOR_WORKSPACE_ROOT` | `.` | default grep/glob root |
| `CURSOR_DEBUG_WIRE` | off | log every inbound field to stderr |

## Tests

Offline only. They patch the H2 loop or parse recorded frames — no
Cursor account required.

```bash
pip install -e ".[dev]"
pytest
```

Redacted wire fixtures live in `tests/fixtures/cursor_frames/`.

Forgejo Actions: `sonarqube.yml` runs coverage, scanner, and the
quality gate. Repo secrets: `SONAR_HOST_URL`, `SONAR_TOKEN`. Project
key `kenzim_cursor-core`. There is no separate build/lint workflow.

## Layout

| Path | What |
|------|------|
| `src/cursor_core/` | public package |
| `tests/` | framing, wire, engine (mocked H2) |
| `tests/fixtures/cursor_frames/` | redacted Run frames |
| `.forgejo/workflows/sonarqube.yml` | SonarQube on Forgejo Actions |
