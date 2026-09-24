"""
``cursor_core`` — a clean, app-agnostic library for Cursor's
``agent.v1.AgentService/Run`` HTTP/2 bidirectional streaming protocol.

Layers (bottom to top):
  - :mod:`cursor_core.framing` — Connect-RPC framing + schema-free protobuf codec
  - :mod:`cursor_core.wire`     — server-message parsing + exec arg/result codec
  - :mod:`cursor_core.auth`     — credentials + endpoint constants
  - :mod:`cursor_core.models`   — live model catalogue + parameter resolution
  - :mod:`cursor_core.engine`   — the inline-exec H2 run loop (:func:`run_turn`)
  - :mod:`cursor_core.client`   — :class:`CursorClient` / :class:`AgentSession`

Mid-stream exec, KV, wait, and checkpoints are answered on the same stream.
Stall and turn clocks pause while a local tool or wait is in flight. A stream
that closes before ``turn_ended`` is ``end_reason=dropped``.
"""
from .auth import AuthError, load_auth
from .client import AgentSession, CursorClient
from .engine import Checkpoint, RunConfig, Timeouts, execute_wait, run_turn

__all__ = [
    "AgentSession",
    "AuthError",
    "Checkpoint",
    "CursorClient",
    "RunConfig",
    "Timeouts",
    "execute_wait",
    "load_auth",
    "run_turn",
]
