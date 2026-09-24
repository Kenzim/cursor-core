"""Public client wrappers around auth, models, and AgentService/Run."""
from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

from .auth import AuthError, load_auth, resolve_access_token
from .engine import (
    ApprovalHandler,
    Checkpoint,
    ExecHandler,
    KvHandler,
    RunConfig,
    Timeouts,
    run_turn,
    transcribe as _transcribe,
)
from .models import get_catalog


class CursorClient:
    """Host-credential Cursor API helper. Safe to construct without a network call."""

    def __init__(self, access_token: str | None = None) -> None:
        self.access_token = access_token

    def token(self) -> str:
        return resolve_access_token(self.access_token)

    def load_auth(self) -> str:
        return self.access_token or load_auth()

    async def list_models(self) -> dict[str, Any]:
        catalog, default_id = await get_catalog(access_token=self.token())
        return {
            "default_id": default_id,
            "models": [
                {"id": m.base_id, "name": m.display_name}
                for m in catalog
            ],
        }

    async def transcribe(
        self,
        audio: bytes,
        mime_type: str = "audio/wav",
        language: str | None = None,
    ) -> str:
        return await _transcribe(audio, mime_type=mime_type, language=language)

    def session(self, checkpoint: Checkpoint | None = None) -> "AgentSession":
        return AgentSession(self, checkpoint=checkpoint)


class AgentSession:
    """One Cursor conversation. Persist ``checkpoint`` across Runs like the IDE."""

    def __init__(
        self,
        client: CursorClient,
        checkpoint: Checkpoint | None = None,
    ) -> None:
        self.client = client
        self.checkpoint = checkpoint or Checkpoint()

    async def run(
        self,
        messages: list[dict],
        model: str = "auto",
        *,
        run_config: RunConfig | None = None,
        exec_handler: ExecHandler | None = None,
        kv_handler: KvHandler | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        cfg = run_config or RunConfig()
        if not cfg.conversation_id:
            cfg.conversation_id = self.checkpoint.conversation_id or None
        if not cfg.conversation_group_id:
            cfg.conversation_group_id = self.checkpoint.group_id or None
        if not cfg.conversation_state:
            cfg.conversation_state = self.checkpoint.state
        if not cfg.initial_blobs and self.checkpoint.blobs:
            cfg.initial_blobs = dict(self.checkpoint.blobs)
        if cfg.access_token is None:
            cfg.access_token = self.client.access_token
        if cfg.session_out is None:
            cfg.session_out = {}

        async def _kv(
            op: str, blob_id: bytes | None, blob_data: bytes | None
        ) -> bytes | None:
            if kv_handler is not None:
                result = await kv_handler(op, blob_id, blob_data)
            else:
                result = None
                if op == "set" and blob_id is not None:
                    self.checkpoint.blobs[blob_id] = blob_data or b""
                elif op == "get" and blob_id is not None:
                    result = self.checkpoint.blobs.get(blob_id)
            if op == "set" and blob_id is not None and kv_handler is not None:
                self.checkpoint.blobs[blob_id] = blob_data or b""
            return result

        async for kind, payload in run_turn(
            messages,
            model,
            run_config=cfg,
            exec_handler=exec_handler,
            kv_handler=_kv if kv_handler is None else kv_handler,
            approval_handler=approval_handler,
        ):
            if kind == "checkpoint" and payload:
                try:
                    self.checkpoint.state = base64.b64decode(payload)
                except Exception:
                    pass
            if cfg.conversation_id:
                self.checkpoint.conversation_id = cfg.conversation_id
            if cfg.conversation_group_id:
                self.checkpoint.group_id = cfg.conversation_group_id
            yield kind, payload


__all__ = [
    "AgentSession",
    "AuthError",
    "Checkpoint",
    "CursorClient",
    "RunConfig",
    "Timeouts",
]
