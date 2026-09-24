"""
Cursor model catalogue for ``cursor_core``: variants, parameters
(reasoning/context/fast), and parameter resolution + defaults.

Fetches the live model list from ``aiserver.v1.AiService/AvailableModels`` (the
same catalogue the IDE/CLI picker shows), expands every parameter-option combo
into a flat, selectable id (e.g. ``gpt-5.5/medium/272k``), and maps a requested
model id back to the ``(cursor_sent_model_id, logical_base, parameters)`` triple
the streaming engine puts on the wire.

This is a near-verbatim extraction of the original ``model_catalog.py``; only the
imports were repointed at ``cursor_core``'s framing/wire/auth modules.
"""
from __future__ import annotations

import hashlib
import itertools
import re
import time
from dataclasses import dataclass, field

from .auth import (
    API2_BASE,
    AVAILABLE_MODELS_PATH,
    GET_DEFAULT_MODEL_PATH,
    GET_USABLE_MODELS_PATH,
    _aiservice_headers,
    resolve_access_token,
)
from .framing import pb_int
from .wire import _read_field

__all__ = [
    "GET_DEFAULT_MODEL_PATH",
    "GET_USABLE_MODELS_PATH",
    "ModelParameterError",
    "ParamOption",
    "ParamDef",
    "ModelVariant",
    "CatalogModel",
    "SelectableModel",
    "openai_param_mapping",
    "catalog_model_for",
    "resolve_model",
    "get_catalog",
    "expand_selectable_models",
    "all_model_ids",
    "selectable_model_for",
    "cursor_sent_model_id",
]

REASONING_PARAM_IDS = ("reasoning", "effort")
CONTEXT_PARAM_IDS = ("context",)
FAST_PARAM_IDS = ("fast",)

_CACHE_TTL = 300.0
_cache: dict[str, dict] = {}
_selectable_cache: dict = {"key": None, "models": None}


def _token_cache_key(token: str | None) -> str:
    raw = (token or "").strip()
    if not raw:
        return "host"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ModelParameterError(ValueError):
    """Requested model parameters are not valid for this model (per API catalogue)."""


@dataclass
class ParamOption:
    value: str
    label: str


@dataclass
class ParamDef:
    id: str
    name: str
    description: str = ""
    options: list[ParamOption] = field(default_factory=list)
    default_index: int | None = None


@dataclass
class ModelVariant:
    id: str
    base_model_id: str
    parameters: dict[str, str]
    display_name: str = ""


@dataclass
class CatalogModel:
    base_id: str
    display_name: str
    supports_agent: bool
    supports_images: bool
    parameter_defs: list[ParamDef] = field(default_factory=list)
    variants: list[ModelVariant] = field(default_factory=list)


def _leaf_option(blob: bytes) -> tuple[str, str] | None:
    """Parse one ParamOption leaf (value + optional label)."""
    val, label = "", ""
    pos = 0
    while (r := _read_field(blob, pos)) is not None:
        f, w, v, pos = r
        if f == 1 and w == 2 and isinstance(v, bytes):
            ipos = 0
            inner_val = ""
            while (r2 := _read_field(v, ipos)) is not None:
                ff, ww, vv, ipos = r2
                if ff == 1 and ww == 2 and isinstance(vv, bytes):
                    inner_val = vv.decode("utf-8", "replace")
            if inner_val:
                val = inner_val
            else:
                val = v.decode("utf-8", "replace")
        elif f == 2 and w == 2 and isinstance(v, bytes):
            label = v.decode("utf-8", "replace")
    if val:
        return val, label or val
    return None


def _parse_options_container(blob: bytes) -> list[ParamOption]:
    """Parse field 4 option list from ModelParameterDefinition."""
    options: list[ParamOption] = []
    seen: set[str] = set()

    def add(val: str, label: str) -> None:
        if val not in seen:
            seen.add(val)
            options.append(ParamOption(val, label))

    def parse_level(b: bytes) -> None:
        f1_subs: list[bytes] = []
        pos = 0
        while (r := _read_field(b, pos)) is not None:
            f, w, v, pos = r
            if f == 1 and w == 2 and isinstance(v, bytes):
                f1_subs.append(v)
            elif f == 2 and w == 2 and isinstance(v, bytes):
                parse_level(v)
        for sub in f1_subs:
            inner_f1s: list[bytes] = []
            ipos = 0
            while (r2 := _read_field(sub, ipos)) is not None:
                ff, ww, vv, ipos = r2
                if ff == 1 and ww == 2 and isinstance(vv, bytes):
                    inner_f1s.append(vv)
            if len(inner_f1s) > 1:
                for inner in inner_f1s:
                    pair = _leaf_option(inner)
                    if pair:
                        add(*pair)
            else:
                pair = _leaf_option(sub)
                if pair:
                    add(*pair)

    parse_level(blob)
    return options


def _parse_param_def(blob: bytes) -> ParamDef | None:
    pid, pname, desc = "", "", ""
    options_blob: bytes | None = None
    default_index: int | None = None
    pos = 0
    while (r := _read_field(blob, pos)) is not None:
        f, w, v, pos = r
        if f == 1 and w == 2 and isinstance(v, bytes):
            pid = v.decode("utf-8", "replace")
        elif f == 2 and w == 2 and isinstance(v, bytes):
            pname = v.decode("utf-8", "replace")
        elif f == 3 and w == 2 and isinstance(v, bytes):
            desc = v.decode("utf-8", "replace")
        elif f == 4 and w == 2 and isinstance(v, bytes):
            options_blob = v
        elif f == 5 and w == 0:
            default_index = int(v)
    if not pid:
        return None
    options = _parse_options_container(options_blob) if options_blob else []
    return ParamDef(pid, pname or pid, desc, options, default_index)


def _parse_variant(blob: bytes) -> ModelVariant | None:
    params: dict[str, str] = {}
    vid, display = "", ""
    pos = 0
    while (r := _read_field(blob, pos)) is not None:
        f, w, v, pos = r
        if f == 1 and w == 2 and isinstance(v, bytes):
            pid, pval = "", ""
            ipos = 0
            while (r3 := _read_field(v, ipos)) is not None:
                ff, ww, vv, ipos = r3
                if ff == 1 and ww == 2 and isinstance(vv, bytes):
                    pid = vv.decode("utf-8", "replace")
                elif ff == 2 and ww == 2 and isinstance(vv, bytes):
                    pval = vv.decode("utf-8", "replace")
            if pid:
                params[pid] = pval
        elif f == 11 and w == 2 and isinstance(v, bytes):
            display = v.decode("utf-8", "replace")
        elif f in (10, 11, 18) and w == 2 and isinstance(v, bytes):
            if not vid:
                vid = v.decode("utf-8", "replace")
    if not vid and not params:
        return None
    return ModelVariant(vid, "", params, display)


def _parse_catalog_model(blob: bytes) -> CatalogModel | None:
    base_id, display = "", ""
    supports_agent = supports_images = False
    param_defs: list[ParamDef] = []
    variants: list[ModelVariant] = []
    variant_ids: list[str] = []

    pos = 0
    while (r := _read_field(blob, pos)) is not None:
        f, w, v, pos = r
        if f == 1 and w == 2:
            base_id = v.decode("utf-8", "replace")
        elif f == 5 and w == 0:
            supports_agent = bool(v)
        elif f == 10 and w == 0:
            supports_images = bool(v)
        elif f == 17 and w == 2:
            display = v.decode("utf-8", "replace")
        elif f == 29 and w == 2:
            pd = _parse_param_def(v)
            if pd:
                param_defs.append(pd)
        elif f == 30 and w == 2:
            var = _parse_variant(v)
            if var:
                variants.append(var)
        elif f == 36 and w == 2:
            variant_ids.append(v.decode("utf-8", "replace"))

    if not base_id:
        return None

    for var in variants:
        var.base_model_id = base_id

    for i, vid in enumerate(variant_ids):
        if i < len(variants):
            if not variants[i].id:
                variants[i].id = vid
            continue
        variants.append(ModelVariant(vid, base_id, {}, ""))

    return CatalogModel(
        base_id=base_id,
        display_name=display or base_id,
        supports_agent=supports_agent,
        supports_images=supports_images,
        parameter_defs=param_defs,
        variants=variants,
    )


def _param_def(cm: CatalogModel, ids: tuple[str, ...]) -> ParamDef | None:
    for pid in ids:
        for p in cm.parameter_defs:
            if p.id == pid:
                return p
    return None


def _allowed_values(pdef: ParamDef | None) -> list[str]:
    if not pdef or not pdef.options:
        return []
    return [o.value for o in pdef.options]


def _match_option(requested: str, pdef: ParamDef | None) -> str | None:
    if not pdef or not pdef.options:
        return requested or None
    raw = requested.strip()
    low = raw.lower().replace("_", "-")
    aliases = {
        "extra-high": "xhigh",
        "extra_high": "xhigh",
        "x-high": "xhigh",
        "xhigh": "xhigh",
        "minimal": "none",
        "default": pdef.options[pdef.default_index].value
        if pdef.default_index is not None and 0 <= pdef.default_index < len(pdef.options)
        else None,
    }
    candidate = aliases.get(low, low)
    for opt in pdef.options:
        if opt.value.lower() == candidate or opt.value.lower() == low:
            return opt.value
        if opt.label.lower() == low:
            return opt.value
    return None


def openai_param_mapping(cm: CatalogModel) -> dict[str, str | None]:
    """Map OpenAI-style request fields to this model's Cursor parameter ids."""
    reasoning = _param_def(cm, REASONING_PARAM_IDS)
    context = _param_def(cm, CONTEXT_PARAM_IDS)
    fast = _param_def(cm, FAST_PARAM_IDS)
    return {
        "reasoning_effort": reasoning.id if reasoning else None,
        "cursor_context_window": context.id if context else None,
        "cursor_fast": fast.id if fast else None,
    }


def catalog_model_for(catalog: list[CatalogModel], model: str) -> CatalogModel | None:
    for cm in catalog:
        if model == cm.base_id:
            return cm
        for var in cm.variants:
            if model == var.id:
                return cm
    for cm in catalog:
        if model.startswith(cm.base_id + "-") or model.startswith(cm.base_id):
            return cm
    return None


def _build_want_params(
    cm: CatalogModel,
    *,
    reasoning_effort: str | None,
    context_window: str | None,
    fast: bool | str | None,
    strict: bool,
) -> dict[str, str]:
    want: dict[str, str] = {}
    errors: list[str] = []

    reasoning_def = _param_def(cm, REASONING_PARAM_IDS)
    if reasoning_effort is not None and reasoning_effort.strip():
        if not reasoning_def:
            if strict:
                errors.append(f"model {cm.base_id!r} does not support reasoning_effort")
        else:
            val = _match_option(reasoning_effort, reasoning_def)
            if val is None:
                errors.append(
                    f"invalid reasoning_effort {reasoning_effort!r} for {cm.base_id}; "
                    f"allowed: {_allowed_values(reasoning_def)}"
                )
            else:
                want[reasoning_def.id] = val

    context_def = _param_def(cm, CONTEXT_PARAM_IDS)
    if context_window is not None and str(context_window).strip():
        if not context_def:
            if strict:
                errors.append(f"model {cm.base_id!r} does not support cursor_context_window")
        else:
            val = _match_option(str(context_window), context_def)
            if val is None:
                val = _match_option(_normalize_context_guess(context_window), context_def)
            if val is None:
                errors.append(
                    f"invalid cursor_context_window {context_window!r} for {cm.base_id}; "
                    f"allowed: {_allowed_values(context_def)}"
                )
            else:
                want[context_def.id] = val

    fast_def = _param_def(cm, FAST_PARAM_IDS)
    fast_val = _normalize_fast(fast)
    if fast_val is not None:
        if not fast_def:
            if strict:
                errors.append(f"model {cm.base_id!r} does not support cursor_fast")
        else:
            val = _match_option(fast_val, fast_def)
            if val is None:
                errors.append(
                    f"invalid cursor_fast {fast!r} for {cm.base_id}; "
                    f"allowed: {_allowed_values(fast_def)}"
                )
            else:
                want[fast_def.id] = val

    if errors and strict:
        raise ModelParameterError("; ".join(errors))
    return want


def _normalize_context_guess(ctx: str | None) -> str | None:
    if not ctx:
        return None
    c = ctx.strip().lower().replace("_", "")
    if c in ("1m", "1000000", "1048576", "1000k"):
        return "1m"
    if c in ("272k", "272000", "300k", "300000", "default"):
        return c
    if re.fullmatch(r"\d+k", c):
        return c
    if re.fullmatch(r"\d+", c):
        n = int(c)
        if n >= 500000:
            return "1m"
        if n >= 250000:
            return "272k"
        return "300k"
    return c


def _normalize_fast(fast: bool | str | None) -> str | None:
    if fast is None:
        return None
    if isinstance(fast, bool):
        return "true" if fast else "false"
    s = str(fast).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return "true"
    if s in ("0", "false", "no", "off"):
        return "false"
    return None


async def _fetch_catalog_raw(access_token: str | None = None) -> tuple[list[CatalogModel], str | None]:
    import httpx
    import certifi

    token = resolve_access_token(access_token)
    body = pb_int(2, 1) + pb_int(5, 1) + pb_int(7, 1)
    headers = _aiservice_headers(token)
    async with httpx.AsyncClient(http2=True, timeout=60, verify=certifi.where()) as client:
        models_resp = await client.post(
            API2_BASE + AVAILABLE_MODELS_PATH, content=body, headers=headers
        )
        default_resp = await client.post(
            API2_BASE + GET_DEFAULT_MODEL_PATH, content=b"", headers=headers
        )
    if models_resp.status_code != 200:
        raise RuntimeError(f"AvailableModels failed ({models_resp.status_code})")

    models: list[CatalogModel] = []
    pos = 0
    while (r := _read_field(models_resp.content, pos)) is not None:
        f, w, v, pos = r
        if f == 2 and w == 2:
            m = _parse_catalog_model(v)
            if m and m.supports_agent:
                models.append(m)

    default_id = None
    if default_resp.status_code == 200:
        pos = 0
        while (r := _read_field(default_resp.content, pos)) is not None:
            f, w, v, pos = r
            if f == 1 and w == 2:
                inner = _parse_catalog_model(v)
                if inner:
                    default_id = inner.base_id
                else:
                    ipos = 0
                    while (r2 := _read_field(v, ipos)) is not None:
                        ff, ww, vv, ipos = r2
                        if ff == 1 and ww == 2:
                            default_id = vv.decode("utf-8", "replace")

    return models, default_id


async def get_catalog(access_token: str | None = None) -> tuple[list[CatalogModel], str | None]:
    now = time.time()
    key = _token_cache_key(access_token)
    hit = _cache.get(key)
    if hit and hit.get("models") is not None and now - hit["ts"] < _CACHE_TTL:
        return hit["models"], hit["default_id"]
    models, default_id = await _fetch_catalog_raw(access_token)
    _cache[key] = {"ts": now, "models": models, "default_id": default_id}
    return models, default_id


@dataclass
class SelectableModel:
    """One OpenAI/OpenClaw-selectable model id with baked-in Cursor parameters."""

    id: str
    display_name: str
    logical_base: str
    parameters: dict[str, str]
    context_window_tokens: int = 128000
    supports_images: bool = False
    supports_reasoning: bool = False
    cursor_default: bool = False


def cursor_sent_model_id(logical_base: str, params: dict[str, str]) -> str:
    """Map logical model + params to the model_id Cursor's H2 API accepts."""
    if logical_base in ("auto", "default"):
        return "default"
    # Explicit composer + fast hangs; fast Composer is reached via default router.
    if logical_base == "composer-2.5" and params.get("fast") == "true":
        return "default"
    return logical_base


def _ordered_param_defs(cm: CatalogModel) -> list[ParamDef]:
    """Parameter defs in stable expansion order (reasoning, context, fast, ...)."""
    ordered: list[ParamDef] = []
    seen: set[str] = set()
    for key in (*REASONING_PARAM_IDS, *CONTEXT_PARAM_IDS, *FAST_PARAM_IDS):
        p = _param_def(cm, (key,))
        if p and p.options and p.id not in seen:
            ordered.append(p)
            seen.add(p.id)
    return ordered


def _auto_reference_model(catalog: list[CatalogModel]) -> CatalogModel | None:
    """Pick one catalogue entry to drive auto parameter expansion."""
    preferred = ("gpt-5.5", "composer-2.5", "claude-sonnet-4-6", "claude-opus-4-8")
    for base_id in preferred:
        for cm in catalog:
            if cm.base_id == base_id and _ordered_param_defs(cm):
                return cm
    for cm in catalog:
        if _ordered_param_defs(cm):
            return cm
    return None


def _auto_param_defs(catalog: list[CatalogModel]) -> list[ParamDef]:
    """Parameter defs for auto/* expanded ids (single family, no mixed effort+reasoning)."""
    ref = _auto_reference_model(catalog)
    return _ordered_param_defs(ref) if ref else []


def _context_tokens_from_params(params: dict[str, str]) -> int:
    ctx = params.get("context", "")
    if not ctx:
        return 128000
    m = re.fullmatch(r"(\d+)(k|m)", ctx.strip().lower())
    if not m:
        return 128000
    n, unit = int(m.group(1)), m.group(2)
    return n * 1_000_000 if unit == "m" else n * 1000


def _format_param_label(pdef: ParamDef, value: str) -> str:
    for opt in pdef.options:
        if opt.value == value:
            return opt.label or value
    return value


def _display_name(base_name: str, param_defs: list[ParamDef], params: dict[str, str]) -> str:
    if not params:
        return base_name
    parts = [
        _format_param_label(pdef, params[pdef.id])
        for pdef in param_defs
        if pdef.id in params
    ]
    return f"{base_name} ({', '.join(parts)})"


def _expand_base(
    *,
    logical_base: str,
    display_base: str,
    param_defs: list[ParamDef],
    supports_images: bool = False,
    supports_reasoning: bool = False,
    cursor_default: bool = False,
) -> list[SelectableModel]:
    if not param_defs:
        return [
            SelectableModel(
                id=logical_base,
                display_name=display_base,
                logical_base=logical_base,
                parameters={},
                supports_images=supports_images,
                supports_reasoning=supports_reasoning,
                cursor_default=cursor_default,
            )
        ]

    combos: list[dict[str, str]] = [{}]
    option_lists = [[(pdef.id, opt.value) for opt in pdef.options] for pdef in param_defs]
    for combo in itertools.product(*option_lists):
        combos.append(dict(combo))

    out: list[SelectableModel] = []
    seen_ids: set[str] = set()
    for params in combos:
        if not params:
            mid = logical_base
        else:
            segments = [logical_base]
            for pdef in param_defs:
                val = params[pdef.id]
                segments.append(val)
            mid = "/".join(segments)
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        out.append(
            SelectableModel(
                id=mid,
                display_name=_display_name(display_base, param_defs, params),
                logical_base=logical_base,
                parameters=params,
                context_window_tokens=_context_tokens_from_params(params),
                supports_images=supports_images,
                supports_reasoning=supports_reasoning,
                cursor_default=cursor_default and not params,
            )
        )
    return out


def expand_selectable_models(
    catalog: list[CatalogModel],
    *,
    default_id: str | None = None,
) -> list[SelectableModel]:
    """Expand every parameter option combo into a distinct selectable model id."""
    cache_key = (id(catalog), default_id, tuple(cm.base_id for cm in catalog))
    if _selectable_cache["key"] == cache_key and _selectable_cache["models"] is not None:
        return _selectable_cache["models"]

    out: list[SelectableModel] = []
    seen: set[str] = set()

    auto_defs = _auto_param_defs(catalog)
    for sm in _expand_base(
        logical_base="auto",
        display_base="Auto",
        param_defs=auto_defs,
        cursor_default=default_id in (None, "auto", "default"),
    ):
        if sm.id not in seen:
            seen.add(sm.id)
            out.append(sm)

    for cm in catalog:
        if not cm.supports_agent:
            continue
        defs = _ordered_param_defs(cm)
        has_reasoning = bool(_param_def(cm, REASONING_PARAM_IDS))
        for sm in _expand_base(
            logical_base=cm.base_id,
            display_base=cm.display_name or cm.base_id,
            param_defs=defs,
            supports_images=cm.supports_images,
            supports_reasoning=has_reasoning,
            cursor_default=cm.base_id == default_id,
        ):
            if sm.id not in seen:
                seen.add(sm.id)
                out.append(sm)

    _selectable_cache.update(key=cache_key, models=out)
    return out


def _match_segment_to_param(segment: str, pdef: ParamDef) -> str | None:
    return _match_option(segment, pdef)


def _parse_slash_model_id(
    model: str, catalog: list[CatalogModel]
) -> tuple[str, dict[str, str]] | None:
    if "/" not in model:
        return None
    base, *segments = model.split("/")
    if base in ("auto", "default"):
        param_defs = _auto_param_defs(catalog)
        logical_base = "auto"
    else:
        cm = catalog_model_for(catalog, base)
        if not cm:
            return None
        param_defs = _ordered_param_defs(cm)
        logical_base = cm.base_id
    if len(segments) > len(param_defs):
        return None
    params: dict[str, str] = {}
    for segment, pdef in zip(segments, param_defs):
        val = _match_segment_to_param(segment, pdef)
        if val is None:
            return None
        params[pdef.id] = val
    return logical_base, params


def _legacy_variant_params(
    model: str,
    catalog: list[CatalogModel],
) -> tuple[str, dict[str, str]] | None:
    for cm in catalog:
        if model == cm.base_id:
            return cm.base_id, {}
        for var in cm.variants:
            if model == var.id:
                return cm.base_id, {
                    k: v for k, v in var.parameters.items() if not k.startswith("_")
                }
    return None


def _merge_encoded_and_body(
    encoded: dict[str, str],
    *,
    cm: CatalogModel | None,
    catalog: list[CatalogModel] | None,
    reasoning_effort: str | None,
    context_window: str | None,
    fast: bool | str | None,
    strict: bool,
) -> dict[str, str]:
    if cm is None and catalog:
        if encoded:
            cm = catalog_model_for(catalog, "auto")
        elif not encoded:
            cm = None
    body_want: dict[str, str] = {}
    if cm:
        body_want = _build_want_params(
            cm,
            reasoning_effort=reasoning_effort,
            context_window=context_window,
            fast=fast,
            strict=strict,
        )
    elif catalog and (reasoning_effort or context_window or fast is not None):
        ref = _auto_reference_model(catalog)
        pseudo = ref or CatalogModel(
            "auto", "Auto", True, False, parameter_defs=_auto_param_defs(catalog)
        )
        body_want = _build_want_params(
            pseudo,
            reasoning_effort=reasoning_effort,
            context_window=context_window,
            fast=fast,
            strict=strict,
        )
    merged = dict(encoded)
    for key, val in body_want.items():
        if key not in merged:
            merged[key] = val
    return merged


def resolve_model(
    model: str,
    *,
    reasoning_effort: str | None = None,
    context_window: str | None = None,
    fast: bool | str | None = None,
    catalog: list[CatalogModel] | None = None,
    strict: bool = True,
) -> tuple[str, str, list[tuple[str, str]]]:
    """
    Return (cursor_sent_model_id, logical_base_id, parameters) for AgentRunRequest.

    Selectable ids use slash-separated parameter values, e.g. ``auto/fast`` or
    ``gpt-5.5/medium/272k``. Legacy variant ids (``gpt-5.5-medium-fast``) are
    still accepted and mapped to base model + parameters.
    """
    encoded: dict[str, str] = {}
    logical_base = model

    if model in ("auto", "default", ""):
        logical_base = "auto"
    elif catalog:
        parsed = _parse_slash_model_id(model, catalog)
        if parsed:
            logical_base, encoded = parsed
        else:
            legacy = _legacy_variant_params(model, catalog)
            if legacy:
                logical_base, encoded = legacy
            else:
                cm = catalog_model_for(catalog, model)
                if cm:
                    logical_base = cm.base_id

    cm_for_body = catalog_model_for(catalog, logical_base) if catalog else None
    if logical_base == "auto" and catalog:
        ref = _auto_reference_model(catalog)
        cm_for_body = ref or CatalogModel(
            "auto",
            "Auto",
            True,
            False,
            parameter_defs=_auto_param_defs(catalog),
        )

    params = _merge_encoded_and_body(
        encoded,
        cm=cm_for_body,
        catalog=catalog,
        reasoning_effort=reasoning_effort,
        context_window=context_window,
        fast=fast,
        strict=strict,
    )
    sent_id = cursor_sent_model_id(logical_base, params)
    return sent_id, logical_base, [(k, v) for k, v in params.items()]


def all_model_ids(catalog: list[CatalogModel], *, default_id: str | None = None) -> list[str]:
    return [sm.id for sm in expand_selectable_models(catalog, default_id=default_id)]


def selectable_model_for(
    catalog: list[CatalogModel],
    model_id: str,
    *,
    default_id: str | None = None,
) -> SelectableModel | None:
    for sm in expand_selectable_models(catalog, default_id=default_id):
        if sm.id == model_id:
            return sm
    return None
