"""Offline tests for the model catalogue helpers."""
from __future__ import annotations

from cursor_core.framing import pb_int, pb_msg, pb_str
from cursor_core.models import (
    CatalogModel,
    ModelParameterError,
    ParamDef,
    ParamOption,
    _context_tokens_from_params,
    _match_option,
    _parse_catalog_model,
    catalog_model_for,
    expand_selectable_models,
    openai_param_mapping,
    resolve_model,
)


def _sample_catalog() -> list[CatalogModel]:
    reasoning = ParamDef(
        "reasoning",
        "Reasoning",
        "",
        [ParamOption("low", "Low"), ParamOption("medium", "Medium")],
        0,
    )
    context = ParamDef(
        "context",
        "Context",
        "",
        [ParamOption("128k", "128k"), ParamOption("272k", "272k")],
        0,
    )
    fast = ParamDef("fast", "Fast", "", [ParamOption("true", "Fast")], 0)
    return [
        CatalogModel(
            base_id="gpt-5.5",
            display_name="GPT 5.5",
            supports_agent=True,
            supports_images=True,
            parameter_defs=[reasoning, context, fast],
        )
    ]


def test_context_tokens_from_params():
    assert _context_tokens_from_params({}) == 128000
    assert _context_tokens_from_params({"context": "272k"}) == 272000
    assert _context_tokens_from_params({"context": "1m"}) == 1_000_000
    assert _context_tokens_from_params({"context": "nope"}) == 128000


def test_match_option_aliases():
    pdef = ParamDef(
        "reasoning",
        "Reasoning",
        "",
        [ParamOption("xhigh", "Extra high"), ParamOption("none", "None")],
        0,
    )
    assert _match_option("extra-high", pdef) == "xhigh"
    assert _match_option("minimal", pdef) == "none"
    assert _match_option("Extra high", pdef) == "xhigh"
    assert _match_option("nope", pdef) is None


def test_catalog_model_for_and_openai_mapping():
    catalog = _sample_catalog()
    assert catalog_model_for(catalog, "gpt-5.5").base_id == "gpt-5.5"
    assert catalog_model_for(catalog, "gpt-5.5-medium").base_id == "gpt-5.5"
    assert catalog_model_for(catalog, "other") is None
    mapping = openai_param_mapping(catalog[0])
    assert mapping["reasoning_effort"] == "reasoning"
    assert mapping["cursor_context_window"] == "context"
    assert mapping["cursor_fast"] == "fast"


def test_resolve_model_auto_and_slash():
    catalog = _sample_catalog()
    sent, logical, params = resolve_model("auto", catalog=catalog, strict=False)
    assert logical == "auto"
    assert sent in {"default", "auto"}

    sent, logical, params = resolve_model(
        "gpt-5.5/medium/272k",
        catalog=catalog,
        strict=True,
    )
    assert logical == "gpt-5.5"
    assert ("reasoning", "medium") in params or any(p[0] == "reasoning" for p in params)


def test_resolve_model_strict_unknown_param():
    catalog = _sample_catalog()
    try:
        resolve_model(
            "gpt-5.5",
            reasoning_effort="not-a-level",
            catalog=catalog,
            strict=True,
        )
    except ModelParameterError:
        return
    raise AssertionError("expected ModelParameterError")


def test_expand_selectable_models():
    catalog = _sample_catalog()
    models = expand_selectable_models(catalog, default_id="gpt-5.5")
    ids = {m.id for m in models}
    assert "auto" in ids
    assert any(mid.startswith("gpt-5.5/") for mid in ids)
    again = expand_selectable_models(catalog, default_id="gpt-5.5")
    assert again is models


def test_parse_catalog_model_bytes():
    blob = (
        pb_str(1, "gpt-5.5")
        + pb_str(2, "GPT 5.5")
        + pb_int(5, 1)
        + pb_int(10, 1)
    )
    parsed = _parse_catalog_model(blob)
    assert parsed is not None
    assert parsed.base_id == "gpt-5.5"
    assert parsed.supports_agent is True
    assert parsed.supports_images is True


def test_normalize_helpers_and_strict_params():
    from cursor_core.models import _normalize_context_guess, _normalize_fast

    assert _normalize_fast(True) == "true"
    assert _normalize_fast(False) == "false"
    assert _normalize_fast("yes") == "true"
    assert _normalize_fast("off") == "false"
    assert _normalize_fast(None) is None
    assert _normalize_context_guess("1m") == "1m"
    assert _normalize_context_guess("272000") == "272000"
    assert _normalize_context_guess("128k") == "128k"
    assert _normalize_context_guess("800000") == "1m"
    assert _normalize_context_guess("260000") == "272k"

    catalog = _sample_catalog()
    _, _, params = resolve_model(
        "gpt-5.5",
        reasoning_effort="medium",
        context_window="272k",
        fast=True,
        catalog=catalog,
        strict=True,
    )
    ids = dict(params)
    assert ids.get("reasoning") == "medium"
    assert ids.get("context") == "272k"
    assert ids.get("fast") == "true"

    from cursor_core.models import _leaf_option, _parse_param_def, _parse_variant

    leaf = pb_msg(1, pb_str(1, "medium")) + pb_str(2, "Medium")
    assert _leaf_option(leaf) == ("medium", "Medium")

    pdef_blob = (
        pb_str(1, "reasoning")
        + pb_str(2, "Reasoning")
        + pb_str(3, "effort")
        + pb_int(5, 0)
    )
    pdef = _parse_param_def(pdef_blob)
    assert pdef is not None
    assert pdef.id == "reasoning"
    assert pdef.name == "Reasoning"

    param_entry = pb_str(1, "reasoning") + pb_str(2, "medium")
    variant = _parse_variant(pb_msg(1, param_entry) + pb_str(11, "shown"))
    assert variant is not None
    assert variant.parameters.get("reasoning") == "medium"


def test_token_cache_and_ids():
    from cursor_core.models import (
        _token_cache_key,
        all_model_ids,
        selectable_model_for,
        cursor_sent_model_id,
        _parse_options_container,
        _leaf_option,
        _allowed_values,
        ParamDef,
    )

    assert _token_cache_key(None) == "host"
    assert _token_cache_key("  ") == "host"
    assert len(_token_cache_key("abc")) == 16
    catalog = _sample_catalog()
    ids = all_model_ids(catalog, default_id="gpt-5.5")
    assert "auto" in ids
    sm = selectable_model_for(catalog, "auto", default_id="gpt-5.5")
    assert sm is not None
    assert selectable_model_for(catalog, "nope", default_id="gpt-5.5") is None
    assert cursor_sent_model_id("auto", {}) in {"default", "auto"}
    pdef = catalog[0].parameter_defs[0]
    assert _allowed_values(pdef)
    assert _allowed_values(ParamDef("x", "x", "", [], 0)) == []
    assert _leaf_option(b"") is None
    nested = pb_msg(1, pb_str(1, "low") + pb_str(2, "Low"))
    opts = _parse_options_container(pb_msg(1, nested))
    assert opts


def test_parse_catalog_with_variants_and_params():
    from cursor_core.models import _parse_param_def, _parse_catalog_model, catalog_model_for

    option = pb_msg(1, pb_str(1, "medium") + pb_str(2, "Medium"))
    pdef = (
        pb_str(1, "reasoning")
        + pb_str(2, "Reasoning")
        + pb_str(3, "effort")
        + pb_msg(4, option)
        + pb_int(5, 0)
    )
    parsed_def = _parse_param_def(pdef)
    assert parsed_def is not None
    assert parsed_def.id == "reasoning"

    variant = pb_str(11, "shown") + pb_msg(1, pb_str(1, "reasoning") + pb_str(2, "medium"))
    blob = (
        pb_str(1, "gpt-5.5")
        + pb_str(17, "GPT 5.5")
        + pb_int(5, 1)
        + pb_int(10, 1)
        + pb_msg(29, pdef)
        + pb_msg(30, variant)
        + pb_str(36, "vid-a")
        + pb_str(36, "vid-b")
    )
    cm = _parse_catalog_model(blob)
    assert cm is not None
    assert cm.base_id == "gpt-5.5"
    assert catalog_model_for([cm], "vid-a") is cm
    assert catalog_model_for([cm], "gpt-5.5-extra") is cm
    assert _parse_catalog_model(pb_str(17, "no-id")) is None
    assert _parse_param_def(pb_str(2, "no-id")) is None

