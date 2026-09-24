"""Unit tests for cursor_core Connect-RPC framing and protobuf helpers."""
from __future__ import annotations

import pytest

from cursor_core.framing import (
    connect_frame,
    decode_frames,
    decode_listvalue,
    decode_raw,
    decode_struct,
    decode_value,
    decode_varint,
    encode_varint,
    json_to_struct,
    json_to_value,
    pb_double,
    pb_int,
    pb_msg,
    pb_str,
)


def test_varint_roundtrip_and_truncated():
    for n in (0, 1, 127, 128, 300, 0x4000, 2**32):
        encoded = encode_varint(n)
        value, pos = decode_varint(encoded, 0)
        assert value == n
        assert pos == len(encoded)
    with pytest.raises(ValueError, match="truncated"):
        decode_varint(b"\x80", 0)


def test_json_value_roundtrip_all_types():
    payload = {
        "n": None,
        "b": True,
        "f": False,
        "i": 3,
        "d": 1.5,
        "s": "hi",
        "obj": {"k": "v"},
        "arr": [1, "x"],
        "other": object(),
    }
    encoded = json_to_struct(payload)
    decoded = decode_struct(encoded)
    assert decoded["n"] is None
    assert decoded["b"] is True
    assert decoded["f"] is False
    assert decoded["i"] == 3
    assert decoded["d"] == 1.5
    assert decoded["s"] == "hi"
    assert decoded["obj"] == {"k": "v"}
    assert decoded["arr"] == [1, "x"]
    assert isinstance(decoded["other"], str)


def test_decode_value_fallbacks():
    assert decode_value(b"") == ""
    assert decode_value(b"plain") == "plain"
    encoded_list = json_to_value(["a", 2])
    assert decode_value(encoded_list) == ["a", 2]
    assert decode_listvalue(b"") == []


def test_connect_frames_and_decode_raw():
    inner = pb_str(1, "hello") + pb_int(2, 7) + pb_double(3, 2.5)
    framed = connect_frame(inner) + connect_frame(b"more", compressed=True)
    frames = list(decode_frames(framed))
    assert frames[0][0] == 0x00
    assert b"hello" in frames[0][1]
    assert frames[1][0] == 0x01

    # truncated frame is ignored
    assert list(decode_frames(b"\x00\x00\x00\x00\x10" + b"xx")) == []

    raw = decode_raw(inner)
    kinds = {k for _f, k, _v in raw}
    assert "str" in kinds
    assert "int" in kinds
    assert "double" in kinds

    # wire-5 float + unknown wire stops
    float_field = pb_msg(1, b"")  # just ensure decode_raw doesn't raise
    assert isinstance(decode_raw(float_field), list)
    unknown = bytes([(1 << 3) | 7])
    assert decode_raw(unknown) == []
