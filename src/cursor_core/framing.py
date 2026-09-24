"""
Connect-RPC framing + schema-free protobuf primitives.

This is the lowest layer of ``cursor_core``: it implements just enough of the
protobuf wire format (varint / 64-bit / length-delimited / 32-bit) to encode and
decode Cursor's ``agent.v1`` messages without a generated schema, plus the
5-byte Connect-RPC envelope used to frame those messages over HTTP/2.

Everything here is app-agnostic and has no Cursor-specific knowledge — it is a
verbatim extraction of the original ``proto.py`` so the higher layers
(``wire``, ``engine``) can build on a stable primitive set.
"""
import struct


def encode_varint(value: int) -> bytes:
    parts = []
    while value > 0x7F:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value & 0x7F)
    return bytes(parts)


def decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    result, shift = 0, 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
    raise ValueError("truncated varint")


def encode_field(field_num: int, wire_type: int, value: bytes) -> bytes:
    tag = (field_num << 3) | wire_type
    return encode_varint(tag) + value


def pb_int(field: int, value: int) -> bytes:
    return encode_field(field, 0, encode_varint(value))


def pb_bytes(field: int, value: bytes) -> bytes:
    return encode_field(field, 2, encode_varint(len(value)) + value)


def pb_str(field: int, value: str) -> bytes:
    return pb_bytes(field, value.encode())


def pb_msg(field: int, content: bytes) -> bytes:
    return pb_bytes(field, content)


def pb_double(field: int, value: float) -> bytes:
    return encode_field(field, 1, struct.pack("<d", value))


def json_to_struct(d: dict) -> bytes:
    """Encode a dict as google.protobuf.Struct (map<string, Value> fields = 1)."""
    body = b""
    for k, v in d.items():
        entry = pb_str(1, str(k)) + pb_msg(2, json_to_value(v))
        body += pb_msg(1, entry)
    return body


def json_to_value(v) -> bytes:
    """Encode a JSON-compatible Python value as google.protobuf.Value bytes.

    Value oneof: 1=null_value(enum), 2=number_value(double), 3=string_value,
    4=bool_value, 5=struct_value(Struct), 6=list_value(ListValue).
    bool is checked before int (bool is a subclass of int in Python).
    """
    if v is None:
        return pb_int(1, 0)
    if isinstance(v, bool):
        return pb_int(4, 1 if v else 0)
    if isinstance(v, (int, float)):
        return pb_double(2, float(v))
    if isinstance(v, str):
        return pb_str(3, v)
    if isinstance(v, dict):
        return pb_msg(5, json_to_struct(v))
    if isinstance(v, (list, tuple)):
        inner = b"".join(pb_msg(1, json_to_value(x)) for x in v)
        return pb_msg(6, inner)
    return pb_str(3, str(v))


def _value_fields(data: bytes):
    """Yield (field, wire, value) for a protobuf message, doubles for wire-1."""
    pos = 0
    while pos < len(data):
        try:
            tag, pos = decode_varint(data, pos)
        except Exception:
            return
        field, wire = tag >> 3, tag & 7
        if field == 0:
            return
        if wire == 0:
            v, pos = decode_varint(data, pos)
            yield field, 0, v
        elif wire == 1:
            if pos + 8 > len(data):
                return
            yield field, 1, struct.unpack("<d", data[pos:pos + 8])[0]
            pos += 8
        elif wire == 2:
            length, pos = decode_varint(data, pos)
            yield field, 2, data[pos:pos + length]
            pos += length
        elif wire == 5:
            if pos + 4 > len(data):
                return
            yield field, 5, struct.unpack("<f", data[pos:pos + 4])[0]
            pos += 4
        else:
            return


def decode_value(data: bytes):
    """Decode google.protobuf.Value bytes to a Python value.

    Value oneof: 1=null, 2=number(double), 3=string, 4=bool, 5=Struct, 6=List.
    Falls back to a UTF-8 string when the bytes aren't a recognizable Value so
    non-Value senders still degrade gracefully.
    """
    for field, wire, v in _value_fields(data):
        if field == 1 and wire == 0:
            return None
        if field == 2 and wire == 1:
            return int(v) if isinstance(v, float) and v.is_integer() else v
        if field == 3 and wire == 2:
            return v.decode("utf-8", "replace")
        if field == 4 and wire == 0:
            return bool(v)
        if field == 5 and wire == 2:
            return decode_struct(v)
        if field == 6 and wire == 2:
            return decode_listvalue(v)
    return data.decode("utf-8", "replace") if data else ""


def decode_struct(data: bytes) -> dict:
    """Decode google.protobuf.Struct bytes (map<string, Value> fields = 1)."""
    out: dict = {}
    for field, wire, v in _value_fields(data):
        if field == 1 and wire == 2:
            key = None
            val_bytes = b""
            for f2, w2, v2 in _value_fields(v):
                if f2 == 1 and w2 == 2:
                    key = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 2:
                    val_bytes = v2
            if key is not None:
                out[key] = decode_value(val_bytes)
    return out


def decode_listvalue(data: bytes) -> list:
    """Decode google.protobuf.ListValue bytes (repeated Value values = 1)."""
    out: list = []
    for field, wire, v in _value_fields(data):
        if field == 1 and wire == 2:
            out.append(decode_value(v))
    return out


def connect_frame(data: bytes, compressed: bool = False) -> bytes:
    """Wrap protobuf bytes in a Connect-RPC 5-byte envelope."""
    flag = 0x00 if not compressed else 0x01
    return bytes([flag]) + struct.pack(">I", len(data)) + data


def decode_frames(data: bytes):
    """Yield (flag, message_bytes) from a stream of Connect-RPC frames."""
    pos = 0
    while pos + 5 <= len(data):
        flag = data[pos]
        msg_len = struct.unpack(">I", data[pos + 1:pos + 5])[0]
        pos += 5
        if pos + msg_len > len(data):
            break
        yield flag, data[pos:pos + msg_len]
        pos += msg_len


def decode_raw(data: bytes, depth: int = 0) -> list:
    """Schema-free recursive protobuf decoder. Returns list of (field, wire, value)."""
    pos = 0
    fields = []
    while pos < len(data):
        try:
            tag_wire, pos = decode_varint(data, pos)
            field = tag_wire >> 3
            wire = tag_wire & 7
            if field == 0:
                break
            if wire == 0:
                val, pos = decode_varint(data, pos)
                fields.append((field, "int", val))
            elif wire == 2:
                length, pos = decode_varint(data, pos)
                chunk = data[pos:pos + length]
                pos += length
                try:
                    text = chunk.decode("utf-8")
                    if all(32 <= ord(c) < 127 or c in "\n\r\t " for c in text):
                        fields.append((field, "str", text))
                        continue
                except Exception:
                    pass
                fields.append((field, "bytes", chunk))
            elif wire == 1:
                val = struct.unpack("<d", data[pos:pos + 8])[0]
                pos += 8
                fields.append((field, "double", val))
            elif wire == 5:
                val = struct.unpack("<f", data[pos:pos + 4])[0]
                pos += 4
                fields.append((field, "float", val))
            else:
                break
        except Exception:
            break
    return fields
