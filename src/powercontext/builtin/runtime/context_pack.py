# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interoperable context packs with digest-verified deltas for incremental transport."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
_DEFAULT_CHUNK_BYTES = 512
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PACK_KEYS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_KEYS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """Raised when pack or delta wire data violates the transport protocol."""


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_item(value: object) -> bytes:
    if not isinstance(value, str):
        raise ContextPackError("pack items must be base64 strings")  # noqa: TRY003
    try:
        item = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise ContextPackError("pack items must be valid base64 strings") from error  # noqa: TRY003
    if base64.b64encode(item).decode("ascii") != value:
        raise ContextPackError("pack items must use canonical base64 encoding")  # noqa: TRY003
    return item


def _items_digest(items: tuple[bytes, ...], schema: str) -> str:
    unsigned = {"schema": schema, "items": _encode_items(items)}
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _validated_digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST_PATTERN.fullmatch(value):
        raise ContextPackError("digest must be sha256 followed by 64 lowercase hex characters")  # noqa: TRY003
    return value


@dataclass(frozen=True)
class ContextPack:
    """An immutable, self-describing chunked context payload."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items) -> ContextPack:
        chunked = tuple(bytes(item) for item in items)
        digest = _items_digest(chunked, PACK_SCHEMA)
        unsigned = {"schema": PACK_SCHEMA, "items": _encode_items(chunked)}
        byte_count = 0
        for _ in range(16):
            payload = {**unsigned, "digest": digest, "byte_count": byte_count}
            encoded = _canonical(payload)
            if len(encoded) == byte_count:
                return cls(items=chunked, digest=digest, byte_count=byte_count)
            byte_count = len(encoded)
        raise ContextPackError("pack byte count did not converge")  # noqa: TRY003

    @classmethod
    def from_content(cls, content, *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> ContextPack:
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        chunks: list[bytes] = []
        current = b""
        if content is not None:
            for character in content:
                encoded = character.encode("utf-8")
                if len(encoded) > chunk_bytes:
                    raise ContextPackError("a single character exceeds the chunk byte limit")  # noqa: TRY003
                if len(current) + len(encoded) > chunk_bytes:
                    chunks.append(current)
                    current = encoded
                else:
                    current += encoded
        if current:
            chunks.append(current)
        return cls.build(tuple(chunks))

    @classmethod
    def from_bytes(cls, content: bytes) -> ContextPack:
        if not isinstance(content, (bytes, bytearray)):
            raise ContextPackError("pack wire data must be bytes")  # noqa: TRY003
        try:
            payload = json.loads(bytes(content).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContextPackError("pack wire data must be UTF-8 JSON") from error  # noqa: TRY003
        if not isinstance(payload, dict) or frozenset(payload) != _PACK_KEYS:
            raise ContextPackError("pack wire data must contain exactly the pack fields")  # noqa: TRY003
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("pack wire data has an unexpected schema")  # noqa: TRY003
        if not isinstance(payload["items"], list):
            raise ContextPackError("pack items must be a list")  # noqa: TRY003
        items = tuple(_decode_item(item) for item in payload["items"])
        digest = _validated_digest(payload["digest"])
        if digest != _items_digest(items, PACK_SCHEMA):
            raise ContextPackError("pack digest does not match its items")  # noqa: TRY003
        byte_count = payload["byte_count"]
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count != len(content):
            raise ContextPackError("pack byte count does not match the wire length")  # noqa: TRY003
        return cls(items=items, digest=digest, byte_count=byte_count)

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": PACK_SCHEMA,
                "items": _encode_items(self.items),
                "digest": self.digest,
                "byte_count": self.byte_count,
            }
        )


@dataclass(frozen=True)
class ContextPackDelta:
    """A prefix-based delta that rebuilds one target pack from its base pack."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        prefix = 0
        limit = min(len(base.items), len(target.items))
        while prefix < limit and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _canonical(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encode_items(self.appended),
            }
        )

    @classmethod
    def from_bytes(cls, content: bytes) -> ContextPackDelta:
        if not isinstance(content, (bytes, bytearray)):
            raise ContextPackError("delta wire data must be bytes")  # noqa: TRY003
        try:
            payload = json.loads(bytes(content).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContextPackError("delta wire data must be UTF-8 JSON") from error  # noqa: TRY003
        if not isinstance(payload, dict) or frozenset(payload) != _DELTA_KEYS:
            raise ContextPackError("delta wire data must contain exactly the delta fields")  # noqa: TRY003
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("delta wire data has an unexpected schema")  # noqa: TRY003
        prefix = payload["prefix_items"]
        if type(prefix) is not int or prefix < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        if not isinstance(payload["appended"], list):
            raise ContextPackError("appended must be a list")  # noqa: TRY003
        return cls(
            base_digest=_validated_digest(payload["base_digest"]),
            target_digest=_validated_digest(payload["target_digest"]),
            prefix_items=prefix,
            appended=tuple(_decode_item(item) for item in payload["appended"]),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the delta's target pack from ``base``, verifying both digests."""

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("delta application requires a pack and a delta")  # noqa: TRY003
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the base pack")  # noqa: TRY003
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base pack length")  # noqa: TRY003
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack does not match the delta target digest")  # noqa: TRY003
    return target


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
