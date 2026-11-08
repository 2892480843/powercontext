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

"""Pi session history: branch-stable event Sources with resumable ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"


class PiSessionInput(BaseModel):
    """One offline Pi session archive plus the selected leaf."""

    model_config = ConfigDict(frozen=True)

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible Pi event with its graph identity and complete record."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """A CAPTURED Source holding one exact Pi event."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str, /) -> str:
    """Build the stable, collision-free identity for one Pi event."""

    for label, value in (("session_id", session_id), ("event_id", event_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")  # noqa: TRY003
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_blocks(content: Any) -> tuple[str, ...]:
    if isinstance(content, str):
        return (content,) if content else ()
    if not isinstance(content, list):
        return ()
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return tuple(parts)


def _message_body(message: dict[str, Any]) -> tuple[str, str] | None:
    """Render one message record as ``(role, body)`` or None when invisible."""

    role = message.get("role")
    if not isinstance(role, str) or not role:
        return None
    content = message.get("content")
    if role == "toolResult":
        texts = _text_blocks(content)
        if not texts:
            return None
        tool_name = message.get("toolName")
        tool_call_id = message.get("toolCallId")
        if not isinstance(tool_name, str) or not isinstance(tool_call_id, str):
            return None
        return role, f"[tool_result {tool_name} {tool_call_id}] " + "\n".join(texts)
    if role == "bashExecution":
        command = message.get("command")
        if not isinstance(command, str) or not command:
            return None
        output = message.get("output")
        return role, f"$ {command}\n{output if isinstance(output, str) else ''}"
    if isinstance(content, str):
        return (role, content) if content else None
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif kind == "toolCall":
                name = block.get("name")
                call_id = block.get("id")
                if isinstance(name, str) and isinstance(call_id, str):
                    parts.append(f"[tool_call {name} {call_id}] {_compact_json(block.get('arguments'))}")
        return (role, "\n".join(parts)) if parts else None
    return None


def _render_event(record: dict[str, Any]) -> tuple[str, str] | None:
    """Render one record as ``(role, content)`` or None when it has no visible body."""

    kind = record.get("type")
    if kind == "message":
        message = record.get("message")
        if isinstance(message, dict):
            rendered = _message_body(message)
            if rendered is not None:
                return rendered[0], f"{rendered[0]}: {rendered[1]}"
        return None
    if kind in {"compaction", "branch_summary"}:
        summary = record.get("summary")
        if isinstance(summary, str) and summary:
            return kind, f"{kind}: {summary}"
        return None
    if kind == "custom_message":
        texts = _text_blocks(record.get("content"))
        if texts:
            return "custom", "custom: " + "\n".join(texts)
        return None
    return None


def _parse_records(content: bytes) -> tuple[str, list[dict[str, Any]]]:
    try:
        text = content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be strict UTF-8") from error  # noqa: TRY003
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Pi session content must be JSONL") from error  # noqa: TRY003
        if not isinstance(record, dict):
            raise ValueError("Pi session records must be JSON objects")  # noqa: TRY003
        records.append(record)
    if not records:
        raise ValueError("Pi session content must contain a session header")  # noqa: TRY003
    header = records[0]
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("Pi session archives must start with a version 3 session header")  # noqa: TRY003
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must carry a non-empty id")  # noqa: TRY003
    return session_id, records


def parse_pi_session(document: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse one archive and return the visible events of the selected branch."""

    session_id, records = _parse_records(document.content)
    by_id: dict[str, dict[str, Any]] = {}
    parent_of: dict[str, str | None] = {}
    order: list[str] = []
    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Pi session events must carry a non-empty id")  # noqa: TRY003
        if event_id in by_id:
            raise ValueError(f"duplicate Pi session event id {event_id!r}")  # noqa: TRY003
        parent = record.get("parentId")
        if parent is None or parent == "":
            resolved: str | None = None
        elif isinstance(parent, str) and parent in by_id:
            resolved = parent
        else:
            raise ValueError(f"Pi session event {event_id!r} references an unknown parent")  # noqa: TRY003
        by_id[event_id] = record
        parent_of[event_id] = resolved
        order.append(event_id)

    leaf_id = document.leaf_id if document.leaf_id is not None else (order[-1] if order else None)
    if leaf_id is None or leaf_id not in by_id:
        raise ValueError("selected Pi session leaf is missing from the archive")  # noqa: TRY003

    chain: list[str] = []
    node: str | None = leaf_id
    while node is not None:
        chain.append(node)
        node = parent_of[node]
    chain.reverse()

    events: list[PiEventInput] = []
    for event_id in chain:
        record = by_id[event_id]
        rendered = _render_event(record)
        if rendered is None:
            continue
        role, content = rendered
        pi_event_source_id(session_id, event_id)
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parent_of[event_id],
                event_type=record.get("type") if isinstance(record.get("type"), str) else "",
                role=role,
                content=content,
                record=record,
            )
        )
    return tuple(events)


class PiEventAdapter:
    """Resolve and read Pi event Sources."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            materialization=SourceMaterialization.CAPTURED,
            source_id=source_id,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose the full event text with its graph identity."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PiEventAdapter(),
    projections=(PiEventTextEvidenceProjection(),),
)


def _event_digest(event: PiEventInput) -> str:
    payload = _compact_json({"content": event.content, "record": event.record})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PiSessionConnector:
    """Incrementally import one Pi session branch with durable checkpoints."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session, /) -> ConnectorRunCompletion:
        events = parse_pi_session(
            PiSessionInput(content=self._path.read_bytes(), leaf_id=self._leaf_id)
        )
        imported: dict[str, str] = {}
        checkpoint = session.checkpoint
        if checkpoint is not None:
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("events"), dict):
                raise ValueError("Pi session checkpoint is malformed")  # noqa: TRY003
            imported = {
                event_id: digest
                for event_id, digest in checkpoint["events"].items()
                if isinstance(event_id, str) and isinstance(digest, str)
            }
        for event in events:
            digest = _event_digest(event)
            known = imported.get(event.event_id)
            if known is not None:
                if known != digest:
                    raise ValueError(f"imported Pi session event {event.event_id!r} was rewritten")  # noqa: TRY003
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            imported[event.event_id] = digest
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"events": imported},
        )


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PiEventAdapter",
    "PiEventInput",
    "PiEventSource",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
