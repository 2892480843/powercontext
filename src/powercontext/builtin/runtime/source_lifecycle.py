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

"""Source withdrawal lifecycle: availability events propagated over a dependency DAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator

from powercontext.artifacts import Artifact, ArtifactRef
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.persistence.family_management import _RepositoryFamilyWriter
from powercontext.builtin.records import ArtifactWrite
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

LIFECYCLE_FAMILY = "source-lifecycle"


class SourceLifecycleContent(BaseModel):
    """The complete serialised lifecycle state of one scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[dict[str, JsonValue], ...] = ()
    state_version: int = 0


class SourceLifecycle(Artifact[SourceLifecycleContent]):
    """One immutable snapshot of a scope's lifecycle bindings and availability."""

    family: ClassVar[str] = LIFECYCLE_FAMILY


class SourceLifecycleManagementWriter(_RepositoryFamilyWriter):
    """Persist lifecycle snapshots through the standard artifact writer."""

    family = LIFECYCLE_FAMILY
    content_type = SourceLifecycleContent

    async def create(self, connection, scope_id, artifact_id, content, direct_source, /):  # noqa: ANN001, ANN202
        return await self._create_artifact(connection, scope_id, artifact_id, content, direct_source)

    async def replace(self, connection, scope_id, current, content, direct_source, /):  # noqa: ANN001, ANN202
        return await self._revise_artifact(connection, scope_id, current, content, direct_source)


class SourceLifecycleError(ValueError):
    """Raised for invalid lifecycle registration, events, or resolution."""


class SourceLifecycleEvent(BaseModel):
    """One availability transition for exactly one registered node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    reason: str
    revision: int
    sequence: int
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def _require_text(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SourceLifecycleError(f"{info.field_name} must be a non-blank string")  # noqa: TRY003
        return value

    @field_validator("revision", "sequence")
    @classmethod
    def _require_positive(cls, value: int, info) -> int:
        if type(value) is not int or value < 1:
            raise SourceLifecycleError(f"{info.field_name} must be a positive integer")  # noqa: TRY003
        return value


class _Node:
    __slots__ = ("node_id", "family", "reference", "parents", "policy", "revision", "available", "last_sequence")

    def __init__(
        self,
        *,
        node_id: str,
        family: str,
        reference: Any,
        parents: tuple[str, ...],
        policy: str | None,
        revision: int,
        available: bool,
        last_sequence: int = 0,
    ) -> None:
        self.node_id = node_id
        self.family = family
        self.reference = reference
        self.parents = parents
        self.policy = policy
        self.revision = revision
        self.available = available
        self.last_sequence = last_sequence

    def matches(self, other: _Node) -> bool:
        return (
            self.family == other.family
            and self.reference == other.reference
            and self.parents == other.parents
            and self.policy == other.policy
            and self.revision == other.revision
        )

    def to_json(self) -> dict:
        record: dict[str, Any] = {
            "node_id": self.node_id,
            "family": self.family,
            "parents": list(self.parents),
            "revision": self.revision,
            "available": self.available,
            "last_sequence": self.last_sequence,
        }
        if self.policy is not None:
            record["policy"] = self.policy
        if isinstance(self.reference, BaseModel):
            record["reference"] = self.reference.model_dump(mode="json")
        else:
            record["reference"] = dict(self.reference)
        return record


def _node_from_json(payload: dict) -> _Node:
    family = payload["family"]
    reference_data = payload["reference"]
    if family == "source":
        reference: Any = SourceRef.model_validate(reference_data)
    elif family == "memory":
        reference = MemoryCitation.model_validate(reference_data)
    else:
        reference = ArtifactRef.model_validate(reference_data)
    return _Node(
        node_id=payload["node_id"],
        family=family,
        reference=reference,
        parents=tuple(payload["parents"]),
        policy=payload.get("policy"),
        revision=int(payload["revision"]),
        available=bool(payload["available"]),
        last_sequence=int(payload.get("last_sequence", 0)),
    )


class ScopedSourceLifecycle:
    """Register evidence dependencies and propagate withdrawal events in one scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id
        self._nodes: dict[str, _Node] | None = None
        self._state_version = 1

    async def _load(self) -> dict[str, _Node]:
        if self._nodes is None:
            self._nodes = {}
            records = self._runtime.records.for_scope(self.scope_id)
            page = await records.query_artifacts(LIFECYCLE_FAMILY, limit=100, cursor=None)
            latest: tuple[int, dict] | None = None
            for item in page.items:
                try:
                    record = await records.get_artifact(LIFECYCLE_FAMILY, item.artifact_id)
                except Exception:  # noqa: BLE001 - unrelated artifacts never break lifecycle state.
                    continue
                version = int(record.content.get("state_version", 0))
                if latest is None or version > latest[0]:
                    latest = (version, record.content)
            self._state_version = latest[0] + 1 if latest else 1
            if latest is not None:
                content = latest[1]
                for payload in content.get("nodes", ()):
                    node = _node_from_json(payload)
                    self._nodes[node.node_id] = node
        return self._nodes

    async def _persist(self) -> None:
        nodes = self._nodes or {}
        version = getattr(self, "_state_version", 1)
        records = self._runtime.records.for_scope(self.scope_id)
        await records.create_artifact(
            LIFECYCLE_FAMILY,
            ArtifactWrite(
                content={
                    "nodes": [node.to_json() for node in nodes.values()],
                    "state_version": version,
                }
            ),
        )
        self._state_version = version + 1

    async def register_source(self, node_id: str, source_ref: SourceRef, /, *, revision: int = 1) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise SourceLifecycleError("node_id must be a non-blank string")  # noqa: TRY003
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("revision must be a positive integer")  # noqa: TRY003
        if not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("source registration requires a SourceRef")  # noqa: TRY003
        await self._register(
            _Node(node_id=node_id, family="source", reference=source_ref, parents=(), policy=None, revision=revision, available=True)
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: tuple[str, ...] = (),
        policy: str = "all",
    ) -> None:
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("memory registration requires a MemoryCitation")  # noqa: TRY003
        await self._register(
            _Node(node_id=node_id, family="memory", reference=citation, parents=tuple(parents), policy=self._validated_policy(policy), revision=1, available=True)
        )

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...] = (),
        policy: str = "any",
    ) -> None:
        if not isinstance(artifact_ref, ArtifactRef):
            raise SourceLifecycleError("topic registration requires an ArtifactRef")  # noqa: TRY003
        await self._register(
            _Node(node_id=node_id, family="topic", reference=artifact_ref, parents=tuple(parents), policy=self._validated_policy(policy), revision=1, available=True)
        )

    @staticmethod
    def _validated_policy(policy: str) -> str:
        if policy not in {"all", "any"}:
            raise SourceLifecycleError("policy must be all or any")  # noqa: TRY003
        return policy

    async def _register(self, node: _Node) -> None:
        if not isinstance(node.node_id, str) or not node.node_id.strip():
            raise SourceLifecycleError("node_id must be a non-blank string")  # noqa: TRY003
        nodes = await self._load()
        if not node.parents and node.family != "source":
            raise SourceLifecycleError("derived nodes require at least one parent")  # noqa: TRY003
        if node.node_id in node.parents:
            raise SourceLifecycleError("nodes cannot depend on themselves")  # noqa: TRY003
        for parent in node.parents:
            if parent not in nodes:
                raise SourceLifecycleError(f"unknown parent node {parent!r}")  # noqa: TRY003
        await self._verify_readable(node)
        existing = nodes.get(node.node_id)
        if existing is not None:
            if not existing.matches(node):
                raise SourceLifecycleError(f"node {node.node_id!r} is already bound differently")  # noqa: TRY003
            return
        if node.family != "source":
            node.available = self._derived_availability(node, nodes)
        nodes[node.node_id] = node
        if node.family == "memory" and not node.available:
            await self._set_memory_entry(node, available=False)
        await self._persist()

    def _derived_availability(self, node: _Node, nodes: dict[str, _Node]) -> bool:
        support = [nodes[parent].available for parent in node.parents if parent in nodes]
        if len(support) != len(node.parents):
            return False
        return all(support) if node.policy == "all" else any(support)

    async def _verify_readable(self, node: _Node) -> None:
        from powercontext.builtin.runtime import GetMemoryEntryRequest

        try:
            if node.family == "source":
                await self._runtime.records.for_scope(self.scope_id).get_source(
                    node.reference.source_type, node.reference.source_id
                )
            elif node.family == "memory":
                await self._runtime.memory.for_scope(self.scope_id).get(
                    GetMemoryEntryRequest(citation=node.reference)
                )
            else:
                await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                    node.reference.family, node.reference.artifact_id, node.reference.revision
                )
        except SourceLifecycleError:
            raise
        except Exception as error:  # noqa: BLE001 - every reader failure is a registration failure.
            raise SourceLifecycleError(f"registered reference must be readable: {error}") from error  # noqa: TRY003

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        nodes = await self._load()
        node = nodes.get(event.node_id)
        if node is None:
            raise SourceLifecycleError(f"unknown node {event.node_id!r}")  # noqa: TRY003
        if event.revision != node.revision:
            raise SourceLifecycleError("event revision does not match the registered node")  # noqa: TRY003
        if event.sequence < node.last_sequence:
            raise SourceLifecycleError("event sequence must not decrease")  # noqa: TRY003
        if event.sequence == node.last_sequence:
            if event.available != node.available:
                raise SourceLifecycleError("a replayed sequence cannot change availability")  # noqa: TRY003
            return sorted(entry.node_id for entry in nodes.values() if entry.available)
        node.available = event.available
        node.last_sequence = event.sequence

        changed = {node.node_id: node.available}
        settled = False
        while not settled:
            settled = True
            for candidate in nodes.values():
                if candidate.family == "source":
                    continue
                updated = self._derived_availability(candidate, nodes)
                if updated != candidate.available:
                    candidate.available = updated
                    changed[candidate.node_id] = updated
                    settled = False

        for node_id, available in changed.items():
            entry = nodes[node_id]
            if entry.family == "memory":
                await self._set_memory_entry(entry, available=available)
        await self._persist()
        return sorted(entry.node_id for entry in nodes.values() if entry.available)

    async def _set_memory_entry(self, node: _Node, *, available: bool) -> None:
        application = self._runtime.memory.for_scope(self.scope_id)
        await application._set_entry_activity(node.reference, available=available)  # noqa: SLF001 - runtime seam.

    async def resolve(self, node_id: str, /, *, purpose: str):
        nodes = await self._load()
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError(f"unknown node {node_id!r}")  # noqa: TRY003
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("purpose must be generation or audit")  # noqa: TRY003
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError(f"node {node_id!r} is withdrawn from generation")  # noqa: TRY003
        return node.reference

    async def inactive_topic_keys(self) -> set[tuple[str, str, int]]:
        nodes = await self._load()
        keys: set[tuple[str, str, int]] = set()
        for node in nodes.values():
            if node.family == "topic" and not node.available:
                keys.add((node.reference.family, node.reference.artifact_id, node.reference.revision))
        return keys


class SourceLifecycleApplication:
    """Select the lifecycle service for one scope."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, scope_id)


__all__ = ["SourceLifecycleApplication", "SourceLifecycleError", "SourceLifecycleEvent", "ScopedSourceLifecycle"]
