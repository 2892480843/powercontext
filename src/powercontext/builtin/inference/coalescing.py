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

"""Embedding request coalescing, bounded batching, and a shared LRU vector cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage

_DEFAULT_MAX_BATCH_SIZE = 64
_DEFAULT_COALESCE_MS = 5.0
_DEFAULT_CACHE_SIZE = 1024


class _Flight:
    """One dispatched batch shared by every waiter interested in its texts."""

    __slots__ = ("batch_id", "future")

    def __init__(self, batch_id: int, future: asyncio.Future) -> None:
        self.batch_id = batch_id
        self.future = future


class CoalescingEmbeddingModel:
    """Deduplicate embedding traffic across concurrent callers.

    Identical texts inside one call, inside the coalescing window, and inside
    still-running requests share one underlying batch. Successful vectors are
    kept in a bounded LRU cache; failures are never cached and can be retried.
    Cancelling one waiter never cancels the shared underlying work.
    """

    def __init__(
        self,
        model,
        *,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = _DEFAULT_COALESCE_MS,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(coalesce_ms, (int, float)) or isinstance(coalesce_ms, bool) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._pending: dict[str, asyncio.Future] = {}
        self._inflight: dict[str, _Flight] = {}
        self._batch_of: dict[str, int] = {}
        self._batch_usage: dict[int, InferenceUsage] = {}
        self._dispatcher: asyncio.Task | None = None
        self._batch_seq = 0

    @property
    def profile(self):  # noqa: ANN201 - mirrors the wrapped model's profile type.
        return self._model.profile

    async def embed(self, texts) -> EmbeddingResult:
        ordered = tuple(texts)
        if not ordered:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        vector_by_text: dict[str, object] = {}
        awaited: dict[str, asyncio.Future] = {}
        for text in dict.fromkeys(ordered):
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                vector_by_text[text] = cached
                continue
            flight = self._inflight.get(text)
            if flight is not None:
                awaited[text] = flight.future
                continue
            future = self._pending.get(text)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._pending[text] = future
                self._ensure_dispatcher()
            awaited[text] = future

        if awaited:
            # asyncio.wait never cancels the shared futures when this caller is cancelled.
            pending = set(awaited.values())
            while pending:
                _, pending = await asyncio.wait(pending)
            for text, future in awaited.items():
                vector_by_text[text] = future.result()

        batch_ids = {self._batch_of[text] for text in awaited}
        vectors = tuple(vector_by_text[text] for text in ordered)
        usage = self._summarize_usage(batch_ids)
        return EmbeddingResult(vectors=vectors, usage=usage)

    def _summarize_usage(self, batch_ids: set[int]) -> InferenceUsage:
        requests = 0
        input_tokens = 0
        output_tokens = 0
        for batch_id in batch_ids:
            usage = self._batch_usage[batch_id]
            requests += usage.requests
            input_tokens += usage.input_tokens or 0
            output_tokens += usage.output_tokens or 0
        return InferenceUsage(
            requests=requests,
            input_tokens=input_tokens if requests else None,
            output_tokens=output_tokens if requests else None,
        )

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.get_running_loop().create_task(self._dispatch_pending())

    async def _dispatch_pending(self) -> None:
        if self._coalesce_seconds:
            await asyncio.sleep(self._coalesce_seconds)
        while self._pending:
            batch = tuple(list(self._pending)[: self._max_batch_size])
            self._batch_seq += 1
            batch_id = self._batch_seq
            futures = {text: self._pending.pop(text) for text in batch}
            for text, future in futures.items():
                self._inflight[text] = _Flight(batch_id, future)
                self._batch_of[text] = batch_id
            await self._run_batch(batch_id, batch, futures)
        self._dispatcher = None

    async def _run_batch(
        self,
        batch_id: int,
        batch: tuple[str, ...],
        futures: dict[str, asyncio.Future],
    ) -> None:
        try:
            result = await self._model.embed(batch)
            if len(result.vectors) != len(batch):
                raise ValueError("underlying embedding vector count does not match the batch size")  # noqa: TRY003
            self._batch_usage[batch_id] = result.usage
            vector_by_text = dict(zip(batch, result.vectors, strict=True))
            for text, vector in vector_by_text.items():
                if self._cache_size:
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
        except BaseException as error:  # noqa: BLE001 - forwarded to every waiter below.
            for future in futures.values():
                if not future.done():
                    future.set_exception(error)
        else:
            for text, future in futures.items():
                if not future.done():
                    future.set_result(vector_by_text[text])
        finally:
            for text in futures:
                if self._inflight.get(text) is not None and self._inflight[text].batch_id == batch_id:
                    del self._inflight[text]

    def __getattr__(self, name: str):  # pragma: no cover - defensive passthrough.
        return getattr(self._model, name)


__all__ = ["CoalescingEmbeddingModel"]
