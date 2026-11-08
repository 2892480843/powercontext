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

"""An HTTP scoring adapter that reranks Memory candidates across request batches."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

_DEFAULT_BATCH_SIZE = 32
_DEFAULT_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score coarse Memory candidates through a batched HTTP service.

    Candidates are split into batches of at most ``batch_size`` entries, every
    global rank is sent exactly once, and the final selection is one global
    stable ordering over the complete candidate pool: score descending, then
    original rank ascending. Identity is never rewritten — the decision only
    references original ranks.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        policy_id: str = _DEFAULT_POLICY_ID,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003

        scored: list[tuple[int, float]] = []
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = tuple(
                (rank, candidate.text)
                for rank, candidate in enumerate(candidates[start : start + self._batch_size], start=start + 1)
            )
            response = await self._client.post(
                self._endpoint,
                json={"query": query, "candidates": [{"rank": rank, "text": text} for rank, text in batch]},
            )
            response.raise_for_status()
            scored.extend(self._parse_scores(response, batch))
            requests += 1

        scored.sort(key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _score in scored[:limit]),
            usage=InferenceUsage(requests=requests),
        )

    def _parse_scores(self, response: httpx.Response, batch: tuple[tuple[int, str], ...]) -> list[tuple[int, float]]:
        try:
            payload = response.json()
        except ValueError as error:
            raise ValueError("rerank service returned invalid JSON") from error  # noqa: TRY003
        if not isinstance(payload, dict) or set(payload) != {"scores"}:
            raise ValueError("rerank service response must contain exactly the scores field")  # noqa: TRY003
        scores = payload["scores"]
        if not isinstance(scores, list):
            raise ValueError("rerank service scores must be a list")  # noqa: TRY003
        expected = {rank for rank, _text in batch}
        observed: dict[int, float] = {}
        for item in scores:
            if not isinstance(item, dict) or set(item) != {"rank", "score"}:
                raise ValueError("each score record must contain exactly rank and score")  # noqa: TRY003
            rank, score = item["rank"], item["score"]
            if type(rank) is not int or rank not in expected:
                raise ValueError("score ranks must cover this batch exactly once")  # noqa: TRY003
            if rank in observed:
                raise ValueError("score ranks must cover this batch exactly once")  # noqa: TRY003
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("scores must be finite numbers")  # noqa: TRY003
            observed[rank] = float(score)
        if set(observed) != expected:
            raise ValueError("score ranks must cover this batch exactly once")  # noqa: TRY003
        return list(observed.items())


__all__ = ["HTTPMemoryReranker"]
