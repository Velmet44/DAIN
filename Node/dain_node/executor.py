"""Executor interface (S3 skeleton) + deterministic FakeExecutor.

The S3 milestone shaped the `Executor` base (no-op lifecycle hooks) and a
deterministic `FakeExecutor` (sha1-derived token stream) used by tests/compat.
Production generation since S4 is driven by the stage pipeline in `llm.py`
(`StageModel.next_token_logits_full`); this module is kept for its documented
determinism contract and milestone lineage, and is not wired into the agent.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator


class Executor:
    """Base executor: optional no-op lifecycle hooks, subclassed per runtime."""

    name: str = "base"

    async def warmup(self) -> None:
        """Load/validate whatever the executor needs before first use."""

    async def shutdown(self) -> None:
        """Release resources; called once during graceful shutdown."""


class FakeExecutor(Executor):
    """Deterministic stub executor: same prompt + max_tokens → same token stream.

    Tokens are derived from a hash of the prompt so tests can assert exact
    output without any model weights.
    """

    name = "fake"

    async def generate(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        prefix = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
        for i in range(max_tokens):
            yield f"{prefix}-{i}"
