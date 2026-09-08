"""Executor interface (S3 skeleton) + deterministic FakeExecutor.

Real executors (HFExecutor for TinyLlama/Qwen, spec §18) arrive in S4; nothing
dispatches jobs yet. The interface is shaped for the pipeline: `generate`
streams tokens for one job on this node.
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
