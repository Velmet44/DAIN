"""S3: FakeExecutor determinism (the contract S4's pipeline will rely on)."""

import asyncio

from dain_node.executor import FakeExecutor


def collect(executor: FakeExecutor, prompt: str, max_tokens: int) -> list[str]:
    async def run() -> list[str]:
        await executor.warmup()
        tokens = [token async for token in executor.generate(prompt, max_tokens)]
        await executor.shutdown()
        return tokens

    return asyncio.run(run())


def test_generate_is_deterministic() -> None:
    first = collect(FakeExecutor(), "hello world", 5)
    second = collect(FakeExecutor(), "hello world", 5)
    assert first == second
    assert len(first) == 5


def test_generate_differs_per_prompt() -> None:
    a = collect(FakeExecutor(), "alpha", 3)
    b = collect(FakeExecutor(), "beta", 3)
    assert a != b


def test_generate_respects_max_tokens() -> None:
    assert len(collect(FakeExecutor(), "x", 7)) == 7
    assert collect(FakeExecutor(), "x", 0) == []
