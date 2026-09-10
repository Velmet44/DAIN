"""Real-model tokenizer (S18): wraps a fast HF `tokenizer.json` served by the
model store. Composition mirrors ByteTokenizer from dain_node.byte_tokenizer,
so JobHandler treats both uniformly (`encode`/`decode`/`feed`). `feed(id)` is
the streaming hook: HF fast tokenizers decode single ids to the best fragment
possible (byte-level subpieces concatenate across tokens).
"""

from __future__ import annotations


class HFTokenizer:
    def __init__(self, path: str) -> None:
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def feed(self, token_id: int) -> str:
        return self._tok.decode([token_id], skip_special_tokens=True)
