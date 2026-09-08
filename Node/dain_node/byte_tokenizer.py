"""Byte-level tokenizer for the dev model (vocab 256).

Byte 0 is reserved as EOS; NUL bytes in text are mapped to 1 so prompts can
never terminate generation accidentally.
"""

from __future__ import annotations


class ByteTokenizer:
    def __init__(self, vocab_size: int = 256, eos_token_id: int = 0) -> None:
        if vocab_size != 256:
            raise ValueError("ByteTokenizer requires a 256-symbol vocabulary")
        self.eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        # Byte 0 is EOS; map NUL to 1 so arbitrary prompts are safe.
        return [b if b != 0 else 1 for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        payload = bytes(i & 0xFF for i in ids if i != self.eos_token_id)
        return payload.decode("utf-8", errors="replace")
