"""Embedding backends.

Everything that needs vectors depends on the `Embedder` protocol rather than on
sentence-transformers directly, so tests can swap in a tiny deterministic
embedder and never download or load a model.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE was trained to embed *queries* with this instruction and passages without
# it. Whether it helps on bge-small-en-v1.5 is measured, not assumed:
# see experiments/retrieval_eval.py.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    @property
    def max_tokens(self) -> int:
        """Longest input the model reads. Anything beyond this is truncated."""
        ...

    def count_tokens(self, text: str) -> int: ...

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        """L2-normalised vectors, shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """L2-normalised vector, shape (dim,)."""
        ...


class SentenceTransformerEmbedder:
    """`Embedder` backed by a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        query_instruction: str = "",
    ) -> None:
        # Imported lazily: importing sentence-transformers pulls in torch, which
        # makes `import pr_agent.search` slow for code that never embeds.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.query_instruction = query_instruction

    @property
    def max_tokens(self) -> int:
        limit = self._model.max_seq_length
        if limit is None:
            raise RuntimeError("The embedding model does not declare max_seq_length")
        return int(limit)

    def count_tokens(self, text: str) -> int:
        return len(
            self._model.tokenizer.encode(
                text, add_special_tokens=True, truncation=False
            )
        )

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._model.encode(
            [self.query_instruction + text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
