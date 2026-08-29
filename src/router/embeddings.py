"""Frozen sentence-embedding extraction with an on-disk cache.

Several experiments reuse the same encoder over the same rows, so embeddings are
computed once per (model, pooling, max_length, text) and memory-mapped back on
subsequent runs. Without this, sweeping heads over a fixed encoder pays the
encoder cost every time.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/processed/embeddings")


def resolve_device(requested: str | None = None) -> str:
    """Pick the best available torch device, honouring an explicit request."""
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pool(hidden: torch.Tensor, mask: torch.Tensor, strategy: str) -> torch.Tensor:
    if strategy == "cls":
        return hidden[:, 0]
    if strategy == "mean":
        expanded = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"unknown pooling {strategy!r}; expected 'cls' or 'mean'")


class EmbeddingEncoder:
    """Wraps a frozen HF encoder as a batched ``list[str] -> np.ndarray``."""

    #: Models whose training used a fixed instruction prefix. Embedding raw
    #: text into these is a silent quality loss -- the encoder was never shown
    #: bare inputs, so the vectors land in a slightly different region of the
    #: space than anything it was optimised for.
    DEFAULT_PREFIXES: dict[str, str] = {
        "intfloat/e5": "query: ",
        "intfloat/multilingual-e5": "query: ",
    }

    def __init__(
        self,
        model_name: str,
        *,
        prefix: str | None = None,
        pooling: str = "mean",
        max_length: int = 256,
        batch_size: int = 64,
        normalize: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        if prefix is None:
            prefix = next((v for k, v in self.DEFAULT_PREFIXES.items()
                           if model_name.startswith(k)), "")
        self.prefix = prefix
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = resolve_device(device)
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("loading encoder %s onto %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @property
    def signature(self) -> str:
        """Identifies the cache namespace for this encoder configuration."""
        raw = f"{self.model_name}|{self.prefix}|{self.pooling}|{self.max_length}|{self.normalize}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self.prefix + t for t in texts[start : start + self.batch_size]]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self._model(**enc).last_hidden_state
            pooled = _pool(hidden, enc["attention_mask"], self.pooling)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)

    def encode_cached(self, texts: list[str], *, tag: str, cache_dir: Path = CACHE_DIR) -> np.ndarray:
        """Encode with a cache keyed by encoder signature, tag and text content.

        ``tag`` names the row set (e.g. ``full_prompt/train``). The content hash
        guards against a stale cache if the underlying split is rebuilt.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        content = hashlib.sha1("\x00".join(texts).encode()).hexdigest()[:12]
        safe_tag = tag.replace("/", "__")
        path = cache_dir / f"{self.signature}__{safe_tag}__{content}.npy"

        if path.exists():
            log.info("embedding cache hit: %s", path.name)
            return np.load(path)

        vectors = self.encode(texts)
        np.save(path, vectors)
        log.info("embedding cache write: %s %s", path.name, vectors.shape)
        return vectors
