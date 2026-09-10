"""Configurable text-encoder pathways + an on-disk embedding store.

Two backends, selected per *pathway* in ``configs/phase0.yaml -> embedding``:

* ``sentence_transformer`` -- a ``sentence-transformers`` model (default
  ``all-MiniLM-L6-v2``). Normalized sentence embeddings, good for kNN / FAISS.
* ``bert`` -- a raw Hugging Face encoder (default ``bert-base-uncased``) with
  ``cls`` or ``mean`` pooling. This is the NIRT-paper text pathway: the same
  encoder embeds queries *and* LLM profile texts so a query's item parameters
  and a model's ability vector live in one text space (and a cold-start model
  can be placed from its profile alone).

Inference is deterministic (eval mode, ``torch.no_grad``, fixed seeds, no
dropout). Phase 0 does **not** fine-tune anything here.

``EmbeddingStore`` persists ``query_id -> vector`` (or ``model_id -> vector``) as
a float32 ``.npy`` matrix + a parquet id index + a JSON manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..config import Config, require_choice, section
from ..determinism import seed_everything as _seed_everything

_BACKENDS = ("sentence_transformer", "bert")


def _resolve_device(pref: str) -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


@dataclass
class EncoderConfig:
    name: str = "default"                     # pathway name
    backend: str = "sentence_transformer"     # {"sentence_transformer", "bert"}
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True
    max_seq_length: int = 256
    pooling: str = "mean"                     # bert only: {"cls", "mean"}
    device: str = "auto"
    seed: int = 42

    def __post_init__(self):
        require_choice(self.backend, _BACKENDS, field="embedding backend")

    def fingerprint(self) -> str:
        payload = json.dumps(
            {k: v for k, v in self.__dict__.items() if k != "name"}, sort_keys=True
        ).encode()
        return hashlib.sha1(payload).hexdigest()[:12]


class TextEncoder:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self._model = None
        self._tokenizer = None

    # -- model loading -------------------------------------------------
    def _load_sentence_transformer(self):
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(self.cfg.model_name, device=self.device)
        if self.cfg.max_seq_length:
            m.max_seq_length = int(self.cfg.max_seq_length)
        m.eval()
        return m

    def _load_bert(self):
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        mdl = AutoModel.from_pretrained(self.cfg.model_name).to(self.device)
        mdl.eval()
        self._tokenizer = tok
        return mdl

    @property
    def model(self):
        if self._model is None:
            _seed_everything(self.cfg.seed)
            self._model = (
                self._load_bert() if self.cfg.backend == "bert"
                else self._load_sentence_transformer()
            )
        return self._model

    @property
    def dim(self) -> int:
        if self.cfg.backend == "bert":
            return int(self.model.config.hidden_size)
        get = getattr(self.model, "get_embedding_dimension", None) or \
            getattr(self.model, "get_sentence_embedding_dimension")
        return int(get())

    # -- encoding ----------------------------------------------------
    def iter_encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        _start_offset: int = 0,
    ):
        """Yield ``(start_row, batch_vectors)`` per batch.

        Streaming -- never accumulates the full matrix in memory. ``start_row``
        is the absolute row index (``_start_offset`` + local offset), so callers
        can resume a partial build.
        """
        _seed_everything(self.cfg.seed)
        bs = int(batch_size or self.cfg.batch_size)
        texts = [("" if t is None else str(t)) for t in texts]
        rng = range(0, len(texts), bs)
        if show_progress:
            try:
                from tqdm import tqdm

                rng = tqdm(rng, desc=f"{self.cfg.backend}:{self.cfg.name}",
                           initial=0, total=len(rng))
            except Exception:
                pass
        for i in rng:
            batch = list(texts[i : i + bs])
            if self.cfg.backend == "bert":
                vec = self._encode_bert_batch(batch)
            else:
                vec = self.model.encode(
                    batch,
                    batch_size=bs,
                    normalize_embeddings=self.cfg.normalize,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            yield _start_offset + i, np.ascontiguousarray(vec, dtype=np.float32)

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [v for _, v in self.iter_encode(texts, show_progress=show_progress)]
        return np.vstack(parts) if parts else np.zeros((0, self.dim), dtype=np.float32)

    def _encode_bert_batch(self, batch: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=int(self.cfg.max_seq_length or 256),
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state          # (b, t, h)
            if self.cfg.pooling == "cls":
                vec = hidden[:, 0]
            else:  # mean pooling over non-pad tokens
                mask = enc["attention_mask"].unsqueeze(-1).type_as(hidden)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            if self.cfg.normalize:
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            return vec.cpu().float().numpy()


# --------------------------------------------------------------------------- #
# config -> encoder                                                           #
# --------------------------------------------------------------------------- #
def _pathway_dict(cfg: Config) -> dict:
    return section(cfg, "embedding")


def available_pathways(cfg: Config) -> list[str]:
    e = _pathway_dict(cfg)
    if isinstance(e.get("pathways"), dict):
        return list(e["pathways"])
    return ["default"]


def load_encoder(cfg: Config, pathway: Optional[str] = None) -> TextEncoder:
    e = _pathway_dict(cfg)
    seed = int(cfg.get("seed", 42))
    shared = {
        "batch_size": int(e.get("batch_size", 64)),
        "device": e.get("device", "auto"),
        "seed": seed,
    }
    pathways = e.get("pathways")
    if isinstance(pathways, dict):
        name = pathway or next(iter(pathways))
        if name not in pathways:
            raise KeyError(f"embedding pathway {name!r} not in config: {list(pathways)}")
        p = pathways[name]
        return TextEncoder(EncoderConfig(
            name=name,
            backend=p.get("backend", "sentence_transformer"),
            model_name=p.get("model_name", EncoderConfig.model_name),
            normalize=bool(p.get("normalize", True)),
            max_seq_length=int(p.get("max_seq_length", 256)),
            pooling=p.get("pooling", "mean"),
            **shared,
        ))
    # flat / legacy form
    return TextEncoder(EncoderConfig(
        name="default",
        backend=e.get("backend", "sentence_transformer"),
        model_name=e.get("model_name", EncoderConfig.model_name),
        normalize=bool(e.get("normalize", True)),
        max_seq_length=int(e.get("max_seq_length", 256)),
        pooling=e.get("pooling", "mean"),
        **shared,
    ))


# --------------------------------------------------------------------------- #
# store                                                                       #
# --------------------------------------------------------------------------- #
def _ids_fingerprint(ids: Sequence[str]) -> str:
    h = hashlib.sha1()
    h.update(str(len(ids)).encode())
    for i in ids:
        h.update(b"\x00")
        h.update(str(i).encode("utf-8"))
    return h.hexdigest()[:16]


class EmbeddingStore:
    """`id -> vector` persisted as vectors.npy + ids.parquet + manifest.json.

    ``id_field`` is ``query_id`` or ``model_id`` depending on what was encoded.
    Large stores are memory-mapped on load (read-only access pattern).
    """

    MATRIX = "vectors.npy"
    IDS = "ids.parquet"
    MANIFEST = "manifest.json"
    PROGRESS = "_progress.json"
    # filenames used before the streaming rewrite -- still readable
    _LEGACY = {
        "vectors.npy": "embeddings.npy",
        "ids.parquet": "embedding_ids.parquet",
        "manifest.json": "embeddings_manifest.json",
    }

    def __init__(self, ids: Sequence[str], matrix: np.ndarray, manifest: dict,
                 id_field: str = "query_id"):
        assert len(ids) == matrix.shape[0], "id / row count mismatch"
        self.ids = list(ids)
        self.matrix = matrix if matrix.dtype == np.float32 else matrix.astype(np.float32)
        self.manifest = manifest
        self.id_field = manifest.get("id_field", id_field)
        self._index = {qid: i for i, qid in enumerate(self.ids)}

    # -- construction --------------------------------------------------
    @classmethod
    def _manifest(cls, encoder: TextEncoder, ids: Sequence[str], id_field: str,
                  dim: int) -> dict:
        return {
            "pathway": encoder.cfg.name,
            "backend": encoder.cfg.backend,
            "model_name": encoder.cfg.model_name,
            "pooling": encoder.cfg.pooling if encoder.cfg.backend == "bert" else None,
            "dim": int(dim),
            "normalized": encoder.cfg.normalize,
            "count": len(ids),
            "id_field": id_field,
            "config_fingerprint": encoder.cfg.fingerprint(),
            "ids_fingerprint": _ids_fingerprint(ids),
            "device": encoder.device,
            "complete": True,
        }

    @classmethod
    def build(cls, ids: Sequence[str], texts: Sequence[str], encoder: TextEncoder,
              id_field: str = "query_id", show_progress: bool = True) -> "EmbeddingStore":
        """In-memory build (fine for small id sets and tests).

        For large corpora on CPU use :func:`build_store` -- it streams to disk
        and is resumable.
        """
        matrix = encoder.encode(texts, show_progress=show_progress)
        dim = int(matrix.shape[1]) if matrix.size else encoder.dim
        return cls(ids, matrix, cls._manifest(encoder, ids, id_field, dim), id_field)

    # -- io ----------------------------------------------------------
    @classmethod
    def _resolve(cls, d: Path, canonical: str) -> Path:
        p = d / canonical
        if p.exists():
            return p
        legacy = d / cls._LEGACY.get(canonical, canonical)
        return legacy if legacy.exists() else p

    def save(self, directory: str | os.PathLike) -> Path:
        import pandas as pd

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / self.MATRIX, np.ascontiguousarray(self.matrix, dtype=np.float32))
        pd.DataFrame({self.id_field: self.ids, "row": range(len(self.ids))}).to_parquet(
            d / self.IDS, index=False
        )
        (d / self.MANIFEST).write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | os.PathLike, mmap: bool = True) -> "EmbeddingStore":
        import pandas as pd

        d = Path(directory)
        manifest = json.loads(cls._resolve(d, cls.MANIFEST).read_text(encoding="utf-8"))
        if manifest.get("complete") is False:
            raise IncompleteEmbeddingStore(
                f"{d} is a partial build (rows_done < count); resume it with build_store()"
            )
        matrix = np.load(cls._resolve(d, cls.MATRIX), mmap_mode="r" if mmap else None)
        id_field = manifest.get("id_field", "query_id")
        ids_df = pd.read_parquet(cls._resolve(d, cls.IDS)).sort_values("row")
        return cls(ids_df[id_field].tolist(), matrix, manifest, id_field=id_field)

    @classmethod
    def exists(cls, directory: str | os.PathLike) -> bool:
        d = Path(directory)
        m = cls._resolve(d, cls.MANIFEST)
        if not m.exists():
            return False
        try:
            return json.loads(m.read_text(encoding="utf-8")).get("complete", True)
        except Exception:
            return False

    # -- access ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def __contains__(self, _id: str) -> bool:
        return _id in self._index

    def row_of(self, _id: str) -> int:
        return self._index[_id]

    def rows_of(self, ids: Sequence[str]) -> np.ndarray:
        return np.fromiter((self._index[i] for i in ids), dtype=np.int64, count=len(ids))

    def get(self, _id: str) -> np.ndarray:
        return np.asarray(self.matrix[self._index[_id]])

    def gather(self, ids: Sequence[str]) -> np.ndarray:
        """Return an ``(len(ids), dim)`` array for the given ids (copies)."""
        return np.asarray(self.matrix[self.rows_of(ids)])

    def subset(self, ids: Iterable[str]) -> "EmbeddingStore":
        wanted = [q for q in ids if q in self._index]
        rows = self.rows_of(wanted)
        return EmbeddingStore(wanted, np.asarray(self.matrix[rows]),
                              dict(self.manifest, count=len(wanted)), self.id_field)

    def as_frame(self):
        import pandas as pd

        return pd.DataFrame(np.asarray(self.matrix), index=self.ids)

    def close(self) -> None:
        """Release the memory-map file handle (matters on Windows before a
        rebuild / delete). The store is unusable afterwards."""
        mm = getattr(self.matrix, "_mmap", None)
        if mm is not None:
            mm.close()
        self.matrix = None  # type: ignore[assignment]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class IncompleteEmbeddingStore(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# resumable streaming build                                                   #
# --------------------------------------------------------------------------- #
def build_store(
    directory: str | os.PathLike,
    ids: Sequence[str],
    texts: Sequence[str],
    encoder: TextEncoder,
    *,
    id_field: str = "query_id",
    flush_every: int = 25,
    resume: bool = True,
    force: bool = False,
    show_progress: bool = True,
    manifest_extra: Optional[dict] = None,
) -> EmbeddingStore:
    """Encode ``texts`` into ``directory`` incrementally and resumably.

    Layout: ``vectors.npy`` (float32 memmap, shape ``(n, dim)``), ``ids.parquet``,
    ``manifest.json``, and while in progress ``_progress.json``
    (``rows_done`` + fingerprints). A killed run (OOM, power loss) leaves a
    valid partial ``vectors.npy``; calling this again with the same ids/config
    picks up from ``rows_done``.
    """
    import pandas as pd

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    n = len(ids)
    dim = encoder.dim
    ids = [str(i) for i in ids]
    id_fp = _ids_fingerprint(ids)
    cfg_fp = encoder.cfg.fingerprint()

    mpath = d / EmbeddingStore.MANIFEST
    ppath = d / EmbeddingStore.PROGRESS
    vpath = d / EmbeddingStore.MATRIX

    # already finished?
    if not force and EmbeddingStore.exists(d):
        man = json.loads(EmbeddingStore._resolve(d, EmbeddingStore.MANIFEST).read_text("utf-8"))
        if man.get("ids_fingerprint") in (id_fp, None) and man.get("count") == n:
            return EmbeddingStore.load(d)

    rows_done = 0
    if resume and not force and ppath.exists() and vpath.exists():
        prog = json.loads(ppath.read_text("utf-8"))
        if (prog.get("ids_fingerprint") == id_fp
                and prog.get("config_fingerprint") == cfg_fp
                and prog.get("dim") == dim):
            rows_done = int(prog.get("rows_done", 0))
            if show_progress:
                print(f"[build_store] resuming {d.name} at row {rows_done}/{n}")
        elif show_progress:
            print(f"[build_store] {d.name}: fingerprint mismatch, restarting")

    mode = "r+" if (rows_done > 0 and vpath.exists()) else "w+"
    mm = np.lib.format.open_memmap(vpath, mode=mode, dtype=np.float32, shape=(n, dim))

    # ids index is stable regardless of progress
    pd.DataFrame({id_field: ids, "row": range(n)}).to_parquet(d / EmbeddingStore.IDS, index=False)

    def _write_progress(done: int) -> None:
        mm.flush()
        ppath.write_text(json.dumps({
            "rows_done": int(done), "count": n, "dim": dim,
            "batch_size": encoder.cfg.batch_size,
            "ids_fingerprint": id_fp, "config_fingerprint": cfg_fp,
        }), encoding="utf-8")

    since_flush = 0
    last = rows_done
    for start, vec in encoder.iter_encode(
        texts[rows_done:], show_progress=show_progress, _start_offset=rows_done
    ):
        mm[start : start + vec.shape[0]] = vec
        last = start + vec.shape[0]
        since_flush += 1
        if since_flush >= flush_every:
            _write_progress(last)
            since_flush = 0
    _write_progress(last)

    if last < n:  # interrupted mid-stream without exception (shouldn't normally happen)
        raise IncompleteEmbeddingStore(f"{d}: only {last}/{n} rows encoded")

    mm.flush()
    del mm
    manifest = EmbeddingStore._manifest(encoder, ids, id_field, dim)
    if manifest_extra:
        manifest.update(manifest_extra)
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ppath.unlink(missing_ok=True)
    return EmbeddingStore.load(d)


def default_store_dir(cfg: Config, kind: str, pathway: str) -> Path:
    """`<embedding.cache_dir>/<kind>__<pathway>` (kind e.g. 'query', 'model_profile')."""
    base = cfg.resolve(cfg.get("embedding.cache_dir", "data/processed/embeddings"))
    return base / f"{kind}__{pathway}"
