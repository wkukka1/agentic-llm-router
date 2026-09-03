"""Cost-aware LLM router research package.

* ``router.data`` -- the Foundation data pipeline: ingest heterogeneous
  benchmark sources into one canonical, leakage-free dataset, exposed through
  the ``router.data.phase1`` facade.
* ``router.embeddings`` -- deterministic text-encoder pathways + on-disk store.
* ``router.models`` -- LLM profile text (for the cold-start ``theta_m`` init).
* ``router.nirt`` -- the NIRT / IRT-Router response model, its trainer, the
  classical-IRT baseline, and routing / cold-start / OOD evaluation.
"""

__version__ = "0.1.0"
