"""Cost-aware LLM router -- serving package.

Everything needed to score a request with an already-fitted model. Must never
import ``training`` (offline data pipeline + fitting) or ``evaluation``
(oracle-relative / label-needing evaluation) -- see ``docs/architecture.md``.

* ``router.data`` -- the serving-side batch container, ``NIRTDataset``.
* ``router.embeddings`` -- deterministic text-encoder pathways + on-disk store.
* ``router.nirt`` -- the NIRT / IRT-Router response model, checkpoint loading,
  and label-free inference (:mod:`router.nirt.predict`).
* ``router.routing`` -- the modular :class:`~router.routing.base.Router`
  interface: one contract (``predict_scores`` -> ``route``) that NIRT, k-NN,
  MLP and future strategies all implement. See ``docs/routing_interface.md``.
* ``router.agentic`` -- the live triage/decompose/orchestrate layer that
  wraps a ``Router`` plus model backends to actually answer a prompt.
"""

__version__ = "0.1.0"
