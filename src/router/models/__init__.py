"""Artifact identity for a fitted router (:mod:`router.models.artifacts`).

Not the inference model classes themselves -- those already exist under
:mod:`router.nirt` / :mod:`router.routing`, fitted by :mod:`training` and
loaded by :func:`router.nirt.checkpoint.load_run`, matching this package's
docstring on ``router.routing.base.Router``. This package exists only to give
that lifecycle a typed identity (:class:`~router.models.artifacts.RouterModelArtifact`)
that a trace can resolve back to an exact checkpoint."""
