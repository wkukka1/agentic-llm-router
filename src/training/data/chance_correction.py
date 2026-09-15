"""Multiple-choice chance correction (data-level).

The project explicitly rejects a 3PL guessing parameter and any item-level
guessing term. Instead we apply a documented, configurable *data-level*
transform to multiple-choice observations, keeping the original score intact.

Formula (``method: normalized``)
--------------------------------
For an observation with ``n`` answer choices, chance level is ``c = 1 / n``.
The corrected score rescales the range ``[c, 1]`` onto ``[0, 1]``::

    corrected = (score - c) / (1 - c)

optionally clipped to ``[0, 1]`` (``chance_correction.clip``). This is the
standard "corrected for guessing" / "adjusted accuracy" rescaling: a model at
pure chance maps to 0 *on average*, a perfect model maps to 1, and a graded
score in between is shifted down by the expected contribution of random guessing.

Caveat (``clip: true``): the transform is applied per observation, so on a
*binary* 0/1 score it is the identity -- ``0 -> -c/(1-c) -> clipped 0`` and
``1 -> 1``. For binary MC items ``score_effective == score`` and the adjustment
only has an effect on graded scores. Use ``clip: false`` to keep the negative
corrected targets (the per-observation mean is then chance-corrected, but the
target leaves ``[0, 1]`` and is no longer a valid Bernoulli / Beta label).

Interpretation: ``corrected`` estimates the probability that the model *knew*
the answer, under the crude assumption that non-knowledge yields a uniform
random guess. It is a per-observation adjustment, not a per-item parameter, so
it introduces no new latent quantities for Phase 1 to fit.

What we do NOT do
-----------------
* We do not touch non-multiple-choice observations.
* We do not correct when ``n_choices`` is unknown -- the original score is kept
  and a warning is emitted (``warn_on_missing_choices``).
* We never overwrite ``score``. Outputs go to new columns.

Per-model bias
--------------
``estimate_model_bias`` additionally reports, per model, the mean
(score - chance) on MC items where the model's graded score equals chance-ish
low values. This is surfaced for inspection / optional use as a per-model
offset in Phase 1; Phase 0 does not bake it into any score.

Output columns added by :func:`apply_chance_correction`
------------------------------------------------------
``corrected_score``    float, null where no correction applied
``chance_level``       float, ``1/n_choices`` where known
``chance_corrected``   bool, whether a correction was applied
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from router.config import Config, section

CORRECTION_COLUMNS = ["corrected_score", "chance_level", "chance_corrected"]


def normalized_correction(score: float, n_choices: int, clip: bool = True) -> float:
    c = 1.0 / float(n_choices)
    corrected = (score - c) / (1.0 - c)
    if clip:
        corrected = min(1.0, max(0.0, corrected))
    return corrected


def apply_chance_correction(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cc = section(cfg, "chance_correction")
    method = cc.get("method", "normalized")
    clip = bool(cc.get("clip", True))
    warn_missing = bool(cc.get("warn_on_missing_choices", True))

    out = df.copy()
    out["corrected_score"] = pd.NA
    out["chance_level"] = pd.NA
    out["chance_corrected"] = False

    if method == "none":
        return out

    is_mc = out["is_multiple_choice"].fillna(False).astype(bool)
    has_n = out["n_choices"].notna()

    correctable = is_mc & has_n
    missing = is_mc & ~has_n
    n_missing = int(missing.sum())
    if n_missing and warn_missing:
        tasks = sorted(out.loc[missing, "dataset"].dropna().unique().tolist())
        warnings.warn(
            f"chance correction: {n_missing} multiple-choice observations have "
            f"no n_choices; leaving them unchanged. Tasks: {tasks[:20]}"
        )

    if correctable.any():
        n = out.loc[correctable, "n_choices"].astype(float).to_numpy()
        s = out.loc[correctable, "score"].astype(float).to_numpy()
        c = 1.0 / n
        corrected = (s - c) / (1.0 - c)
        if clip:
            corrected = np.clip(corrected, 0.0, 1.0)
        out.loc[correctable, "corrected_score"] = corrected
        out.loc[correctable, "chance_level"] = c
        out.loc[correctable, "chance_corrected"] = True

    out["chance_level"] = pd.to_numeric(out["chance_level"], errors="coerce")
    out["corrected_score"] = pd.to_numeric(out["corrected_score"], errors="coerce")
    return out


def estimate_model_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model diagnostic on multiple-choice items.

    Returns one row per model with:
      mc_n            number of MC observations
      mc_mean_score   mean raw MC score
      mc_mean_chance  mean chance level over those items
      mc_excess       mc_mean_score - mc_mean_chance  (crude skill-above-chance)
    """
    mc = df[df["is_multiple_choice"].fillna(False).astype(bool) & df["n_choices"].notna()]
    if mc.empty:
        return pd.DataFrame(
            columns=["model_id", "mc_n", "mc_mean_score", "mc_mean_chance", "mc_excess"]
        )
    g = mc.assign(_chance=1.0 / mc["n_choices"].astype(float)).groupby("model_id")
    res = g.agg(
        mc_n=("score", "size"),
        mc_mean_score=("score", "mean"),
        mc_mean_chance=("_chance", "mean"),
    ).reset_index()
    res["mc_excess"] = res["mc_mean_score"] - res["mc_mean_chance"]
    return res
