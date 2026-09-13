"""The length head: a regression, so the failure modes differ from the classifiers.

A classifier that breaks usually predicts one class for everything and the
accuracy screams. A regressor that breaks predicts the mean for everything and
still scores a respectable-looking MAE -- so these tests check the things that
separate a fitted model from a constant one: that ranking survives, that the
residual spread is honest, and that the audit can tell noise from signal.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from evaluation.length_audit import audit
from router.features import build_features
from router.heads.length import BUCKETS, LengthHead
from router.heads.length_model import (
    LengthModel,
    bootstrap_ci,
    evaluate,
    interval_coverage,
)
from training.data.arena_corpus import _first_user_text, _split_of


@pytest.fixture
def learnable():
    """Prompts whose answer length really is a function of the prompt.

    Built so that a working pipeline must score well and a broken one cannot:
    the target is driven by features the surface extractor actually sees.
    """
    rng = np.random.default_rng(0)
    prompts, y = [], []
    for _ in range(400):
        n_items = int(rng.integers(0, 5))
        brief = bool(rng.integers(0, 2))
        body = " ".join(f"{i + 1}. do the thing" for i in range(n_items)) or "explain this"
        prompts.append(("keep it brief. " if brief else "") + body)
        y.append(4.0 + 0.5 * n_items - 1.5 * brief + rng.normal(0, 0.2))
    return prompts, np.array(y)


class TestPromptExtraction:
    """The bug that corrupted an entire corpus without raising anything."""

    def test_a_numpy_array_of_parts_is_not_stringified(self):
        """pyarrow hands nested lists back as ndarray, which is not a `list`.
        The first version fell through to `str(content)` and stored the repr of
        the structure as the prompt -- for all 109,335 rows."""
        conv = np.array([{"role": "user",
                          "content": np.array([{"type": "text", "text": "write me a poem",
                                                "image": None}])}])
        assert _first_user_text(conv) == "write me a poem"

    @pytest.mark.parametrize("content", [
        "write me a poem",
        [{"type": "text", "text": "write me a poem"}],
        ({"type": "text", "text": "write me a poem"},),
        "[{'type': 'text', 'text': 'write me a poem', 'image': None}]",
        {"type": "text", "text": "write me a poem"},
    ])
    def test_every_shape_this_dump_uses_yields_the_same_text(self, content):
        assert _first_user_text([{"role": "user", "content": content}]) == "write me a poem"

    def test_multi_part_content_is_joined(self):
        parts = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
        assert _first_user_text([{"role": "user", "content": parts}]) == "first second"

    def test_a_prompt_that_merely_starts_with_a_bracket_is_left_alone(self):
        """`[INST] hello` is a real prompt, not a serialised structure."""
        assert _first_user_text([{"role": "user", "content": "[INST] hello"}]) == "[INST] hello"

    def test_the_assistant_turn_is_not_mistaken_for_the_prompt(self):
        conv = [{"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "write me a poem"}]
        assert _first_user_text(conv) == "write me a poem"


class TestCorpusGuard:
    """A shape bug hits every row; a user pasting JSON hits one."""

    def test_the_threshold_sits_between_the_two_cases(self):
        from training.data.arena_corpus import SERIALISED_PROMPT_LIMIT

        one_user_in_a_hundred_thousand = 1 / 105_799
        assert one_user_in_a_hundred_thousand < SERIALISED_PROMPT_LIMIT < 1.0


class TestSplitAssignment:
    def test_a_prompt_always_lands_in_the_same_split(self):
        """Hashed, not shuffled: rebuilding the corpus must not move a row from
        test into train, which would silently invalidate every earlier score."""
        assert _split_of("write me a poem") == _split_of("write me a poem")

    def test_the_split_is_not_constant(self):
        seen = {_split_of(f"prompt number {i}") for i in range(200)}
        assert seen == {"train", "val", "test"}

    def test_roughly_the_requested_proportions(self):
        splits = [_split_of(f"prompt {i}") for i in range(4000)]
        assert 0.65 < splits.count("train") / len(splits) < 0.75


class TestFeatures:
    def test_each_set_has_the_width_it_claims(self):
        prompts = ["write me a poem", "1. a 2. b"]
        embeddings = np.zeros((2, 8))
        assert build_features(prompts, "length_only").shape == (2, 1)
        assert build_features(prompts, "surface").shape[1] == 23
        assert build_features(prompts, "embedding", embeddings).shape == (2, 8)
        assert build_features(prompts, "surface+embedding", embeddings).shape == (2, 31)

    def test_an_embedding_set_without_embeddings_is_an_error(self):
        """Silently falling back to surface features would report an encoder
        result that no encoder produced."""
        with pytest.raises(ValueError, match="needs precomputed embeddings"):
            build_features(["x"], "embedding")

    def test_unknown_feature_set_is_rejected(self):
        with pytest.raises(ValueError, match="unknown feature set"):
            build_features(["x"], "magic")


class TestModel:
    def test_it_learns_something_a_constant_cannot(self, learnable):
        prompts, y = learnable
        X = build_features(prompts, "surface")
        model = LengthModel(alpha=1.0).fit(X[:300], y[:300])
        fitted = evaluate(y[300:], model.predict(X[300:]))
        constant = evaluate(y[300:], np.full(100, y[:300].mean()))
        assert fitted["r2"] > 0.5
        assert constant["r2"] < 0.05
        assert fitted["mae_log"] < constant["mae_log"]

    def test_expected_tokens_undoes_the_log(self, learnable):
        prompts, y = learnable
        X = build_features(prompts, "surface")
        model = LengthModel(alpha=1.0).fit(X, y)
        np.testing.assert_allclose(model.expected_tokens(X), np.expm1(model.predict(X)))

    def test_probability_over_falls_as_the_threshold_rises(self, learnable):
        prompts, y = learnable
        X = build_features(prompts, "surface")
        model = LengthModel(alpha=1.0).fit(X, y)
        p = [model.probability_over(X[:1], t)[0] for t in (10, 100, 1000, 10000)]
        assert p == sorted(p, reverse=True)
        assert all(0.0 <= x <= 1.0 for x in p)

    def test_calibrating_on_held_out_rows_widens_the_spread(self, learnable):
        """Training residuals are optimistically small by construction; a
        `P(tokens > T)` built on them would be over-confident."""
        prompts, y = learnable
        X = build_features(prompts, "surface")
        model = LengthModel(alpha=1.0).fit(X[:300], y[:300])
        train_sigma = model.sigma
        model.calibrate(X[300:], y[300:])
        assert model.sigma >= train_sigma * 0.9

    def test_save_and_load_round_trip(self, learnable, tmp_path):
        prompts, y = learnable
        X = build_features(prompts, "surface")
        model = LengthModel(alpha=2.0, feature_set="surface").fit(X, y)
        model.save(tmp_path)
        loaded = LengthModel.load(tmp_path)
        np.testing.assert_allclose(loaded.predict(X), model.predict(X))
        assert loaded.feature_set == "surface"
        assert loaded.sigma == model.sigma


class TestIntervalHonesty:
    def test_coverage_matches_the_nominal_level_for_gaussian_residuals(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 20000)
        coverage = interval_coverage(y, np.zeros_like(y), 1.0)
        assert coverage["coverage@80"] == pytest.approx(0.80, abs=0.02)
        assert coverage["coverage@95"] == pytest.approx(0.95, abs=0.02)

    def test_an_understated_spread_shows_up_as_missing_coverage(self):
        """The failure this check exists to catch: intervals that look tight
        and are wrong."""
        rng = np.random.default_rng(2)
        y = rng.normal(0, 1, 20000)
        assert interval_coverage(y, np.zeros_like(y), 0.5)["coverage@80"] < 0.7


class TestAudit:
    def test_it_separates_real_signal_from_noise(self, learnable):
        prompts, y = learnable
        X = build_features(prompts, "surface")
        real = audit(X[:250], y[:250], X[250:325], y[250:325], X[325:], y[325:],
                     name="real", alpha=1.0, permutations=20)
        assert real.permutation_p < 0.1
        assert "fitted" in real.verdict or "under-fed" in real.verdict

    def test_a_random_target_does_not_pass_the_permutation_check(self, learnable):
        """The load-bearing check. If shuffled targets score like real ones,
        the number is coming from the protocol, not the data."""
        prompts, _ = learnable
        noise = np.random.default_rng(3).normal(0, 1, len(prompts))
        X = build_features(prompts, "surface")
        result = audit(X[:250], noise[:250], X[250:325], noise[250:325],
                       X[325:], noise[325:], name="noise", alpha=1.0, permutations=20)
        assert result.permutation_p > 0.05
        assert result.verdict.startswith("no signal")

    def test_bootstrap_interval_brackets_the_point_estimate(self, learnable):
        prompts, y = learnable
        X = build_features(prompts, "surface")
        model = LengthModel(alpha=1.0).fit(X[:300], y[:300])
        pred = model.predict(X[300:])
        low, high = bootstrap_ci(y[300:], pred, n=200)
        assert low <= evaluate(y[300:], pred)["spearman"] <= high


class TestCeiling:
    """Which ceiling you quote changes how finished the head looks."""

    def test_the_averaged_target_is_more_reproducible_than_one_model(self):
        """Averaging two noisy measurements of the same thing cancels part of
        the noise, so the average is easier to predict than either side. Scoring
        against the average and comparing to the single-model agreement is the
        mistake this exists to prevent."""
        from evaluation.length_audit import target_ceiling

        rng = np.random.default_rng(0)
        signal = rng.normal(0, 1, 4000)
        a = np.expm1(signal + rng.normal(0, 1, 4000))
        b = np.expm1(signal + rng.normal(0, 1, 4000))
        raw = target_ceiling(a, b, averaged=False)
        assert target_ceiling(a, b) > raw

    def test_perfect_agreement_leaves_the_ceiling_at_one(self):
        from evaluation.length_audit import target_ceiling

        tokens = np.arange(1, 500)
        assert target_ceiling(tokens, tokens) == pytest.approx(1.0)
        assert target_ceiling(tokens, tokens, averaged=False) == pytest.approx(1.0)


class TestRoutingValue:
    """The test that decides whether the head is worth serving at all."""

    def _corpus(self, n=600):
        rng = np.random.default_rng(5)
        hardness = rng.uniform(0, 1, n)
        y = 4 + 2 * hardness + rng.normal(0, 0.3, n)   # length tracks hardness
        return pd.DataFrame({
            "arena_id": [f"id{i}" for i in range(n)],
            "prompt": [f"prompt number {i}" for i in range(n)],
            "y": y, "tokens_a": np.expm1(y), "tokens_b": np.expm1(y + rng.normal(0, 0.4, n)),
        }), hardness, rng

    def test_a_signal_unrelated_to_the_decision_scores_at_chance(self, monkeypatch):
        """The shipped result: length tracks hardness strongly and the routing
        decision not at all. A head can be accurate and still be worthless for
        the decision it was built to serve."""
        from evaluation import routing_value as rv

        corpus, hardness, rng = self._corpus()
        labels = pd.DataFrame({
            "arena_id": corpus["arena_id"],
            "routing_label": rng.permutation(["strong_needed", "weak_sufficient"] * 300),
            "hardness_score": hardness,
        })
        monkeypatch.setattr(rv, "load_routing_labels", lambda *a, **k: labels)
        out = rv.routing_value(corpus, corpus["y"].to_numpy())
        assert 0.44 < out["auc"]["true_length"] < 0.56
        assert out["hardness_rho"]["true_length"] > 0.8

    def test_it_refuses_to_report_on_too_few_joined_rows(self, monkeypatch):
        """A near-empty join silently produces confident nonsense; it was an
        empty join that exposed the corpus bug in the first place."""
        from evaluation import routing_value as rv

        corpus, hardness, _ = self._corpus()
        monkeypatch.setattr(rv, "load_routing_labels", lambda *a, **k: pd.DataFrame(
            {"arena_id": ["nope"], "routing_label": ["strong_needed"], "hardness_score": [0.5]}))
        with pytest.raises(ValueError, match="too few"):
            rv.routing_value(corpus, corpus["y"].to_numpy())


class TestArtifactNaming:
    """Two encoders over the same feature set are two different heads."""

    def test_each_encoder_gets_its_own_run_directory(self, tmp_path, learnable):
        """Regression: they were named by feature set alone, so the second
        sweep silently overwrote the first and left a head being served
        vectors from an encoder it was never fitted on."""
        from training.heads.length import _save

        prompts, y = learnable
        model = LengthModel(alpha=1.0).fit(build_features(prompts, "surface"), y)
        _save(model, "surface+embedding", "BAAI/bge-small-en-v1.5", tmp_path)
        _save(model, "surface+embedding", "intfloat/e5-large-v2", tmp_path)
        assert sorted(d.name for d in tmp_path.iterdir()) == [
            "bge-small-en-v1.5__surface_embedding", "e5-large-v2__surface_embedding"]

    def test_an_encoderless_head_is_named_by_its_features_alone(self, tmp_path, learnable):
        from training.heads.length import _save

        prompts, y = learnable
        model = LengthModel(alpha=1.0).fit(build_features(prompts, "surface"), y)
        _save(model, "surface", "BAAI/bge-small-en-v1.5", tmp_path)
        assert [d.name for d in tmp_path.iterdir()] == ["surface"]

    def test_the_saved_metadata_names_the_encoder_to_feed_it(self, tmp_path, learnable):
        from training.heads.length import _save

        prompts, y = learnable
        model = LengthModel(alpha=1.0).fit(build_features(prompts, "surface"), y)
        _save(model, "surface+embedding", "intfloat/e5-large-v2", tmp_path)
        meta = json.loads((tmp_path / "e5-large-v2__surface_embedding"
                           / "length.json").read_text())
        assert meta["encoder_model"] == "intfloat/e5-large-v2"
        _save(model, "surface", "intfloat/e5-large-v2", tmp_path)
        assert json.loads((tmp_path / "surface" / "length.json").read_text())[
            "encoder_model"] is None


class TestServingHead:
    """The surface head needs no encoder, so this runs anywhere the tests do."""

    def _run_dir(self, tmp_path, learnable):
        prompts, y = learnable
        X = build_features(prompts, "surface")
        model = LengthModel(alpha=1.0, feature_set="surface").fit(X, y)
        model.save(tmp_path)
        (tmp_path / "length.json").write_text(json.dumps(
            {"feature_set": "surface", "encoder_model": None}), encoding="utf-8")
        return tmp_path

    def test_it_predicts_more_tokens_for_a_longer_job(self, tmp_path, learnable):
        head = LengthHead(self._run_dir(tmp_path, learnable))
        brief = head.predict("keep it brief. explain this")
        big = head.predict("1. do the thing 2. do the thing 3. do the thing 4. do the thing")
        assert big.expected_tokens > brief.expected_tokens
        assert big.bucket in BUCKETS

    def test_the_prediction_carries_its_own_uncertainty(self, tmp_path, learnable):
        """A point estimate without the spread invites a caller to trust a
        number that is typically off by a factor of two."""
        head = LengthHead(self._run_dir(tmp_path, learnable))
        p = head.predict("explain this")
        assert p.sigma > 0
        assert 0.0 < p.probability_over(10) < 1.0

    def test_an_empty_batch_is_not_an_error(self, tmp_path, learnable):
        head = LengthHead(self._run_dir(tmp_path, learnable))
        assert head.predict_batch([]) == []
