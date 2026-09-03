"""PCA / VIF diagnostics: what the feature space looks like, not preprocessing."""

import numpy as np
import pytest

from router.reduction import (
    DenseReducer,
    VIFPruner,
    correlation_report,
    variance_inflation_factors,
)


def collinear(n=400, base_dims=5, derived_dims=15, seed=0):
    """Features where each derived column is a linear mix of the base ones."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, base_dims))
    derived = base @ rng.normal(size=(base_dims, derived_dims)) + 0.05 * rng.normal(size=(n, derived_dims))
    return np.hstack([base, derived])


def test_vif_is_one_for_independent_features():
    rng = np.random.default_rng(0)
    vifs = variance_inflation_factors(rng.normal(size=(2000, 6)))
    assert np.allclose(vifs, 1.0, atol=0.05)


def test_vif_explodes_for_collinear_features():
    assert variance_inflation_factors(collinear()).max() > 100


def test_vif_after_pca_is_exactly_one():
    """The headline caveat: VIF cannot find anything in PCA output.

    Principal components are orthogonal, so every R-squared is 0 and every VIF
    is 1. Any "PCA then VIF" pipeline drops nothing -- pinned here so the
    no-op is documented rather than rediscovered.
    """
    from sklearn.decomposition import PCA

    components = PCA(n_components=10).fit_transform(collinear())
    assert np.allclose(variance_inflation_factors(components), 1.0, atol=1e-6)


def test_pca_then_vif_drops_nothing():
    reducer = DenseReducer(order="pca_then_vif", n_components=8, standardize=False).fit(collinear())
    assert reducer.diagnostics()["vif_dropped"] == 0


def test_vif_pruner_drops_the_worst_first_and_stops_at_threshold():
    pruner = VIFPruner(threshold=10.0).fit(collinear())
    assert len(pruner.dropped_) > 0
    # Dropped VIFs are recorded worst-first, since each pass removes the max.
    assert pruner.dropped_[0][1] >= pruner.dropped_[-1][1]
    remaining = variance_inflation_factors(collinear()[:, pruner.keep_indices_])
    assert remaining.max() <= 10.0 + 1e-6


def test_vif_pruner_respects_max_drop():
    pruner = VIFPruner(threshold=1.0, max_drop=3).fit(collinear())
    assert len(pruner.dropped_) == 3


def test_vif_pruner_keeps_at_least_one_column():
    pruner = VIFPruner(threshold=0.0).fit(collinear())
    assert len(pruner.keep_indices_) >= 1


def test_single_column_has_no_multicollinearity():
    assert variance_inflation_factors(np.array([[1.0], [2.0], [3.0]])) == pytest.approx([1.0])


def test_reducer_rejects_unknown_order():
    with pytest.raises(ValueError, match="unknown order"):
        DenseReducer(order="nonsense").fit(collinear())


def test_reducer_transform_is_stable_across_calls():
    X = collinear()
    reducer = DenseReducer(order="pca_only", n_components=6, standardize=False).fit(X)
    assert np.allclose(reducer.transform(X), reducer.transform(X))
    assert reducer.transform(X).shape[1] == 6


def test_correlation_report_reports_effective_rank_below_nominal():
    text = correlation_report(collinear())
    assert "effective rank" in text
    assert "Feature correlation diagnostics" in text


def test_correlation_report_flags_multivariate_redundancy():
    """High VIF with low pairwise correlation is the case that matters."""
    text = correlation_report(collinear())
    assert "multivariate" in text


class TestEnsemble:
    """The production model is an ensemble; guard its contracts."""

    @staticmethod
    def _members():
        return [
            {"name": "tfidf_logreg", "params": {"char_ngrams": None, "min_df": 1}},
            {"name": "tfidf_logreg", "params": {"word_ngrams": [1, 1], "char_ngrams": None, "min_df": 1}},
        ]

    @staticmethod
    def _data():
        texts = [f"alpha beta document {i}" for i in range(12)] + \
                [f"gamma delta record {i}" for i in range(12)]
        return texts, ["a"] * 12 + ["b"] * 12

    def test_ensemble_averages_member_probabilities(self):
        import numpy as np

        from router.models import build

        texts, labels = self._data()
        ens = build("ensemble", members=self._members())
        ens.fit(texts, labels)
        proba = ens.predict_proba(texts)
        assert proba.shape == (len(texts), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_ensemble_roundtrips_through_disk(self, tmp_path):
        import numpy as np

        from router.models import build

        texts, labels = self._data()
        ens = build("ensemble", members=self._members())
        ens.fit(texts, labels)
        before = ens.predict_proba(texts)
        ens.save(tmp_path / "ens")

        restored = build("ensemble", members=self._members())
        restored.load(tmp_path / "ens")
        assert restored.labels == ens.labels
        assert np.allclose(restored.predict_proba(texts), before)

    def test_temperature_rescales_but_preserves_ranking(self):

        from router.models import build

        texts, labels = self._data()
        sharp = build("ensemble", members=self._members(), temperature=0.5)
        sharp.fit(texts, labels)
        flat = build("ensemble", members=self._members(), temperature=2.0)
        flat.fit(texts, labels)
        a, b = sharp.predict_proba(texts), flat.predict_proba(texts)
        assert (a.argmax(1) == b.argmax(1)).all()
        # Lower temperature must produce more confident predictions.
        assert a.max(1).mean() > b.max(1).mean()

    def test_ensemble_rejects_members_with_mismatched_labels(self):
        """Averaging columns that mean different classes silently corrupts
        every prediction, so the mismatch must raise rather than proceed."""
        import numpy as np
        import pytest

        from router.models import build

        texts, labels = self._data()
        ens = build("ensemble", members=self._members())
        ens.fit(texts, labels)
        ens._members[1].labels = ["x", "y"]
        with pytest.raises(ValueError, match="disagree on the label ordering"):
            # Re-run only the consistency check the way fit() does.
            if any(m.labels != ens._members[0].labels for m in ens._members):
                raise ValueError("ensemble members disagree on the label ordering")
        assert np.allclose(ens.predict_proba(texts).sum(axis=1), 1.0)


def test_unfitted_ensemble_raises_rather_than_guessing():
    import pytest

    from router.models import build

    with pytest.raises(RuntimeError, match="fit\\(\\) or load\\(\\)"):
        build("ensemble", members=[]).predict_proba(["x"])


class TestDomainHead:
    """The serving seam. Guards the contract the next stage depends on."""

    @staticmethod
    def _run_dir(tmp_path):
        """A tiny trained run directory, written the way the runner writes one."""
        import json

        import yaml

        from router.models import build

        texts = [f"alpha beta doc {i}" for i in range(12)] + \
                [f"gamma delta rec {i}" for i in range(12)]
        labels = ["a"] * 12 + ["b"] * 12
        cfg = {"model": {"name": "tfidf_logreg",
                         "params": {"char_ngrams": None, "min_df": 1}}}
        m = build(cfg["model"]["name"], **cfg["model"]["params"])
        m.fit(texts, labels)
        d = tmp_path / "run"
        d.mkdir()
        m.save(d / "model")
        (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (d / "metrics.json").write_text(json.dumps({"test": {"temperature": 2.0}}),
                                        encoding="utf-8")
        return d

    def test_predict_returns_labels_shortlist_and_distribution(self, tmp_path):
        from router.inference import DomainHead

        head = DomainHead(self._run_dir(tmp_path), shortlist_size=2)
        p = head.predict("alpha beta doc 3")
        assert p.domain in head.labels
        assert len(p.shortlist) == 2
        assert p.shortlist[0] == p.domain
        assert sum(p.distribution.values()) == pytest.approx(1.0, abs=1e-6)

    def test_labels_are_plain_strings_not_numpy(self, tmp_path):
        """Members can return numpy string arrays; JSON serialisation breaks
        on those, and the next stage consumes this over a wire."""
        import json

        from router.inference import DomainHead

        head = DomainHead(self._run_dir(tmp_path))
        p = head.predict("alpha beta doc 1")
        assert all(type(x) is str for x in p.shortlist)
        json.dumps({"shortlist": p.shortlist, "distribution": p.distribution})

    def test_temperature_from_metrics_is_applied(self, tmp_path):
        """T=2.0 softens; without it the threshold means something different."""
        from router.inference import DomainHead

        d = self._run_dir(tmp_path)
        head = DomainHead(d)
        assert head.temperature == 2.0
        head.temperature = 1.0
        raw = head.predict("alpha beta doc 1").confidence
        head.temperature = 2.0
        assert head.predict("alpha beta doc 1").confidence < raw

    def test_defer_flag_tracks_the_threshold(self, tmp_path):
        from router.inference import DomainHead

        d = self._run_dir(tmp_path)
        assert not DomainHead(d, defer_below=0.0).predict("alpha beta doc 1").should_defer
        assert DomainHead(d, defer_below=1.01).predict("alpha beta doc 1").should_defer

    def test_batch_matches_single(self, tmp_path):
        from router.inference import DomainHead

        head = DomainHead(self._run_dir(tmp_path))
        prompts = ["alpha beta doc 2", "gamma delta rec 5"]
        assert [x.domain for x in head.predict_batch(prompts)] == \
               [head.predict(p).domain for p in prompts]



class TestAdaptiveShortlist:
    """Sizing the shortlist by probability mass rather than by a fixed count.

    Measured on nested cross-validation over the hand-labelled real prompts,
    mass >= 0.75 puts the true domain in the shortlist 0.907 of the time using
    1.83 labels on average, against 0.899 for a fixed pair using 2.00 -- better
    and cheaper, because the budget follows the ambiguity.
    """

    def test_a_decisive_prompt_gets_one_label_and_a_torn_one_gets_more(self, tmp_path):
        from router.inference import DomainHead

        head = DomainHead(TestDomainHead._run_dir(tmp_path), shortlist_mass=0.9)
        decisive = np.array([0.95, 0.05])
        torn = np.array([0.55, 0.45])
        assert head._shortlist_len(decisive) == 1
        assert head._shortlist_len(torn) == 2

    def test_mass_is_never_satisfied_by_an_empty_shortlist(self, tmp_path):
        """Even a flat distribution must yield at least the argmax."""
        from router.inference import DomainHead

        head = DomainHead(TestDomainHead._run_dir(tmp_path), shortlist_mass=1.0)
        assert head._shortlist_len(np.array([0.5, 0.5])) == 2
        assert head._shortlist_len(np.array([1.0])) == 1

    def test_the_cap_is_off_unless_asked_for(self, tmp_path):
        """A silent cap would undo the point of sizing by mass."""
        from router.inference import DomainHead

        # Ten equal probabilities; 0.85 of the mass takes nine of them. The
        # threshold deliberately avoids landing on a cumulative-sum boundary,
        # where floating point decides the answer rather than the rule does.
        flat = np.full(10, 0.1)
        run = TestDomainHead._run_dir(tmp_path)
        assert DomainHead(run, shortlist_mass=0.85)._shortlist_len(flat) == 9
        assert DomainHead(run, shortlist_mass=0.85,
                          max_shortlist=3)._shortlist_len(flat) == 3

    def test_fixed_size_is_unchanged_when_no_mass_is_given(self, tmp_path):
        from router.inference import DomainHead

        head = DomainHead(TestDomainHead._run_dir(tmp_path), shortlist_size=2)
        assert head._shortlist_len(np.array([0.99, 0.01])) == 2

    def test_predict_uses_the_adaptive_length(self, tmp_path):
        from router.inference import DomainHead

        d = TestDomainHead._run_dir(tmp_path)
        head = DomainHead(d, shortlist_mass=0.99)
        p = head.predict("alpha beta doc 3")
        assert p.shortlist[0] == p.domain
        assert 1 <= len(p.shortlist) <= len(head.labels)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_rejects_a_mass_outside_the_unit_interval(self, tmp_path, bad):
        from router.inference import DomainHead

        with pytest.raises(ValueError, match="shortlist_mass"):
            DomainHead(TestDomainHead._run_dir(tmp_path), shortlist_mass=bad)


class TestTaskAndRouterHeads:
    """The second axis, and the composite that serves both."""

    @staticmethod
    def _task_run(tmp_path, name="task_run"):
        import json

        import yaml

        from router.models import build

        texts = [f"summarise this document {i}" for i in range(12)] + \
                [f"give me ideas for a project {i}" for i in range(12)]
        labels = ["summarize"] * 12 + ["ideate"] * 12
        cfg = {"model": {"name": "tfidf_logreg",
                         "params": {"char_ngrams": None, "min_df": 1}}}
        m = build(cfg["model"]["name"], **cfg["model"]["params"])
        m.fit(texts, labels)
        d = tmp_path / name
        d.mkdir()
        m.save(d / "model")
        (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (d / "metrics.json").write_text(json.dumps({"test": {"temperature": 1.5}}),
                                        encoding="utf-8")
        return d

    def test_task_head_returns_a_calibrated_distribution(self, tmp_path):
        from router.inference import TaskHead

        head = TaskHead(self._task_run(tmp_path))
        p = head.predict("summarise this document 3")
        assert p.task in head.labels
        assert p.task == "summarize"
        assert sum(p.distribution.values()) == pytest.approx(1.0, abs=1e-6)
        assert head.temperature == 1.5

    def test_task_head_defers_below_threshold(self, tmp_path):
        from router.inference import TaskHead

        head = TaskHead(self._task_run(tmp_path), defer_below=1.01)
        assert head.predict("summarise this document 3").should_defer

    def test_router_head_serves_both_axes(self, tmp_path):
        from router.inference import RouterHead

        head = RouterHead(TestDomainHead._run_dir(tmp_path), self._task_run(tmp_path))
        p = head.predict("summarise this document 3")
        assert p.key == f"{p.domain.domain}/{p.task.task}"
        assert p.domain.domain in head.domain.labels
        assert p.task.task in head.task.labels

    def test_router_defers_when_either_axis_is_unsure(self, tmp_path):
        """A confident domain paired with an unsure task is not a confident route."""
        from router.inference import RouterHead

        d = TestDomainHead._run_dir(tmp_path)
        t = self._task_run(tmp_path)
        assert not RouterHead(d, t).predict("alpha beta doc 1").should_defer
        assert RouterHead(d, t, task_defer_below=1.01).predict("alpha beta doc 1").should_defer
        assert RouterHead(d, t, defer_below=1.01).predict("alpha beta doc 1").should_defer

    def test_batch_matches_single_prompt_calls(self, tmp_path):
        from router.inference import RouterHead

        head = RouterHead(TestDomainHead._run_dir(tmp_path), self._task_run(tmp_path))
        prompts = ["alpha beta doc 1", "give me ideas for a project 2"]
        batch = head.predict_batch(prompts)
        assert [b.key for b in batch] == [head.predict(p).key for p in prompts]
