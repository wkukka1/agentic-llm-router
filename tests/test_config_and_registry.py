import pytest
import yaml

from router.config import ExperimentConfig, load_experiments
from router.models import available, build


def test_all_expected_models_are_registered():
    assert set(available()) == {
        "tfidf_logreg", "tfidf_linear_svm", "embed_logreg", "embed_mlp", "finetune_transformer",
    }


def test_build_unknown_model_lists_the_alternatives():
    with pytest.raises(KeyError, match="unknown model"):
        build("does_not_exist")


def test_config_roundtrips_through_dict():
    config = ExperimentConfig.from_dict({
        "name": "x",
        "model": {"name": "tfidf_logreg", "params": {"C": 2.0}},
        "data": {"variant": "question_only"},
    })
    assert config.model.params["C"] == 2.0
    assert config.data.variant == "question_only"
    assert ExperimentConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_config_requires_name_and_model():
    with pytest.raises(ValueError, match="requires 'name' and 'model'"):
        ExperimentConfig.from_dict({"name": "x"})


def test_shipped_experiment_configs_all_load_and_are_uniquely_named():
    configs = load_experiments(["experiments/domain", "experiments/preference"])
    assert len(configs) >= 14
    names = [c.name for c in configs]
    assert len(names) == len(set(names)), "duplicate experiment names would overwrite each other"
    for config in configs:
        assert config.model.name in available()


def test_shipped_config_name_matches_its_filename():
    """The run directory is named from `name`, so a mismatch is silently confusing."""
    from pathlib import Path

    for path in sorted(Path("experiments").glob("*/*.yaml")):
        assert yaml.safe_load(path.read_text())["name"] == path.stem
