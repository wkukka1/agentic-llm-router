import numpy as np
import torch

from experiment import (QueryBottleneck, compress_prompt_text, compress_texts, fit_quality_head,
                        full_linear_head, replace_rows)


def test_compress_texts_rate_one_is_identity():
    texts = ["What is 2 + 2?", "Name the capital city of France."]
    assert compress_texts(texts, rate=1.0) == texts


def test_replace_rows_swaps_by_id_and_keeps_the_rest():
    ids = np.array(["q1", "q2", "q1", "q3"])           # cell-level: q1 appears twice
    base = np.zeros((4, 2), np.float32)
    new = replace_rows(ids, base, {"q1": np.array([1.0, 1.0]), "q3": np.array([3.0, 3.0])})
    assert new.tolist() == [[1, 1], [0, 0], [1, 1], [3, 3]]
    assert base.sum() == 0                             # input untouched


def test_fit_quality_head_learns_which_model_is_better():
    # low-rank embeddings (like BERT's); model 0 is right when latent z[0] > 0, model 1 otherwise
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(2000, 16))
    X = (Z @ rng.normal(size=(16, 768)) + 0.1 * rng.normal(size=(2000, 768))).astype(np.float32)
    Y = np.stack([Z[:, 0] > 0, Z[:, 0] <= 0], axis=1).astype(np.float32)
    for net in (QueryBottleneck(n_models=2), full_linear_head(n_models=2)):
        pred = fit_quality_head(net, X[:1600], Y[:1600], X[1600:], Y[1600:], epochs=30, seed=0)
        assert pred(X[1600:]).shape == (400, 2)
        picks = pred(X[1600:]).argmax(1)
        assert (Y[1600:][np.arange(400), picks] == 1).mean() > 0.9


def test_bottleneck_maps_768_to_64():
    model = QueryBottleneck(n_models=11)
    z = model(torch.randn(8, 768))
    assert z.shape == (8, 64)


def test_bottleneck_head_scores_every_model():
    model = QueryBottleneck(n_models=11)
    logits = model.classify(torch.randn(8, 768))
    assert logits.shape == (8, 11)


def test_compress_prompt_text_shortens_and_keeps_query():
    prompt = (
        "Hello there, I hope you are doing well today. I was just wondering, if it is "
        "not too much trouble and you happen to have a moment, could you please tell me "
        "what the capital city of France is? Thank you so much in advance for your help, "
        "I really appreciate it a lot."
    )
    out = compress_prompt_text(prompt, rate=0.5)
    assert len(out) < len(prompt)
    for word in ("capital", "France"):
        assert word in out
