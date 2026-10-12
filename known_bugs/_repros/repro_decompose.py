"""Repro: decompose.* bugs (LENGTHHEAD-001, FROZENENCODERMODEL-001, PROMPTDECOMPOSER-002, decompose MOD-001 x2). Run from repo root."""
import sys
sys.path.insert(0, "src")
import json, tempfile, os
from pathlib import Path
import numpy as np

from decompose.classifiers.prompt_decomposition.signals import extract

print("== 1. has_code_shape / has_math on plain prose")
for p in ["Let me know what you think about this plan",
          "Translate this sentence from French to English",
          "What class should I take next semester?",
          "How much is $5 in euros?",
          "What happened on 2024-01-15?",
          "Convert 10-20 into a range",
          "Explain the French revolution"]:
    s = extract(p)
    print(f"  code_shape={s['has_code_shape']:.0f} math={s['has_math']:.0f}  {p!r}")

print("== 2. LengthHead negative expected_tokens")
from decompose.classifiers.prompt_decomposition.heads.length_model import LengthModel
from decompose.classifiers.prompt_decomposition.heads.length import LengthHead
from decompose.classifiers.prompt_decomposition.features import build_features
rng = np.random.default_rng(0)
train = ["w " * int(n) for n in rng.integers(5, 400, 400)]
y = np.log1p(np.array([len(t) * 0.8 for t in train]) / 4)   # tokens ~ prompt size
X = build_features(train, "surface")
m = LengthModel(feature_set="surface").fit(X, y)
d = Path(tempfile.mkdtemp())
m.save(d)
(d / "length.json").write_text(json.dumps({"feature_set": "surface"}))
h = LengthHead(d)
for p in ["", "hi", "a " * 200]:
    pr = h.predict(p)
    print(f"  {p[:10]!r:>14} expected_tokens={pr.expected_tokens:9.2f} log={pr.log_tokens:.3f} bucket={pr.bucket}")

print("== 3. EmbeddingLogisticRegression ignores `prefix` / `normalize`")
from decompose.classifiers.prompt_decomposition.models import build
mm = build("embed_logreg", encoder_model="BAAI/bge-small-en-v1.5", prefix="Represent this sentence: ", normalize=False)
print("  params.prefix =", repr(mm.params.get("prefix")), "| encoder.prefix =", repr(mm.encoder.prefix),
      "| params.normalize =", mm.params.get("normalize"), "| encoder.normalize =", mm.encoder.normalize)

print("== 4. PromptDecomposer duplicate / remove semantics")
from decompose.decomposer import PromptDecomposer
from decompose.classifiers.prompt_heads import SurfaceClassifier
from decompose.classifiers.base import ClassificationInput
pd = PromptDecomposer([SurfaceClassifier()])
pd.add_classifier(SurfaceClassifier())          # accepted silently
try:
    pd.extract_signals(ClassificationInput(prompt="hi"))
except ValueError as e:
    print("  add_classifier accepted a duplicate, extract_signals raised:", e)

print("== 5. .env inline comment")
from decompose.classifiers.prompt_decomposition import settings as S
tmp = Path(tempfile.mkdtemp()) / ".env"
tmp.write_text("ROUTER_SEED=123  # my seed\n")
S.settings.cache_clear()
try:
    print("  seed =", S.settings(str(tmp)).seed)
except Exception as e:
    print("  settings() raised:", type(e).__name__, e)
