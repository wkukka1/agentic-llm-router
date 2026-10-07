"""TextEncoder.encode() reseeds process-global RNGs on every call (incl. serving)."""
import random
import numpy as np
from router.embeddings.encoder import EncoderConfig, TextEncoder


class Stub(TextEncoder):
    def _encode_bert_batch(self, batch):
        return np.zeros((len(batch), 3), np.float32)


enc = Stub(EncoderConfig(backend="bert"))
random.seed(1234); np.random.seed(1234)          # the *application's* own seeding
draws = []
for _ in range(3):
    draws.append((random.random(), float(np.random.rand())))
    enc.encode(["a request"])                     # one routing request's encode
print("app draws between encodes:", [tuple(round(x, 4) for x in d) for d in draws])
print("draws after the first encode identical ->", len(set(draws[1:])) == 1)
