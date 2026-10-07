"""build_store: a partial build is NOT resumed after a device/batch_size change,
although EncoderConfig.vector_fingerprint() says those fields don't change vectors."""
import tempfile
import numpy as np
from router.embeddings.encoder import EncoderConfig, TextEncoder, build_store


class Fake(TextEncoder):
    calls = 0
    boom_after = None

    @property
    def dim(self):
        return 4

    def iter_encode(self, texts, batch_size=None, show_progress=False, _start_offset=0):
        bs = int(batch_size or self.cfg.batch_size)
        for i in range(0, len(texts), bs):
            if Fake.boom_after is not None and Fake.calls >= Fake.boom_after:
                raise KeyboardInterrupt("simulated kill")
            Fake.calls += 1
            yield _start_offset + i, np.ones((len(texts[i:i + bs]), 4), np.float32)


ids = [f"q{i}" for i in range(40)]
texts = [f"text {i}" for i in range(40)]
d = tempfile.mkdtemp()

cpu = Fake(EncoderConfig(device="cpu", batch_size=4))
Fake.boom_after = 5
try:
    build_store(d, ids, texts, cpu, flush_every=1, show_progress=True)
except KeyboardInterrupt:
    print("-- run 1 (cpu) killed after 5 batches (20 rows)")

Fake.boom_after, Fake.calls = None, 0
gpu = Fake(EncoderConfig(device="cuda", batch_size=4))
assert cpu.cfg.vector_fingerprint() == gpu.cfg.vector_fingerprint()
print("-- run 2 (cuda, same vectors):")
build_store(d, ids, texts, gpu, flush_every=1, show_progress=True)
print("batches encoded in run 2:", Fake.calls, "(10 total; 5 if it had resumed)")
