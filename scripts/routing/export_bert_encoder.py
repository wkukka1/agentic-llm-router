"""Export a pathway's raw BERT encoder to ONNX for aar-gateway (no PyTorch at serve time).

The gateway encodes prompts with fastembed (ONNX). fastembed ships MiniLM and
bge-small but no raw BERT, and the public ONNX exports of bert-base-uncased are
masked-LM heads (vocabulary logits, not hidden states). This writes the encoder's
``last_hidden_state`` as ``model.onnx`` plus its tokenizer files into
``~/.aar-gateway/encoders/<model>-<pooling>/``, the folder a NIRT router file
exported from that pathway names (``encoder.model_path``), then checks fastembed
reproduces the training encoder on short and truncated-long prompts.

    python scripts/routing/export_bert_encoder.py                 # the `irt` pathway
    python scripts/routing/export_bert_encoder.py --out D:/encoders/bert-base-uncased-mean

Needs PyTorch, transformers and ``onnx`` (export only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PARITY_TOL = 1e-4


def gateway_encoder_folder(model_name: str, pooling: str) -> str:
    """The folder name under the gateway's ``encoders/`` for a raw encoder + pooling."""
    return f"{model_name.rstrip('/').split('/')[-1]}-{pooling}"


def export(model_name: str, out: Path) -> int:
    """Write the ONNX encoder + tokenizer to ``out``; returns the hidden size."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name).eval()

    class Hidden(torch.nn.Module):
        """``last_hidden_state`` only, inputs by name (forward's positional order changed upstream)."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state

    enc = tok(["hello world"], return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    torch.onnx.export(Hidden(bert).eval(), tuple(enc[k] for k in names), out / "model.onnx",
                      input_names=names, output_names=["last_hidden_state"],
                      dynamic_axes={k: {0: "batch", 1: "seq"} for k in [*names, "last_hidden_state"]},
                      opset_version=17, dynamo=False)
    tok.save_pretrained(out)
    bert.config.save_pretrained(out)
    return int(bert.config.hidden_size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pathway", default="irt", help="the embedding pathway whose encoder to export")
    ap.add_argument("--config", default=None, help="phase-0 config (default configs/phase0.yaml)")
    ap.add_argument("--out", default=None, help="default: ~/.aar-gateway/encoders/<model>-<pooling>")
    args = ap.parse_args()

    from aar_gateway._core.encoder import EncoderConfig, TextEncoder
    from aar_gateway.config import encoders_dir

    from router.config import load_config
    from router.embeddings.encoder import load_encoder

    train_enc = load_encoder(load_config(args.config), args.pathway)
    cfg = train_enc.cfg
    if cfg.backend != "bert":
        print(f"pathway {args.pathway!r} is backend {cfg.backend!r}; the gateway serves it already")
        return 1
    folder = gateway_encoder_folder(cfg.model_name, cfg.pooling)
    out = Path(args.out).expanduser() if args.out else encoders_dir() / folder
    dim = export(cfg.model_name, out)

    texts = ["hi", "Write a Python function that merges two sorted lists.",
             "Explain " + "the history of the Roman empire in great detail " * 80]   # > max_seq_length
    served = TextEncoder(EncoderConfig(backend="fastembed", model_name=f"aar/{folder}", model_path=str(out),
                                       pooling=cfg.pooling, dim=dim, normalize=bool(cfg.normalize),
                                       max_seq_length=int(cfg.max_seq_length or 256)))
    diff = float(np.abs(served.encode(texts) - train_enc.encode(texts)).max())
    if not diff <= PARITY_TOL:
        print(f"parity FAILED: max |d| {diff:.3g} > {PARITY_TOL:g} -- don't serve {out}")
        return 1
    print(f"wrote {out} ({(out / 'model.onnx').stat().st_size / 1e6:.0f} MB); fastembed vs training encoder "
          f"max |d| {diff:.2g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
