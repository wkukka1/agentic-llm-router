"""End-to-end fine-tuned transformer classifier.

Deliberately a hand-rolled loop rather than ``transformers.Trainer``: the
training set is small enough that the loop is short, and it keeps early stopping
on validation macro-F1 explicit -- which is the metric that matters here, since
the domain distribution is imbalanced 2:1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from router.embeddings import resolve_device
from router.models import DomainClassifier, register

log = logging.getLogger(__name__)


class _TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None = None) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return (self.texts[idx], -1 if self.labels is None else self.labels[idx])


@register("finetune_transformer")
class FineTunedTransformer(DomainClassifier):
    def __init__(self, *, model_name: str, max_length: int = 256, batch_size: int = 32,
                 eval_batch_size: int = 64, epochs: int = 4, lr: float = 3e-5,
                 weight_decay: float = 0.01, warmup_ratio: float = 0.1,
                 class_weighted_loss: bool = True, seed: int = 20260824,
                 device: str | None = None, **kw):
        super().__init__(model_name=model_name, max_length=max_length, batch_size=batch_size,
                         eval_batch_size=eval_batch_size, epochs=epochs, lr=lr,
                         weight_decay=weight_decay, warmup_ratio=warmup_ratio,
                         class_weighted_loss=class_weighted_loss, seed=seed, device=device, **kw)
        self.device = resolve_device(device)
        self.tokenizer = None
        self.model = None

    def _collate(self, batch):
        texts, labels = zip(*batch, strict=True)
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.params["max_length"], return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    def fit(self, train_texts, train_labels, val_texts=None, val_labels=None) -> None:
        p = self.params
        torch.manual_seed(p["seed"])
        np.random.seed(p["seed"])

        self.labels = sorted(set(train_labels))
        index = {label: i for i, label in enumerate(self.labels)}
        y_train = [index[label] for label in train_labels]

        self.tokenizer = AutoTokenizer.from_pretrained(p["model_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            p["model_name"],
            num_labels=len(self.labels),
            id2label=dict(enumerate(self.labels)),
            label2id=index,
        ).to(self.device)

        loader = DataLoader(
            _TextDataset(list(train_texts), y_train),
            batch_size=p["batch_size"], shuffle=True, collate_fn=self._collate,
        )

        # Imbalance here is mild (2:1) but systematic -- the three 1400-row
        # domains would otherwise absorb the smaller six.
        loss_weights = None
        if p["class_weighted_loss"]:
            counts = np.bincount(y_train, minlength=len(self.labels)).astype(np.float64)
            weights = counts.sum() / (len(self.labels) * np.maximum(counts, 1))
            loss_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
        )
        total_steps = max(1, len(loader) * p["epochs"])
        warmup_steps = int(total_steps * p["warmup_ratio"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (
                step / max(1, warmup_steps)
                if step < warmup_steps
                else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
            ),
        )

        best_f1, best_state = -1.0, None
        for epoch in range(p["epochs"]):
            self.model.train()
            running = 0.0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                targets = batch.pop("labels")
                # Always compute the loss in fp32. Some checkpoints (DeBERTa-v3)
                # load in half precision, and a Half/Float mismatch against the
                # class-weight tensor fails outright; fp32 loss is also the
                # numerically safer default.
                logits = self.model(**batch).logits.float()
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running += loss.item()

            message = f"epoch {epoch + 1}/{p['epochs']} train_loss={running / len(loader):.4f}"
            if val_texts is not None and val_labels is not None:
                predictions = self.predict(list(val_texts))
                val_f1 = f1_score(list(val_labels), predictions, average="macro")
                message += f" val_macro_f1={val_f1:.4f}"
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    message += " *"
            log.info(message)

        # Restore the best epoch rather than the last: with 4-6 epochs on 6k
        # rows the last epoch is frequently past the overfitting point.
        if best_state is not None:
            self.model.load_state_dict(best_state)
            log.info("restored best checkpoint (val_macro_f1=%.4f)", best_f1)

    @torch.inference_mode()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(
            _TextDataset(list(texts)),
            batch_size=self.params["eval_batch_size"], shuffle=False, collate_fn=self._collate,
        )
        chunks = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch.pop("labels")
            logits = self.model(**batch).logits
            chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.vstack(chunks) if chunks else np.zeros((0, len(self.labels)), dtype=np.float32)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "labels.json").write_text(json.dumps(self.labels), encoding="utf-8")

    def load(self, path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.labels = json.loads((path / "labels.json").read_text(encoding="utf-8"))

    def size_bytes(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() * p.element_size() for p in self.model.parameters())
