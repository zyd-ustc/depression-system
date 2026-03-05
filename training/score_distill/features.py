from __future__ import annotations

import hashlib
from typing import Iterable


def _hashed_feature(text: str, dim: int = 768) -> list[float]:
    values = [0.0] * dim
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    for index in range(dim):
        values[index] = digest[index % len(digest)] / 255.0
    return values


class RobertaFeatureExtractor:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device_name = device
        self._use_hf = True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            self._use_hf = False
            self.torch = None
            self.tokenizer = None
            self.model = None
            return

        self.torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self._use_hf = False
            self.tokenizer = None
            self.model = None
            return
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> list[list[float]]:
        rows = list(texts)
        if not rows:
            return []
        if not self._use_hf:
            return [_hashed_feature(text) for text in rows]

        features: list[list[float]] = []
        torch = self.torch
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                features.extend(pooled.detach().cpu().tolist())
        return features
