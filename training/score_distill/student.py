from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 32
    random_seed: int = 42


class StudentRegressor:
    def __init__(self, input_dim: int = 768, device: str = "auto") -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn
        self.input_dim = input_dim
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        nn = self.nn

        class _Model(nn.Module):
            def __init__(self, input_dim: int) -> None:
                super().__init__()
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=8,
                    dim_feedforward=input_dim * 2,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.regressor = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                )

            def forward(self, features):
                # Shape: [batch, dim] -> [batch, 1, dim]
                x = features.unsqueeze(1)
                encoded = self.encoder(x).squeeze(1)
                score = self.regressor(encoded).squeeze(-1) * 100.0
                return score

        return _Model(self.input_dim)

    def fit(self, x_train: list[list[float]], y_train: list[float], cfg: TrainConfig) -> None:
        torch = self.torch
        nn = self.nn
        torch.manual_seed(cfg.random_seed)

        x_tensor = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(cfg.epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

    def predict(self, x_values: list[list[float]], batch_size: int = 128) -> list[float]:
        torch = self.torch
        self.model.eval()
        outputs: list[float] = []
        with torch.no_grad():
            for start in range(0, len(x_values), batch_size):
                batch = x_values[start : start + batch_size]
                x_tensor = torch.tensor(batch, dtype=torch.float32, device=self.device)
                preds = self.model(x_tensor).detach().cpu().tolist()
                outputs.extend([float(max(0.0, min(100.0, score))) for score in preds])
        return outputs

    def save(self, path: str) -> None:
        import torch
        from pathlib import Path

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load(self, path: str) -> None:
        import torch

        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

