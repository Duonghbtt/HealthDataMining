from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_core_metrics
from src.training.trainer import Trainer


class _ValidationEchoModel(nn.Module):
    def __init__(self, ddi_matrix: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("ddi_matrix", ddi_matrix)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = self.weight * 0.0 + batch["loss_value"].mean()
        return {
            "total_loss": base + 1.0,
            "prediction_loss": base + 0.5,
            "ddi_loss": base + 0.1,
            "drug_probs": batch["drug_probs"],
            "target_current": batch["target_current"],
            "final_target_drugs": batch["target_current"],
        }


def test_trainer_validate_one_epoch_reports_core_validation_metrics(tmp_path) -> None:
    ddi_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    dataset = [
        {
            "visit_mask": torch.tensor([1], dtype=torch.bool),
            "drug_probs": torch.tensor([0.9, 0.2, 0.8], dtype=torch.float32),
            "target_current": torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
            "loss_value": torch.tensor(0.2, dtype=torch.float32),
        },
        {
            "visit_mask": torch.tensor([1], dtype=torch.bool),
            "drug_probs": torch.tensor([0.4, 0.7, 0.6], dtype=torch.float32),
            "target_current": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
            "loss_value": torch.tensor(0.3, dtype=torch.float32),
        },
    ]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = _ValidationEchoModel(ddi_matrix)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        validation_threshold=0.5,
    )

    metrics = trainer.validate_one_epoch(dataloader)
    expected = compute_core_metrics(
        y_true=torch.stack([dataset[0]["target_current"], dataset[1]["target_current"]], dim=0),
        y_score=torch.stack([dataset[0]["drug_probs"], dataset[1]["drug_probs"]], dim=0),
        threshold=0.5,
        ddi_matrix=ddi_matrix,
    )

    assert metrics["val_jaccard"] == pytest.approx(expected["jaccard"])
    assert metrics["val_f1"] == pytest.approx(expected["f1"])
    assert metrics["val_prauc"] == pytest.approx(expected["prauc"])
    assert metrics["val_ddi_rate"] == pytest.approx(expected["ddi_rate"])


def test_trainer_fit_writes_machine_readable_metric_artifacts(tmp_path) -> None:
    ddi_matrix = torch.zeros((3, 3), dtype=torch.float32)
    dataset = [
        {
            "visit_mask": torch.tensor([1], dtype=torch.bool),
            "drug_probs": torch.tensor([0.9, 0.2, 0.8], dtype=torch.float32),
            "target_current": torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32),
            "loss_value": torch.tensor(0.2, dtype=torch.float32),
        },
        {
            "visit_mask": torch.tensor([1], dtype=torch.bool),
            "drug_probs": torch.tensor([0.4, 0.7, 0.6], dtype=torch.float32),
            "target_current": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
            "loss_value": torch.tensor(0.3, dtype=torch.float32),
        },
    ]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = _ValidationEchoModel(ddi_matrix)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        validation_threshold=0.5,
        use_self_history=False,
        use_ddi=False,
    )

    result = trainer.fit(
        train_dataloader=dataloader,
        val_dataloader=dataloader,
        epochs=1,
    )

    assert Path(result["metrics_per_epoch_json"]).exists()
    assert Path(result["metrics_per_epoch_csv"]).exists()
    assert Path(result["best_metrics_json"]).exists()
