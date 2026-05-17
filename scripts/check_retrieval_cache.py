from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.runtime_builder import (  # noqa: E402
    build_core_model,
    build_dataset,
    build_runtime_data_config_file,
    load_vocab_sizes,
    select_collate_fn,
)
from src.utils.io import load_yaml_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check offline retrieval cache wiring.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    data_config = load_yaml_config(args.config)
    train_config = load_yaml_config(args.train_config)
    model_config = load_yaml_config(args.model_config)
    project_root = Path(data_config["_project_root"]).resolve()
    processed_root = resolve_path(project_root, data_config["paths"]["processed_root"]).resolve()
    vocab_root = resolve_path(project_root, data_config["paths"]["vocab_root"]).resolve()
    ddi_matrix_path = resolve_path(project_root, train_config["paths"]["ddi_matrix_path"]).resolve()

    with tempfile.TemporaryDirectory(prefix="clinrec_check_retrieval_cache_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            project_root=project_root,
            data_config=data_config,
            processed_root=processed_root,
            vocab_root=vocab_root,
            temp_dir=Path(temp_dir_name),
            retrieval_cache_config=train_config.get("retrieval_cache"),
        )
        vocab_sizes = load_vocab_sizes(vocab_root)
        dataset = build_dataset(
            split=args.split,
            runtime_data_config_path=runtime_data_config_path,
            processed_root=processed_root,
            drug_vocab_size=int(vocab_sizes["drug"]),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=select_collate_fn(dataset),
        )
        batch = next(iter(dataloader))
        required_keys = (
            "retrieval_neighbor_ids",
            "retrieval_scores",
            "retrieval_medication_ids",
            "retrieval_medication_multi_hot",
            "retrieval_mask",
        )
        missing = [key for key in required_keys if key not in batch]
        if missing:
            raise RuntimeError(f"Missing retrieval batch fields: {missing}")
        print("retrieval batch shapes:")
        for key in required_keys:
            value = batch[key]
            print(f"  {key}: {tuple(value.shape)} {value.dtype}")
        self_matches = (batch["retrieval_neighbor_ids"] == torch.arange(4).unsqueeze(1)) & batch["retrieval_mask"]
        if args.split == "train" and bool(self_matches.any().item()):
            raise RuntimeError("Found a self neighbor in train retrieval cache batch.")

        if args.forward:
            device = torch.device(args.device)
            model = build_core_model(
                train_config=train_config,
                model_config=model_config,
                train_dataset=dataset,
                runtime_data_config_path=runtime_data_config_path,
                processed_root=processed_root,
                vocab_root=vocab_root,
                ddi_matrix_path=ddi_matrix_path,
            ).to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(_move_batch_to_device(batch, device))
            print("forward ok:")
            print(f"  drug_logits: {tuple(outputs['drug_logits'].shape)}")
            print(
                "  retrieval_fraction_with_context: "
                f"{float((outputs['retrieval_valid_candidate_counts'] > 0).float().mean().item()):.4f}"
            )


if __name__ == "__main__":
    main()
