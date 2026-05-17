from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.training import runtime_builder
from src.training.trainer import Trainer
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the core ClinRec recommendation model.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Path to configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to configs/model.yaml")
    parser.add_argument("--processed-root", default=None, help="Optional override for processed data root")
    parser.add_argument("--vocab-root", default=None, help="Optional override for vocab directory")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional override for DDI matrix artifact")
    parser.add_argument("--device", default=None, help="Optional override for runtime device")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for random seed")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for the fixed prediction threshold used in validation/evaluation",
    )
    parser.add_argument(
        "--baseline-mode",
        default=None,
        choices=("current_only", "current_self_history", "current_self_history_ddi"),
        help="Select the baseline protocol without manually editing configs",
    )
    parser.add_argument(
        "--monitor-metric",
        default=None,
        help="Optional override for the checkpoint monitor metric (defaults to current repo behavior)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for number of training epochs")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short end-to-end smoke path with limited train/validation/eval batches",
    )
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional cap for validation batches per epoch")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional cap for post-train evaluation batches")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _first_existing_path(
    candidates: Sequence[Path | None],
    *,
    kind: str,
) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to resolve {kind}. Checked candidates: {checked if checked else ['<none>']}"
    )


def resolve_runtime_paths(
    *,
    project_root: Path,
    train_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path]:
    train_paths = dict(train_config.get("paths", {}))
    data_paths = dict(data_config.get("paths", {}))

    processed_root = _first_existing_path(
        [
            None if args.processed_root is None else Path(args.processed_root).resolve(),
            None if train_paths.get("processed_root") is None else resolve_path(project_root, train_paths["processed_root"]).resolve(),
            None if data_paths.get("processed_root") is None else resolve_path(project_root, data_paths["processed_root"]).resolve(),
            (project_root / "handover_data" / "processed").resolve(),
        ],
        kind="processed_root",
    )
    vocab_root = _first_existing_path(
        [
            None if args.vocab_root is None else Path(args.vocab_root).resolve(),
            None if train_paths.get("vocab_root") is None else resolve_path(project_root, train_paths["vocab_root"]).resolve(),
            None if data_paths.get("interim_root") is None else (resolve_path(project_root, data_paths["interim_root"]).resolve() / "vocab"),
            (project_root / "handover_data" / "vocab").resolve(),
        ],
        kind="vocab_root",
    )
    ddi_matrix_path = _first_existing_path(
        [
            None if args.ddi_matrix_path is None else Path(args.ddi_matrix_path).resolve(),
            None if train_paths.get("ddi_matrix_path") is None else resolve_path(project_root, train_paths["ddi_matrix_path"]).resolve(),
            (project_root / "handover_data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        ],
        kind="ddi_matrix_path",
    )

    checkpoint_dir = ensure_dir(resolve_path(project_root, train_paths.get("checkpoint_dir", "outputs/checkpoints")).resolve())
    log_dir = ensure_dir(resolve_path(project_root, train_paths.get("log_dir", "outputs/logs")).resolve())
    report_dir = ensure_dir(resolve_path(project_root, train_paths.get("report_dir", "outputs/reports")).resolve())

    print("Resolved runtime paths:")
    print(f"  processed_root: {processed_root}")
    print(f"  vocab_root: {vocab_root}")
    print(f"  ddi_matrix_path: {ddi_matrix_path}")
    print(f"  checkpoint_dir: {checkpoint_dir}")
    print(f"  log_dir: {log_dir}")
    print(f"  report_dir: {report_dir}")

    return {
        "processed_root": processed_root,
        "vocab_root": vocab_root,
        "ddi_matrix_path": ddi_matrix_path,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "report_dir": report_dir,
    }


def resolve_baseline_settings(
    *,
    train_config: Mapping[str, Any],
    baseline_mode: str | None,
) -> dict[str, Any]:
    requested_mode = str(
        baseline_mode
        or train_config.get("baseline", {}).get("mode")
        or "current_self_history_ddi"
    ).strip()
    configured_lambda_ddi = float(train_config.get("loss", {}).get("lambda_ddi", 0.0))

    if requested_mode == "current_only":
        return {
            "mode": requested_mode,
            "use_self_history": False,
            "use_ddi": False,
            "lambda_ddi": 0.0,
        }
    if requested_mode == "current_self_history":
        return {
            "mode": requested_mode,
            "use_self_history": True,
            "use_ddi": False,
            "lambda_ddi": 0.0,
        }
    if requested_mode == "current_self_history_ddi":
        return {
            "mode": requested_mode,
            "use_self_history": True,
            "use_ddi": True,
            "lambda_ddi": configured_lambda_ddi,
        }
    raise ValueError(f"Unsupported baseline_mode: {requested_mode!r}")


def apply_baseline_settings(train_config: Mapping[str, Any], baseline_settings: Mapping[str, Any]) -> dict[str, Any]:
    runtime_train_config = copy.deepcopy(dict(train_config))
    runtime_train_config.setdefault("loss", {})
    runtime_train_config["loss"]["lambda_ddi"] = float(baseline_settings["lambda_ddi"])
    runtime_train_config["baseline"] = {
        "mode": str(baseline_settings["mode"]),
        "use_self_history": bool(baseline_settings["use_self_history"]),
        "use_ddi": bool(baseline_settings["use_ddi"]),
        "lambda_ddi": float(baseline_settings["lambda_ddi"]),
    }
    return runtime_train_config


def build_optimizer(*, model: torch.nn.Module, train_config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optimization_cfg = dict(train_config.get("optimization", {}))
    optimizer_name = str(optimization_cfg.get("optimizer", "adam")).strip().lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer `{optimizer_name}`. Only `adam` is supported in train_core.py.")
    learning_rate = float(optimization_cfg.get("learning_rate", 1.0e-3))
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def build_scheduler(*, optimizer: torch.optim.Optimizer, train_config: Mapping[str, Any]) -> Any | None:
    scheduler_name = str(train_config.get("optimization", {}).get("scheduler", "none")).strip().lower()
    if scheduler_name != "none":
        raise ValueError(f"Unsupported scheduler `{scheduler_name}`. Only `none` is supported in train_core.py.")
    _ = optimizer
    return None


def main() -> None:
    args = parse_args()
    original_train_config = load_yaml_config(args.config)
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)
    baseline_settings = resolve_baseline_settings(
        train_config=original_train_config,
        baseline_mode=args.baseline_mode,
    )
    train_config = apply_baseline_settings(original_train_config, baseline_settings)

    project_root = Path(train_config["_project_root"]).resolve()
    resolved_paths = resolve_runtime_paths(
        project_root=project_root,
        train_config=train_config,
        data_config=data_config,
        args=args,
    )

    runtime_cfg = dict(train_config.get("runtime", {}))
    run_cfg = dict(train_config.get("run", {}))
    device = runtime_builder.resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    seed = int(args.seed if args.seed is not None else data_config.get("seed", 17))
    num_workers = int(runtime_cfg.get("num_workers", 0))
    train_num_workers = int(runtime_cfg.get("train_num_workers", num_workers))
    val_num_workers = int(runtime_cfg.get("val_num_workers", num_workers))
    pin_memory = bool(runtime_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers_value = runtime_cfg.get("persistent_workers")
    prefetch_factor_value = runtime_cfg.get("prefetch_factor")
    threshold = float(args.threshold if args.threshold is not None else train_config.get("prediction", {}).get("threshold", 0.5))
    training_cfg = dict(train_config.get("training", {}))
    configured_selection_metric = str(training_cfg.get("selection_metric", "")).strip().lower()
    if args.monitor_metric is not None:
        monitor_metric = str(args.monitor_metric)
    elif runtime_cfg.get("monitor_metric") is not None:
        monitor_metric = str(runtime_cfg.get("monitor_metric"))
    elif configured_selection_metric in {"jaccard", "f1", "prauc"}:
        monitor_metric = f"val_{configured_selection_metric}"
    else:
        monitor_metric = "val_total_loss"
    if runtime_cfg.get("monitor_mode") is not None:
        monitor_mode = str(runtime_cfg.get("monitor_mode"))
    elif monitor_metric in {"val_jaccard", "val_f1", "val_prauc"}:
        monitor_mode = "max"
    else:
        monitor_mode = "min"
    set_seed(seed)
    smoke_test = bool(args.smoke_test or run_cfg.get("smoke_test", False))
    max_train_batches = (
        args.max_train_batches
        if args.max_train_batches is not None
        else run_cfg.get("max_train_batches")
    )
    max_val_batches = (
        args.max_val_batches
        if args.max_val_batches is not None
        else run_cfg.get("max_val_batches")
    )
    max_eval_batches = (
        args.max_eval_batches
        if args.max_eval_batches is not None
        else run_cfg.get("max_eval_batches")
    )
    if smoke_test:
        if max_train_batches is None:
            max_train_batches = 2
        if max_val_batches is None:
            max_val_batches = 2
        if max_eval_batches is None:
            max_eval_batches = 2

    print(f"Using device: {device}")
    print(f"Using seed: {seed}")
    print("Run configuration:")
    print(f"  baseline_mode: {baseline_settings['mode']}")
    print(f"  use_self_history: {baseline_settings['use_self_history']}")
    print(f"  use_ddi: {baseline_settings['use_ddi']}")
    print(f"  lambda_ddi: {float(baseline_settings['lambda_ddi']):.6f}")
    retrieval_enabled_for_run = bool(
        train_config.get("core", {}).get(
            "use_retrieval",
            train_config.get("extended", {}).get("use_retrieval", model_config.get("retrieval", {}).get("enabled", False)),
        )
    )
    if bool(train_config.get("retrieval_cache", {}).get("enabled", False)):
        retrieval_enabled_for_run = True
    print(f"  retrieval_enabled: {retrieval_enabled_for_run}")
    print(f"  retrieval_cache_enabled: {bool(train_config.get('retrieval_cache', {}).get('enabled', False))}")
    print(f"  history_mode: {'self_retrieval' if retrieval_enabled_for_run else str(model_config.get('full_model', {}).get('history_mode', 'self_only'))}")
    print(f"  stage_training: {bool(train_config.get('loss', {}).get('stage_training', False))}")
    print(f"  stage1_epochs: {int(train_config.get('training', {}).get('stage1_epochs', 0))}")
    print(f"  stage2_epochs: {int(train_config.get('training', {}).get('stage2_epochs', 0))}")
    print(f"  threshold: {threshold:.4f}")
    print(f"  monitor_metric: {monitor_metric}")
    print(f"  monitor_mode: {monitor_mode}")
    print(f"  smoke_test: {smoke_test}")
    print(f"  max_train_batches: {max_train_batches}")
    print(f"  max_val_batches: {max_val_batches}")
    print(f"  max_eval_batches: {max_eval_batches}")

    with tempfile.TemporaryDirectory(prefix="clinrec_runtime_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_data_config_path = runtime_builder.build_runtime_data_config_file(
            project_root=project_root,
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=temp_dir,
            retrieval_cache_config=train_config.get("retrieval_cache"),
        )
        med_vocab_path = resolved_paths["vocab_root"] / "med_vocab_main.json"
        legacy_drug_vocab_path = resolved_paths["vocab_root"] / "drug_vocab.json"
        resolved_drug_vocab_path = med_vocab_path if med_vocab_path.exists() else legacy_drug_vocab_path
        drug_vocab_size = int(read_json(resolved_drug_vocab_path)["size"])

        train_loader, val_loader, train_dataset = runtime_builder.build_dataloaders(
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=int(runtime_cfg.get("batch_size", 16)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
            train_num_workers=train_num_workers,
            val_num_workers=val_num_workers,
            persistent_workers=None if persistent_workers_value is None else bool(persistent_workers_value),
            prefetch_factor=None if prefetch_factor_value is None else int(prefetch_factor_value),
        )
        model = runtime_builder.build_core_model(
            train_config=train_config,
            model_config=model_config,
            train_dataset=train_dataset,
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
        )

    optimizer = build_optimizer(model=model, train_config=train_config)
    scheduler = build_scheduler(optimizer=optimizer, train_config=train_config)
    optimization_config = train_config.get("optimization", {})
    max_grad_norm = optimization_config.get("max_grad_norm", 1.0)
    sanitize_code_embedding_grads = optimization_config.get(
        "sanitize_code_embedding_grads",
        optimization_config.get("sanitize_diagnosis_embedding_grad", True),
    )
    code_embedding_grad_max_norm = optimization_config.get(
        "code_embedding_grad_max_norm",
        optimization_config.get("diagnosis_embedding_grad_max_norm", 0.5),
    )
    freeze_code_embedding_epochs = optimization_config.get(
        "freeze_code_embedding_epochs",
        optimization_config.get("diagnosis_embedding_freeze_epochs", 1),
    )
    debug_checks_config = dict(train_config.get("debug_checks", {}))
    debug_checks_enabled = debug_checks_config.get("enabled", True)
    debug_checks_light_mode = debug_checks_config.get("light_mode", True)
    debug_check_every_n_steps = debug_checks_config.get("check_every_n_steps", 100)
    sync_timing = debug_checks_config.get("sync_timing", False)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=resolved_paths["checkpoint_dir"],
        log_dir=resolved_paths["log_dir"],
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        validation_threshold=threshold,
        use_self_history=bool(baseline_settings["use_self_history"]),
        use_ddi=bool(baseline_settings["use_ddi"]),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_grad_norm=max_grad_norm,
        sanitize_code_embedding_grads=sanitize_code_embedding_grads,
        code_embedding_grad_max_norm=code_embedding_grad_max_norm,
        freeze_code_embedding_epochs=freeze_code_embedding_epochs,
        debug_checks_enabled=debug_checks_enabled,
        debug_checks_light_mode=debug_checks_light_mode,
        debug_check_every_n_steps=debug_check_every_n_steps,
        sync_timing=sync_timing,
    )

    training_epochs = int(
        args.epochs
        if args.epochs is not None
        else train_config.get("optimization", {}).get("epochs", 10)
    )
    if smoke_test:
        training_epochs = 1
    print(f"  epochs: {training_epochs}")

    fit_result = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=training_epochs,
        extra_checkpoint_state={
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
            "resolved_paths": {key: str(value) for key, value in resolved_paths.items()},
            "seed": seed,
            "baseline_mode": str(baseline_settings["mode"]),
            "threshold": threshold,
            "smoke_test": smoke_test,
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
            "max_eval_batches": max_eval_batches,
        },
    )

    print(f"Best checkpoint: {fit_result['best_checkpoint_path']}")
    print(f"Monitor metric: {fit_result['monitor_metric']} (best={fit_result['best_metric']:.6f})")

    best_checkpoint_path = fit_result.get("best_checkpoint_path")
    if not best_checkpoint_path:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    best_lambda_config_path = ensure_dir(resolved_paths["report_dir"]) / f"best_lambda_ddi_config_{baseline_settings['mode']}.json"
    write_json(
        best_lambda_config_path,
        {
            "baseline_mode": str(baseline_settings["mode"]),
            "best_checkpoint_path": str(best_checkpoint_path),
            "monitor_metric": fit_result["monitor_metric"],
            "best_metric": float(fit_result["best_metric"]),
            "loss": copy.deepcopy(dict(train_config.get("loss", {}))),
            "training": copy.deepcopy(dict(train_config.get("training", {}))),
        },
    )

    from src.evaluation.evaluate_core import evaluate_checkpoint

    eval_config_path = resolve_path(project_root, "configs/eval.yaml")
    evaluation_report = evaluate_checkpoint(
        checkpoint_path=Path(str(best_checkpoint_path)),
        eval_config_path=eval_config_path,
        split="test",
        threshold=args.threshold,
        device=device,
        max_eval_batches=max_eval_batches,
    )
    if hasattr(model, "get_retrieval_policy"):
        retrieval_policy_path = ensure_dir(resolved_paths["report_dir"]) / "retrieval_policy.json"
        write_json(retrieval_policy_path, model.get_retrieval_policy())
    else:
        retrieval_policy_path = None
    baseline_report_path = ensure_dir(resolved_paths["report_dir"]) / f"baseline_report_{baseline_settings['mode']}.json"
    baseline_report_payload = {
        "baseline_mode": str(baseline_settings["mode"]),
        "seed": seed,
        "threshold": threshold,
        "smoke_test": smoke_test,
        "monitor_metric": fit_result["monitor_metric"],
        "best_metric": float(fit_result["best_metric"]),
        "best_checkpoint_path": str(best_checkpoint_path),
        "training_artifacts": {
            "metrics_per_epoch_json": fit_result["metrics_per_epoch_json"],
            "metrics_per_epoch_csv": fit_result["metrics_per_epoch_csv"],
            "best_metrics_json": fit_result["best_metrics_json"],
            "best_lambda_ddi_config_json": str(best_lambda_config_path),
            "retrieval_policy_json": None if retrieval_policy_path is None else str(retrieval_policy_path),
        },
        "evaluation": evaluation_report,
    }
    write_json(baseline_report_path, baseline_report_payload)
    print(f"Baseline report: {baseline_report_path}")


if __name__ == "__main__":
    main()
