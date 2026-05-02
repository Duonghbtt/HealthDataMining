from __future__ import annotations

from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.data.build_vocab import load_vocab_bundle
from src.data.dataset import MIMICTrajectoryDataset
from src.models.ddi_regularization import load_ddi_matrix
from src.models.fusion import FusionModule
from src.models.history_selector import HistorySelector, SelfHistorySelector
from src.models.medication_decoder import MedicationDecoder
from src.models.patient_state_encoder import PatientStateEncoder
from src.retrieval.memory_bank import VisitMemoryBank
from src.retrieval.topk_retriever import TopKVisitRetriever
from src.training.losses import (
    build_medication_loss_config,
    compute_medication_losses,
    extract_last_valid_targets,
)
from src.utils.io import load_yaml_config, resolve_path

_VALID_HISTORY_MODES = {"self_only", "retrieval_only", "self_retrieval"}


def _tensor_debug_summary(name: str, value: torch.Tensor) -> str:
    resolved = value.detach()
    shape = tuple(resolved.shape)
    dtype = resolved.dtype
    if resolved.numel() == 0:
        return f"{name}: shape={shape} dtype={dtype} numel=0"
    if not (resolved.is_floating_point() or resolved.is_complex()):
        return (
            f"{name}: shape={shape} dtype={dtype} "
            f"min={resolved.min().item()} max={resolved.max().item()}"
        )
    nan_count = int(torch.isnan(resolved).sum().item())
    inf_count = int(torch.isinf(resolved).sum().item())
    finite = resolved[torch.isfinite(resolved)]
    if finite.numel() == 0:
        return (
            f"{name}: shape={shape} dtype={dtype} nan_count={nan_count} "
            f"inf_count={inf_count} finite_values=0"
        )
    return (
        f"{name}: shape={shape} dtype={dtype} nan_count={nan_count} "
        f"inf_count={inf_count} min={finite.min().item():.6g} "
        f"max={finite.max().item():.6g} mean={finite.mean().item():.6g}"
    )


def _assert_finite(name: str, value: Any) -> None:
    if not isinstance(value, torch.Tensor):
        return
    if value.is_floating_point() or value.is_complex():
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Non-finite tensor detected in full_model at `{name}`; "
                f"{_tensor_debug_summary(name, value)}"
            )


def _assert_finite_tree(prefix: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _assert_finite(prefix, value)
        return
    if isinstance(value, MappingABC):
        for key, child in value.items():
            _assert_finite_tree(f"{prefix}.{key}", child)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(f"{prefix}[{index}]", child)


def _should_debug_check(batch: Mapping[str, Any] | None) -> bool:
    if batch is None:
        return True
    return bool(batch.get("_debug_check_now", True))


def _log_progress_line(message: str) -> None:
    writer = getattr(tqdm, "write", None)
    if callable(writer):
        writer(message)
        return
    print(message)


def extract_last_valid_state(state_sequence: torch.Tensor, visit_mask: torch.Tensor) -> torch.Tensor:
    """Extract the hidden state of the last valid visit.

    Shapes:
    - state_sequence: [B, T, H]
    - visit_mask: [B, T]
    - return: [B, H]
    """

    if state_sequence.ndim != 3:
        raise ValueError(f"state_sequence must have shape (B, T, H), got {tuple(state_sequence.shape)}")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    if tuple(state_sequence.shape[:2]) != tuple(visit_mask.shape):
        raise ValueError(
            "state_sequence and visit_mask must align on batch/time dimensions: "
            f"got {tuple(state_sequence.shape[:2])} and {tuple(visit_mask.shape)}"
        )

    resolved_mask = visit_mask.to(device=state_sequence.device, dtype=torch.bool)
    valid_counts = resolved_mask.sum(dim=1)
    if bool((valid_counts <= 0).any().item()):
        raise ValueError("Each sample must contain at least one valid visit")

    last_indices = valid_counts.to(dtype=torch.long) - 1
    batch_indices = torch.arange(state_sequence.shape[0], device=state_sequence.device)
    return state_sequence[batch_indices, last_indices]


def _infer_numeric_feature_sizes(data_config_path: str | Path) -> tuple[int, int]:
    dataset = MIMICTrajectoryDataset("train", data_config_path)
    lab_feature_size = int(getattr(dataset, "default_lab_feature_size", 0))
    vital_feature_size = int(getattr(dataset, "default_vital_feature_size", 0))
    if lab_feature_size > 0 or vital_feature_size > 0:
        return lab_feature_size, vital_feature_size

    if len(dataset) <= 0:
        raise ValueError("Training split is empty; cannot infer lab/vital feature sizes from dataset.")
    sample = dataset[0]
    return int(sample.get("lab_feature_size", 0)), int(sample.get("vital_feature_size", 0))


def _load_optional_ddi_matrix(
    train_config_path: str | Path | None,
) -> tuple[torch.Tensor | None, float]:
    if train_config_path is None:
        return None, 0.0

    train_config = load_yaml_config(train_config_path)
    lambda_ddi = float(train_config.get("loss", {}).get("lambda_ddi", 0.0))
    ddi_path_value = train_config.get("paths", {}).get("ddi_matrix_path")
    if not ddi_path_value:
        return None, lambda_ddi

    ddi_path = resolve_path(train_config["_project_root"], ddi_path_value)
    if not ddi_path.exists():
        return None, lambda_ddi
    return load_ddi_matrix(ddi_path, device="cpu"), lambda_ddi


class FullMedicationModel(nn.Module):
    """Core medication recommendation model with optional self-history and retrieval branches."""

    def __init__(
        self,
        encoder: PatientStateEncoder,
        history_selector: SelfHistorySelector | HistorySelector,
        fusion_module: FusionModule,
        *,
        medication_decoder: MedicationDecoder | None = None,
        decoder: MedicationDecoder | None = None,
        retriever: TopKVisitRetriever | None = None,
        ddi_matrix: torch.Tensor | None = None,
        lambda_ddi: float = 0.0,
        use_self_history: bool = True,
        history_mode: str = "self_only",
        return_retrieval_aux: bool = True,
        loss_config: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.self_history_selector = history_selector
        self.fusion_module = fusion_module
        self.medication_decoder = medication_decoder if medication_decoder is not None else decoder
        self.retriever = retriever
        self.lambda_ddi = float(lambda_ddi)
        self.use_self_history = bool(use_self_history)
        self.history_mode = str(history_mode).strip().lower()
        if self.history_mode not in _VALID_HISTORY_MODES:
            raise ValueError(f"history_mode must be one of {_VALID_HISTORY_MODES}, got {history_mode!r}")
        self.return_retrieval_aux = bool(return_retrieval_aux)
        self.loss_config = build_medication_loss_config(loss_config=loss_config)
        self.loss_config["lambda_ddi"] = float(lambda_ddi)
        self.retrieval_memory_bank: VisitMemoryBank | None = None
        if ddi_matrix is None:
            self.register_buffer("ddi_matrix", None)
        else:
            self.register_buffer("ddi_matrix", torch.as_tensor(ddi_matrix, dtype=torch.float32))

    @classmethod
    def from_config(
        cls,
        *,
        data_config_path: str | Path,
        model_config_path: str | Path,
        train_config_path: str | Path | None = None,
    ) -> "FullMedicationModel":
        data_config = load_yaml_config(data_config_path)
        model_config = load_yaml_config(model_config_path)
        vocab_bundle = load_vocab_bundle(data_config)
        lab_feature_size, vital_feature_size = _infer_numeric_feature_sizes(data_config_path)

        hidden_dim = int(model_config.get("model", {}).get("hidden_dim", 128))
        model_dropout = float(model_config.get("model", {}).get("dropout", 0.1))
        embedding_cfg = dict(model_config.get("embedding", {}))
        encoder_cfg = dict(model_config.get("encoder", {}))
        history_cfg = dict(model_config.get("history_selector", {}))
        retrieval_cfg = dict(model_config.get("retrieval", {}))
        debug_cfg = dict(model_config.get("debug", {}))
        full_model_cfg = dict(model_config.get("full_model", {}))
        fusion_cfg = dict(model_config.get("fusion", {}))
        decoder_cfg: dict[str, Any] = {}
        loss_cfg: dict[str, Any] = {}

        code_embedding_dim = int(embedding_cfg.get("diag_dim", hidden_dim))
        proc_dim = int(embedding_cfg.get("proc_dim", code_embedding_dim))
        if proc_dim != code_embedding_dim:
            raise ValueError(
                "PatientStateEncoder requires matching diagnosis/procedure embedding widths in the new core pipeline."
            )

        numeric_projection_dim = int(embedding_cfg.get("lab_dim", 64))
        vital_dim = int(embedding_cfg.get("vital_dim", numeric_projection_dim))
        if vital_dim != numeric_projection_dim:
            raise ValueError(
                "PatientStateEncoder requires matching lab/vital projection widths in the new core pipeline."
            )

        ddi_matrix, lambda_ddi = _load_optional_ddi_matrix(train_config_path)
        use_self_history = True
        if train_config_path is not None:
            train_config = load_yaml_config(train_config_path)
            baseline_cfg = dict(train_config.get("baseline", {}))
            decoder_cfg = dict(train_config.get("decoder", {}))
            loss_cfg = build_medication_loss_config(
                loss_config=train_config.get("loss", {}),
                training_config=train_config.get("training", {}),
            )
            use_self_history = bool(baseline_cfg.get("use_self_history", True))
            if not bool(baseline_cfg.get("use_ddi", True)):
                lambda_ddi = 0.0
                loss_cfg["use_ddi"] = False
                loss_cfg["lambda_ddi"] = 0.0
            elif baseline_cfg.get("lambda_ddi") is not None:
                lambda_ddi = float(baseline_cfg["lambda_ddi"])
                loss_cfg["lambda_ddi"] = lambda_ddi
        encoder = PatientStateEncoder(
            diagnosis_vocab_size=len(vocab_bundle["diagnosis"]["idx_to_token"]),
            procedure_vocab_size=len(vocab_bundle["procedure"]["idx_to_token"]),
            drug_vocab_size=len(vocab_bundle["drug"]["idx_to_token"]),
            num_lab_features=lab_feature_size,
            num_vital_features=vital_feature_size,
            code_embedding_dim=code_embedding_dim,
            medication_embedding_dim=int(embedding_cfg.get("drug_dim", hidden_dim)),
            numeric_projection_dim=numeric_projection_dim,
            time_embedding_dim=int(embedding_cfg.get("time_dim", 32)),
            visit_hidden_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=model_dropout,
            encoder_mode=str(encoder_cfg.get("mode", "legacy_gru")),
            modality_hidden_dim=encoder_cfg.get("modality_hidden_dim"),
            fusion_hidden_dim=encoder_cfg.get("fusion_hidden_dim"),
            modality_dropout=encoder_cfg.get("modality_dropout"),
            use_temporal_attention=bool(encoder_cfg.get("use_temporal_attention", True)),
            temporal_attention_heads=int(encoder_cfg.get("temporal_attention_heads", 1)),
            temporal_attention_dropout=encoder_cfg.get("temporal_attention_dropout"),
        )
        history_selector = SelfHistorySelector(
            hidden_dim=hidden_dim,
            dropout=float(history_cfg.get("dropout", model_dropout)),
            self_top_k=history_cfg.get("top_k", history_cfg.get("self_top_k")),
            selection_mode=str(
                history_cfg.get(
                    "mode",
                    "visit_only" if bool(history_cfg.get("enabled", True)) else "none",
                )
            ),
            attention_type=str(history_cfg.get("attention_type", "softmax_topk")),
            return_attention_weights=bool(history_cfg.get("return_attention_weights", True)),
            save_selected_indices=bool(history_cfg.get("save_selected_indices", True)),
        )
        fusion_module = FusionModule(
            hidden_dim=hidden_dim,
            dropout=float(fusion_cfg.get("dropout", model_dropout)),
            strategy=str(fusion_cfg.get("mode", fusion_cfg.get("strategy", "gated"))),
        )
        retriever = None
        if bool(retrieval_cfg.get("enabled", False)):
            retriever = TopKVisitRetriever(
                hidden_dim=hidden_dim,
                drug_vocab_size=len(vocab_bundle["drug"]["idx_to_token"]),
                top_k=int(retrieval_cfg.get("top_k", 5)),
                backend=str(retrieval_cfg.get("mode", retrieval_cfg.get("backend", "bruteforce"))),
                use_faiss_if_available=bool(retrieval_cfg.get("use_faiss_if_available", True)),
                similarity_mode=str(retrieval_cfg.get("similarity_mode", "cosine_decay")),
                temporal_decay_alpha=float(retrieval_cfg.get("temporal_decay_alpha", 0.05)),
                allow_same_patient=bool(retrieval_cfg.get("allow_same_patient", False)),
                exclude_future=bool(retrieval_cfg.get("exclude_future", True)),
                exclude_exact_match=bool(retrieval_cfg.get("exclude_exact_match", True)),
                exclude_future_same_patient=retrieval_cfg.get("exclude_future_same_patient"),
                exclude_future_all_patients_if_absolute_time=bool(
                    retrieval_cfg.get("exclude_future_all_patients_if_absolute_time", True)
                ),
                require_absolute_time_for_cross_patient_temporal_filter=bool(
                    retrieval_cfg.get("require_absolute_time_for_cross_patient_temporal_filter", False)
                ),
                use_time_gap=bool(retrieval_cfg.get("use_time_gap", True)),
                dropout=float(retrieval_cfg.get("dropout", model_dropout)),
            )
        decoder = MedicationDecoder(
            hidden_dim=hidden_dim,
            drug_vocab_size=len(vocab_bundle["drug"]["idx_to_token"]),
            dropout=float(decoder_cfg.get("dropout", model_dropout)),
            hidden_multiplier=int(decoder_cfg.get("hidden_multiplier", 2)),
            activation=str(decoder_cfg.get("activation", "relu")),
            layer_norm=bool(decoder_cfg.get("layer_norm", True)),
            decoder_type=str(decoder_cfg.get("type", "residual_mlp")),
            decoder_mode=decoder_cfg.get("mode", "legacy"),
            gate_type=str(decoder_cfg.get("gate_type", "scalar")),
            use_history_copy=bool(decoder_cfg.get("use_history_copy", True)),
            use_retrieval_copy=bool(decoder_cfg.get("use_retrieval_copy", True)),
            use_memory_copy=bool(decoder_cfg.get("use_memory_copy", False)),
            copy_projection=str(decoder_cfg.get("copy_projection", "none")),
            gate_hidden_dim=decoder_cfg.get("gate_hidden_dim"),
        )
        return cls(
            encoder,
            history_selector,
            fusion_module,
            medication_decoder=decoder,
            retriever=retriever,
            ddi_matrix=ddi_matrix,
            lambda_ddi=lambda_ddi,
            use_self_history=use_self_history,
            history_mode=str(
                full_model_cfg.get(
                    "history_mode",
                    "self_retrieval" if retriever is not None and use_self_history else "retrieval_only" if retriever is not None else "self_only",
                )
            ),
            return_retrieval_aux=bool(debug_cfg.get("return_retrieval_aux", True)),
            loss_config=loss_cfg,
        )

    def _resolve_ddi_matrix(self, batch: Mapping[str, Any]) -> torch.Tensor | None:
        ddi_adj = batch.get("ddi_adj")
        if ddi_adj is None:
            return self.ddi_matrix
        return torch.as_tensor(ddi_adj, dtype=torch.float32)

    def _resolve_current_visit_indices(
        self,
        *,
        batch: Mapping[str, Any],
        visit_mask: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(batch.get("visit_index"), torch.Tensor):
            return torch.as_tensor(batch["visit_index"], device=visit_mask.device, dtype=torch.long)
        return visit_mask.sum(dim=1).to(dtype=torch.long) - 1

    @property
    def use_retrieval(self) -> bool:
        return self.retriever is not None and self.history_mode in {"retrieval_only", "self_retrieval"}

    def set_retrieval_memory_bank(self, memory_bank: VisitMemoryBank | None) -> None:
        self.retrieval_memory_bank = memory_bank
        if self.retriever is not None:
            self.retriever.set_memory_bank(memory_bank)

    def get_retrieval_policy(self) -> dict[str, Any]:
        if self.retriever is None:
            return {
                "memory_bank_split": None,
                "has_absolute_time": False,
                "all_visits_have_absolute_time": False,
                "exact_match_blocked": False,
                "same_patient_future_blocked": False,
                "cross_patient_absolute_temporal_filter": False,
                "notes": "Retrieval branch is disabled in this model.",
            }
        return self.retriever.describe_leakage_policy(memory_bank=self.retrieval_memory_bank)

    def clear_retrieval_memory_bank(self) -> None:
        self.set_retrieval_memory_bank(None)

    def _move_batch_to_device(self, batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def refresh_retrieval_memory_bank(
        self,
        dataloader: DataLoader,
        *,
        split_name: str = "train",
        device: torch.device | None = None,
        progress_desc: str | None = None,
    ) -> VisitMemoryBank | None:
        if self.retriever is None:
            return None

        resolved_device = device or next(self.parameters()).device
        memory_bank = VisitMemoryBank(split_name=split_name, time_is_absolute=False)
        was_training = self.training
        self.eval()
        resolved_progress_desc = str(progress_desc or f"Refreshing retrieval bank ({split_name})")
        progress = tqdm(
            dataloader,
            desc=resolved_progress_desc,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
        try:
            with torch.no_grad():
                for batch in progress:
                    batch_on_device = self._move_batch_to_device(batch, resolved_device)
                    encoder_batch = dict(batch_on_device)
                    encoder_batch["_debug_check_now"] = False
                    enc_out = self.encoder(encoder_batch)
                    state_sequence = enc_out.get("history_states", enc_out["state_sequence"]).detach().cpu()
                    visit_mask = enc_out["visit_mask"].detach().cpu()
                    target_drugs = batch_on_device.get("target_drugs")
                    if not isinstance(target_drugs, torch.Tensor):
                        raise RuntimeError("Retrieval memory bank build requires `target_drugs` in each batch.")
                    patient_ids = batch.get("patient_ids", batch.get("subject_ids"))
                    if patient_ids is None:
                        raise RuntimeError("Retrieval memory bank build requires `patient_ids` or `subject_ids`.")
                    batch_metadata = {
                        "subject_id": list(batch.get("subject_ids", [])),
                        "hadm_id": list(batch.get("hadm_ids", [])),
                        "stay_id": list(batch.get("stay_ids", [])),
                    }
                    memory_bank.add_batch(
                        patient_ids=patient_ids,
                        visit_embeddings=state_sequence,
                        medication_evidence=target_drugs.detach().cpu(),
                        visit_mask=visit_mask,
                        time_delta_hours=None
                        if not isinstance(batch_on_device.get("time_delta_hours"), torch.Tensor)
                        else batch_on_device["time_delta_hours"].detach().cpu(),
                        visit_time_absolute_hours=None
                        if not isinstance(batch_on_device.get("visit_time_absolute_hours"), torch.Tensor)
                        else batch_on_device["visit_time_absolute_hours"].detach().cpu(),
                        visit_time_absolute_mask=None
                        if not isinstance(batch_on_device.get("visit_time_absolute_mask"), torch.Tensor)
                        else batch_on_device["visit_time_absolute_mask"].detach().cpu(),
                        batch_metadata=batch_metadata,
                    )
        finally:
            close = getattr(progress, "close", None)
            if callable(close):
                close()
            if was_training:
                self.train()
        memory_bank.validate()
        self.set_retrieval_memory_bank(memory_bank)
        _log_progress_line(
            f"Finished retrieval memory bank refresh: split={split_name} visits={memory_bank.num_visits}"
        )
        return memory_bank

    def _zero_selection_outputs(
        self,
        *,
        current_state: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "selection_mode": "none",
            "selected_history_context": torch.zeros_like(current_state),
            "history_context": torch.zeros_like(current_state),
            "visit_context": torch.zeros_like(current_state),
            "self_history_summary": torch.zeros_like(current_state),
            "medication_history_context": torch.zeros_like(current_state),
            "visit_attention_weights": torch.zeros(
                visit_mask.shape[0],
                visit_mask.shape[1],
                device=visit_mask.device,
                dtype=current_state.dtype,
            ),
            "self_attention_weights": torch.zeros(
                visit_mask.shape[0],
                visit_mask.shape[1],
                device=visit_mask.device,
                dtype=current_state.dtype,
            ),
            "attribute_contexts": {},
            "attribute_attention_weights": {},
            "attribute_fusion_weights": {},
            "selected_visit_indices": torch.empty(
                visit_mask.shape[0],
                0,
                device=visit_mask.device,
                dtype=torch.long,
            ),
            "selected_visit_mask": torch.zeros_like(visit_mask, dtype=torch.bool),
        }

    def _resolve_current_visit_metadata(
        self,
        *,
        batch: Mapping[str, Any],
        visit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        patient_ids = batch.get("patient_ids", batch.get("subject_ids"))
        if patient_ids is None:
            raise RuntimeError("Retrieval branch requires `patient_ids` or `subject_ids` in the batch.")
        patient_id_tensor = torch.as_tensor(patient_ids, device=visit_mask.device, dtype=torch.long)
        visit_indices = self._resolve_current_visit_indices(batch=batch, visit_mask=visit_mask)

        absolute_visit_times = batch.get("visit_time_absolute_hours")
        absolute_visit_time_mask = batch.get("visit_time_absolute_mask")
        batch_indices = torch.arange(visit_mask.shape[0], device=visit_mask.device)
        if isinstance(absolute_visit_times, torch.Tensor):
            absolute_visit_time_tensor = torch.as_tensor(
                absolute_visit_times,
                device=visit_mask.device,
                dtype=torch.float32,
            )
            absolute_time_available = (
                torch.as_tensor(
                    absolute_visit_time_mask,
                    device=visit_mask.device,
                    dtype=torch.bool,
                )
                if isinstance(absolute_visit_time_mask, torch.Tensor)
                else visit_mask.to(dtype=torch.bool)
            )
            current_absolute_available = absolute_time_available[batch_indices, visit_indices]
            current_visit_times = absolute_visit_time_tensor[batch_indices, visit_indices]
            return patient_id_tensor, visit_indices, current_visit_times, current_absolute_available
        return patient_id_tensor, visit_indices, None, None

    def _zero_drug_bag(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        drug_vocab_size = int(
            getattr(self.medication_decoder, "drug_vocab_size", 0)
            if self.medication_decoder is not None
            else 0
        )
        return torch.zeros(batch_size, drug_vocab_size, device=device, dtype=dtype)

    def _build_history_med_bag(
        self,
        *,
        batch: Mapping[str, Any],
        visit_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = int(visit_mask.shape[0])
        device = visit_mask.device
        history_med_bag = self._zero_drug_bag(batch_size=batch_size, device=device, dtype=dtype)
        if not (self.use_self_history and self.history_mode in {"self_only", "self_retrieval"}):
            return history_med_bag

        visit_indices = self._resolve_current_visit_indices(batch=batch, visit_mask=visit_mask)
        batch_indices = torch.arange(batch_size, device=device)
        drug_vocab_size = int(history_med_bag.shape[1])

        # The final medication label space is the drug vocab itself, so the copy
        # branch should reuse only valid medication ids and skip PAD/UNK.
        med_history = batch.get("med_history")
        if isinstance(med_history, torch.Tensor) and med_history.ndim == 3:
            med_history_tensor = torch.as_tensor(med_history, device=device, dtype=torch.long)
            current_ids = med_history_tensor[batch_indices, visit_indices]
            current_mask_tensor = batch.get("med_history_mask")
            if isinstance(current_mask_tensor, torch.Tensor):
                current_mask = torch.as_tensor(
                    current_mask_tensor,
                    device=device,
                    dtype=torch.bool,
                )[batch_indices, visit_indices]
            else:
                current_mask = current_ids > 0
            valid_ids = current_mask & (current_ids >= 2) & (current_ids < drug_vocab_size)
            if bool(valid_ids.any().item()):
                row_index = batch_indices.unsqueeze(1).expand_as(current_ids)
                history_med_bag[row_index[valid_ids], current_ids[valid_ids]] = 1.0
            return history_med_bag

        target_drugs = batch.get("target_drugs")
        if isinstance(target_drugs, torch.Tensor) and target_drugs.ndim == 3 and int(target_drugs.shape[-1]) == drug_vocab_size:
            target_tensor = torch.as_tensor(target_drugs, device=device, dtype=dtype)
            step_index = torch.arange(target_tensor.shape[1], device=device).unsqueeze(0)
            history_only_mask = visit_mask.to(device=device, dtype=torch.bool) & (step_index < visit_indices.unsqueeze(1))
            if target_tensor.shape[1] > 0:
                history_med_bag = (
                    target_tensor * history_only_mask.unsqueeze(-1).to(dtype=target_tensor.dtype)
                ).amax(dim=1)
                history_med_bag[:, : min(2, drug_vocab_size)] = 0.0
            return history_med_bag.to(dtype=dtype)
        return history_med_bag

    def _zero_retrieval_outputs(self, *, current_state: torch.Tensor) -> dict[str, Any]:
        if self.retriever is not None:
            return self.retriever.retrieve(
                current_state=current_state,
                current_patient_ids=torch.zeros(current_state.shape[0], device=current_state.device, dtype=torch.long),
                current_visit_indices=torch.zeros(current_state.shape[0], device=current_state.device, dtype=torch.long),
                memory_bank=None,
                return_metadata=False,
            )
        batch_size = current_state.shape[0]
        drug_vocab_size = int(
            getattr(self.medication_decoder, "drug_vocab_size", 0)
            if self.medication_decoder is not None
            else 0
        )
        return {
            "aggregated_retrieval_context": torch.zeros_like(current_state),
            "retrieval_medication_context": torch.zeros(
                batch_size,
                drug_vocab_size,
                device=current_state.device,
                dtype=current_state.dtype,
            ),
            "retrieved_indices": torch.empty(batch_size, 0, device=current_state.device, dtype=torch.long),
            "retrieved_scores": torch.empty(batch_size, 0, device=current_state.device, dtype=current_state.dtype),
            "retrieval_weights": torch.empty(batch_size, 0, device=current_state.device, dtype=current_state.dtype),
            "retrieved_medication_evidence": torch.empty(
                batch_size,
                0,
                drug_vocab_size,
                device=current_state.device,
                dtype=current_state.dtype,
            ),
            "retrieved_metadata": [[] for _ in range(batch_size)],
            "valid_candidate_counts": torch.zeros(batch_size, device=current_state.device, dtype=torch.long),
            "avg_valid_candidates": 0.0,
            "avg_retrieved_score": 0.0,
        }

    def forward(self, batch: Mapping[str, Any], **_: Any) -> dict[str, Any]:
        return_aux = bool(_.get("return_aux", False)) or self.return_retrieval_aux
        debug_checks_enabled = _should_debug_check(batch)
        encoder_batch = dict(batch)
        encoder_batch["_debug_check_now"] = debug_checks_enabled
        if not self.use_self_history and isinstance(encoder_batch.get("med_history"), torch.Tensor):
            encoder_batch["med_history"] = torch.zeros_like(encoder_batch["med_history"])
            if isinstance(encoder_batch.get("med_history_mask"), torch.Tensor):
                encoder_batch["med_history_mask"] = torch.zeros_like(encoder_batch["med_history_mask"])

        enc_out = self.encoder(encoder_batch)
        state_sequence = enc_out.get("history_states", enc_out["state_sequence"])  # [B, T, H]
        visit_mask = enc_out["visit_mask"]                                       # [B, T]
        modality_history_states = enc_out.get("modality_history_states")
        current_state = enc_out.get("current_state")
        if current_state is None:
            current_state = extract_last_valid_state(state_sequence, visit_mask)  # [B, H]
        if debug_checks_enabled:
            _assert_finite("encoder.visit_repr", enc_out["visit_repr"])
            _assert_finite("encoder.state_sequence", state_sequence)
            _assert_finite("encoder.current_state", current_state)
            _assert_finite("encoder.pooled_state", enc_out.get("pooled_state"))
            _assert_finite("encoder.temporal_attention_weights", enc_out.get("temporal_attention_weights"))
            _assert_finite_tree("encoder.modality_summary", enc_out.get("modality_summary"))
            _assert_finite_tree("encoder.modality_history_states", modality_history_states)
            _assert_finite("encoder.modality_gate_weights", enc_out.get("modality_gate_weights"))
            _assert_finite("encoder.numeric_gate_weights", enc_out.get("numeric_gate_weights"))

        use_self_branch = self.use_self_history and self.history_mode in {"self_only", "self_retrieval"}
        if use_self_branch:
            sel_out = self.self_history_selector(
                current_state=current_state,
                history_states=state_sequence,
                visit_mask=visit_mask,
                modality_history_states=modality_history_states,
            )
        else:
            sel_out = self._zero_selection_outputs(current_state=current_state, visit_mask=visit_mask)
        if debug_checks_enabled:
            _assert_finite("history.selected_history_context", sel_out.get("selected_history_context"))
            _assert_finite("history.self_history_summary", sel_out.get("self_history_summary"))
            _assert_finite("history.medication_history_context", sel_out.get("medication_history_context"))
            _assert_finite("history.visit_attention_weights", sel_out.get("visit_attention_weights"))
            _assert_finite("history.self_attention_weights", sel_out.get("self_attention_weights"))
            _assert_finite_tree("history.attribute_contexts", sel_out.get("attribute_contexts"))
            _assert_finite_tree("history.attribute_attention_weights", sel_out.get("attribute_attention_weights"))

        if self.use_retrieval:
            (
                current_patient_ids,
                current_visit_indices,
                current_visit_times,
                current_visit_times_are_absolute,
            ) = self._resolve_current_visit_metadata(
                batch=batch,
                visit_mask=visit_mask,
            )
            retrieval_out = self.retriever.retrieve(
                current_state=current_state,
                current_patient_ids=current_patient_ids,
                current_visit_indices=current_visit_indices,
                current_visit_times=current_visit_times,
                current_visit_times_are_absolute=current_visit_times_are_absolute,
                memory_bank=self.retrieval_memory_bank,
                return_metadata=return_aux,
            )
        else:
            retrieval_out = self._zero_retrieval_outputs(current_state=current_state)
        if debug_checks_enabled:
            _assert_finite("retrieval.aggregated_retrieval_context", retrieval_out.get("aggregated_retrieval_context"))
            _assert_finite("retrieval.medication_context", retrieval_out.get("retrieval_medication_context"))
            _assert_finite("retrieval.scores", retrieval_out.get("retrieved_scores"))
            _assert_finite("retrieval.weights", retrieval_out.get("retrieval_weights"))
            _assert_finite("retrieval.medication_evidence", retrieval_out.get("retrieved_medication_evidence"))

        fusion_out = self.fusion_module(
            current_state=current_state,
            self_history_summary=sel_out["self_history_summary"],
            selected_self_history=sel_out.get("selected_history_context"),
            medication_history_context=sel_out.get("medication_history_context"),
            retrieval_context=retrieval_out.get("aggregated_retrieval_context"),
            attribute_contexts=sel_out.get("attribute_contexts"),
        )
        if debug_checks_enabled:
            _assert_finite("fusion.context_vector", fusion_out.get("context_vector"))
            _assert_finite_tree("fusion.gates", fusion_out.get("fusion_gates"))
            _assert_finite_tree("fusion.components", fusion_out.get("fusion_components"))

        if self.medication_decoder is None:
            raise RuntimeError("FullMedicationModel requires a MedicationDecoder for forward inference.")
        history_med_bag = self._build_history_med_bag(
            batch=batch,
            visit_mask=visit_mask,
            dtype=current_state.dtype,
        )
        retrieval_med_bag = torch.as_tensor(
            retrieval_out.get("retrieval_medication_context", self._zero_drug_bag(
                batch_size=current_state.shape[0],
                device=current_state.device,
                dtype=current_state.dtype,
            )),
            device=current_state.device,
            dtype=current_state.dtype,
        )
        if debug_checks_enabled:
            _assert_finite("decoder.history_med_bag", history_med_bag)
            _assert_finite("decoder.retrieval_med_bag", retrieval_med_bag)
        dec_out = self.medication_decoder(
            fusion_out["context_vector"],
            current_state=current_state,
            history_context=sel_out.get("selected_history_context", sel_out["self_history_summary"]),
            retrieval_context=retrieval_out.get("aggregated_retrieval_context"),
            history_med_bag=history_med_bag,
            retrieval_med_bag=retrieval_med_bag,
        )
        if debug_checks_enabled:
            _assert_finite("decoder.logits_new", dec_out.get("logits_new"))
            _assert_finite("decoder.logits_copy", dec_out.get("logits_copy"))
            _assert_finite("decoder.gate", dec_out.get("gate"))
            _assert_finite("decoder.copy_signal", dec_out.get("copy_signal"))
            _assert_finite("decoder.drug_logits", dec_out["drug_logits"])
            _assert_finite("decoder.drug_probs", dec_out.get("drug_probs"))

        target_drugs = batch.get("target_drugs")
        target_current: torch.Tensor | None = None
        prediction_loss: torch.Tensor | None = None
        pred_bce_loss: torch.Tensor | None = None
        margin_loss: torch.Tensor | None = None
        weighted_margin_loss: torch.Tensor | None = None
        ddi_loss: torch.Tensor | None = None
        weighted_ddi_loss: torch.Tensor | None = None
        lambda_ddi_current: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None

        if target_drugs is not None:
            target_tensor = torch.as_tensor(
                target_drugs,
                device=dec_out["drug_logits"].device,
                dtype=dec_out["drug_logits"].dtype,
            )
            target_current = extract_last_valid_targets(target_tensor, visit_mask)
            epoch_value = batch.get("_current_epoch")
            loss_outputs = compute_medication_losses(
                drug_logits=dec_out["drug_logits"],
                drug_probs=dec_out["drug_probs"],
                target_drugs=target_tensor,
                visit_mask=visit_mask,
                ddi_matrix=self._resolve_ddi_matrix(batch),
                lambda_ddi=self.lambda_ddi,
                reduction="mean",
                loss_config=self.loss_config,
                current_epoch=epoch_value,
            )
            prediction_loss = loss_outputs["prediction_loss"]
            pred_bce_loss = loss_outputs["pred_bce_loss"]
            margin_loss = loss_outputs["margin_loss"]
            weighted_margin_loss = loss_outputs["weighted_margin_loss"]
            ddi_loss = loss_outputs["ddi_loss"]
            weighted_ddi_loss = loss_outputs["weighted_ddi_loss"]
            lambda_ddi_current = loss_outputs["lambda_ddi_current"]
            total_loss = loss_outputs["total_loss"]
            target_current = loss_outputs["target_current"]

        return {
            "visit_repr": enc_out["visit_repr"],
            "state_sequence": state_sequence,
            "pooled_state": enc_out["pooled_state"],
            "visit_mask": visit_mask,
            "current_state": current_state,
            "history_states": state_sequence,
            "modality_summary": enc_out.get("modality_summary"),
            "modality_gate_weights": enc_out.get("modality_gate_weights"),
            "numeric_gate_weights": enc_out.get("numeric_gate_weights"),
            "temporal_attention_weights": enc_out.get("temporal_attention_weights"),
            "self_history_summary": sel_out["self_history_summary"],
            "self_attention_weights": sel_out["self_attention_weights"],
            "visit_attention_weights": sel_out.get("visit_attention_weights", sel_out["self_attention_weights"]),
            "attribute_contexts": sel_out.get("attribute_contexts"),
            "attribute_attention_weights": sel_out.get("attribute_attention_weights"),
            "attribute_fusion_weights": sel_out.get("attribute_fusion_weights"),
            "selected_history_context": sel_out.get("selected_history_context", sel_out["self_history_summary"]),
            "history_context": sel_out.get("history_context", sel_out["self_history_summary"]),
            "visit_context": sel_out.get("visit_context", sel_out["self_history_summary"]),
            "medication_history_context": sel_out.get("medication_history_context"),
            "selected_visit_indices": sel_out["selected_visit_indices"],
            "selected_visit_mask": sel_out.get("selected_visit_mask"),
            "selection_mode": sel_out.get("selection_mode"),
            "history_mode": self.history_mode,
            "context_vector": fusion_out["context_vector"],
            "fusion_gates": fusion_out.get("fusion_gates"),
            "fusion_components": fusion_out.get("fusion_components"),
            "fusion_attribute_gates": fusion_out.get("attribute_fusion_gates"),
            "retrieval_context": retrieval_out.get("aggregated_retrieval_context"),
            "retrieval_medication_context": retrieval_out.get("retrieval_medication_context"),
            "history_med_bag": history_med_bag,
            "retrieval_med_bag": retrieval_med_bag,
            "retrieved_indices": retrieval_out.get("retrieved_indices"),
            "retrieved_scores": retrieval_out.get("retrieved_scores"),
            "retrieval_weights": retrieval_out.get("retrieval_weights"),
            "retrieved_medication_evidence": retrieval_out.get("retrieved_medication_evidence"),
            "retrieved_metadata": retrieval_out.get("retrieved_metadata"),
            "retrieval_valid_candidate_counts": retrieval_out.get("valid_candidate_counts"),
            "retrieval_avg_valid_candidates": retrieval_out.get("avg_valid_candidates"),
            "retrieval_avg_score": retrieval_out.get("avg_retrieved_score"),
            "logits_new": dec_out.get("logits_new"),
            "logits_copy": dec_out.get("logits_copy"),
            "decoder_gate": dec_out.get("gate"),
            "decoder_gate_raw": dec_out.get("gate_raw"),
            "copy_signal": dec_out.get("copy_signal"),
            "copy_source_weights": dec_out.get("copy_source_weights"),
            "copy_source_mask": dec_out.get("copy_source_mask"),
            "decoder_mode": dec_out.get("decoder_mode"),
            "drug_logits": dec_out["drug_logits"],
            "drug_probs": dec_out["drug_probs"],
            "target_current": target_current,
            "final_target_drugs": target_current,
            "use_self_history": self.use_self_history,
            "use_retrieval": self.use_retrieval,
            "prediction_loss": prediction_loss,
            "pred_bce_loss": pred_bce_loss,
            "margin_loss": margin_loss,
            "weighted_margin_loss": weighted_margin_loss,
            "ddi_loss": ddi_loss,
            "weighted_ddi_loss": weighted_ddi_loss,
            "lambda_ddi_current": lambda_ddi_current,
            "total_loss": total_loss,
        }

__all__ = ["FullMedicationModel", "extract_last_valid_state"]
