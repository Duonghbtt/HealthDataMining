# ClinRec

ClinRec hiện được giữ ở trạng thái **truthful end-to-end** cho mức hệ thống:

**`core + history`**

Điều đó có nghĩa là luồng đã được verify và chạy xuyên suốt là:

```text
encoder -> history_selector -> fusion -> decoder -> loss
```

Trong core path hiện tại:

- `history` active thật và ảnh hưởng logits.
- `retrieval` không active trong `train_core.py`, `evaluate_core.py`, `evaluate_safety.py`, `evaluate_subgroup.py`.
- `DDI` có đi vào loss và report. Với local repo hiện tại, artifact active mặc định được build từ **TwoSIDES**; `manual_smoke` chỉ còn là fallback rõ ràng khi nguồn thật không activate được end-to-end.

> Repo này phục vụ nghiên cứu / học tập. Không phải hệ thống triển khai lâm sàng.

## Currently Implemented And Verified

### 1. Data pipeline

Các bước preprocess hiện có implementation thật và nối được end-to-end:

```bash
python -m src.data.build_cohort --config configs/data.yaml
python -m src.data.stage_filtered_tables --config configs/data.yaml
python -m src.data.build_vocab --config configs/data.yaml
python -m src.data.build_drugbank_metadata --config configs/data.yaml
python -m src.data.build_ddi_matrix --config configs/data.yaml
python -m src.data.build_drugbank_ddi_matrix --config configs/data.yaml
python -m src.data.build_trajectories --config configs/data.yaml
```

Artifact chính sau preprocess:

- `data/interim/cohort/`
- `data/interim/spark_cache/`
- `data/interim/vocab/`
- `data/interim/vocab/drugbank_drug_metadata.json`
- `data/processed/drugbank/drugbank_summary.json`
- `data/processed/drugbank/drugbank_drugs.jsonl.gz`
- `data/processed/trajectories/manifest.json`
- `data/processed/trajectories/metadata.json`
- `data/processed/trajectories/<split>/*.parquet`
- `data/processed/ddi/drug_ddi.pt`
- `data/processed/ddi/drug_ddi_report.json`
- `data/processed/ddi/drug_ddi_drugbank.pt`
- `data/processed/ddi/drug_ddi_drugbank_report.json`

### 2. Core model path

Core model path dùng các module sau:

- `src/models/patient_state_encoder.py`
- `src/models/history_selector.py`
- `src/models/fusion.py`
- `src/models/medication_decoder.py`
- `src/training/losses.py`
- `src/models/ddi_regularization.py`
- `src/models/full_model.py`

Core mode luôn được build qua:

```bash
python -m src.training.train_core
```

và evaluate qua:

```bash
python -m src.evaluation.evaluate_core
python -m src.evaluation.evaluate_safety
python -m src.evaluation.evaluate_subgroup
```

### 3. DDI trong repo này đang ở mức nào

`configs/data.yaml` hiện ưu tiên nguồn DDI thật:

```text
data/raw/ddi/twosides/TWOSIDES.csv
```

Builder sẽ canonicalize `drug_1_concept_name` / `drug_2_concept_name` sang token thuốc hiện tại bằng name-based normalization, rồi aggregate condition-level TwoSIDES rows thành pair-level DDI artifact.

Khi build TwoSIDES thành công với `nonzero_pairs > 0`, artifact sẽ được gắn nhãn:

- `ddi_type: twosides_real_condition_aggregated`
- `ddi_research_grade: true`

Artifact canonical trung gian mặc định:

```text
data/processed/ddi/drug_ddi_pairs.csv.gz
```

Vì vocab thuốc của repo hiện là token `NAME:` và chưa có RxNorm crosswalk layer, mapping DDI thật trong pass này vẫn là **name-based**, không phải ID-level alignment.

Nếu TwoSIDES thiếu file, không map được, hoặc build ra zero matched pairs, builder sẽ fallback rõ ràng về:

- `ddi_type: manual_smoke`
- `ddi_research_grade: false`

`manual_smoke` chỉ dùng để giữ wiring train/eval/report chạy được; nó **không** được coi là benchmark safety research-grade.

Ngoài path mặc định ở trên, repo hiện có thêm **optional parallel DrugBank path**:

- source XML mặc định: `data/raw/drugbank/full database.xml`
- metadata builder: `python -m src.data.build_drugbank_metadata --config configs/data.yaml`
- DDI builder: `python -m src.data.build_drugbank_ddi_matrix --config configs/data.yaml`
- artifact riêng: `data/processed/ddi/drug_ddi_drugbank.pt`
- report riêng: `data/processed/ddi/drug_ddi_drugbank_report.json`

DrugBank-derived DDI hiện được đánh dấu là **auxiliary / benchmark opt-in only** với `ddi_research_grade: false`. Nó **không** thay thế default TwoSIDES artifact và không nên được claim là tốt hơn nếu chưa có benchmark riêng công bằng.

## Partial / Experimental

Những phần dưới đây còn tồn tại trong repo nhưng **không phải verified core path**:

- `src/retrieval/memory_bank.py`
- `src/retrieval/topk_retriever.py`
- `src/training/train_extended.py`
- `src/graph/group_encoder.py`

Trạng thái hiện tại:

- retrieval chỉ được dùng trong **experimental extension path**.
- `train_extended.py` được giữ lại như đường thử nghiệm, không phải entrypoint mặc định.
- core checkpoint/report luôn khai báo `retrieval_active: false`.

Config tương ứng vẫn được giữ trong repo để phục vụ extension, nhưng đã được đánh dấu rõ là experimental:

- `configs/model.yaml -> retrieval`
- `configs/model.yaml -> hypergraph`
- `configs/train.yaml -> extended`

## Deferred / Not Implemented As Verified System

Các phần sau **không được coi là feature active của hệ thống hiện tại**:

- hypergraph/group reasoning như một phần verified train-eval-report path
- counterfactual training objective
- natural-language explanation / explanation branch
- research-grade safety evaluation
- full `core + history + retrieval` reporting pipeline

Có file mã nguồn cho một số phần trên, nhưng hiện chưa được coi là capability production hoặc benchmark-ready.

## Config Semantics

### `configs/model.yaml`

- `model`, `embedding`, `sequence`, `history_selector`, `fusion` là phần có effect trong core path.
- `retrieval` và `hypergraph` chỉ dành cho extension/experimental path.

### `configs/train.yaml`

- `runtime.mode: core` là cấu hình chính cho `train_core.py`.
- `core.use_retrieval` và `core.use_group_encoder` phải giữ `false`.
- `extended` chỉ dành cho `train_extended.py` và được xem là experimental.
- `threshold_tuning` chọn threshold trên **validation split** và checkpoint sẽ lưu `effective_threshold` cùng `threshold_selection`.
- `loss.pos_weight_mode: log_balanced` bật capped positive weighting để giảm under-prediction do class imbalance.

### `configs/eval.yaml`

- evaluator shipped trong repo là **core-only**.
- retrieval không được bật từ file này.
- `prediction.threshold` chỉ là fallback. Evaluator sẽ ưu tiên:
  1. CLI `--threshold`
  2. threshold đã lưu trong checkpoint
  3. threshold fallback từ config

## Quick Start

### 1. Cài môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt MIMIC-IV dưới:

```text
data/raw/mimic-iv/
```

Sau đó chạy preprocess:

```bash
./scripts/preprocess.sh --config configs/data.yaml
```

Nếu `data/raw/drugbank/full database.xml` tồn tại, preprocess wrapper sẽ build thêm DrugBank metadata và DrugBank-derived DDI artifact riêng. Nếu file này thiếu, hai bước DrugBank sẽ skip gracefully và chỉ ghi inactive summary/report.

hoặc từng bước:

```bash
python -m src.data.build_cohort --config configs/data.yaml
python -m src.data.stage_filtered_tables --config configs/data.yaml
python -m src.data.build_vocab --config configs/data.yaml
python -m src.data.build_drugbank_metadata --config configs/data.yaml
python -m src.data.build_ddi_matrix --config configs/data.yaml
python -m src.data.build_drugbank_ddi_matrix --config configs/data.yaml
python -m src.data.build_trajectories --config configs/data.yaml
```

### 3. Train core

```bash
python -m src.training.train_core --config configs/train.yaml
```

Checkpoint tốt nhất mặc định:

```text
outputs/checkpoints/train_core_best.pt
```

Checkpoint sẽ luôn lưu metadata truthful như:

- `pipeline_level`
- `history_active`
- `retrieval_active`
- `fusion_strategy`
- `ddi_type`
- `ddi_research_grade`
- `effective_threshold`
- `threshold_selection`
- `train_mode`

### 4. Evaluate core

```bash
python -m src.evaluation.evaluate_core --config configs/eval.yaml
python -m src.evaluation.evaluate_safety --config configs/eval.yaml
python -m src.evaluation.evaluate_subgroup --config configs/eval.yaml
```

Report JSON mặc định nằm ở:

- `outputs/reports/evaluate_core_<split>.json`
- `outputs/reports/evaluate_safety_<split>.json`
- `outputs/reports/evaluate_subgroup_<split>.json`

Các report này phản ánh đúng runtime hiện tại:

- `pipeline_level: core+history`
- `history_active: true`
- `retrieval_active: false`
- `fusion_strategy: <effective strategy>`
- `ddi_type: twosides_real_condition_aggregated` nếu TwoSIDES artifact active
- `ddi_research_grade: true` nếu TwoSIDES artifact active
- `ddi_type: manual_smoke` và `ddi_research_grade: false` chỉ khi builder fallback
- `threshold_source` và `threshold_selection` để chỉ ra threshold thực sự đã dùng khi evaluate

## Loss Thật Đang Được Dùng

Core training hiện dùng đúng công thức:

```text
total_loss = prediction_loss + lambda_ddi * ddi_loss
```

Repo hiện **không** claim các loss component sau là active trong verified path:

- `lambda_sim`
- `lambda_cf`

## Notes On Safety Claims

- `evaluate_safety.py` và `evaluate_subgroup.py` dùng cùng artifact DDI với `evaluate_core.py` và carry cùng DDI metadata vào report.
- Khi artifact active là TwoSIDES với `ddi_research_grade: true`, DDI rate trong report được tính trên artifact thật đó.
- Nếu builder đang fallback về `manual_smoke`, hãy hiểu report như kiểm tra wiring và bookkeeping, không phải evidence cho kết luận nghiên cứu.

## Outputs

Sau train/eval, artifact được ghi vào:

```text
outputs/
├── checkpoints/
├── logs/
├── predictions/
└── reports/
```

Training log JSONL và report JSON đều đã được đồng bộ để cùng phản ánh một sự thật runtime.

## Benchmark Snapshot

- Benchmark summary mới nhất nằm ở `outputs/benchmarks/benchmark_summary.json`.
- Frozen baseline hiện tại với real TwoSIDES là `frozen_real_ddi_current` tại threshold `0.25`.
- Hai clean warm-start rerun trên GPU (`warm_current_ddi_safe`, `warm_tuned_ddi_safe`) đều thắng frozen baseline về test `F1` và `Jaccard`.
- `warm_tuned_ddi_safe` có test `F1` cao nhất rất nhẹ (`0.4691`), nhưng cũng có `DDI rate` cao hơn (`0.1631`) và `PRAUC` thấp hơn nhẹ so với `warm_current_ddi_safe`.
- Vì chênh lệch giữa `ddi_lambda=0.05` và `ddi_lambda=0.01` là rất nhỏ trên test, repo không nên claim `ddi_lambda=0.01` là default winner rõ ràng chỉ từ pass này; điểm chắc chắn hơn là warm-start fine-tune từ frozen checkpoint mạnh hiện tại có giúp thật.

## Tests

Có test cho:

- data pipeline và dataset contract
- encoder / history selector / fusion / decoder
- core runtime
- retrieval module-level behavior

Nên chạy tối thiểu:

```bash
pytest -q tests/test_core_runtime.py tests/test_history_selector.py tests/test_fusion.py tests/test_retrieval.py
```

## License

MIT
