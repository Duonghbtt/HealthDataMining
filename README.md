# ClinRec - HealthDataMining

ClinRec là pipeline nghiên cứu cho bài toán khuyến nghị thuốc an toàn từ dữ liệu hồ sơ bệnh án điện tử theo chuỗi lần khám. Dự án dùng MIMIC-IV làm nguồn dữ liệu chính, biểu diễn trạng thái bệnh nhân theo thời gian, chọn lịch sử quan trọng của chính bệnh nhân và tối ưu đầu ra thuốc với ràng buộc tương tác thuốc-thuốc.

> Lưu ý: dự án chỉ phục vụ học thuật và nghiên cứu. Kết quả mô hình không phải chỉ định lâm sàng và không thay thế quyết định của bác sĩ.

## Mục tiêu

Với một bệnh nhân tại lần khám hiện tại, hệ thống sử dụng:

- mã chẩn đoán,
- mã thủ thuật,
- xét nghiệm,
- dấu hiệu sinh tồn,
- lịch sử thuốc và các lần khám trước,

để dự đoán tập thuốc phù hợp cho lần khám hiện tại, đồng thời giảm nguy cơ Drug-Drug Interaction (DDI) trong tập thuốc được khuyến nghị.

Bài toán được mô hình hóa dưới dạng multi-label medication recommendation trên chuỗi visit EHR.

## Điểm chính của pipeline

- Xây dựng cohort bệnh nhân từ MIMIC-IV.
- Xây dựng vocabulary cho diagnosis, procedure, lab, vital và medication.
- Chuẩn hóa medication vocabulary chính qua RxNorm.
- Build ma trận DDI từ RxNorm, DrugBank Vocabulary và DDInter.
- Tạo trajectory theo bệnh nhân, sau đó export sang dạng tensorized để train nhanh hơn.
- Train mô hình core gồm Patient State Encoder, Self-History Selector, Fusion và Medication Decoder.
- Đánh giá bằng Jaccard, F1, PRAUC, DDI Rate, threshold trade-off và phân tích theo nhóm history.

## Kiến trúc tổng quan

```text
MIMIC-IV tables
  -> build_cohort
  -> build_vocab
  -> build_ddi_matrix
  -> build_trajectories
  -> export_tensorized_trajectories
  -> train_core
  -> evaluate_core
```

Luồng mô hình core:

```text
diag_codes + proc_codes + lab_values + vital_values + med_history
  -> PatientStateEncoder
  -> SelfHistorySelector
  -> Offline cached retrieval context
  -> Fusion / gated fusion
  -> MedicationDecoder + history/retrieval copy branch
  -> DDI-aware loss / DDI metrics
```

Các thành phần chính:

- `PatientStateEncoder`: mã hóa trạng thái bệnh nhân theo từng visit.
- `SelfHistorySelector`: chọn các visit quan trọng trong lịch sử của chính bệnh nhân.
- `src.retrieval`: nạp ngữ cảnh truy hồi đã được tính trước từ offline cache.
- `FusionModule`: hợp nhất trạng thái hiện tại, tóm tắt lịch sử và retrieval context.
- `MedicationDecoder`: dự đoán xác suất cho từng thuốc trong vocabulary, có hỗ trợ history/retrieval copy branch.
- `ddi_regularization`: nạp ma trận DDI và hỗ trợ regularization/rerank an toàn.

## Cấu trúc thư mục

```text
HealthDataMining/
|-- configs/
|   |-- data.yaml
|   |-- model.yaml
|   |-- train.yaml
|   `-- eval.yaml
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- artifacts/
|-- outputs/
|   |-- checkpoints/
|   |-- logs/
|   |-- predictions/
|   |-- figures/
|   `-- reports/
|-- scripts/
|   |-- preprocess.ps1
|   `-- export_tensorized.ps1
|-- src/
|   |-- data/
|   |-- evaluation/
|   |-- features/
|   |-- models/
|   |-- retrieval/
|   |-- training/
|   `-- utils/
|-- tests/
|-- requirements.txt
`-- README.md
```

Ghi chú:

- `src/retrieval/` là đường train mặc định hiện tại qua offline cached retrieval trong `configs/train.yaml`.
- Repo hiện không có app/demo UI riêng; các entrypoint chính là CLI trong `scripts/`, `src/training/train_core.py` và `src/evaluation/evaluate_core.py`.
- `data/` và `outputs/` được ignore trong git để tránh commit dữ liệu MIMIC-IV, checkpoint lớn và artifact sinh ra khi chạy.

## Dữ liệu cần chuẩn bị

### 1. MIMIC-IV

Đặt dữ liệu MIMIC-IV đã giải nén vào:

```text
data/raw/
|-- hosp/
`-- icu/
```

Các bảng thường được pipeline dùng:

- `hosp/patients.csv.gz`
- `hosp/admissions.csv.gz`
- `hosp/diagnoses_icd.csv.gz`
- `hosp/procedures_icd.csv.gz`
- `hosp/labevents.csv.gz`
- `hosp/prescriptions.csv.gz`
- `icu/icustays.csv.gz`
- `icu/chartevents.csv.gz`

Tùy bước xử lý, pipeline cũng có thể đọc thêm các bảng dictionary như `d_labitems.csv.gz`, `d_items.csv.gz`, `d_icd_diagnoses.csv.gz` và `d_icd_procedures.csv.gz`.

### 2. Dữ liệu ngoài cho DDI

Pipeline DDI hiện tại dùng đường chính sau:

```text
med_vocab_main
  -> RxNorm ingredient/name index
  -> DrugBank Vocabulary alias bridge
  -> DDInter entity matching
  -> drug_ddi.pt + drug_ddi_report.json
```

Các path mặc định nằm trong `configs/data.yaml`:

```yaml
paths:
  rxnorm_root: data/processed/ddi/RxNorm_full_04062026
  drugbank_vocab_path: data/processed/ddi/drugbank vocabulary.csv
  ddinter_root: data/processed/ddi
  ddinter_glob: ddinter_downloads_code_*.csv
  ddi_root: data/processed/ddi
```

Sau khi build thành công, cần có:

```text
data/processed/ddi/drug_ddi.pt
data/processed/ddi/drug_ddi_report.json
```

Các file mapping cũ kiểu `ndc2RXCUI.txt`, `RXCUI2atc4.csv`, `drug-atc.csv` hoặc `drug-DDI.csv` không phải đường chính của `src.data.build_ddi_matrix` hiện tại. Nếu còn xuất hiện trong thư mục dữ liệu, hãy xem chúng là dữ liệu cũ hoặc dữ liệu đối chiếu, trừ khi bạn chủ động mở lại một pipeline legacy.

## Cài đặt môi trường

Yêu cầu tối thiểu:

- Python 3.9 trở lên.
- JDK 17 nếu chạy pipeline Spark.
- Trên Windows, cần `HADOOP_HOME` và `winutils.exe` khi Spark ghi Parquet.
- GPU là tùy chọn. Nếu không có GPU, dùng `--device cpu` hoặc sửa `device` trong config.

Tạo môi trường ảo và cài dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu dùng Linux hoặc macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cấu hình

Các file cấu hình chính:

- `configs/data.yaml`: đường dẫn dữ liệu, Spark, split train/val/test và tham số feature.
- `configs/model.yaml`: encoder, selector, fusion, decoder và chế độ core.
- `configs/train.yaml`: batch size, epoch, optimizer, DDI loss schedule và output checkpoint/log.
- `configs/eval.yaml`: split đánh giá, checkpoint, threshold search, DDI trade-off và report.

Khi chạy CPU, có thể ghi đè trực tiếp:

```powershell
python -m src.training.train_core --config configs/train.yaml --device cpu --smoke-test
```

## Offline cached retrieval / cross-patient memory

Pipeline core mặc định trong `configs/train.yaml` dùng offline cached retrieval:

```yaml
core:
  use_retrieval: true

extended:
  use_retrieval: true

retrieval_cache:
  enabled: true
  use_precomputed: true
```

Config train chính hiện tại là bản cross-patient memory theo hướng offline cached retrieval:

```text
configs/train.yaml
```

Config này bật:

- `core.use_retrieval: true`
- `extended.use_retrieval: true`
- `decoder.use_retrieval_copy: true`
- `retrieval_cache.enabled: true`
- `retrieval_cache.use_precomputed: true`
- `retrieval_cache.memory_split_for_eval: train`
- `retrieval_cache.allow_cross_split: false`
- `retrieval_cache.allow_same_patient: false`

Lưu ý quan trọng: train loop không search neighbor online. Neighbor được precompute trước và lưu vào cache. Dataset/DataLoader chỉ đọc `neighbor_ids`, `scores`, `mask` và medication evidence từ cache.

### Build retrieval cache

Chạy sau khi đã có tensorized trajectories trong `data/processed/{train,val,test}`:

```powershell
python src\data\build_retrieval_cache.py --config configs\data.yaml --train-config configs\train.yaml --splits train val test --top-k 3 --overwrite
```

Artifact sinh ra:

```text
data/artifacts/retrieval_cache/train_topk.pt
data/artifacts/retrieval_cache/val_topk.pt
data/artifacts/retrieval_cache/test_topk.pt
outputs/reports/retrieval_cache_report.json
```

Cache mỗi sample gồm các trường chính:

- `retrieval_neighbor_ids`
- `retrieval_neighbor_patient_ids`
- `retrieval_neighbor_visit_indices`
- `retrieval_scores`
- `retrieval_mask`
- `retrieval_medication_ids`

Báo cáo cache nên được kiểm tra trước khi train:

- `fraction_with_neighbors`
- `avg_valid_neighbors`
- `avg_score`
- `leakage_check_counts`
- `backend_used`

Nếu FAISS không cài được, builder có thể fallback sang brute force. Cách này đúng về logic nhưng có thể chậm hơn nhiều trên full train memory.

### Leakage policy

Mặc định `allow_cross_split: false`:

- train query chỉ retrieve từ train memory.
- val/test query chỉ retrieve từ train memory.
- không dùng val/test labels làm memory khi evaluate.
- không lấy chính sample/visit đó làm neighbor.
- không lấy future visit của cùng patient.
- `allow_same_patient: false` nên cache ưu tiên cross-patient memory; nếu bật same-patient về sau thì vẫn chặn self/future visit.

Query representation trong cache builder dùng diagnosis ids, procedure ids và medication history ids. Target medication của chính sample không được dùng để tạo query representation.

### Kiểm tra cache và forward pass

Sau khi build cache, chạy script kiểm tra nhanh:

```powershell
python scripts\check_retrieval_cache.py --config configs\data.yaml --train-config configs\train.yaml --model-config configs\model.yaml --split val --forward --device cuda
```

Script này kiểm tra cache load được, batch có retrieval fields đúng shape, và model forward được với `use_retrieval=true`.

### Train với cached retrieval

Smoke test 2 batch:

```powershell
python src\training\train_core.py --config configs\train.yaml --data-config configs\data.yaml --model-config configs\model.yaml --baseline-mode current_self_history_ddi --seed 42 --smoke-test --epochs 1 --max-train-batches 2 --max-val-batches 2
```

Train full:

```powershell
python src\training\train_core.py --config configs\train.yaml --data-config configs\data.yaml --model-config configs\model.yaml --baseline-mode current_self_history_ddi --seed 42
```

Trong log mỗi epoch sẽ có thêm retrieval metrics:

- `train_retrieval_fraction_with_context`
- `train_retrieval_avg_valid_candidates`
- `train_retrieval_avg_score`
- `val_retrieval_fraction_with_context`
- `val_retrieval_avg_valid_candidates`
- `val_retrieval_avg_score`

Nếu retrieval đã bật nhưng `fraction_with_context = 0`, cần dừng lại kiểm tra cache path, split, top-k và leakage filter.

### Evaluate checkpoint retrieval

`train_core` tự gọi `evaluate_core` trên best checkpoint sau khi huấn luyện xong. Nếu cần chạy lại evaluation, dùng:

```powershell
python src\evaluation\evaluate_core.py --config configs\eval.yaml --checkpoint outputs\checkpoints\train_core_best.pt --split test
```

Retrieval policy trong report cần thể hiện:

- retrieval branch enabled
- use precomputed cache
- allow cross split false
- memory split for eval train

## Chạy preprocessing

Cách nhanh nhất trên PowerShell:

```powershell
.\scripts\preprocess.ps1
```

Script này chạy lần lượt:

```powershell
python -m src.data.build_cohort --config configs/data.yaml
python -m src.data.build_vocab --config configs/data.yaml
python -m src.data.build_ddi_matrix --config configs/data.yaml
python -m src.data.build_trajectories --config configs/data.yaml
python -m src.data.export_tensorized_trajectories --config configs/data.yaml --overwrite
```

Nếu chỉ cần export lại tensorized trajectories:

```powershell
.\scripts\export_tensorized.ps1 -Overwrite
```

Các artifact quan trọng sau preprocessing:

- `data/interim/cohort/cohort.csv.gz`
- `data/interim/vocab/med_vocab_main.json`
- `data/interim/vocab/med_vocab_main_metadata.json`
- `data/processed/ddi/drug_ddi.pt`
- `data/processed/ddi/drug_ddi_report.json`
- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`

## Huấn luyện

Train chính hiện tại là bản retrieval cached trong `configs/train.yaml`. Nên dùng lệnh đầy đủ để khóa rõ data/model config, baseline mode và seed:

```powershell
python src\training\train_core.py --config configs\train.yaml --data-config configs\data.yaml --model-config configs\model.yaml --baseline-mode current_self_history_ddi --seed 42
```

Chạy smoke test ngắn để kiểm tra pipeline:

```powershell
python src\training\train_core.py --config configs\train.yaml --data-config configs\data.yaml --model-config configs\model.yaml --baseline-mode current_self_history_ddi --seed 42 --smoke-test --epochs 1 --max-train-batches 2 --max-val-batches 2
```

Checkpoint và log được ghi vào:

```text
outputs/checkpoints/
outputs/logs/
```

`outputs/checkpoints/train_core_best.pt` là checkpoint tốt nhất theo validation monitor, không phải mặc định là epoch cuối. Sau khi train xong, `train_core` tự gọi `evaluate_core` trên checkpoint này và ghi report test vào `outputs/reports/`.

Loss chính:

```text
total_loss = prediction_loss + lambda_ddi * ddi_loss
```

Trong đó:

- `prediction_loss`: BCE loss cho bài toán multi-label.
- `ddi_loss`: regularization dựa trên ma trận DDI.
- `lambda_ddi`: hệ số cân bằng giữa độ chính xác và an toàn.

## Đánh giá

Nếu đã train bằng `train_core`, core evaluation trên test đã được chạy tự động sau train. Các lệnh dưới đây dùng để chạy lại evaluation hoặc chạy thêm phân tích bổ sung.

Đánh giá core model:

```powershell
python src\evaluation\evaluate_core.py --config configs\eval.yaml --checkpoint outputs\checkpoints\train_core_best.pt --split test
```

Metric chính:

- Jaccard
- F1
- PRAUC
- DDI Rate
- Avg #Drugs

Báo cáo được ghi vào:

```text
outputs/reports/
outputs/predictions/
```

Các report thường dùng trong bản hiện tại:

- `evaluate_core_test.json/csv`: metric chính trên test set.
- `evaluate_core_test_threshold_comparison.json/csv`: so sánh threshold/top-k/percentile.
- `evaluate_core_test_tradeoff_accuracy_safety.json/csv`: trade-off accuracy và DDI.
- `evaluate_core_test_subgroup_metrics.json/csv`: phân tích first/short/long history.
- `evaluate_core_test_best_threshold_config.json`: threshold tốt nhất được chọn.
- `evaluate_core_test_retrieval_policy.json`: chính sách retrieval/cache và leakage policy khi evaluate.

## Kiểm thử

Chạy toàn bộ test:

```powershell
python -m pytest
```

Chạy một nhóm test cụ thể:

```powershell
python -m pytest tests/test_data.py
python -m pytest tests/test_encoder.py
python -m pytest tests/test_decoder.py
```

## Lỗi thường gặp

### CUDA không khả dụng

Nếu máy không có GPU hoặc CUDA chưa được cài đúng, chạy bằng CPU:

```powershell
python -m src.training.train_core --config configs/train.yaml --device cpu --smoke-test
```

### Spark lỗi trên Windows

Kiểm tra:

- đã cài JDK 17,
- `JAVA_HOME` trỏ đúng thư mục JDK,
- `HADOOP_HOME` trỏ tới thư mục có `bin/winutils.exe`,
- thư mục Spark temp trong `configs/data.yaml` có quyền ghi.

### Không build được DDI matrix

Kiểm tra các path trong `configs/data.yaml`:

- `rxnorm_root`
- `drugbank_vocab_path`
- `ddinter_root`
- `ddinter_glob`
- `vocab_root`

Sau khi chạy `build_ddi_matrix`, mở `drug_ddi_report.json` và kiểm tra số medication vocabulary item match được với DDInter. Nếu ma trận toàn 0, thường là do lệch path, thiếu RxNorm/DrugBank/DDInter hoặc `med_vocab_main` chưa được build đúng.

## Thành viên nhóm

| Thành viên | Vai trò chính | Module/file chính |
|---|---|---|
| Bùi Đức Đại | Dữ liệu, đặc trưng và Patient State Encoder | `src/data/*`, `src/features/*`, `src/models/patient_state_encoder.py`, `configs/data.yaml` |
| Đỗ Mạnh Cường | Self-history selection và phân tích lịch sử bệnh nhân | `src/models/history_selector.py` |
| Nguyễn Văn Phúc | Offline cached retrieval và fusion | `src/retrieval/*`, `src/data/retrieval_cache.py`, `src/models/fusion.py`, `scripts/check_retrieval_cache.py` |
| Nguyễn Thế Dương | Decoder, loss, training/evaluation pipeline và tài liệu chạy | `src/models/medication_decoder.py`, `src/training/*`, `src/evaluation/*`, `configs/train.yaml`, `configs/eval.yaml`, `README.md`, `docs/*` |

Các file tích hợp chung như `src/models/full_model.py`, `src/training/runtime_builder.py` và `configs/model.yaml` cần phối hợp khi thay đổi giao diện giữa các module.

## Quy tắc sử dụng repo

- Không commit dữ liệu gốc MIMIC-IV.
- Không commit checkpoint lớn, log tạm, cache hoặc file nhạy cảm.
- Giữ logic chính trong `src/` và `scripts/`; các thử nghiệm phụ chỉ nên thêm khi thật sự cần và có tài liệu đi kèm.
- Khi mở rộng mô hình, ưu tiên giữ pipeline core ổn định trước rồi mới thêm retrieval, graph hoặc hypergraph.

## Trích dẫn và miễn trừ trách nhiệm

Nếu dùng repo hoặc ý tưởng từ dự án này cho báo cáo/nghiên cứu, hãy trích dẫn MIMIC-IV, RxNorm, DrugBank, DDInter và các paper baseline liên quan.

ClinRec là hệ thống nghiên cứu. Mọi đầu ra chỉ có ý nghĩa hỗ trợ phân tích và không được xem là chỉ định lâm sàng thực tế.
