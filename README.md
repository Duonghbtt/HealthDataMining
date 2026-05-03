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
- Đánh giá bằng Jaccard, F1, PRAUC, DDI Rate và các báo cáo safety.

## Kiến trúc tổng quan

```text
MIMIC-IV tables
  -> build_cohort
  -> build_vocab
  -> build_ddi_matrix
  -> build_trajectories
  -> export_tensorized_trajectories
  -> train_core
  -> evaluate_core / evaluate_safety / evaluate_ablation
```

Luồng mô hình core:

```text
diag_codes + proc_codes + lab_values + vital_values + med_history
  -> PatientStateEncoder
  -> SelfHistorySelector
  -> Fusion
  -> MedicationDecoder
  -> DDI-aware loss / safety evaluation
```

Các thành phần chính:

- `PatientStateEncoder`: mã hóa trạng thái bệnh nhân theo từng visit.
- `SelfHistorySelector`: chọn các visit quan trọng trong lịch sử của chính bệnh nhân.
- `FusionModule`: hợp nhất trạng thái hiện tại và tóm tắt lịch sử.
- `MedicationDecoder`: dự đoán xác suất cho từng thuốc trong vocabulary.
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
|-- notebooks/
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

- `src/retrieval/` vẫn có trong repo, nhưng cấu hình core hiện tại đang tắt retrieval (`use_retrieval: false`).
- Một số notebook cũ như temporal similarity hoặc hypergraph có thể dùng cho thử nghiệm, không phải đường chính của pipeline core.
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
- `configs/eval.yaml`: split đánh giá, checkpoint, threshold search, safety decoding và report.

Khi chạy CPU, có thể ghi đè trực tiếp:

```powershell
python -m src.training.train_core --config configs/train.yaml --device cpu --smoke-test
```

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

Train mô hình core:

```powershell
python -m src.training.train_core --config configs/train.yaml
```

Chạy smoke test ngắn để kiểm tra pipeline:

```powershell
python -m src.training.train_core --config configs/train.yaml --smoke-test --device cpu
```

Checkpoint và log được ghi vào:

```text
outputs/checkpoints/
outputs/logs/
```

Loss chính:

```text
total_loss = prediction_loss + lambda_ddi * ddi_loss
```

Trong đó:

- `prediction_loss`: BCE loss cho bài toán multi-label.
- `ddi_loss`: regularization dựa trên ma trận DDI.
- `lambda_ddi`: hệ số cân bằng giữa độ chính xác và an toàn.

## Đánh giá

Đánh giá core model:

```powershell
python -m src.evaluation.evaluate_core --config configs/eval.yaml
```

Đánh giá safety-aware decoding:

```powershell
python -m src.evaluation.evaluate_safety --config configs/eval.yaml
```

Đánh giá ablation:

```powershell
python -m src.evaluation.evaluate_ablation --config configs/eval.yaml
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

| Thành viên | Vai trò chính |
|---|---|
| Bùi Đức Đại | Data, feature engineering, Patient State Encoder |
| Đỗ Mạnh Cường | Self-history selection, integration support |
| Nguyễn Văn Phúc | Fusion, ablation, model integration |
| Nguyễn Thế Dương | Decoder, training, evaluation, documentation |

## Quy tắc sử dụng repo

- Không commit dữ liệu gốc MIMIC-IV.
- Không commit checkpoint lớn, log tạm, cache hoặc file nhạy cảm.
- Giữ logic chính trong `src/` và `scripts/`; notebook chỉ nên dùng để phân tích hoặc thử nghiệm.
- Khi mở rộng mô hình, ưu tiên giữ pipeline core ổn định trước rồi mới thêm retrieval, graph hoặc hypergraph.

## Trích dẫn và miễn trừ trách nhiệm

Nếu dùng repo hoặc ý tưởng từ dự án này cho báo cáo/nghiên cứu, hãy trích dẫn MIMIC-IV, RxNorm, DrugBank, DDInter và các paper baseline liên quan.

ClinRec là hệ thống nghiên cứu. Mọi đầu ra chỉ có ý nghĩa hỗ trợ phân tích và không được xem là chỉ định lâm sàng thực tế.
