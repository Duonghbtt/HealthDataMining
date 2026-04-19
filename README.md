# 🏥 ClinRec

<div align="center">

**Dynamic Temporal Patient Similarity + Safe Medication Recommendation**  
**Cross-Patient Grouping + Counterfactual Explanation on MIMIC-IV**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV-00897B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Hệ thống hỗ trợ quyết định lâm sàng tích hợp khai phá dữ liệu và khuyến nghị cá nhân hóa trên dữ liệu bệnh nhân ICU**

*CS246 Data Mining + Recommender Systems · MIMIC-IV Dataset*

[Tính năng](#-tính-năng) · [Kiến trúc](#-kiến-trúc-hệ-thống) · [Cài đặt](#-cài-đặt) · [Nhóm](#-nhóm-thực-hiện)

</div>

---

## 1. Tổng quan

**ClinRec** là một pipeline nghiên cứu end-to-end cho bài toán **safe medication recommendation** từ dữ liệu EHR nhiều lần khám. So với README cũ theo hướng tách rời các module khai phá dữ liệu và khuyến nghị, phiên bản hiện tại đã được cập nhật theo kiến trúc mở rộng mới:

- **Dynamic temporal patient similarity retrieval**
- **Multi-granularity evidence selection**
- **DDI-aware medication recommendation**
- **Cross-patient grouping bằng hypergraph / clustering**
- **Counterfactual + natural-language explanation**

Hệ thống không chỉ trả lời câu hỏi **"nên kê thuốc gì"**, mà còn hướng tới trả lời thêm:

1. **Ca nào trong lịch sử là bằng chứng?**
2. **Bệnh nhân hiện tại thuộc nhóm lâm sàng nào?**
3. **Nếu một vài đặc trưng thay đổi, khuyến nghị có đổi không?**

> ⚠️ **Lưu ý:** Dự án phục vụ mục đích nghiên cứu và học thuật. Đây không phải là hệ thống triển khai lâm sàng thực tế và không thay thế quyết định chuyên môn của bác sĩ.

---

## 2. Bài toán

Cho một bệnh nhân tại thời điểm $t$ với dữ liệu lâm sàng hiện tại và lịch sử trước đó:

- diagnosis codes
- procedure codes
- lab results
- vital signs
- medication history
- time features

mục tiêu của hệ thống là dự đoán:

- **tập thuốc phù hợp** cho lần khám hiện tại,
- đồng thời cung cấp:
  - **Top-K ca bệnh tương tự**,
  - **group / cohort evidence**,
  - **cảnh báo DDI**,
  - **giải thích bằng evidence và counterfactual**.

---

## 3. Đóng góp chính

- Mã hóa **trạng thái bệnh nhân theo thời gian** thay vì xử lý từng visit độc lập.
- Truy hồi **ca bệnh tương tự có điều kiện thời gian** bằng cosine similarity kết hợp temporal decay.
- Chọn lọc **self-history**, **neighbor-history** và **group evidence** thay vì dùng toàn bộ lịch sử thô.
- Tối ưu **drug recommendation đa nhãn có kiểm soát DDI**.
- Mở rộng từ evidence mức cá thể sang **cross-patient grouping** bằng hypergraph.
- Bổ sung **counterfactual explanation** để tăng interpretability và giảm over-reliance.

---

## 4. Kiến trúc hệ thống

### 4.1. Pipeline mức cao

```text
Input EHR tại thời điểm hiện tại
  → Module 1. Clinical Input & Temporal Representation
  → Module 2. Dynamic Patient State Encoder
  → Module 3. Dynamic Patient Graph / Retrieval Index
  → Module 4. Cross-Patient Hypergraph Grouping
  → Module 5. Multi-Granularity Evidence Selection
  → Module 6. Joint Drug Recommendation
  → Module 7. Safety & DDI Control
  → Module 8. Counterfactual + Natural-Language Explanation
  → Output: Thuốc + ca tương tự + nhóm bệnh nhân + giải thích + cảnh báo DDI
```

### 4.2. Các module chính

1. **Clinical Input & Temporal Representation**  
   Chuẩn hóa diagnosis / procedure / labs / vitals / medication history / time features thành biểu diễn visit-level $z_{p}^{t}$.

2. **Dynamic Patient State Encoder**  
   Mã hóa chuỗi visits thành patient state động $h_{p}^{t}$ bằng GRU hoặc temporal encoder.

3. **Dynamic Patient Graph / Retrieval Index**  
   Truy hồi Top-$K$ ca bệnh tương tự theo trạng thái động bằng cosine similarity, temporal decay và ANN/FAISS.

4. **Cross-Patient Hypergraph Grouping**  
   Tạo group embedding, cluster hoặc hyperedge từ các neighbor và pattern lâm sàng để sinh cohort-level evidence.

5. **Multi-Granularity Evidence Selection**  
   Chọn self-history, neighbor-history và group evidence ở mức visit và attribute.

6. **Joint Drug Recommendation**  
   Hợp nhất current state $h_{p}^{t}$ và selected evidences để dự đoán drug logits hoặc regimen context.

7. **Safety & DDI Control**  
   Thêm DDI-aware regularization và/hoặc constrained decoding để giảm tương tác thuốc nguy hiểm.

8. **Counterfactual & NL Explanation**  
   Sinh giải thích dạng evidence-based và nếu-thì bằng perturbation hoặc prototype comparison.

### 4.3. Ký hiệu chính

- Visit representation: $z_{p}^{t}$
- Dynamic patient state: $h_{p}^{t}$
- Medication probability vector: $\hat{y}_{p}^{t}$
- Retrieved neighbor set: $\mathcal{N}_{p}^{t}$
- Group embedding / cluster label: $c_{p}^{t}$ hoặc $g_{p}^{t}$
- Selected evidence set: $\mathcal{E}_{p}^{t}$
- Counterfactual explanation: $\mathrm{CF}_{p}^{t}$

### 4.4. Input / Output toàn hệ thống

**Input**

$$
x_{p}^{t} = \{\text{diagnosis},\ \text{procedure},\ \text{labs},\ \text{vitals},\ \text{medication history},\ \text{time features}\}
$$

$$
\mathcal{D}_{\text{hist}} = \{X_{q}\}_{q=1}^{N}
$$

**Output**

1. **Vector xác suất thuốc**

$$
\hat{y}_{p}^{t}
$$

2. **Top-$K$ ca tương tự**

$$
\mathcal{N}_{p}^{t} = \{(q, \tau, s)\}_{k=1}^{K}
$$

3. **Group embedding hoặc cluster label**

$$
c_{p}^{t} \quad \text{hoặc} \quad g_{p}^{t}
$$

4. **Selected self / neighbor / group evidence**

$$
\mathcal{E}_{p}^{t}
$$

5. **Counterfactual explanation + alternative recommendation**

$$
\mathrm{CF}_{p}^{t}
$$

---

## 5. Dataset

Dự án sử dụng **MIMIC-IV** từ PhysioNet.

- Trang dataset: https://physionet.org/content/mimiciv/
- Để truy cập cần tài khoản PhysioNet và hoàn thành training theo yêu cầu của dự án.

### Các bảng thường dùng

- `patients`
- `admissions`
- `icustays`
- `transfers`
- `diagnoses_icd`
- `procedures_icd`
- `labevents`
- `chartevents`
- `prescriptions`
- `emar`, `emar_detail`, `pharmacy`

> Dữ liệu gốc **không commit lên git**. Chỉ lưu trong `data/raw/mimic-iv/`.

---

## 6. Cấu trúc thư mục mới

Cấu trúc dưới đây đã được cập nhật để khớp với bản **thuyết minh đầy đủ cấu trúc thư mục, chức năng file, luồng gọi và lộ trình xây dựng**.

```text
clinrec/
├── data/
│   ├── raw/
│   │   └── mimic-iv/
│   ├── interim/
│   │   ├── cohort/
│   │   ├── trajectories/
│   │   └── vocab/
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   └── ddi/
│   └── artifacts/
│       ├── memory_bank/
│       ├── faiss/
│       └── hypergraph/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   └── eval.yaml
├── src/
│   ├── data/
│   │   ├── load_mimic.py
│   │   ├── build_cohort.py
│   │   ├── build_trajectories.py
│   │   ├── build_vocab.py
│   │   ├── build_ddi_matrix.py
│   │   └── dataset.py
│   ├── features/
│   │   ├── diagnosis_encoder.py
│   │   ├── procedure_encoder.py
│   │   ├── lab_processor.py
│   │   ├── vital_processor.py
│   │   └── medication_history.py
│   ├── models/
│   │   ├── patient_state_encoder.py
│   │   ├── temporal_similarity.py
│   │   ├── history_selector.py
│   │   ├── fusion.py
│   │   ├── medication_decoder.py
│   │   ├── ddi_regularization.py
│   │   └── full_model.py
│   ├── retrieval/
│   │   ├── memory_bank.py
│   │   ├── topk_retriever.py
│   │   ├── faiss_index.py
│   │   └── dynamic_graph.py
│   ├── graph/
│   │   ├── hypergraph_builder.py
│   │   ├── hypergraph_layers.py
│   │   └── group_encoder.py
│   ├── explainability/
│   │   ├── attention_export.py
│   │   ├── similar_case_report.py
│   │   ├── counterfactual.py
│   │   └── nl_explainer.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   ├── train_core.py
│   │   └── train_extended.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate_core.py
│   │   ├── evaluate_ablation.py
│   │   ├── evaluate_safety.py
│   │   └── evaluate_subgroup.py
│   └── utils/
│       ├── seed.py
│       ├── logger.py
│       ├── io.py
│       └── device.py
├── notebooks/
│   ├── 01_eda_mimic_iv.ipynb
│   ├── 02_build_cohort.ipynb
│   ├── 03_train_base.ipynb
│   ├── 04_train_tempsim.ipynb
│   ├── 05_train_full_core.ipynb
│   ├── 06_hypergraph_extension.ipynb
│   └── 07_counterfactual_cases.ipynb
├── scripts/
│   ├── preprocess.ps1
│   ├── preprocess.sh
│   ├── train_core.ps1
│   ├── train_extended.ps1
│   └── evaluate.ps1
├── tests/
│   ├── test_data.py
│   ├── test_encoder.py
│   ├── test_retrieval.py
│   ├── test_history_selector.py
│   ├── test_fusion.py
│   ├── test_decoder.py
│   └── test_counterfactual.py
├── app/
│   ├── streamlit_app.py
│   ├── pages/
│   │   ├── 1_similar_cases.py
│   │   ├── 2_recommendation.py
│   │   ├── 3_safety_ddi.py
│   │   └── 4_counterfactual.py
│   └── components/
│       ├── patient_form.py
│       ├── similarity_panel.py
│       ├── recommendation_panel.py
│       └── explanation_panel.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   ├── figures/
│   └── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 7. Luồng gọi chính giữa các module

### 7.1. Pha dữ liệu

```text
load_mimic.py
  → build_cohort.py
  → stage_filtered_tables.py
  → build_vocab.py / build_ddi_matrix.py
  → build_trajectories.py
  → dataset.py
```

### 7.2. Pha core model

```text
patient_state_encoder.py
  → temporal_similarity.py
  → memory_bank.py / topk_retriever.py
  → history_selector.py
  → fusion.py
  → medication_decoder.py
  → ddi_regularization.py
  → full_model.py
```

### 7.3. Pha train

**Môn học:** Khai phá dữ liệu lớn

### 7.4. Pha đánh giá

```text
metrics.py
  → evaluate_core.py
  → evaluate_ablation.py / evaluate_safety.py / evaluate_subgroup.py
```

### 7.5. Pha mở rộng

```text
dynamic_graph.py
  → hypergraph_builder.py
  → hypergraph_layers.py
  → group_encoder.py
  → counterfactual.py
  → nl_explainer.py
```

### 7.6. Pha demo

```text
streamlit_app.py
  → pages/*
  → components/*
```

---

## 8. Cài đặt môi trường

### Yêu cầu tối thiểu

- Python 3.9+
- Khuyến nghị dùng môi trường ảo
- RAM đủ lớn để xử lý cohort và trajectory từ MIMIC-IV
- GPU là tùy chọn, nhưng hữu ích khi train extended model

### Clone repo

```bash
git clone https://github.com/your-username/clinrec.git
cd clinrec
```

### Tạo môi trường ảo

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Cài dependencies

```bash
pip install -r requirements.txt
```

---

## 9. Chuẩn bị dữ liệu

### Bước 1. Đặt MIMIC-IV vào đúng thư mục

```text
data/raw/mimic-iv/
```

### Bước 2. Kiểm tra cấu hình dữ liệu

Chỉnh các file sau cho phù hợp với máy của bạn:

- `configs/data.yaml`
- `configs/model.yaml`
- `configs/train.yaml`
- `configs/eval.yaml`

### Bước 3. Chạy preprocessing

**PowerShell**

```powershell
./scripts/preprocess.ps1 -Config configs/data.yaml
```

**Bash / Linux / macOS**

```bash
./scripts/preprocess.sh --config configs/data.yaml
```

Hoặc chạy từng bước bằng Python:

```bash
python -m src.data.build_cohort --config configs/data.yaml
python -m src.data.stage_filtered_tables --config configs/data.yaml
python -m src.data.build_vocab --config configs/data.yaml
python -m src.data.build_ddi_matrix --config configs/data.yaml
python -m src.data.build_trajectories --config configs/data.yaml
```

Neu ban vua chay lai `build_cohort` hoac thay doi cohort / Spark-related config,
hay chay lai `stage_filtered_tables` truoc `build_vocab` va `build_trajectories`
de refresh `data/interim/spark_cache/`.

Sau bước này, các thư mục quan trọng cần xuất hiện:

- `data/interim/cohort/`
- `data/interim/trajectories/`
- `data/interim/vocab/`
- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`
- `data/processed/ddi/`

Luu y:
Mac dinh local repo nay tro `ddi_source_path` den `data/raw/ddi/drug_ddi_smoke.csv`.
Day la source `manual_smoke`, chi dung cho local wiring / smoke test cua DDI path, khong duoc dung de claim ket qua nghien cuu.
Muon co `research-grade real DDI`, ban phai thay bang source DDI that va rebuild de artifact/report cho thay `matched_pairs > 0` tu source do.

---

## 10. Huấn luyện

### 10.1. Train bản core

```powershell
./scripts/train_core.ps1
```

hoặc:

```bash
python -m src.training.train_core
```

Checkpoint va report core se carry `ddi_context` cua artifact dang dung.
Neu ban dang chay voi `manual_smoke` source, training van co the `DDI active` de test wiring, nhung khong duoc dien giai nhu DDI research-grade.

### 10.2. Train bản mở rộng

```powershell
./scripts/train_extended.ps1
```

hoặc:

```bash
python -m src.training.train_extended
```

### 10.3. Loss tổng quát

$$
\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_{\text{ddi}}\,\mathcal{L}_{\text{ddi}} + \lambda_{\text{sim}}\,\mathcal{L}_{\text{sim}} + \lambda_{\text{cf}}\,\mathcal{L}_{\text{cf}}
$$

Trong đó:

- $\mathcal{L}_{\text{pred}}$: prediction loss cho multi-label medication recommendation
- $\mathcal{L}_{\text{ddi}}$: regularization cho drug-drug interaction
- $\mathcal{L}_{\text{sim}}$: regularization cho retrieval / structure stability nếu dùng
- $\mathcal{L}_{\text{cf}}$: loss cho explanation branch nếu huấn luyện đồng thời
- $\lambda_{\text{ddi}}, \lambda_{\text{sim}}, \lambda_{\text{cf}}$: các hệ số trade-off giữa accuracy, safety, retrieval stability và explanation

---

## 11. Đánh giá

### Các metric chính

- Jaccard
- F1 Score
- PRAUC
- DDI Rate
- Avg #Drugs

### Chạy đánh giá

```powershell
./scripts/evaluate.ps1
```

hoặc:

```bash
python -m src.evaluation.evaluate_core
python -m src.evaluation.evaluate_ablation
python -m src.evaluation.evaluate_safety
python -m src.evaluation.evaluate_subgroup
```

### Các nhóm so sánh ablation

- Base
- TempSim
- SelfSel
- NbrSel
- Full Core
- Extended

---

## 12. Demo ứng dụng

Sau khi đã có checkpoint ổn định:

```bash
streamlit run app/streamlit_app.py
```

Các trang chính:

- `1_similar_cases.py`: hiển thị Top-K ca bệnh tương tự
- `2_recommendation.py`: hiển thị thuốc gợi ý
- `3_safety_ddi.py`: hiển thị cảnh báo DDI
- `4_counterfactual.py`: hiển thị giải thích counterfactual

---

## 13. Lộ trình xây dựng khuyến nghị

### Pha 1 — Khóa dữ liệu và tiền xử lý

- `load_mimic.py`
- `build_cohort.py`
- `build_vocab.py`
- `build_ddi_matrix.py`
- `build_trajectories.py`
- `dataset.py`

**Điều kiện đạt:** cohort sạch, vocab ổn định, DDI matrix đúng kích thước, batch đầu tiên load được.

### Pha 2 — Dựng baseline nhỏ nhất

- `patient_state_encoder.py`
- `medication_decoder.py`
- `losses.py`
- `metrics.py`

**Điều kiện đạt:** train được 1–2 epoch, loss giảm, metric tính được.

### Pha 3 — Thêm retrieval

- `temporal_similarity.py`
- `memory_bank.py`
- `topk_retriever.py`

**Điều kiện đạt:** Top-K hợp lý, phân biệt được static và temporal retrieval.

### Pha 4 — Thêm selection và fusion

- `history_selector.py`
- `fusion.py`
- `full_model.py`

**Điều kiện đạt:** full core forward pass chạy end-to-end.

### Pha 5 — Huấn luyện và đánh giá core

- `trainer.py`
- `train_core.py`
- `evaluate_core.py`
- `evaluate_ablation.py`
- `evaluate_safety.py`

**Điều kiện đạt:** có checkpoint tốt nhất, bảng metric, ablation và safety report.

### Pha 6 — Mở rộng hypergraph và explanation

- `dynamic_graph.py`
- `hypergraph_builder.py`
- `hypergraph_layers.py`
- `group_encoder.py`
- `counterfactual.py`
- `nl_explainer.py`

**Điều kiện đạt:** extended model có thêm cohort evidence và case study explainability.

### Pha 7 — Script hóa, test hóa và demo

- `scripts/*`
- `tests/*`
- `app/*`
- `README.md`

**Điều kiện đạt:** người khác clone repo vẫn có thể cài, chạy preprocess tối thiểu, train, evaluate và xem demo.

---

## 14. Kiểm thử

Các test chính:

- `tests/test_data.py`
- `tests/test_encoder.py`
- `tests/test_retrieval.py`
- `tests/test_history_selector.py`
- `tests/test_fusion.py`
- `tests/test_decoder.py`
- `tests/test_counterfactual.py`

Khuyến nghị chạy test sớm theo từng pha thay vì để cuối kỳ.

---

## 15. Baselines và paper liên quan

### Nhóm baseline / papers nên so sánh

- GAMENet
- SafeDrug
- MICRON
- COGNet
- MoleRec
- DAPSNet
- VITA
- RaVSNet
- HypeMed

### Gợi ý theo module

- **Temporal encoding:** RETAIN, BEHRT, Med-BERT
- **Temporal retrieval:** DAPSNet, RaVSNet, FAISS
- **Hypergraph grouping:** HGNN, HypeMed, BH³-MedRec
- **Evidence selection:** VITA, REFINE, COGNet
- **DDI-aware objective:** SafeDrug, MoleRec
- **Counterfactual explanation:** DiCE và các paper CF cho Clinical AI / EHR

---

## 16. Artifact đầu ra

Sau khi train / evaluate, các kết quả được lưu tại:

```text
outputs/
├── checkpoints/
├── logs/
├── predictions/
├── figures/
└── reports/
```

Đây là nơi lưu:

- checkpoint tốt nhất
- log huấn luyện
- prediction export
- biểu đồ loss / metric / ablation
- báo cáo tổng hợp cuối cùng

---

## 17. Thành viên nhóm

| Thành viên | Vai trò chính |
|---|---|
| Bùi Đức Đại | Data + Features + Patient State Encoder |
| Đỗ Mạnh Cường | Temporal Similarity + Retrieval + Dynamic Graph |
| Nguyễn Văn Phúc | History Selection + Fusion + Hypergraph + Ablation |
| Nguyễn Thế Dương | Decoder + Training + Evaluation + Counterfactual + App |

---

## 18. Ghi chú sử dụng repo

- Không commit dữ liệu gốc MIMIC-IV.
- Không commit checkpoint lớn, log tạm, cache và file nhạy cảm.
- Chỉ bắt đầu phần **hypergraph**, **counterfactual** và **app** sau khi core pipeline đã ổn định.
- Logic production nên nằm trong `src/` và `scripts/`, không để notebook là nơi duy nhất chứa code chính.

---

## 19. License

MIT License

---

## 20. Citation

Nếu bạn sử dụng repo hoặc ý tưởng từ dự án này cho báo cáo / nghiên cứu, hãy trích dẫn repo và các paper baseline liên quan trong phần tài liệu tham khảo.

---

## 21. Disclaimer

ClinRec là hệ thống nghiên cứu phục vụ học thuật. Mọi đầu ra của hệ thống chỉ mang tính chất hỗ trợ phân tích và không được xem là chỉ định lâm sàng thực tế.
