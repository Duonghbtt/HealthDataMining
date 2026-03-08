# 🏥 Khai phá dữ liệu MIMIC-IV cho hỗ trợ quyết định lâm sàng

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV-00897B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Hệ thống hỗ trợ quyết định lâm sàng tích hợp khai phá dữ liệu và khuyến nghị cá nhân hóa trên dữ liệu bệnh nhân ICU**

*CS246 Data Mining + Recommender Systems · MIMIC-IV Dataset*

[Tính năng](#-tính-năng) · [Kiến trúc](#-kiến-trúc-hệ-thống) · [Cài đặt](#-cài-đặt) · [Nhóm](#-nhóm-thực-hiện)

</div>

---

## 📋 Tổng quan

**ClinRec** là hệ thống hỗ trợ quyết định lâm sàng (Clinical Decision Support System) được xây dựng trên dữ liệu ICU thực tế từ bộ dữ liệu **MIMIC-IV**. Hệ thống tích hợp hai lớp kỹ thuật:

- **Lớp khai phá dữ liệu (CS246):** Similarity Search, Clustering, Frequent Pattern Mining
- **Lớp khuyến nghị (RecSys):** Collaborative Filtering, Matrix Factorization, Sequential Recommendation

Khi bác sĩ nhập hồ sơ bệnh nhân mới, ClinRec tự động:
1. Tìm các ca bệnh tương tự trong lịch sử
2. Phân nhóm bệnh nhân theo đặc điểm bệnh lý
3. Gợi ý phác đồ / thuốc phù hợp với giải thích rõ ràng
4. Cảnh báo tương tác thuốc nguy hiểm
5. Dự đoán xét nghiệm tiếp theo cần thực hiện

> ⚠️ **Lưu ý:** ClinRec là công cụ hỗ trợ nghiên cứu và học thuật, không thay thế quyết định lâm sàng của bác sĩ.

---

## ✨ Tính năng

| Module | Kỹ thuật | Chức năng |
|--------|----------|-----------|
| 🔍 **Patient Similarity** | Shingling · MinHash · LSH | Tìm Top-K bệnh nhân giống nhất |
| 🗂️ **Patient Clustering** | K-means · Cosine Similarity | Phân nhóm bệnh nhân theo bệnh lý |
| 📊 **Pattern Mining** | FP-Growth · Association Rules | Khai phá mẫu phác đồ phổ biến |
| 💊 **Drug Recommender** | UserCF · ItemCF · SVD | Gợi ý thuốc cá nhân hóa + giải thích |
| ⚠️ **Risk Alert** | Association Rules | Cảnh báo tương tác thuốc nguy hiểm |
| 🔬 **Next Lab Predictor** | Markov Chain · Co-occurrence | Dự đoán xét nghiệm tiếp theo |

---

## 🏗️ Kiến trúc hệ thống

```
INPUT: Hồ sơ bệnh nhân mới
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
┌─────────────────────┐              ┌──────────────────────┐
│  MODULE 1           │              │  MODULE 2            │
│  SIMILARITY (CS246) │              │  CLUSTERING (CS246)  │
│  Shingling→MinHash  │              │  K-means + Cosine    │
│  →LSH               │              │                      │
│  Output: Top-K BN   │              │  Output: Cluster ID  │
└──────────┬──────────┘              └───────────┬──────────┘
           │                                     │
           │         ┌───────────────────┐        │
           │         │  MODULE 3         │        │
           │         │  FP-GROWTH(CS246) │        │
           │         │  Frequent Mining  │        │
           │         │  Output: Rules    │        │
           │         └────────┬──────────┘        │
           │                  │                   │
           └──────────────────┼───────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │   INTEGRATION LAYER           │
              │   Similar patients +          │
              │   Cluster context +           │
              │   Association Rules           │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
  ┌───────────────────────┐     ┌─────────────────────────┐
  │  MODULE 4             │     │  MODULE 6               │
  │  RECOMMENDER (RecSys) │     │  SEQUENTIAL (RecSys)    │
  │  UserCF+ItemCF+SVD    │     │  Markov Chain           │
  │  Output: Top-N thuốc  │     │  Output: Next lab test  │
  └───────────┬───────────┘     └─────────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │  MODULE 5             │
  │  RISK ALERT           │
  │  (CS246 + RecSys)     │
  │  Output: ⚠️ Cảnh báo  │
  └───────────────────────┘
              │
              ▼
     STREAMLIT DASHBOARD
  Gợi ý + Giải thích + Cảnh báo
```

---

## 📊 Dataset — MIMIC-IV

Dự án sử dụng **[MIMIC-IV](https://physionet.org/content/mimiciv/)** — bộ dữ liệu ICU lớn nhất thế giới công bố công khai cho nghiên cứu học thuật.

| Bảng dữ liệu | Nội dung | Số bản ghi (ước tính) |
|---|---|---|
| `patients` | Thông tin cơ bản bệnh nhân | ~315,000 |
| `admissions` | Lịch sử nhập viện | ~431,000 |
| `diagnoses_icd` | Chẩn đoán ICD-9/10 | ~4,900,000 |
| `prescriptions` | Đơn thuốc | ~15,000,000 |
| `labevents` | Kết quả xét nghiệm | ~122,000,000 |
| `procedures_icd` | Quy trình điều trị | ~670,000 |

### Cách lấy dataset

> MIMIC-IV yêu cầu đăng ký tài khoản PhysioNet và hoàn thành khóa đào tạo CITI.

1. Đăng ký tại [PhysioNet.org](https://physionet.org/register/)
2. Hoàn thành **CITI Data or Specimens Only Research** training (~2–3 giờ)
3. Yêu cầu quyền truy cập MIMIC-IV tại [trang dự án](https://physionet.org/content/mimiciv/2.2/)
4. Tải về và đặt vào thư mục `data/mimic-iv/`

---

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.9+
- RAM ≥ 16GB (để xử lý MIMIC-IV đầy đủ)
- Disk ≥ 50GB

### 1. Clone repository

```bash
git clone https://github.com/your-username/clinrec.git
cd clinrec
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình đường dẫn dataset

```bash
cp config/config.example.yaml config/config.yaml
# Chỉnh sửa đường dẫn MIMIC-IV trong config.yaml
```

---

## 📁 Cấu trúc thư mục

```
clinrec/
├── data/
│   ├── mimic-iv/              # Dataset gốc (không commit lên git)
│   │   ├── patients.csv
│   │   ├── admissions.csv
│   │   ├── diagnoses_icd.csv
│   │   ├── prescriptions.csv
│   │   └── labevents.csv
│   └── test_data/             # Mock data để test nhanh
│       ├── patients_db.json
│       ├── module1_similarity_input.json
│       ├── module2_clustering_input.json
│       └── ...
│
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py     # Load & join MIMIC-IV tables
│   │   ├── feature_engineer.py # ICD encoding, lab normalization
│   │   └── cleaner.py
│   │
│   ├── module1_similarity/
│   │   ├── shingling.py       # k-shingle generation
│   │   ├── minhash.py         # MinHash signature matrix
│   │   ├── lsh.py             # LSH banding + querying
│   │   └── evaluate.py        # Precision vs brute-force
│   │
│   ├── module2_clustering/
│   │   ├── kmeans_patient.py  # K-means on patient vectors
│   │   ├── cluster_viz.py     # PCA + t-SNE visualization
│   │   └── evaluate.py        # Silhouette, Davies-Bouldin
│   │
│   ├── module3_fpgrowth/
│   │   ├── transactions.py    # Build transaction sets
│   │   ├── fpgrowth_mine.py   # FP-Growth implementation
│   │   ├── rules.py           # Association rule generation
│   │   └── evaluate.py        # Support/confidence/lift analysis
│   │
│   ├── module4_recommender/
│   │   ├── user_cf.py         # User-based Collaborative Filtering
│   │   ├── item_cf.py         # Item-based Collaborative Filtering
│   │   ├── svd_model.py       # Matrix Factorization (SVD)
│   │   ├── hybrid.py          # SVD + Cluster context filter
│   │   ├── explainer.py       # Generate recommendation explanations
│   │   └── evaluate.py        # Precision@K, Recall@K, NDCG, Hit Rate
│   │
│   ├── module5_riskalert/
│   │   ├── risk_rules.py      # Filter high-risk association rules
│   │   └── alert_engine.py    # Match patient drugs to risk rules
│   │
│   └── module6_sequential/
│       ├── markov_chain.py    # Build Markov transition matrix
│       ├── cooccurrence.py    # Lab co-occurrence matrix
│       └── evaluate.py        # Accuracy@1, Hit Rate@3
│
├── app/
│   ├── streamlit_app.py       # Main Streamlit dashboard
│   ├── components/
│   │   ├── patient_input.py
│   │   ├── similarity_panel.py
│   │   ├── recommender_panel.py
│   │   └── risk_panel.py
│   └── assets/
│
├── notebooks/
│   ├── 01_EDA_MIMIC4.ipynb
│   ├── 02_Module1_Similarity.ipynb
│   ├── 03_Module2_Clustering.ipynb
│   ├── 04_Module3_FPGrowth.ipynb
│   ├── 05_Module4_Recommender.ipynb
│   ├── 06_Module5_RiskAlert.ipynb
│   └── 07_Module6_Sequential.ipynb
│
├── tests/
│   ├── test_module1.py
│   ├── test_module2.py
│   ├── test_module3.py
│   ├── test_module4.py
│   ├── test_module5.py
│   └── test_module6.py
│
├── config/
│   ├── config.example.yaml
│   └── config.yaml            # (gitignored)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---




## 👥 Nhóm thực hiện

| Thành viên       | MSSV       | Phụ trách |
|------------------|------------|---|
| Bùi Đức Đại      | B22DCKH025 | Module 1 (Similarity) + Module 4 (Recommender)|
| Đỗ Mạnh Cường    | B22DCKH011 | Module 2 (Clustering)|
| Nguyễn Thế Dương | B22DCKH023 | Module 6 (Sequential)|
| Nguyễn Văn Phúc  | B22DCKH089 | Module 3 (FP-Growth) + Module 5 (Risk Alert)|

**Giảng viên hướng dẫn:** TS.Đặng Hoàng Long

**Môn học:** CS246 - Mining Massive Datasets / Recommender Systems

**Trường:** Học viện Công nghệ Bưu chính Viễn thông (PTIT)

---



