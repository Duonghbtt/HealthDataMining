# Module 1 -- Patient Similarity Search (MinHash + LSH)

## Overview

Module 1 xây dựng hệ thống **tìm bệnh nhân tương tự (Patient Similarity
Search)** trên dataset **MIMIC‑IV** bằng thuật toán **MinHash + Locality
Sensitive Hashing (LSH)**.

Mục tiêu: - Tìm **Top‑K bệnh nhân có hồ sơ y tế tương tự** - Xử lý
dataset lớn (\~300k bệnh nhân) - Chạy được trên máy cá nhân (\~16GB RAM)

Feature sử dụng: - ICD (diagnosis codes) - DRUG (medication codes) - LAB
(laboratory tests)

---

# 1. System Pipeline

    MIMIC-IV Raw Data
    │
    ├── diagnoses_icd.csv.gz
    ├── prescriptions.csv.gz
    ├── labevents.csv.gz
    │
    ▼
    Feature Engineering
    (ICD / DRUG / LAB tokens)
    │
    ▼
    Patient Feature Set
    │
    ▼
    MinHash Signature (128 permutations)
    │
    ▼
    LSH Index (banding)
    │
    ▼
    Candidate Patients
    │
    ▼
    Jaccard Similarity
    │
    ▼
    Top‑K Similar Patients

---

# 2. Feature Representation

Mỗi bệnh nhân được biểu diễn bằng **set tokens**.

Example:

    Patient 10000032

    {
    ICD9:07054
    ICD9:5715
    DRUGCD:APAP500
    DRUGCD:HEPA5I
    LAB:50861
    LAB:50862
    }

Token Ý nghĩa

---

ICD diagnosis code
DRUG medication code
LAB lab itemid

---

# 3. Similarity Metric

## Jaccard Similarity

    J(A,B) = |A ∩ B| / |A ∪ B|

Trong đó:

- A = feature set bệnh nhân A
- B = feature set bệnh nhân B

Brute‑force so sánh tất cả cặp bệnh nhân:

    O(N²)

Không khả thi với dataset lớn.

---

# 4. MinHash

MinHash dùng để **xấp xỉ Jaccard similarity**.

Nguyên lý:

    signature_i = min( h_i(feature) )

Với:

- h_i = hash function
- num_perm = 128

Ví dụ signature:

    [39960851, 12539112, 36740512, ...]

Property quan trọng:

    P(signature_i(A) == signature_i(B)) = J(A,B)

Do đó **tỷ lệ hash trùng ≈ Jaccard similarity**.

---

# 5. Locality Sensitive Hashing (LSH)

Signature được chia thành **bands**.

Ví dụ:

    num_perm = 128
    bands = 32
    rows = 4

Signature:

    [s1 s2 s3 s4 | s5 s6 s7 s8 | ...]

Mỗi band được hash vào bucket.

Nếu 2 bệnh nhân có **band giống nhau → candidate pair**.

---

# 6. LSH S‑Curve (Threshold)

Xác suất trở thành candidate:

    P(candidate) = 1 - (1 - s^r)^b

Trong đó:

- s = Jaccard similarity
- r = rows
- b = bands

Ví dụ:

    b = 32
    r = 4
    threshold ≈ 0.5

Đồ thị có dạng **S‑curve**.

Ý nghĩa:

- similarity thấp → xác suất gần 0
- similarity cao → xác suất gần 1

---

# 7. Training Model

Fit hệ thống:

    python -m module1.cli_demo fit \
      --mimic_root data/raw \
      --use_labs \
      --top_k_labs 200 \
      --model_path module1_full_labs.pkl \
      --num_perm 128 \
      --auto_threshold 0.5

Output:

    Top lab size: 200
    Số bệnh nhân có feature: 316314
    Auto chọn bands=32 rows=4
    Model saved

---

# 8. Query Similar Patients

    python -m module1.cli_demo query \
      --model_path module1_full_labs.pkl \
      --patient_id 10000032 \
      --top_k 10

Example:

    Top‑10 similar patients
    sid=17060477 score=0.51
    sid=17984005 score=0.51
    sid=10132988 score=0.50
    ...

---

# 9. RAM Estimation

Giả sử:

    patients ≈ 316k
    tokens/patient ≈ 150
    num_perm = 128

MinHash signatures:

    316k × 128 × 8 bytes ≈ 323 MB

Patient tokens:

    ≈ 1‑2 GB

LSH index:

    ≈ 300‑500 MB

Tổng RAM:

    ~2‑4 GB

Chạy tốt trên **16GB RAM**.

---

# 10. Applications

Hệ thống có thể dùng cho:

- Patient cohort discovery
- Clinical decision support
- Disease similarity analysis
- Drug recommendation systems

---

# Author

Mining Massive Data Sets -- Health Data Mining Project
