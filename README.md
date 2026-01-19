# 🏥 HealthDataMining

## 🎯 Mục tiêu
Hệ thống hỗ trợ quyết định lâm sàng dựa trên khai phá dữ liệu bệnh nhân lớn, tích hợp Recommender, giúp:
- Tìm bệnh nhân tương tự
- Gợi ý điều trị / xét nghiệm
- Phát hiện bất thường
- Phân tích xu hướng y tế

---

## A. 4 Module CS246 thuần
| Module | Thuật toán | CSV dùng | Ghi chú |
|--------|------------|----------|---------|
| 1. Patient Similarity Search | Shingling, MinHash, LSH, Jaccard | patients.csv, conditions.csv, procedures.csv, observations.csv | Tạo set đặc trưng để tính Jaccard, output Top-K bệnh nhân giống |
| 2. Patient Stratification (Clustering) | K-means / Hierarchical, Jaccard/Cosine | patients.csv, conditions.csv, medications.csv | Vector hóa tuổi, giới, danh sách bệnh/thuốc, output cluster |
| 3. Anomaly Detection | Rare itemset, Distance-based outlier | medications.csv, procedures.csv, observations.csv | Phát hiện phác đồ hiếm / xét nghiệm bất thường |
| 4. Dashboard Analytics | Visualization | Tất cả bảng: patients, conditions, medications, observations, procedures, encounters | Thống kê phác đồ phổ biến, nhóm bệnh nhân, dòng xét nghiệm |

---

## B. 4 Module Recommender System
| Module | Thuật toán | CSV dùng | Mapping | Output |
|--------|------------|----------|---------|--------|
| 5. Treatment / Drug Recommendation | Collaborative Filtering, Frequent Itemset, Association Rules | medications.csv, conditions.csv, encounters.csv | User=patient_id, Item=medication_code, Context=condition_code | Top-N thuốc/phác đồ + giải thích |
| 6. Next Test Recommendation | Sequential Pattern Mining (PrefixSpan), Co-occurrence, Markov | observations.csv, encounters.csv | Session=encounter_id, Sequence=lab/test_code theo thời gian | Xét nghiệm tiếp theo gợi ý |
| 7. Risk Alert | Association Rules, Rule Mining | conditions.csv, medications.csv, observations.csv | {condition, drug} → {risk} | Cảnh báo biến chứng tiềm ẩn |
| 8. Care / Follow-up Recommendation | Frequent Pattern Mining, Association Rules | careplans.csv, procedures.csv, conditions.csv | User=patient_id, Item=care_action/follow-up | Top-N gợi ý hành động tiếp theo |

---

## 📊 Dataset sử dụng
### ⭐ Chính (cho Recommender)
- **MIMIC-IV**: ICU dataset lớn, chuẩn nghiên cứu
- Bảng chính:
  - patients, diagnoses_icd, prescriptions, labevents, procedures_icd, admissions

### 🟡 Phụ / Demo
- **UCI Diabetes**: nhẹ, dễ xử lý, prototype
- **Synthea (Synthetic EHR)**: dữ liệu giả lập, không cần quyền, demo pipeline

### 💡 Mapping Synthea → MIMIC-IV
| Synthea | MIMIC-IV |
|---------|----------|
| patients.csv | patients.csv |
| encounters.csv | admissions.csv |
| conditions.csv | diagnoses_icd.csv |
| medications.csv | prescriptions.csv |
| procedures.csv | procedures_icd.csv |
| observations.csv | labevents.csv |
| careplans.csv | careplans.csv |

> Pipeline giữ nguyên, chỉ đổi tên bảng khi dùng MIMIC-IV.

---

## ⚡ Lợi ích
- Hỗ trợ ra quyết định y tế dựa trên dữ liệu lớn
- Phát hiện bất thường và rủi ro
- Gợi ý điều trị, xét nghiệm và chăm sóc tiếp theo
- Dashboard trực quan, phân tích xu hướng
