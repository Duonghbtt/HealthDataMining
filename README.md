# ðŸ¥ ClinRec

<div align="center">

**Patient State Encoding + Self-History Selection + Safe Medication Recommendation**  
**Core pipeline on MIMIC-IV**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MIMIC-IV](https://img.shields.io/badge/Dataset-MIMIC--IV-00897B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Há»‡ thá»‘ng khuyáº¿n nghá»‹ thuá»‘c an toÃ n tá»« dá»¯ liá»‡u EHR nhiá»u láº§n khÃ¡m trÃªn MIMIC-IV**

*Data Mining + Recommender Systems Â· MIMIC-IV Dataset*

[Tá»•ng quan](#1-tá»•ng-quan) Â· [Kiáº¿n trÃºc](#4-kiáº¿n-trÃºc-há»‡-thá»‘ng) Â· [CÃ i Ä‘áº·t](#8-cÃ i-Ä‘áº·t-mÃ´i-trÆ°á»ng) Â· [Huáº¥n luyá»‡n](#10-huáº¥n-luyá»‡n)

</div>

---

## 1. Tá»•ng quan

> Canonical data pipeline hiá»‡n táº¡i: build_cohort -> build_vocab -> build_ddi_matrix -> build_trajectories -> export_tensorized_trajectories.
> New benchmark inputs: MIMIC-IV + RxNorm + DrugBank Vocabulary + DDInter. Legacy DDI mapping text later in this README is deprecated.



**ClinRec** lÃ  pipeline nghiÃªn cá»©u end-to-end cho bÃ i toÃ¡n **safe medication recommendation** tá»« dá»¯ liá»‡u EHR theo chuá»—i visit. PhiÃªn báº£n hiá»‡n táº¡i Ä‘Ã£ Ä‘Æ°á»£c chá»‘t theo **hÆ°á»›ng má»›i** vÃ  chá»‰ giá»¯ láº¡i pháº§n core cá»§a há»‡ thá»‘ng:

- **Patient State Encoder**
- **Self-history selection**
- **Fusion giá»¯a current state vÃ  self-history**
- **Medication prediction**
- **DDI-aware objective**

Há»‡ thá»‘ng táº­p trung tráº£ lá»i cÃ¢u há»i chÃ­nh:

**"NÃªn khuyáº¿n nghá»‹ thuá»‘c gÃ¬ cho visit hiá»‡n táº¡i, dá»±a trÃªn tráº¡ng thÃ¡i hiá»‡n táº¡i vÃ  lá»‹ch sá»­ quan trá»ng cá»§a chÃ­nh bá»‡nh nhÃ¢n, Ä‘á»“ng thá»i háº¡n cháº¿ tÆ°Æ¡ng tÃ¡c thuá»‘c nguy hiá»ƒm?"**

> âš ï¸ **LÆ°u Ã½:** Dá»± Ã¡n phá»¥c vá»¥ má»¥c Ä‘Ã­ch nghiÃªn cá»©u vÃ  há»c thuáº­t. ÄÃ¢y khÃ´ng pháº£i lÃ  há»‡ thá»‘ng triá»ƒn khai lÃ¢m sÃ ng thá»±c táº¿ vÃ  khÃ´ng thay tháº¿ quyáº¿t Ä‘á»‹nh chuyÃªn mÃ´n cá»§a bÃ¡c sÄ©.

---

## 2. BÃ i toÃ¡n

Cho má»™t bá»‡nh nhÃ¢n táº¡i thá»i Ä‘iá»ƒm hiá»‡n táº¡i vá»›i dá»¯ liá»‡u lÃ¢m sÃ ng vÃ  lá»‹ch sá»­ trÆ°á»›c Ä‘Ã³:

- diagnosis codes
- procedure codes
- lab values
- vital signs
- medication history

má»¥c tiÃªu cá»§a há»‡ thá»‘ng lÃ  dá»± Ä‘oÃ¡n:

- **táº­p thuá»‘c phÃ¹ há»£p** cho visit hiá»‡n táº¡i,
- Ä‘á»“ng thá»i **giáº£m nguy cÆ¡ drug-drug interaction (DDI)** trong Ä‘áº§u ra.

BÃ i toÃ¡n Ä‘Æ°á»£c mÃ´ hÃ¬nh hÃ³a nhÆ° **multi-label medication recommendation** trÃªn chuá»—i visit EHR.

---

## 3. ÄÃ³ng gÃ³p chÃ­nh

- MÃ£ hÃ³a **tráº¡ng thÃ¡i bá»‡nh nhÃ¢n theo chuá»—i visit** báº±ng encoder thá»i gian.
- Chá»‰ giá»¯ láº¡i **self-history selection** trÃªn lá»‹ch sá»­ cá»§a chÃ­nh bá»‡nh nhÃ¢n.
- Há»£p nháº¥t **current state** vÃ  **self-history summary** Ä‘á»ƒ táº¡o context vector cho dá»± Ä‘oÃ¡n thuá»‘c.
- Tá»‘i Æ°u **drug recommendation Ä‘a nhÃ£n cÃ³ kiá»ƒm soÃ¡t DDI**.
- Tá»• chá»©c pipeline gá»n hÆ¡n, dá»… train hÆ¡n vÃ  phÃ¹ há»£p hÆ¡n vá»›i Ä‘iá»u kiá»‡n cháº¡y local.

---

## 4. Kiáº¿n trÃºc há»‡ thá»‘ng

### 4.1. Pipeline má»©c cao

```text
Input (diag, proc, lab, vital, med_history)
  â†’ PatientStateEncoder
  â†’ Self-history selection
  â†’ Fusion
  â†’ Medication prediction
  â†’ DDI-aware loss
  â†’ Output: drug_logits, drug_probs
```

### 4.2. Diá»…n giáº£i tá»«ng khá»‘i

1. **PatientStateEncoder**  
   Nháº­n Ä‘áº§u vÃ o gá»“m `diag_codes`, `proc_codes`, `lab_values`, `vital_values`, `med_history`, `visit_mask` vÃ  sinh ra:
   - `visit_repr`
   - `state_sequence`
   - `pooled_state`
   - `visit_mask`

2. **Self-history selection**  
   Chá»n cÃ¡c visit quan trá»ng tá»« **lá»‹ch sá»­ cá»§a chÃ­nh bá»‡nh nhÃ¢n**, thÆ°á»ng báº±ng visit-level attention trÃªn `state_sequence`.

3. **Fusion**  
   Há»£p nháº¥t:
   - `current_state`
   - `self_history_summary`

   Ä‘á»ƒ táº¡o `context_vector`.

4. **Medication prediction**  
   Dá»± Ä‘oÃ¡n:
   - `drug_logits`
   - `drug_probs`

5. **DDI-aware loss**  
   Tá»‘i Æ°u:

   ```text
   total_loss = prediction_loss + lambda_ddi * ddi_loss
   ```

### 4.3. KÃ½ hiá»‡u chÃ­nh

- `visit_repr`: biá»ƒu diá»…n cá»§a tá»«ng visit
- `state_sequence`: chuá»—i tráº¡ng thÃ¡i bá»‡nh nhÃ¢n theo thá»i gian
- `pooled_state`: biá»ƒu diá»…n gá»™p toÃ n trajectory
- `current_state`: tráº¡ng thÃ¡i hiá»‡n táº¡i dÃ¹ng Ä‘á»ƒ dá»± Ä‘oÃ¡n
- `self_history_summary`: tÃ³m táº¯t lá»‹ch sá»­ quan trá»ng cá»§a chÃ­nh bá»‡nh nhÃ¢n
- `context_vector`: vector há»£p nháº¥t cuá»‘i cÃ¹ng
- `drug_logits`, `drug_probs`: Ä‘áº§u ra dá»± Ä‘oÃ¡n thuá»‘c


## 5. Dataset

Dá»± Ã¡n sá»­ dá»¥ng **MIMIC-IV** tá»« PhysioNet.

- Dataset page: https://physionet.org/content/mimiciv/
- Äá»ƒ truy cáº­p cáº§n tÃ i khoáº£n PhysioNet vÃ  hoÃ n thÃ nh training báº¯t buá»™c cá»§a PhysioNet.

### CÃ¡c báº£ng thÆ°á»ng dÃ¹ng

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

> Dá»¯ liá»‡u gá»‘c **khÃ´ng commit lÃªn git**. Chá»‰ lÆ°u trong `data/raw/`.

Legacy note: the benchmark no longer uses ndc2RXCUI.txt, RXCUI2atc4.csv, drug-atc.csv, drug-DDI.csv, or drugbank_drugs_info.csv. Use RxNorm + DrugBank Vocabulary + DDInter instead.
### Dá»¯ liá»‡u ngoÃ i MIMIC-IV Ä‘á»ƒ build DDI matrix

NgoÃ i dá»¯ liá»‡u gá»‘c MIMIC-IV, pipeline cÃ²n cáº§n thÃªm cÃ¡c file mapping/pharmacology Ä‘á»ƒ táº¡o ma tráº­n tÆ°Æ¡ng tÃ¡c thuá»‘c (DDI):

- `drug-atc.csv`: tá»‡p Ã¡nh xáº¡ mÃ£ thuá»‘c sang mÃ£ ATC.
- `ndc2RXCUI.txt`: tá»‡p Ã¡nh xáº¡ NDC sang RXCUI.
- `drugbank_drugs_info.csv`: báº£ng thÃ´ng tin thuá»‘c Ä‘Æ°á»£c táº£i tá»« DrugBank, dÃ¹ng Ä‘á»ƒ Ã¡nh xáº¡ tÃªn thuá»‘c sang chuá»—i SMILES cá»§a thuá»‘c. File nÃ y há»¯u Ã­ch cho cÃ¡c baseline/pháº§n má»Ÿ rá»™ng liÃªn quan tá»›i biá»ƒu diá»…n phÃ¢n tá»­, nhÆ°ng **khÃ´ng báº¯t buá»™c** náº¿u chá»‰ build ma tráº­n DDI nhá»‹ phÃ¢n cÆ¡ báº£n.
- `drug-DDI.csv`: tá»‡p lá»›n chá»©a thÃ´ng tin vá» tÆ°Æ¡ng tÃ¡c thuá»‘c (DDI), Ä‘Æ°á»£c mÃ£ hÃ³a báº±ng CID/STITCH.
- `RXCUI2atc4.csv`: tá»‡p Ã¡nh xáº¡ NDC-RXCUI-ATC4; trong pipeline hiá»‡n táº¡i chá»‰ dÃ¹ng pháº§n Ã¡nh xáº¡ **RXCUI â†’ ATC4**.

### Nguá»“n táº£i cÃ¡c file ngoÃ i
- `drug-atc.csv`,`ndc2RXCUI.txt`: https://github.com/kybinn/DrugDoctor/tree/main
- `drug-DDI.csv`: https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
- `drugbank_drugs_info.csv`: https://drive.google.com/file/d/1EzIlVeiIR6LFtrBnhzAth4fJt6H_ljxk/view?usp=sharing
- `RXCUI2atc4.csv`: láº¥y tá»« repo GAMENet, file gá»‘c cÃ³ tÃªn `ndc2atc_level4.csv`: https://github.com/sjy1203/GAMENet

### Luá»“ng táº¡o DDI matrix

MIMIC-IV **khÃ´ng cung cáº¥p sáºµn DDI matrix**. Ma tráº­n DDI cá»§a project Ä‘Æ°á»£c build theo chuá»—i Ã¡nh xáº¡ sau:

```text
prescriptions.csv.gz (cá»™t ndc)
â†’ ndc2RXCUI.txt
â†’ RXCUI2atc4.csv
â†’ drug-atc.csv
â†’ drug-DDI.csv
â†’ drug_ddi.pt / drug_ddi_report.json
```

Trong Ä‘Ã³:

- `prescriptions.csv.gz` cung cáº¥p thuá»‘c kÃª Ä‘Æ¡n tá»« MIMIC-IV.
- `ndc2RXCUI.txt` ná»‘i NDC trong MIMIC-IV sang RXCUI.
- `RXCUI2atc4.csv` ná»‘i RXCUI sang ATC4.
- `drug-atc.csv` ná»‘i CID/STITCH vá»›i ATC.
- `drug-DDI.csv` cung cáº¥p cÃ¡c cáº·p thuá»‘c cÃ³ tÆ°Æ¡ng tÃ¡c á»Ÿ má»©c CID/STITCH.

---

## 6. Cáº¥u trÃºc thÆ° má»¥c

```text
clinrec/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â””â”€â”€ hosp/
â”‚   â”‚   â””â”€â”€ icu/
â”‚   â”œâ”€â”€ interim/
â”‚   â”‚   â”œâ”€â”€ cohort/
â”‚   â”‚   â”œâ”€â”€ trajectories/
â”‚   â”‚   â””â”€â”€ vocab/
â”‚   â”œâ”€â”€ processed/
â”‚   â”‚   â”œâ”€â”€ train/
â”‚   â”‚   â”œâ”€â”€ val/
â”‚   â”‚   â”œâ”€â”€ test/
â”‚   â”‚   â””â”€â”€ ddi/
â”‚   â””â”€â”€ artifacts/
â”‚       â””â”€â”€ encoder/
â”œâ”€â”€ configs/
â”‚   â”œâ”€â”€ data.yaml
â”‚   â”œâ”€â”€ model.yaml
â”‚   â”œâ”€â”€ train.yaml
â”‚   â””â”€â”€ eval.yaml
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ load_mimic.py
â”‚   â”‚   â”œâ”€â”€ build_cohort.py
â”‚   â”‚   â”œâ”€â”€ build_trajectories.py
â”‚   â”‚   â”œâ”€â”€ build_vocab.py
â”‚   â”‚   â”œâ”€â”€ build_ddi_matrix.py
â”‚   â”‚   â””â”€â”€ dataset.py
â”‚   â”œâ”€â”€ features/
â”‚   â”‚   â”œâ”€â”€ diagnosis_encoder.py
â”‚   â”‚   â”œâ”€â”€ procedure_encoder.py
â”‚   â”‚   â”œâ”€â”€ lab_processor.py
â”‚   â”‚   â”œâ”€â”€ vital_processor.py
â”‚   â”‚   â””â”€â”€ medication_history.py
â”‚   â”œâ”€â”€ models/
â”‚   â”‚   â”œâ”€â”€ patient_state_encoder.py
â”‚   â”‚   â”œâ”€â”€ history_selector.py
â”‚   â”‚   â”œâ”€â”€ fusion.py
â”‚   â”‚   â”œâ”€â”€ medication_decoder.py
â”‚   â”‚   â”œâ”€â”€ ddi_regularization.py
â”‚   â”‚   â””â”€â”€ full_model.py
â”‚   â”œâ”€â”€ training/
â”‚   â”‚   â”œâ”€â”€ losses.py
â”‚   â”‚   â”œâ”€â”€ trainer.py
â”‚   â”‚   â””â”€â”€ train_core.py
â”‚   â”œâ”€â”€ evaluation/
â”‚   â”‚   â”œâ”€â”€ metrics.py
â”‚   â”‚   â”œâ”€â”€ evaluate_core.py
â”‚   â”‚   â”œâ”€â”€ evaluate_safety.py
â”‚   â”‚   â””â”€â”€ evaluate_ablation.py
â”‚   â””â”€â”€ utils/
â”‚       â”œâ”€â”€ seed.py
â”‚       â”œâ”€â”€ logger.py
â”‚       â”œâ”€â”€ io.py
â”‚       â””â”€â”€ device.py
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ 01_eda_mimic_iv.ipynb
â”‚   â”œâ”€â”€ 02_build_cohort.ipynb
â”‚   â”œâ”€â”€ 03_train_base.ipynb
â”‚   â””â”€â”€ 05_train_full_core.ipynb
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ preprocess.ps1
â”‚   â”œâ”€â”€ train_core.ps1
â”‚   â””â”€â”€ evaluate.ps1
â”œâ”€â”€ tests/
â”‚   â”œâ”€â”€ test_data.py
â”‚   â”œâ”€â”€ test_encoder.py
â”‚   â”œâ”€â”€ test_history_selector.py
â”‚   â”œâ”€â”€ test_fusion.py
â”‚   â””â”€â”€ test_decoder.py
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ checkpoints/
â”‚   â”œâ”€â”€ logs/
â”‚   â”œâ”€â”€ predictions/
â”‚   â”œâ”€â”€ figures/
â”‚   â””â”€â”€ reports/
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .gitignore
â””â”€â”€ README.md
```

### Ghi chÃº

- Náº¿u repo hiá»‡n táº¡i váº«n cÃ³ `src/retrieval/`, `src/graph/`, `src/explainability/` hoáº·c cÃ¡c notebook/train script cá»§a hÆ°á»›ng cÅ©, cÃ³ thá»ƒ giá»¯ táº¡m trong repo nhÆ°ng **khÃ´ng dÃ¹ng trong pipeline má»›i**.
- Khi dá»n repo sáº¡ch hÆ¡n, nÃªn chuyá»ƒn cÃ¡c pháº§n Ä‘Ã³ sang nhÃ¡nh khÃ¡c hoáº·c Ä‘Ã¡nh dáº¥u deprecated.

---

## 7. Luá»“ng gá»i chÃ­nh giá»¯a cÃ¡c module

### 7.1. Pha dá»¯ liá»‡u

```text
load_mimic.py
  â†’ build_cohort.py
  â†’ build_vocab.py / build_ddi_matrix.py
  â†’ build_trajectories.py
  â†’ dataset.py
```

### 7.2. Pha core model

```text
patient_state_encoder.py
  â†’ history_selector.py
  â†’ fusion.py
  â†’ medication_decoder.py
  â†’ ddi_regularization.py
  â†’ full_model.py
```

### 7.3. Pha train

```text
losses.py
  â†’ trainer.py
  â†’ train_core.py
```

### 7.4. Pha Ä‘Ã¡nh giÃ¡

```text
metrics.py
  â†’ evaluate_core.py
  â†’ evaluate_safety.py / evaluate_ablation.py
```

---

## 8. CÃ i Ä‘áº·t mÃ´i trÆ°á»ng

### YÃªu cáº§u tá»‘i thiá»ƒu

- Python 3.9+
- Khuyáº¿n nghá»‹ dÃ¹ng mÃ´i trÆ°á»ng áº£o
- RAM Ä‘á»§ lá»›n Ä‘á»ƒ xá»­ lÃ½ cohort vÃ  trajectory tá»« MIMIC-IV
- GPU lÃ  tÃ¹y chá»n nhÆ°ng há»¯u Ã­ch khi train

### Clone repo

```bash
git clone https://github.com/your-username/clinrec.git
cd clinrec
```

### Táº¡o mÃ´i trÆ°á»ng áº£o

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

### CÃ i dependencies

```bash
pip install -r requirements.txt
```

---

## 9. Chuáº©n bá»‹ dá»¯ liá»‡u

### BÆ°á»›c 1. Äáº·t dá»¯ liá»‡u MIMIC-IV vÃ o Ä‘Ãºng thÆ° má»¥c

Tá»‘i thiá»ƒu cáº§n báº£ng thuá»‘c sau:

```text
data/raw/hosp/prescriptions.csv.gz
```

NgoÃ i ra cÃ¡c bÆ°á»›c preprocess cohort / trajectory cÃ³ thá»ƒ dÃ¹ng thÃªm cÃ¡c báº£ng khÃ¡c trong `hosp/` vÃ  `icu/`.

Legacy note: the file list below is deprecated. Current DDI inputs are the extracted RxNorm release directory, DrugBank Vocabulary CSV, and DDInter files under data/processed/ddi/.
### BÆ°á»›c 2. Äáº·t cÃ¡c file ngoÃ i Ä‘á»ƒ build DDI matrix

Äáº·t cÃ¡c file sau vÃ o thÆ° má»¥c `data/processed/ddi/` hoáº·c sá»­a láº¡i path tÆ°Æ¡ng á»©ng trong config:

```text
data/processed/ddi/drug-DDI.csv
data/processed/ddi/drug-atc.csv
data/processed/ddi/ndc2RXCUI.txt
data/processed/ddi/RXCUI2atc4.csv
```

Náº¿u báº¡n dÃ¹ng thÃªm baseline/pháº§n má»Ÿ rá»™ng liÃªn quan molecular graph, cÃ³ thá»ƒ Ä‘áº·t thÃªm:

```text
data/processed/ddi/drugbank_drugs_info.csv
```

### BÆ°á»›c 3. Kiá»ƒm tra cáº¥u hÃ¬nh

Chá»‰nh cÃ¡c file sau cho phÃ¹ há»£p vá»›i mÃ¡y cá»§a báº¡n:

- `configs/data.yaml`
- `configs/model.yaml`
- `configs/train.yaml`
- `configs/eval.yaml`

Canonical configs/data.yaml paths now point to rxnorm_root, drugbank_vocab_path, ddinter_root, ddi_root, and trajectory_interim_root.
RiÃªng `configs/data.yaml`, cáº§n kiá»ƒm tra cÃ¡c path cho bÆ°á»›c build DDI, vÃ­ dá»¥:

```yaml
paths:
  ddi_source_path: data/processed/ddi/drug-DDI.csv
  mimic_prescriptions_path: data/raw/hosp/prescriptions.csv.gz
  ndc_to_rxcui_path: data/processed/ddi/ndc2RXCUI.txt
  rxcui_to_atc4_path: data/processed/ddi/RXCUI2atc4.csv
  drug_atc_path: data/processed/ddi/drug-atc.csv
```

### BÆ°á»›c 4. Cháº¡y preprocessing

**PowerShell**

```powershell
./scripts/preprocess.ps1
```

Hoáº·c cháº¡y tá»«ng bÆ°á»›c báº±ng Python:

```bash
python -m src.data.build_cohort
python -m src.data.build_vocab
python -m src.data.build_ddi_matrix
python -m src.data.build_trajectories
```

### BÆ°á»›c 5. Kiá»ƒm tra build DDI thÃ nh cÃ´ng

Sau khi cháº¡y `build_ddi_matrix`, cáº§n cÃ³:

```text
data/processed/ddi/drug_ddi.pt
data/processed/ddi/drug_ddi_report.json
```

Má»Ÿ `drug_ddi_report.json` Ä‘á»ƒ kiá»ƒm tra:

- `matched_pairs > 0`
- `matrix_shape` khá»›p vá»›i drug vocab
- khÃ´ng cÃ²n tráº¡ng thÃ¡i `fallback_zero`

Náº¿u `matched_pairs = 0`, nghÄ©a lÃ  pipeline DDI chÆ°a ná»‘i Ä‘Æ°á»£c Ä‘Ãºng giá»¯a:

- thuá»‘c trong MIMIC-IV (`prescriptions.csv.gz`)
- file mapping NDC/RXCUI/ATC
- file `drug-DDI.csv`

Sau bÆ°á»›c nÃ y, cÃ¡c thÆ° má»¥c quan trá»ng cáº§n xuáº¥t hiá»‡n:

- `data/interim/cohort/`
- `data/interim/trajectories/`
- `data/interim/vocab/`
- `data/processed/train/`
- `data/processed/val/`
- `data/processed/test/`
- `data/processed/ddi/`

---

## 10. Huáº¥n luyá»‡n

### 10.1. Train báº£n core

```powershell
./scripts/train_core.ps1
```

hoáº·c:

```bash
python -m src.training.train_core
```

### 10.2. CÃ´ng thá»©c loss

```text
total_loss = prediction_loss + lambda_ddi * ddi_loss
```

Trong Ä‘Ã³:

- `prediction_loss`: BCEWithLogitsLoss cho bÃ i toÃ¡n multi-label medication recommendation
- `ddi_loss`: regularization dá»±a trÃªn ma tráº­n DDI
- `lambda_ddi`: há»‡ sá»‘ cÃ¢n báº±ng giá»¯a accuracy vÃ  safety

### 10.3. Output cáº§n theo dÃµi khi train

- `prediction_loss`
- `ddi_loss`
- `total_loss`
- Jaccard
- F1
- PRAUC
- DDI Rate

---

## 11. ÄÃ¡nh giÃ¡

### CÃ¡c metric chÃ­nh

- Jaccard
- F1 Score
- PRAUC
- DDI Rate
- Avg #Drugs

### Cháº¡y Ä‘Ã¡nh giÃ¡

```powershell
./scripts/evaluate.ps1
```

hoáº·c:

```bash
python -m src.evaluation.evaluate_core
python -m src.evaluation.evaluate_safety
python -m src.evaluation.evaluate_ablation
```

### Gá»£i Ã½ ablation Ä‘Ãºng hÆ°á»›ng má»›i

- Base encoder + decoder
- + Self-history selection
- + Fusion
- + DDI-aware loss
- Full core

---

## 12. Quy trÃ¬nh xÃ¢y dá»±ng khuyáº¿n nghá»‹

### Pha 1 â€” KhÃ³a dá»¯ liá»‡u vÃ  tiá»n xá»­ lÃ½

- `load_mimic.py`
- `build_cohort.py`
- `build_vocab.py`
- `build_ddi_matrix.py`
- `build_trajectories.py`
- `dataset.py`

**Äiá»u kiá»‡n Ä‘áº¡t:** cohort sáº¡ch, vocab á»•n Ä‘á»‹nh, DDI matrix Ä‘Ãºng kÃ­ch thÆ°á»›c, batch Ä‘áº§u tiÃªn load Ä‘Æ°á»£c.

### Pha 2 â€” Dá»±ng encoder

- `patient_state_encoder.py`

**Äiá»u kiá»‡n Ä‘áº¡t:** forward pass á»•n Ä‘á»‹nh, sinh Ä‘Æ°á»£c:
- `visit_repr`
- `state_sequence`
- `pooled_state`
- `visit_mask`

### Pha 3 â€” ThÃªm self-history selection

- `history_selector.py`

**Äiá»u kiá»‡n Ä‘áº¡t:** chá»n Ä‘Æ°á»£c visit quan trá»ng tá»« chÃ­nh lá»‹ch sá»­ bá»‡nh nhÃ¢n, attention mask Ä‘Ãºng, khÃ´ng dÃ¹ng neighbor branch.

### Pha 4 â€” ThÃªm fusion vÃ  decoder

- `fusion.py`
- `medication_decoder.py`
- `full_model.py`

**Äiá»u kiá»‡n Ä‘áº¡t:** full forward pass cháº¡y end-to-end tá»« batch Ä‘áº¿n `drug_logits`.

### Pha 5 â€” Huáº¥n luyá»‡n vÃ  Ä‘Ã¡nh giÃ¡

- `losses.py`
- `trainer.py`
- `train_core.py`
- `evaluate_core.py`
- `evaluate_safety.py`
- `evaluate_ablation.py`

**Äiá»u kiá»‡n Ä‘áº¡t:** cÃ³ checkpoint tá»‘t nháº¥t, báº£ng metric vÃ  safety report.

### Pha 6 â€” Script hÃ³a vÃ  test hÃ³a

- `scripts/*`
- `tests/*`
- `README.md`

**Äiá»u kiá»‡n Ä‘áº¡t:** ngÆ°á»i khÃ¡c clone repo cÃ³ thá»ƒ preprocess, train vÃ  evaluate báº£n core.

---

## 13. Kiá»ƒm thá»­

CÃ¡c test chÃ­nh:

- `tests/test_data.py`
- `tests/test_encoder.py`
- `tests/test_history_selector.py`
- `tests/test_fusion.py`
- `tests/test_decoder.py`

Khuyáº¿n nghá»‹ cháº¡y test sá»›m theo tá»«ng pha thay vÃ¬ Ä‘á»ƒ Ä‘áº¿n cuá»‘i.

---

## 14. Baselines vÃ  paper liÃªn quan

### Baselines nÃªn biáº¿t

- GAMENet
- SafeDrug
- MICRON
- COGNet
- MoleRec
- VITA

### Gá»£i Ã½ theo module

- **Patient state encoding:** RETAIN, BEHRT, Med-BERT
- **Relevant visit selection:** VITA
- **DDI-aware objective:** SafeDrug, MoleRec

> README hiá»‡n táº¡i chá»‰ mÃ´ táº£ **pipeline core Ä‘ang dÃ¹ng**. Má»™t sá»‘ paper nhÆ° DAPSNet, RaVSNet, HypeMed hoáº·c cÃ¡c hÆ°á»›ng retrieval / hypergraph váº«n cÃ³ thá»ƒ xuáº¥t hiá»‡n trong pháº§n related work cá»§a bÃ¡o cÃ¡o, nhÆ°ng khÃ´ng pháº£i lÃ  thÃ nh pháº§n cá»§a code pipeline má»›i.

---

## 15. Artifact Ä‘áº§u ra

Sau khi train / evaluate, cÃ¡c káº¿t quáº£ Ä‘Æ°á»£c lÆ°u táº¡i:

```text
outputs/
â”œâ”€â”€ checkpoints/
â”œâ”€â”€ logs/
â”œâ”€â”€ predictions/
â”œâ”€â”€ figures/
â””â”€â”€ reports/
```

ÄÃ¢y lÃ  nÆ¡i lÆ°u:

- checkpoint tá»‘t nháº¥t
- log huáº¥n luyá»‡n
- prediction export
- biá»ƒu Ä‘á»“ loss / metric / ablation
- bÃ¡o cÃ¡o tá»•ng há»£p cuá»‘i cÃ¹ng

RiÃªng artifact cá»§a bÆ°á»›c build DDI Ä‘Æ°á»£c lÆ°u táº¡i:

```text
data/processed/ddi/drug_ddi.pt
data/processed/ddi/drug_ddi_report.json
```

- `drug_ddi.pt`: ma tráº­n DDI dÃ¹ng khi train/evaluate
- `drug_ddi_report.json`: bÃ¡o cÃ¡o mapping vÃ  sá»‘ cáº·p DDI match Ä‘Æ°á»£c

---

## 16. ThÃ nh viÃªn nhÃ³m

| ThÃ nh viÃªn | Vai trÃ² chÃ­nh |
|---|---|
| BÃ¹i Äá»©c Äáº¡i | Data + Features + Patient State Encoder |
| Äá»— Máº¡nh CÆ°á»ng | Self-history selection + Integration support |
| Nguyá»…n VÄƒn PhÃºc | Fusion + Ablation + Model integration |
| Nguyá»…n Tháº¿ DÆ°Æ¡ng | Decoder + Training + Evaluation + Documentation |

---

## 17. Ghi chÃº sá»­ dá»¥ng repo

- KhÃ´ng commit dá»¯ liá»‡u gá»‘c MIMIC-IV.
- KhÃ´ng commit checkpoint lá»›n, log táº¡m, cache vÃ  file nháº¡y cáº£m.
- Æ¯u tiÃªn á»•n Ä‘á»‹nh **core pipeline** trÆ°á»›c khi thá»­ nghiá»‡m báº¥t ká»³ má»Ÿ rá»™ng nÃ o.
- Logic production nÃªn náº±m trong `src/` vÃ  `scripts/`, khÃ´ng Ä‘á»ƒ notebook lÃ  nÆ¡i duy nháº¥t chá»©a code chÃ­nh.
- Náº¿u cÃ²n file retrieval / graph / explainability trong repo, hÃ£y xem Ä‘Ã³ lÃ  pháº§n cÅ© vÃ  trÃ¡nh import vÃ o pipeline má»›i.

---

## 18. License

MIT License

---

## 19. Citation

Náº¿u báº¡n sá»­ dá»¥ng repo hoáº·c Ã½ tÆ°á»Ÿng tá»« dá»± Ã¡n nÃ y cho bÃ¡o cÃ¡o / nghiÃªn cá»©u, hÃ£y trÃ­ch dáº«n repo vÃ  cÃ¡c paper baseline liÃªn quan trong pháº§n tÃ i liá»‡u tham kháº£o.

---

## 20. Disclaimer

ClinRec lÃ  há»‡ thá»‘ng nghiÃªn cá»©u phá»¥c vá»¥ há»c thuáº­t. Má»i Ä‘áº§u ra cá»§a há»‡ thá»‘ng chá»‰ mang tÃ­nh cháº¥t há»— trá»£ phÃ¢n tÃ­ch vÃ  khÃ´ng Ä‘Æ°á»£c xem lÃ  chá»‰ Ä‘á»‹nh lÃ¢m sÃ ng thá»±c táº¿.




