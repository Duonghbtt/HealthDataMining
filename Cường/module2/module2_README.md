# Module 2 — Patient Stratification / Clustering (PySpark)


## Cấu trúc file

```
module2/
├── __init__.py
├── settings.py          
├── reader.py            
├── vectorizer.py        
├── model.py             
├── plots.py             
└── Module2.ipynb  
```
## Pipeline

```
mimic-iv-3.1.zip
    ↓  extract_zip()              Cell 4  [Python zipfile, 1 lần]
mimic_extracted/*.csv.gz
    ↓  Spark read CSV             Cell 5  [song song, lazy]
Spark DataFrames (20M rows)
    ↓  Spark groupBy top tokens   Cell 6  [song song]
    ↓  Spark pivot + join         Cell 7  [song song]
pandas DataFrame (211k × 102)
    ↓  StandardScaler + PCA       Cell 7  [sklearn]
X_pca / X_scaled / X_binary
    ↓  find_best_k                Cell 9  [sklearn KMeans]
    ↓  KMeans + Hierarchical      Cell 10 [sklearn]
Clusters
    ↓  save outputs               Cell 14
patient_clusters.csv →  Dashboard
```

