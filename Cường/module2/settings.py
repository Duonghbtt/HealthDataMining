from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    #  Đường dẫn 
    zip_path: Path                             

    #  Output 
    output_dir: Path = Path("output_module2")

    #  PySpark 
    extracted_dir: Path = Path("mimic_extracted")   
    spark_driver_memory:   str = "8g"
    spark_executor_memory: str = "4g"
    spark_cores:           str = "4"           

    #  Feature engineering 
    top_icd:  int = 50
    top_drug: int = 50
    limit_patients: Optional[int] = None       

    #  PCA 
    pca_dims: int = 20

    #  Clustering 
    k_range_min:  int = 2
    k_range_max:  int = 9
    best_k:       int = 4
    random_state: int = 42

    #  Similarity metrics 
    use_euclidean: bool = True
    use_cosine:    bool = True
    use_jaccard:   bool = True

    #  Sampling (Hierarchical không scale với N lớn) 
    cosine_sample:  int = 5_000
    jaccard_sample: int = 3_000

    # Visualization 
    tsne_sample:   int = 5_000
    dendro_sample: int = 200
