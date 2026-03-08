
from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering   
from sklearn.metrics import adjusted_rand_score       


#  PHẦN 1 — THUẬT TOÁN CODE 

def euclidean_distance_scratch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tính ma trận Euclidean distance giữa mỗi điểm trong A và B.

    Công thức:
        d(a, b) = sqrt( Σ(a_i - b_i)² )

    Dùng trick numpy để tính nhanh, tránh vòng for O(N*K*D):
        ||a - b||² = ||a||² + ||b||² - 2·aᵀb

    Args:
        A : (N, D)
        B : (K, D)
    Returns:
        dist : (N, K)  —  dist[i,j] = khoảng cách từ A[i] đến B[j]
    """
    sq_A    = (A ** 2).sum(axis=1, keepdims=True)    # (N, 1)
    sq_B    = (B ** 2).sum(axis=1, keepdims=True).T  # (1, K)
    dot_AB  = A @ B.T                                 # (N, K)
    sq_dist = sq_A + sq_B - 2.0 * dot_AB
    return np.sqrt(np.clip(sq_dist, 0.0, None))       # clip tránh âm do float error


def cosine_similarity_scratch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tính ma trận Cosine similarity giữa mỗi điểm trong A và B.

    Công thức:
        cos(a, b) = dot(a, b) / (||a|| · ||b||)

    Giá trị trong [-1, 1]:
        1  → cùng hướng (rất giống)
        0  → vuông góc (không liên quan)
       -1  → ngược hướng (rất khác)

    Args:
        A : (N, D)
        B : (K, D)
    Returns:
        sim : (N, K)
    """
    dot    = A @ B.T                                      # (N, K)
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)    # (N, 1)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True).T  # (1, K)
    return dot / (norm_A * norm_B + 1e-10)


def jaccard_similarity_scratch(a: np.ndarray, b: np.ndarray) -> float:
    """
    Tính Jaccard similarity giữa 2 vector nhị phân.

    Công thức:
        J(a, b) = |a ∩ b| / |a ∪ b|
                = Σ min(a_i, b_i) / Σ max(a_i, b_i)

    Giá trị trong [0, 1]:  1 = giống hoàn toàn, 0 = không có gì chung.

    Args:
        a, b : vector nhị phân 0/1, shape (D,)
    Returns:
        float: Jaccard similarity
    """
    intersection = np.minimum(a, b).sum()
    union        = np.maximum(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def kmeans_scratch(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    n_init: int = 10,
    random_state: int = 42,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    K-means clustering — thuần numpy.

    Thuật toán Lloyd's algorithm:
        1. Khởi tạo K centroids bằng K-means++ (chọn thông minh, không random thuần)
        2. Assign: mỗi điểm gán vào centroid Euclidean gần nhất
        3. Update: centroid mới = mean của tất cả điểm thuộc cluster đó
        4. Lặp 2-3 cho đến khi centroid không dịch chuyển đáng kể (< tol) hoặc hết max_iter
        5. Chạy lại n_init lần, giữ kết quả có inertia thấp nhất

    K-means++ initialization:
        Thay vì chọn K centroid ngẫu nhiên, mỗi centroid tiếp theo được chọn
        với xác suất tỉ lệ d²(x, centroid_gần_nhất).
        Ưu điểm: hội tụ nhanh hơn, ít bị local minimum hơn.

    Args:
        X            : (N, D) — feature matrix
        k            : số cluster
        max_iter     : số vòng lặp tối đa mỗi lần chạy
        n_init       : số lần khởi tạo lại (lấy kết quả tốt nhất)
        random_state : seed
        tol          : ngưỡng hội tụ (centroid dịch < tol thì dừng)

    Returns:
        labels    : (N,)   — cluster id của từng điểm (0..k-1)
        centroids : (K, D) — centroid cuối cùng
        inertia   : float  — tổng Σ||x - centroid||² (thấp hơn = tốt hơn)
    """
    rng            = np.random.RandomState(random_state)
    best_labels    = None
    best_centroids = None
    best_inertia   = np.inf

    for _ in range(n_init):

        # ─ K-means++ initialization 
        first_idx = rng.randint(0, len(X))
        centroids = [X[first_idx].copy()]

        for _ in range(k - 1):
            cent_arr = np.array(centroids)                      # (c, D)
            dists    = euclidean_distance_scratch(X, cent_arr)  # (N, c)
            min_dist = dists.min(axis=1)                        # (N,)
            probs    = min_dist ** 2
            probs   /= probs.sum()
            next_idx = rng.choice(len(X), p=probs)
            centroids.append(X[next_idx].copy())

        centroids = np.array(centroids)   # (K, D)
        labels    = np.zeros(len(X), dtype=np.int32)

        # ─ Lloyd's iterations 
        for _ in range(max_iter):

            # Assign: mỗi điểm → centroid gần nhất
            dists      = euclidean_distance_scratch(X, centroids)  # (N, K)
            new_labels = dists.argmin(axis=1)                      # (N,)

            # Update: centroid = mean của cluster
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = new_labels == c
                if mask.sum() > 0:
                    new_centroids[c] = X[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]   # cluster rỗng: giữ nguyên

            # Kiểm tra hội tụ
            shift     = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            labels    = new_labels
            if shift < tol:
                break

        # Tính inertia = tổng bình phương khoảng cách đến centroid
        inertia = 0.0
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                diff     = X[mask] - centroids[c]
                inertia += float((diff ** 2).sum())

        if inertia < best_inertia:
            best_inertia   = inertia
            best_labels    = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids, best_inertia


def silhouette_scratch(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    sample_size: int = 5_000,
    random_state: int = 42,
) -> float:
    """
    Silhouette Score — CODE CHAY thuần numpy.

    Công thức cho mỗi điểm i:
        a(i) = mean( d(i, j) ) với mọi j cùng cluster          (cohesion)
        b(i) = min_c≠cluster_i  mean( d(i, j) ) với mọi j ∈ c  (separation)
        s(i) = ( b(i) - a(i) ) / max( a(i), b(i) )
        Score = mean( s(i) )

    s(i) ∈ [-1, 1]:
        ~1  = cluster gọn + tách biệt tốt
        ~0  = điểm nằm ở ranh giới
        ~-1 = bị gán nhầm cluster

    Args:
        metric : "euclidean", "cosine", hoặc "jaccard"
    """
    rng = np.random.RandomState(random_state)
    n   = len(X)

    if n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X_s, L_s = X[idx], labels[idx]
    else:
        X_s, L_s = X, labels

    n_s             = len(X_s)
    unique_clusters = np.unique(L_s)

    if len(unique_clusters) < 2:
        return 0.0

    # Tính ma trận distance
    if metric == "euclidean":
        D = euclidean_distance_scratch(X_s, X_s)
    elif metric == "cosine":
        D = np.clip(1.0 - cosine_similarity_scratch(X_s, X_s), 0.0, None)
    elif metric == "jaccard":
        # O(n_s²) — chỉ chạy trên sample nhỏ nên chấp nhận được
        D = np.zeros((n_s, n_s))
        for i in range(n_s):
            for j in range(i + 1, n_s):
                d = 1.0 - jaccard_similarity_scratch(X_s[i], X_s[j])
                D[i, j] = d
                D[j, i] = d
    else:
        D = euclidean_distance_scratch(X_s, X_s)

    scores = np.zeros(n_s)
    for i in range(n_s):
        c_i          = L_s[i]
        same_mask    = (L_s == c_i)
        same_mask[i] = False

        if same_mask.sum() == 0:
            scores[i] = 0.0
            continue

        a_i = D[i, same_mask].mean()

        b_i = np.inf
        for c_other in unique_clusters:
            if c_other == c_i:
                continue
            other_mask = (L_s == c_other)
            mean_d     = D[i, other_mask].mean()
            if mean_d < b_i:
                b_i = mean_d

        denom     = max(a_i, b_i)
        scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(scores.mean())


def davies_bouldin_scratch(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Davies-Bouldin Score — CODE CHAY thuần numpy.

    Công thức:
        s_i  = mean( ||x - centroid_i|| )  cho x ∈ cluster i  (scatter)
        d_ij = ||centroid_i - centroid_j||                      (separation)
        R_ij = (s_i + s_j) / d_ij
        DB   = (1/K) Σ_i  max_{j≠i} R_ij

    Giá trị nhỏ = tốt (cluster gọn + cách xa nhau).
    """
    unique_clusters = np.unique(labels)
    k               = len(unique_clusters)

    if k < 2:
        return 0.0

    centroids = np.zeros((k, X.shape[1]))
    scatters  = np.zeros(k)

    for idx, c in enumerate(unique_clusters):
        mask           = labels == c
        centroids[idx] = X[mask].mean(axis=0)
        diffs          = X[mask] - centroids[idx]
        scatters[idx]  = np.sqrt((diffs ** 2).sum(axis=1)).mean()

    D_vals = np.zeros(k)
    for i in range(k):
        max_R = -np.inf
        for j in range(k):
            if i == j:
                continue
            diff = centroids[i] - centroids[j]
            d_ij = float(np.sqrt((diff ** 2).sum()))
            if d_ij == 0:
                continue
            R_ij = (scatters[i] + scatters[j]) / d_ij
            if R_ij > max_R:
                max_R = R_ij
        D_vals[i] = max_R if max_R > -np.inf else 0.0

    return float(D_vals.mean())
#  PHẦN 2 — HÀM TÌM K TỐI ƯU

def find_best_k(
    X: np.ndarray,
    k_range: range,
    random_state: int = 42,
    sample_size: int = 20_000,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Tính Inertia, Silhouette, Davies-Bouldin cho mỗi K.
    Dùng hoàn toàn hàm code chay: kmeans_scratch, silhouette_scratch, davies_bouldin_scratch.

    Returns:
        inertia_dict    : {k: inertia}
        silhouette_dict : {k: score}
        db_dict         : {k: score}
    """
    n = len(X)
    if n > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, sample_size, replace=False)
        X_s = X[idx]
    else:
        X_s = X

    inertia_d, sil_d, db_d = {}, {}, {}

    for k in k_range:
        lbl, _, inertia = kmeans_scratch(
            X_s, k=k, max_iter=300, n_init=5, random_state=random_state
        )
        inertia_d[k] = inertia
        sil_d[k]     = silhouette_scratch(
            X_s, lbl, metric="euclidean",
            sample_size=5_000, random_state=random_state,
        )
        db_d[k]      = davies_bouldin_scratch(X_s, lbl)

        print(f"  K={k}: inertia={inertia:,.0f}  "
              f"silhouette={sil_d[k]:.4f}  davies_bouldin={db_d[k]:.4f}")

    return inertia_d, sil_d, db_d


def auto_best_k(silhouette_dict: Dict[int, float]) -> int:
    """Trả về K có Silhouette score cao nhất."""
    return max(silhouette_dict, key=silhouette_dict.get)

#  PHẦN 3 — DATACLASSES
@dataclass
class MetricResult:
    name: str            # 'Euclidean', 'Cosine', 'Jaccard'
    algorithm: str       # 'KMeans (scratch)', 'Hierarchical (sklearn)'
    labels: np.ndarray
    silhouette: float
    davies_bouldin: float
    n_patients: int
    note: str = ""


@dataclass
class AssignResult:
    cluster_id: int
    cluster_name: str
    confidence: float
    distances_euclidean:     Dict[str, float]
    cosine_sim_to_centroids: Dict[str, float]


#  PHẦN 4 — MAIN MODEL CLASS

class PatientClustering:
    """
    Fit 3 clustering models với 3 similarity metrics.

    CODE CHAY (thuần numpy):
        [1] K-means Euclidean  → kmeans_scratch()
        [2] Cosine similarity  → cosine_similarity_scratch()
        [3] Euclidean distance → euclidean_distance_scratch()
        [4] Silhouette score   → silhouette_scratch()
        [5] Davies-Bouldin     → davies_bouldin_scratch()

    Dùng sklearn (lý do có ghi):
        AgglomerativeClustering — Hierarchical cần heap/union-find/dendrogram,
                                  quá phức tạp để code chay trong scope CS246
    """

    def __init__(self, best_k: int = 4, random_state: int = 42):
        self.best_k       = best_k
        self.random_state = random_state

        self.centroids_pca:      np.ndarray = None
        self.centroids_scaled:   np.ndarray = None
        self.labels_kmeans:      np.ndarray = None
        self.labels_cosine:      np.ndarray = None
        self.labels_jaccard:     np.ndarray = None
        self.cosine_sample_idx:  np.ndarray = None
        self.jaccard_sample_idx: np.ndarray = None

        # Chỉ 2 model này dùng sklearn
        self.hier_cosine:  Optional[AgglomerativeClustering] = None
        self.hier_jaccard: Optional[AgglomerativeClustering] = None

        self.results:       List[MetricResult] = []
        self.cluster_names: Dict[int, str]     = {}
        self.patient_ids:   List[int]          = []

    def fit(
        self,
        X_pca:       np.ndarray,
        X_scaled:    np.ndarray,
        X_binary:    np.ndarray,
        patient_ids: List[int],
        cosine_sample:  int = 5_000,
        jaccard_sample: int = 3_000,
    ) -> "PatientClustering":
        """
        Fit cả 3 clustering models.

        Tại sao Hierarchical dùng sample?
            AgglomerativeClustering cần ma trận N×N pairwise distances.
            N=211k → 211k × 211k × 8 bytes ≈ 167 GB RAM → MemoryError.
            K-means không bị vì chỉ tính khoảng cách đến K centroids.
        """
        self.patient_ids = patient_ids
        n   = len(X_pca)
        rng = np.random.RandomState(self.random_state)

        print(f"\n{'='*65}")
        print(f"FIT CLUSTERING  K={self.best_k}  N={n:,}")
        print(f"{'='*65}")
        print(f"[1] K-means      → CODE CHAY (kmeans_scratch)")
        print(f"[2] Hier Cosine  → sklearn   (sample {min(cosine_sample, n):,})")
        print(f"[3] Hier Jaccard → sklearn   (sample {min(jaccard_sample, n):,})")

        #  [1/3] K-MEANS 
        print(f"\n[1/3] K-means (Euclidean) — CODE CHAY — {n:,} benh nhan ...")

        self.labels_kmeans, self.centroids_pca, inertia = kmeans_scratch(
            X_pca, k=self.best_k,
            max_iter=300, n_init=10, random_state=self.random_state,
        )

        # Centroids trên X_scaled dùng cho cosine assign
        self.centroids_scaled = np.array([
            X_scaled[self.labels_kmeans == c].mean(axis=0)
            for c in range(self.best_k)
        ])

        # Đánh giá bằng hàm code chay
        sil_eu = silhouette_scratch(
            X_pca, self.labels_kmeans,
            metric="euclidean", sample_size=10_000, random_state=self.random_state,
        )
        db_eu = davies_bouldin_scratch(X_pca, self.labels_kmeans)

        self.results.append(MetricResult(
            name="Euclidean", algorithm="KMeans (scratch)",
            labels=self.labels_kmeans,
            silhouette=sil_eu, davies_bouldin=db_eu,
            n_patients=n,
            note=f"Inertia={inertia:,.0f}  Toan bo dataset  CODE CHAY",
        ))
        print(f"  Inertia={inertia:,.0f}  Silhouette={sil_eu:.4f}  Davies-Bouldin={db_eu:.4f}")

        # [2/3] HIERARCHICAL COSINE — sklearn (sample) 
        actual_cosine = min(cosine_sample, n)
        print(f"\n[2/3] Hierarchical (Cosine) — sklearn — sample {actual_cosine:,}/{n:,} ...")
        print(f"  [sklearn: Agglomerative can heap+dendrogram phuc tap]")
        print(f"  [sample : N={n:,} -> {n*n*8//1024**3} GB RAM neu full]")

        self.cosine_sample_idx = rng.choice(n, actual_cosine, replace=False)
        X_cos = X_scaled[self.cosine_sample_idx]

        self.hier_cosine = AgglomerativeClustering(
            n_clusters=self.best_k, metric="cosine", linkage="average"
        )
        labels_cos_sample = self.hier_cosine.fit_predict(X_cos)

        # Assign toàn bộ → dùng cosine_similarity_scratch 
        cos_centroids  = np.array([
            X_cos[labels_cos_sample == c].mean(axis=0)
            for c in range(self.best_k)
        ])
        cos_sim_matrix     = cosine_similarity_scratch(X_scaled, cos_centroids) 
        self.labels_cosine = cos_sim_matrix.argmax(axis=1)

        sil_cos = silhouette_scratch(
            X_cos, labels_cos_sample,
            metric="cosine", sample_size=min(3_000, actual_cosine),
            random_state=self.random_state,
        )
        db_cos  = davies_bouldin_scratch(X_cos, labels_cos_sample)
        ari_cos = adjusted_rand_score(
            self.labels_kmeans[self.cosine_sample_idx], labels_cos_sample
        )
        self.results.append(MetricResult(
            name="Cosine", algorithm="Hierarchical (sklearn)",
            labels=self.labels_cosine,
            silhouette=sil_cos, davies_bouldin=db_cos,
            n_patients=actual_cosine,
            note=f"Sample {actual_cosine:,}  ARI vs KMeans={ari_cos:.4f}",
        ))
        print(f"  Silhouette={sil_cos:.4f}  Davies-Bouldin={db_cos:.4f}  ARI={ari_cos:.4f}")

        #  [3/3] HIERARCHICAL JACCARD — sklearn (sample) 
        actual_jac = min(jaccard_sample, n)
        print(f"\n[3/3] Hierarchical (Jaccard) — sklearn — sample {actual_jac:,}/{n:,} ...")

        self.jaccard_sample_idx = rng.choice(n, actual_jac, replace=False)
        X_bin = X_binary[self.jaccard_sample_idx]

        self.hier_jaccard = AgglomerativeClustering(
            n_clusters=self.best_k, metric="jaccard", linkage="average"
        )
        self.labels_jaccard = self.hier_jaccard.fit_predict(X_bin)

        sil_jac = silhouette_scratch(
            X_bin, self.labels_jaccard,
            metric="jaccard", sample_size=2_000, random_state=self.random_state,
        )
        db_jac  = davies_bouldin_scratch(X_bin, self.labels_jaccard)
        ari_jac = adjusted_rand_score(
            self.labels_kmeans[self.jaccard_sample_idx], self.labels_jaccard
        )
        self.results.append(MetricResult(
            name="Jaccard", algorithm="Hierarchical (sklearn)",
            labels=self.labels_jaccard,
            silhouette=sil_jac, davies_bouldin=db_jac,
            n_patients=actual_jac,
            note=f"Sample {actual_jac:,}  ARI vs KMeans={ari_jac:.4f}",
        ))
        print(f"  Silhouette={sil_jac:.4f}  Davies-Bouldin={db_jac:.4f}  ARI={ari_jac:.4f}")

        #  Tổng kết
        print(f"\n{'='*65}")
        print("SO SANH 3 METRICS:")
        print(f"{'Metric':<20} {'Algorithm':<28} {'Silhouette':>10} {'DB Score':>10}")
        print(f"{'-'*65}")
        for r in self.results:
            print(f"{r.name:<20} {r.algorithm:<28} "
                  f"{r.silhouette:>10.4f} {r.davies_bouldin:>10.4f}")
        best = max(self.results, key=lambda r: r.silhouette)
        print(f"\nMetric tot nhat: {best.name} ({best.algorithm})")
        print(f"{'='*65}\n")

        return self

    def set_cluster_names(self, names: Dict[int, str]) -> None:
        self.cluster_names = names

    def assign_new_patient(
        self,
        vec_scaled: np.ndarray,
        vec_pca:    np.ndarray,
    ) -> AssignResult:
        """
        Gán cluster cho bệnh nhân mới.
        Dùng CODE CHAY: euclidean_distance_scratch + cosine_similarity_scratch.
        """
        # Euclidean distance đến K centroids 
        dists      = euclidean_distance_scratch(vec_pca, self.centroids_pca)[0]  # (K,)
        cluster_id = int(dists.argmin())
        min_d      = dists[cluster_id]
        max_d      = dists.max()
        confidence = float(1.0 - min_d / (max_d + 1e-8))

        # Cosine similarity
        cos_sims = cosine_similarity_scratch(vec_scaled, self.centroids_scaled)[0]  # (K,)

        return AssignResult(
            cluster_id   = cluster_id,
            cluster_name = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
            confidence   = round(confidence, 4),
            distances_euclidean = {
                f"C{i}:{self.cluster_names.get(i,'')}": round(float(d), 4)
                for i, d in enumerate(dists)
            },
            cosine_sim_to_centroids = {
                f"C{i}:{self.cluster_names.get(i,'')}": round(float(s), 4)
                for i, s in enumerate(cos_sims)
            },
        )

    def get_comparison_df(self):
        import pandas as pd
        rows = []
        for r in self.results:
            rows.append({
                "Metric"          : r.name,
                "Algorithm"       : r.algorithm,
                "Silhouette ↑"    : round(r.silhouette, 4),
                "Davies-Bouldin ↓": round(r.davies_bouldin, 4),
                "N Patients"      : r.n_patients,
                "Note"            : r.note,
            })
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        payload = {
            "best_k"             : self.best_k,
            "random_state"       : self.random_state,
            "centroids_pca"      : self.centroids_pca,
            "centroids_scaled"   : self.centroids_scaled,
            "hier_cosine"        : self.hier_cosine,
            "hier_jaccard"       : self.hier_jaccard,
            "labels_kmeans"      : self.labels_kmeans,
            "labels_cosine"      : self.labels_cosine,
            "labels_jaccard"     : self.labels_jaccard,
            "cosine_sample_idx"  : self.cosine_sample_idx,
            "jaccard_sample_idx" : self.jaccard_sample_idx,
            "cluster_names"      : self.cluster_names,
            "patient_ids"        : self.patient_ids,
            "results"            : self.results,
            "created_at"         : datetime.now().isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "PatientClustering":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = PatientClustering(
            best_k=data["best_k"], random_state=data["random_state"]
        )
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj