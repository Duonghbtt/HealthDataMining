"""
plots.py — Tất cả biểu đồ của Module 2.
Tách riêng để notebook gọn hơn.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

COLORS = ["#e74c3c","#3498db","#2ecc71","#f39c12",
          "#9b59b6","#1abc9c","#e67e22","#34495e"]

plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"]  = 10


def plot_eda(feature_df: pd.DataFrame, icd_df: pd.DataFrame,
             output_dir: Path) -> None:
    """Biểu đồ 1: EDA cơ bản."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("EDA — Phân tích bệnh nhân MIMIC-IV", fontsize=13, fontweight="bold")

    # Phân phối tuổi
    axes[0].hist(feature_df["age"], bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(feature_df["age"].mean(), color="red", linestyle="--",
                    label=f"Mean={feature_df['age'].mean():.0f}")
    axes[0].set_title("Phân phối tuổi"); axes[0].set_xlabel("Tuổi")
    axes[0].set_ylabel("Số bệnh nhân"); axes[0].legend()

    # Giới tính
    gc = feature_df["gender_num"].map({1: "Nam (M)", 0: "Nữ (F)"}).value_counts()
    axes[1].bar(gc.index, gc.values, color=["#3498db","#e74c3c"], alpha=0.8)
    axes[1].set_title("Phân phối giới tính"); axes[1].set_ylabel("Số bệnh nhân")
    for i, v in enumerate(gc.values):
        axes[1].text(i, v + 50, f"{v:,}", ha="center", fontsize=9)

    # Top 10 ICD
    icd_freq = icd_df["icd_token"].value_counts().head(10)
    axes[2].barh(range(10), icd_freq.values, color="#2ecc71", alpha=0.8)
    axes[2].set_yticks(range(10))
    axes[2].set_yticklabels(icd_freq.index.tolist(), fontsize=8)
    axes[2].set_title("Top 10 ICD codes"); axes[2].invert_yaxis()

    plt.tight_layout()
    _save(fig, output_dir / "01_eda.png")


def plot_pca_variance(pca, output_dir: Path) -> None:
    """Biểu đồ 2: PCA explained variance."""
    n = len(pca.explained_variance_ratio_)
    cumsum = pca.explained_variance_ratio_.cumsum()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, n+1), pca.explained_variance_ratio_*100,
           color="#3498db", alpha=0.7, label="Mỗi chiều")
    ax2 = ax.twinx()
    ax2.plot(range(1, n+1), cumsum*100, "s--", color="#e74c3c", label="Tích lũy")
    ax2.axhline(80, color="gray", linestyle=":", alpha=0.7)
    ax2.text(n*0.7, 81, "80%", color="gray", fontsize=9)
    ax.set_xlabel("Số chiều PCA"); ax.set_ylabel("Variance (%)", color="#3498db")
    ax2.set_ylabel("Variance tích lũy (%)", color="#e74c3c")
    ax.set_title("PCA Explained Variance")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, loc="center right")
    plt.tight_layout()
    _save(fig, output_dir / "02_pca_variance.png")


def plot_choose_k(inertia_d: dict, sil_d: dict, db_d: dict,
                  best_k: int, output_dir: Path) -> None:
    """Biểu đồ 3: Elbow + Silhouette + Davies-Bouldin."""
    K = sorted(inertia_d.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Chọn K tối ưu", fontsize=13, fontweight="bold")

    axes[0].plot(K, [inertia_d[k] for k in K], "o-", color="#3498db", lw=2)
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    axes[0].set_title("Elbow (Inertia)"); axes[0].set_xlabel("K")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(K, [sil_d[k] for k in K], "o-", color="#2ecc71", lw=2)
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
    axes[1].set_title("Silhouette ↑"); axes[1].set_xlabel("K")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(K, [db_d[k] for k in K], "o-", color="#e74c3c", lw=2)
    axes[2].axvline(best_k, color="blue", linestyle="--", label=f"Best K={best_k}")
    axes[2].set_title("Davies-Bouldin ↓"); axes[2].set_xlabel("K")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir / "03_choose_K.png")


def plot_metric_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Biểu đồ 4: So sánh 3 similarity metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("So sánh 3 Similarity Metrics", fontsize=13, fontweight="bold")

    labels = [f"{r['Metric']}\n({r['Algorithm']})" for _, r in comparison_df.iterrows()]
    sil_vals = comparison_df["Silhouette ↑"].tolist()
    db_vals  = comparison_df["Davies-Bouldin ↓"].tolist()

    bars1 = axes[0].bar(labels, sil_vals, color=COLORS[:len(labels)], alpha=0.85, edgecolor="white")
    axes[0].set_title("Silhouette Score (↑ cao hơn tốt hơn)")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_ylim(0, max(sil_vals)*1.2)
    for bar, v in zip(bars1, sil_vals):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                     f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    bars2 = axes[1].bar(labels, db_vals, color=COLORS[:len(labels)], alpha=0.85, edgecolor="white")
    axes[1].set_title("Davies-Bouldin Score (↓ thấp hơn tốt hơn)")
    axes[1].set_ylabel("DB Score")
    for bar, v in zip(bars2, db_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    _save(fig, output_dir / "04_metric_comparison.png")


def plot_dendrogram(X_pca: np.ndarray, labels_kmeans: np.ndarray,
                    best_k: int, sample: int, random_state: int,
                    output_dir: Path) -> None:
    """Biểu đồ 5: Hierarchical dendrogram (Ward)."""
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X_pca), min(sample, len(X_pca)), replace=False)
    X_s = X_pca[idx]

    Z = linkage(X_s, method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, no_labels=True,
               color_threshold=0.7*max(Z[:,2]),
               above_threshold_color="gray")
    ax.axhline(y=Z[-best_k+1, 2], color="red", linestyle="--",
               label=f"Cắt tại K={best_k}")
    ax.set_title(f"Hierarchical Dendrogram (Ward linkage, sample {len(X_s):,})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Bệnh nhân (sample)"); ax.set_ylabel("Distance")
    ax.legend()
    plt.tight_layout()
    _save(fig, output_dir / "05_dendrogram.png")


def plot_cluster_heatmap(feature_df: pd.DataFrame, icd_cols: List[str],
                         drug_cols: List[str], cluster_names: Dict[int, str],
                         best_k: int, output_dir: Path) -> None:
    """Biểu đồ 6: Cluster profile heatmap."""
    profile = feature_df.groupby("cluster_kmeans")[icd_cols + drug_cols].mean()

    # Lấy top-5 features đặc trưng nhất mỗi cluster
    top_feats = []
    for k in range(best_k):
        if k not in profile.index:
            continue
        top5 = profile.loc[k].sort_values(ascending=False).head(5).index.tolist()
        top_feats.extend(top5)
    top_feats = list(dict.fromkeys(top_feats))

    hm = profile[top_feats].copy()
    hm.index = [f"C{k}: {cluster_names.get(k,'?')[:18]}" for k in hm.index]
    hm.columns = [c.replace("ICD_ICD","ICD:").replace("DRUG_DRUG:","DRUG:")
                   .replace("ICD_","").replace("DRUG_","") for c in hm.columns]

    fig, ax = plt.subplots(figsize=(max(12, len(top_feats)*0.65), best_k*1.3+2))
    sns.heatmap(hm, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Tỉ lệ bệnh nhân"})
    ax.set_title("Cluster Profile Heatmap", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    _save(fig, output_dir / "06_cluster_heatmap.png")


def plot_tsne(X_pca: np.ndarray, labels: np.ndarray, cluster_names: Dict[int, str],
              best_k: int, sample: int, random_state: int,
              title_suffix: str, filename: str, output_dir: Path) -> None:
    """Biểu đồ t-SNE 2D (dùng cho K-means và Cosine)."""
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X_pca), min(sample, len(X_pca)), replace=False)
    X_s  = X_pca[idx]
    lbl_s = labels[idx]

    print(f"  t-SNE {title_suffix} trên {len(X_s):,} bệnh nhân ...")
    X_2d = TSNE(n_components=2, random_state=random_state,
            perplexity=40, max_iter=1000, verbose=0).fit_transform(X_s)
    fig, ax = plt.subplots(figsize=(10, 7))
    for k in range(best_k):
        mask = lbl_s == k
        if mask.sum() == 0:
            continue
        ax.scatter(X_2d[mask,0], X_2d[mask,1],
                   c=COLORS[k], alpha=0.5, s=15, edgecolors="none",
                   label=f"C{k}: {cluster_names.get(k,'?')} ({mask.sum():,})")
    ax.set_title(f"t-SNE — {title_suffix} (K={best_k}, sample {len(X_s):,})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout()
    _save(fig, output_dir / filename)


def plot_distribution(feature_df: pd.DataFrame, cluster_names: Dict[int, str],
                      best_k: int, output_dir: Path) -> None:
    """Biểu đồ 8: Phân phối cluster + Boxplot tuổi."""
    cc = feature_df["cluster_kmeans"].value_counts().sort_index()
    full_names = [cluster_names.get(k, f"C{k}") for k in cc.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phân phối Cluster", fontsize=13, fontweight="bold")

    bars = axes[0].bar([f"C{k}" for k in cc.index], cc.values,
                       color=COLORS[:best_k], alpha=0.85, edgecolor="white")
    for bar, cnt in zip(bars, cc.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                     f"{cnt:,}", ha="center", fontsize=9)
    axes[0].set_title("Số bệnh nhân theo Cluster")
    axes[0].set_ylabel("Số bệnh nhân")

    axes[1].pie(cc.values,
                labels=[f"C{k}:\n{n[:15]}" for k, n in zip(cc.index, full_names)],
                colors=COLORS[:best_k], autopct="%1.1f%%", startangle=140,
                pctdistance=0.8, textprops={"fontsize": 9})
    axes[1].set_title("Tỉ lệ phần trăm")
    plt.tight_layout()
    _save(fig, output_dir / "08_cluster_distribution.png")

    # Boxplot tuổi
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    data = [feature_df[feature_df["cluster_kmeans"]==k]["age"].values
            for k in range(best_k)]
    bp = ax2.boxplot(data, patch_artist=True)
    for patch, color in zip(bp["boxes"], COLORS[:best_k]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax2.set_xticklabels([f"C{k}:\n{cluster_names.get(k,'?')[:15]}"
                         for k in range(best_k)], fontsize=9)
    ax2.set_title("Phân phối tuổi theo Cluster"); ax2.set_ylabel("Tuổi")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig2, output_dir / "09_age_by_cluster.png")


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"  Lưu: {path}")
