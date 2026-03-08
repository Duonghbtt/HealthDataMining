from __future__ import annotations
import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from module1.model import PatientSimilarity


def exact_jaccard(a: Set[str], b: Set[str]) -> float:
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0


def brute_force_topk(
    query_pid: int,
    features: Dict[int, Set[str]],
    top_k: int
) -> List[Tuple[int, float]]:
    q = features[query_pid]
    scores: List[Tuple[int, float]] = []

    for pid, feats in features.items():
        if pid == query_pid:
            continue
        score = exact_jaccard(q, feats)
        if score > 0:
            scores.append((pid, score))

    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_k]


def lsh_topk(
    model: PatientSimilarity,
    query_pid: int,
    top_k: int,
    shortlist_factor: int
) -> List[Tuple[int, float]]:
    res = model.query(query_pid, top_k=top_k, shortlist_factor=shortlist_factor)
    return [(r.subject_id, r.score) for r in res]


def precision_at_k(pred_ids: List[int], true_ids: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_set = set(pred_ids[:k])
    true_set = set(true_ids[:k])
    return len(pred_set & true_set) / k


def evaluate(
    model: PatientSimilarity,
    sample_size: int,
    top_k: int,
    shortlist_factor: int,
    seed: int
) -> dict:
    features = model.features
    patient_ids = list(features.keys())

    if len(patient_ids) < 2:
        raise ValueError("Không đủ bệnh nhân để đánh giá.")

    rng = random.Random(seed)

    valid_ids = [pid for pid in patient_ids if features.get(pid)]
    if len(valid_ids) == 0:
        raise ValueError("Không có bệnh nhân nào có feature.")

    sample_size = min(sample_size, len(valid_ids))
    eval_ids = rng.sample(valid_ids, sample_size)

    precisions: List[float] = []
    lsh_times: List[float] = []
    brute_times: List[float] = []

    empty_lsh = 0
    empty_brute = 0

    for pid in eval_ids:
        t0 = time.perf_counter()
        pred = lsh_topk(model, pid, top_k=top_k, shortlist_factor=shortlist_factor)
        t1 = time.perf_counter()
        lsh_times.append(t1 - t0)

        t2 = time.perf_counter()
        gt = brute_force_topk(pid, features, top_k=top_k)
        t3 = time.perf_counter()
        brute_times.append(t3 - t2)

        pred_ids = [x[0] for x in pred]
        true_ids = [x[0] for x in gt]

        if len(pred_ids) == 0:
            empty_lsh += 1
        if len(true_ids) == 0:
            empty_brute += 1

        precisions.append(precision_at_k(pred_ids, true_ids, top_k))

    avg_precision = statistics.mean(precisions) if precisions else 0.0
    avg_lsh_time = statistics.mean(lsh_times) if lsh_times else 0.0
    avg_brute_time = statistics.mean(brute_times) if brute_times else 0.0
    median_lsh_time = statistics.median(lsh_times) if lsh_times else 0.0
    median_brute_time = statistics.median(brute_times) if brute_times else 0.0
    speedup = (avg_brute_time / avg_lsh_time) if avg_lsh_time > 0 else float("inf")

    return {
        "num_patients_total": len(features),
        "num_eval_queries": len(eval_ids),
        "top_k": top_k,
        "avg_precision_at_k": avg_precision,
        "avg_lsh_time_sec": avg_lsh_time,
        "avg_bruteforce_time_sec": avg_brute_time,
        "median_lsh_time_sec": median_lsh_time,
        "median_bruteforce_time_sec": median_brute_time,
        "speedup": speedup,
        "empty_lsh_queries": empty_lsh,
        "empty_bruteforce_queries": empty_brute,
        "bands": model.bands,
        "rows": model.rows,
        "num_perm": model.num_perm,
        "shortlist_factor": shortlist_factor,
        "seed": seed,
    }


def build_report_text(metrics: dict) -> str:
    lines = []
    lines.append("========== EVALUATION REPORT ==========")
    lines.append(f"Tổng số bệnh nhân trong model   : {metrics['num_patients_total']}")
    lines.append(f"Số query dùng để đánh giá       : {metrics['num_eval_queries']}")
    lines.append(f"Top-K                           : {metrics['top_k']}")
    lines.append(f"num_perm                        : {metrics['num_perm']}")
    lines.append(f"bands                           : {metrics['bands']}")
    lines.append(f"rows                            : {metrics['rows']}")
    lines.append(f"shortlist_factor                : {metrics['shortlist_factor']}")
    lines.append(f"seed                            : {metrics['seed']}")
    lines.append(f"Precision@{metrics['top_k']} trung bình     : {metrics['avg_precision_at_k']:.4f}")
    lines.append(f"LSH query time trung bình       : {metrics['avg_lsh_time_sec']:.6f} giây")
    lines.append(f"Brute-force time trung bình     : {metrics['avg_bruteforce_time_sec']:.6f} giây")
    lines.append(f"LSH query time median           : {metrics['median_lsh_time_sec']:.6f} giây")
    lines.append(f"Brute-force time median         : {metrics['median_bruteforce_time_sec']:.6f} giây")
    lines.append(f"Speedup trung bình              : {metrics['speedup']:.2f}x")
    lines.append(f"Số query LSH trả về rỗng        : {metrics['empty_lsh_queries']}")
    lines.append(f"Số query brute-force trả về rỗng: {metrics['empty_bruteforce_queries']}")
    lines.append("=======================================")
    lines.append("")
    lines.append("Điền vào bảng báo cáo:")
    lines.append(f"- Precision@10 (so với brute-force): {metrics['avg_precision_at_k'] * 100:.2f}%")
    lines.append(f"- Speedup so với brute-force      : {metrics['speedup']:.2f}x")
    lines.append(f"- Query time                      : {metrics['avg_lsh_time_sec']:.6f} giây")
    return "\n".join(lines)


def save_text_report(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def save_json_report(output_path: Path, metrics: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Module 1: Precision@K, speedup, query time")
    parser.add_argument("--model_path", type=str, required=True, help="Đường dẫn file .pkl model đã fit")
    parser.add_argument("--sample_size", type=int, default=100, help="Số bệnh nhân query để đánh giá")
    parser.add_argument("--top_k", type=int, default=10, help="K cho Precision@K")
    parser.add_argument("--shortlist_factor", type=int, default=50, help="Shortlist factor cho LSH query")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # file output
    parser.add_argument("--output_txt", type=str, default="artifacts/module1_eval_report.txt", help="File txt lưu report")
    parser.add_argument("--output_json", type=str, default="artifacts/module1_eval_report.json", help="File json lưu metrics")

    args = parser.parse_args()

    model = PatientSimilarity.load(Path(args.model_path))
    metrics = evaluate(
        model=model,
        sample_size=args.sample_size,
        top_k=args.top_k,
        shortlist_factor=args.shortlist_factor,
        seed=args.seed,
    )

    report_text = build_report_text(metrics)
    print(report_text)

    save_text_report(Path(args.output_txt), report_text)
    save_json_report(Path(args.output_json), metrics)

    print(f"\nĐã lưu report txt : {args.output_txt}")
    print(f"Đã lưu report json: {args.output_json}")


if __name__ == "__main__":
    main()