import argparse
from pathlib import Path

from module1.settings import Settings
from module1.reader import compute_top_lab_itemids, build_patient_tokens
from module1.model import PatientSimilarity


def cmd_fit(args):
    s = Settings(
        mimic_root=Path(args.mimic_root),
        chunksize=args.chunksize,
        use_diagnoses=True,
        use_prescriptions=True,
        use_labs=args.use_labs,
        top_k_labs=args.top_k_labs,
        max_unique_icd_per_patient=args.max_icd,
        max_unique_drug_per_patient=args.max_drug,
        max_unique_lab_per_patient=args.max_lab,
        num_perm=args.num_perm,
        bands=args.bands,
        rows=args.rows,
        auto_threshold=args.auto_threshold,
        top_k=args.top_k,
        shortlist_factor=args.shortlist_factor,
        limit_patients=args.limit_patients,
        top_n_explain_tokens=args.top_n_tokens,
    )

    top_lab = None
    if s.use_labs:
        print("Đang tính top lab itemid (streaming)...")
        top_lab = compute_top_lab_itemids(s.mimic_root, s.chunksize, s.top_k_labs)
        print("Top lab size:", len(top_lab))

    print("Đang build patient tokens (pandas chunks)...")
    feats = build_patient_tokens(
        mimic_root=s.mimic_root,
        chunksize=s.chunksize,
        use_diagnoses=s.use_diagnoses,
        use_prescriptions=s.use_prescriptions,
        use_labs=s.use_labs,
        top_lab_itemids=top_lab,
        max_icd=s.max_unique_icd_per_patient,
        max_drug=s.max_unique_drug_per_patient,
        max_lab=s.max_unique_lab_per_patient,
        limit_patients=s.limit_patients,
    )
    print("Số bệnh nhân có feature:", len(feats))

    print("Đang fit MinHash + LSH ...")
    if s.auto_threshold is not None:
        model = PatientSimilarity(num_perm=s.num_perm, auto_threshold=s.auto_threshold).fit(feats)
        print(f"Auto chọn (bands,rows) theo threshold={s.auto_threshold}: bands={model.bands}, rows={model.rows}")
    else:
        # manual b,r
        if s.bands * s.rows != s.num_perm:
            raise ValueError("Cần bands*rows == num_perm (vd: 32*4=128).")
        model = PatientSimilarity(num_perm=s.num_perm, bands=s.bands, rows=s.rows).fit(feats)

    out = Path(args.model_path)
    model.save(out)
    print("Đã lưu model:", out)


def cmd_query(args):
    model = PatientSimilarity.load(Path(args.model_path))
    pid = int(args.patient_id)

    res = model.query(pid, top_k=args.top_k, shortlist_factor=args.shortlist_factor)

    print(f"Top-{len(res)} similar cho subject_id={pid} (bands={model.bands}, rows={model.rows}):")
    for i, r in enumerate(res, 1):
        bd = r.breakdown
        print(
            f"{i:02d}. sid={r.subject_id} score={r.score:.4f} "
            f"overlap={r.overlap}/{r.union} | "
            f"ICD {bd['ICD']['pct']*100:.1f}% "
            f"DRUG {bd['DRUG']['pct']*100:.1f}% "
            f"LAB {bd['LAB']['pct']*100:.1f}%"
        )

        if args.show_tokens:
            exp = model.explain_pair(pid, r.subject_id, top_n_each_group=args.top_n_tokens)
            print("    ├─ ICD overlap:", exp["overlap_tokens"]["ICD"])
            print("    ├─ DRUG overlap:", exp["overlap_tokens"]["DRUG"])
            print("    └─ LAB overlap:", exp["overlap_tokens"]["LAB"])


def main():
    parser = argparse.ArgumentParser(description="Module 1 - Patient Similarity (pandas reader + code-chay MinHash/LSH)")
    sub = parser.add_subparsers(dest="mode", required=True)

    # ===== fit =====
    p_fit = sub.add_parser("fit")
    p_fit.add_argument("--mimic_root", type=str, required=True)
    p_fit.add_argument("--model_path", type=str, default="module1_model.pkl")
    p_fit.add_argument("--chunksize", type=int, default=300_000)

    p_fit.add_argument("--use_labs", action="store_true")
    p_fit.add_argument("--top_k_labs", type=int, default=200)

    p_fit.add_argument("--max_icd", type=int, default=200)
    p_fit.add_argument("--max_drug", type=int, default=200)
    p_fit.add_argument("--max_lab", type=int, default=200)

    p_fit.add_argument("--num_perm", type=int, default=128)

    # manual (bands, rows)
    p_fit.add_argument("--bands", type=int, default=32)
    p_fit.add_argument("--rows", type=int, default=4)

    # auto choose (bands, rows) via S-curve threshold
    p_fit.add_argument(
        "--auto_threshold",
        type=float,
        default=None,
        help="Nếu set, tự chọn (bands,rows) sao cho P(threshold)≈0.5 (S-curve).",
    )

    p_fit.add_argument("--limit_patients", type=int, default=None)

    # query params stored in settings but not required for fitting logic
    p_fit.add_argument("--top_k", type=int, default=20)
    p_fit.add_argument("--shortlist_factor", type=int, default=50)

    # explain token cap (chỉ để thống nhất settings)
    p_fit.add_argument("--top_n_tokens", type=int, default=15)

    p_fit.set_defaults(func=cmd_fit)

    # ===== query =====
    p_query = sub.add_parser("query")
    p_query.add_argument("--model_path", type=str, default="module1_model.pkl")
    p_query.add_argument("--patient_id", type=int, required=True)
    p_query.add_argument("--top_k", type=int, default=20)
    p_query.add_argument("--shortlist_factor", type=int, default=50)

    p_query.add_argument("--show_tokens", action="store_true", help="In overlap tokens (ICD/DRUG/LAB) để giải thích.")
    p_query.add_argument("--top_n_tokens", type=int, default=15)
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()