from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import hashlib
import pickle
import random


# =========================
# Stable hashing utilities
# =========================
def stable_u64(s: str) -> int:
    """
    Hash ổn định 64-bit (không phụ thuộc PYTHONHASHSEED).
    Dùng blake2b digest_size=8.
    """
    h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=8)
    return int.from_bytes(h.digest(), "little", signed=False)


# =========================
# MinHash (code chay)
# =========================
P61 = (1 << 61) - 1  # prime kiểu Mersenne

class MinHasher:
    """
    Universal hashing trên base(token):
      base = stable_u64(token) % P
      h_i(token) = (a_i * base + b_i) % P
    signature[i] = min h_i(token) với mọi token.
    """
    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = int(num_perm)
        rng = random.Random(seed)
        self.a = [rng.randrange(1, P61) for _ in range(self.num_perm)]
        self.b = [rng.randrange(0, P61) for _ in range(self.num_perm)]

    def signature(self, tokens: Set[str]) -> Tuple[int, ...]:
        sig = [P61] * self.num_perm
        for t in tokens:
            base = stable_u64(t) % P61
            for i in range(self.num_perm):
                hv = (self.a[i] * base + self.b[i]) % P61
                if hv < sig[i]:
                    sig[i] = hv
        return tuple(sig)


def approx_jaccard(sigA: Tuple[int, ...], sigB: Tuple[int, ...]) -> float:
    """Ước lượng Jaccard bằng tỉ lệ phần tử signature trùng nhau."""
    eq = 0
    n = len(sigA)
    for a, b in zip(sigA, sigB):
        if a == b:
            eq += 1
    return eq / n if n else 0.0


# =========================
# LSH banding (code chay)
# =========================
class LSHIndex:
    def __init__(self, bands: int, rows: int):
        self.bands = int(bands)
        self.rows = int(rows)
        # key=(band_id, hash(band_slice)) -> list[pid]
        self.buckets: Dict[Tuple[int, int], List[int]] = {}

    @staticmethod
    def _band_hash(band: Tuple[int, ...]) -> int:
        h = hashlib.blake2b(digest_size=8)
        for x in band:
            h.update(x.to_bytes(8, "little", signed=False))
        return int.from_bytes(h.digest(), "little", signed=False)

    def insert(self, pid: int, sig: Tuple[int, ...]) -> None:
        for band_id in range(self.bands):
            start = band_id * self.rows
            band = sig[start:start + self.rows]
            key = (band_id, self._band_hash(band))
            self.buckets.setdefault(key, []).append(pid)

    def query(self, sig: Tuple[int, ...]) -> List[int]:
        cands = set()
        for band_id in range(self.bands):
            start = band_id * self.rows
            band = sig[start:start + self.rows]
            key = (band_id, self._band_hash(band))
            for pid in self.buckets.get(key, []):
                cands.add(pid)
        return list(cands)


# =========================
# S-curve: auto choose (b,r)
# =========================
def lsh_candidate_prob(s: float, b: int, r: int) -> float:
    # P(s) = 1 - (1 - s^r)^b
    return 1.0 - (1.0 - (s ** r)) ** b

def choose_br(num_perm: int, target_threshold: float, target_prob: float = 0.5) -> tuple[int, int, float]:
    """
    Tìm (b,r) sao cho b*r == num_perm và P(threshold) gần target_prob nhất.
    Trả: (b, r, err)
    """
    best = None
    for r in range(1, num_perm + 1):
        if num_perm % r != 0:
            continue
        b = num_perm // r
        p = lsh_candidate_prob(target_threshold, b, r)
        err = abs(p - target_prob)
        if best is None or err < best[2]:
            best = (b, r, err)
    assert best is not None
    return best


# =========================
# Explain helpers
# =========================
def group_of_token(tok: str) -> str:
    if tok.startswith("ICD"):
        return "ICD"
    if tok.startswith("DRUG") or tok.startswith("DRUGCD"):
        return "DRUG"
    if tok.startswith("LAB:"):
        return "LAB"
    return "OTHER"


@dataclass
class SimilarResult:
    subject_id: int
    score: float
    overlap: int
    union: int
    breakdown: dict  # {"ICD":{"overlap":..,"pct":..}, ...}


class PatientSimilarity:
    """
    Pipeline:
      - fit(): build MinHash signatures + LSH buckets
      - query(): LSH candidates -> shortlist (approx) -> exact rerank + breakdown
      - explain_pair(): list overlap tokens theo group
      - save/load: demo nhanh
    """
    def __init__(
        self,
        *,
        num_perm: int = 128,
        bands: Optional[int] = None,
        rows: Optional[int] = None,
        auto_threshold: Optional[float] = None,
        seed: int = 42
    ):
        self.num_perm = int(num_perm)

        if auto_threshold is not None:
            b, r, _ = choose_br(self.num_perm, float(auto_threshold), target_prob=0.5)
            self.bands, self.rows = b, r
        else:
            if bands is None or rows is None:
                raise ValueError("Cần (bands, rows) hoặc bật auto_threshold.")
            self.bands, self.rows = int(bands), int(rows)

        if self.bands * self.rows != self.num_perm:
            raise ValueError("Cần bands*rows == num_perm (vd: 32*4=128).")

        self.hasher = MinHasher(num_perm=self.num_perm, seed=seed)
        self.lsh = LSHIndex(bands=self.bands, rows=self.rows)

        self.features: Dict[int, Set[str]] = {}
        self.signatures: Dict[int, Tuple[int, ...]] = {}

    def fit(self, patient_tokens: Dict[int, Set[str]]) -> "PatientSimilarity":
        self.features = patient_tokens

        for pid, toks in patient_tokens.items():
            if not toks:
                continue
            sig = self.hasher.signature(toks)
            self.signatures[pid] = sig
            self.lsh.insert(pid, sig)

        return self

    def _breakdown(self, A: Set[str], B: Set[str]) -> dict:
        inter = A & B
        cnt = {"ICD": 0, "DRUG": 0, "LAB": 0}

        for t in inter:
            g = group_of_token(t)
            if g in cnt:
                cnt[g] += 1

        total = len(inter)
        return {g: {"overlap": cnt[g], "pct": (cnt[g] / total if total else 0.0)} for g in cnt}

    def explain_pair(self, pid: int, cid: int, top_n_each_group: int = 15) -> dict:
        """
        Trả về token giao nhau để giải thích vì sao giống:
        - ICD overlap tokens
        - DRUG overlap tokens
        - LAB overlap tokens
        """
        pid = int(pid)
        cid = int(cid)
        A = self.features.get(pid, set())
        B = self.features.get(cid, set())

        inter = sorted(A & B)
        by_group = {"ICD": [], "DRUG": [], "LAB": [], "OTHER": []}

        for t in inter:
            by_group[group_of_token(t)].append(t)

        for g in by_group:
            by_group[g] = by_group[g][:top_n_each_group]

        return {
            "pid": pid,
            "cid": cid,
            "overlap_total": len(A & B),
            "union_total": len(A | B),
            "overlap_tokens": by_group,
        }

    def query(self, patient_id: int, top_k: int = 20, shortlist_factor: int = 50) -> List[SimilarResult]:
        pid = int(patient_id)
        if pid not in self.features:
            raise KeyError(f"Không có subject_id={pid} trong features (có thể do limit_patients).")

        q_tokens = self.features[pid]
        q_sig = self.signatures[pid]

        # 1) LSH candidates
        cands = self.lsh.query(q_sig)
        cands = [x for x in cands if x != pid]
        if not cands:
            return []

        # 2) shortlist bằng approx_jaccard(signature)
        approx: List[Tuple[int, float]] = []
        for cid in cands:
            sig = self.signatures.get(cid)
            if sig is None:
                continue
            approx.append((cid, approx_jaccard(q_sig, sig)))

        approx.sort(key=lambda x: x[1], reverse=True)
        shortlist_n = min(len(approx), top_k * shortlist_factor)
        shortlist = [cid for cid, _ in approx[:shortlist_n]]

        # 3) rerank exact Jaccard + breakdown
        out: List[SimilarResult] = []
        for cid in shortlist:
            t = self.features.get(cid)
            if not t:
                continue
            inter = len(q_tokens & t)
            uni = len(q_tokens | t)
            score = inter / uni if uni else 0.0
            if score > 0:
                out.append(SimilarResult(
                    subject_id=cid,
                    score=float(score),
                    overlap=inter,
                    union=uni,
                    breakdown=self._breakdown(q_tokens, t)
                ))

        out.sort(key=lambda r: r.score, reverse=True)
        return out[:top_k]

    def save(self, path: Path) -> None:
        payload = {
            "num_perm": self.num_perm,
            "bands": self.bands,
            "rows": self.rows,
            "features": self.features,
            "signatures": self.signatures,
            "buckets": self.lsh.buckets,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "PatientSimilarity":
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = PatientSimilarity(num_perm=data["num_perm"], bands=data["bands"], rows=data["rows"])
        obj.features = data["features"]
        obj.signatures = data["signatures"]
        obj.lsh.buckets = data["buckets"]
        return obj