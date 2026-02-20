import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

from scipy.optimize import nnls
from scipy.stats import spearmanr

# pyDecision
import pyDecision as madm
from pyDecision.algorithm import cradis_method
from pyDecision.algorithm import edas_method
from pyDecision.algorithm import macbeth_method
from pyDecision.algorithm import mairca_method
from pyDecision.algorithm import marcos_method
from pyDecision.algorithm import topsis_method


# =========================
# Tunable parameters
# =========================
LAMBDA_RIDGE = 1e-6
EPS = 1e-12
BETA_ZERO_TOL = 1e-10

HARA_ALPHA = 2
HARA_BETA = 1.0
HARA_GAMMA = 1.5


# =========================
# IMPORTANT: score direction per method
# True  => higher score is better
# False => lower score is better (distance/deviation/gap)
# If you find any method still reversed, just flip it here.
# =========================
METHOD_SCORE_DIRECTION: Dict[str, bool] = {
    "GOPA-RUE": True,

    "TOPSIS": True,
    "MARCOS": True,
    "EDAS": True,

    # These two are commonly "lower is better" depending on implementation in libraries
    "MAIRCA": False,
    "CRADIS": False,

    # MACBETH often returns a value function where larger is better
    "MACBETH": True,
}


# =========================
# Reference density v(x) and normalized integral F(x)
# =========================
def v_uniform(x: np.ndarray, R: float) -> np.ndarray:
    return np.ones_like(x) / max(R, EPS)

def F_uniform(x: np.ndarray, R: float) -> np.ndarray:
    return np.clip(x, 0, R) / max(R, EPS)

def v_hara(x: np.ndarray, R: float,
           alpha: float = HARA_ALPHA,
           beta: float = HARA_BETA,
           gamma: float = HARA_GAMMA) -> np.ndarray:
    base = beta + (alpha / gamma) * x
    base = np.maximum(base, EPS)
    return alpha * np.power(base, -gamma)

def F_hara(x: np.ndarray, R: float,
           alpha: float = HARA_ALPHA,
           beta: float = HARA_BETA,
           gamma: float = HARA_GAMMA) -> np.ndarray:
    x = np.clip(x, 0, R)
    y0 = max(beta, EPS)
    yx = beta + (alpha / gamma) * x
    yR = beta + (alpha / gamma) * R
    yx = np.maximum(yx, EPS)
    yR = np.maximum(yR, EPS)

    if abs(gamma - 1.0) > 1e-10:
        def I(y):
            return gamma * (np.power(y, 1 - gamma) / (1 - gamma))
        num = I(yx) - I(y0)
        den = I(yR) - I(y0)
    else:
        num = gamma * (np.log(yx) - np.log(y0))
        den = gamma * (np.log(yR) - np.log(y0))

    den = np.maximum(den, EPS)
    return num / den

def v_logistic(x: np.ndarray, R: float) -> np.ndarray:
    s = (x - R/2.0)
    sigma = 1.0 / (1.0 + np.exp(-s))
    return sigma * (1.0 - sigma)

def F_logistic(x: np.ndarray, R: float) -> np.ndarray:
    x = np.clip(x, 0, R)
    def V(t):
        return 1.0 / (1.0 + np.exp(-(t - R/2.0)))
    V0 = V(0.0)
    VR = V(R)
    denom = max(VR - V0, EPS)
    return (V(x) - V0) / denom

def get_vF(ref_type: int):
    if ref_type == 1:
        return v_uniform, F_uniform
    if ref_type == 2:
        return v_hara, F_hara
    if ref_type == 3:
        return v_logistic, F_logistic
    raise ValueError(f"Unknown reference function type: {ref_type}")


# =========================
# Data structure
# =========================
@dataclass
class ConstraintSpec:
    rb: int
    rb2: Optional[int] = None
    alpha: float = 1.0
    beta: float = 0.0
    kind: str = "TWO_POINT"
    mode: str = ""


def flip_rank(r: int, R: int) -> int:
    if r < 1 or r > R:
        raise ValueError(f"Rank out of range: r={r}, R={R}")
    return R + 1 - r


def parse_constraints_block(block: np.ndarray, R: int) -> List[ConstraintSpec]:
    specs: List[ConstraintSpec] = []
    for row in block:
        judge = row[:R].astype(float)
        degree = float(row[R])

        nz = np.where(np.abs(judge) > EPS)[0]
        if len(nz) == 0:
            continue

        if len(nz) == 1:
            r_raw = int(nz[0] + 1)
            r = flip_rank(r_raw, R)
            specs.append(
                ConstraintSpec(rb=r, rb2=None, alpha=1.0, beta=degree, kind="LOWER_BOUND", mode="LOWER_BOUND")
            )
            continue

        r1_raw = int(nz[0] + 1)
        r2_raw = int(nz[1] + 1)
        r1 = flip_rank(r1_raw, R)
        r2 = flip_rank(r2_raw, R)

        if r1 <= r2:
            rb, rb2 = r1, r2
            coeff_rb2 = judge[nz[1]]
        else:
            rb, rb2 = r2, r1
            coeff_rb2 = judge[nz[0]]

        if abs(degree) <= BETA_ZERO_TOL:
            alpha = abs(coeff_rb2)
            if alpha <= EPS:
                raise ValueError("Detected RATIO constraint (beta≈0) but cannot infer alpha.")
            specs.append(ConstraintSpec(rb=rb, rb2=rb2, alpha=alpha, beta=0.0, kind="TWO_POINT", mode="RATIO"))
        else:
            alpha = abs(coeff_rb2)
            if alpha <= EPS:
                alpha = 1.0
            specs.append(ConstraintSpec(rb=rb, rb2=rb2, alpha=alpha, beta=degree, kind="TWO_POINT", mode="ABS_DIFF"))
    return specs


def build_breakpoints_from_constraints(specs: List[ConstraintSpec], R: int) -> np.ndarray:
    pts = {0, R}
    for s in specs:
        pts.add(int(s.rb))
        if s.rb2 is not None:
            pts.add(int(s.rb2))
    return np.unique(np.array(sorted(pts), dtype=float))

def interval_masses(F, breakpoints: np.ndarray, R: float) -> np.ndarray:
    left = breakpoints[:-1]
    right = breakpoints[1:]
    return (F(right, R) - F(left, R)).astype(float)

def bar_phi_vector(r_point: int, breakpoints: np.ndarray) -> np.ndarray:
    ends = breakpoints[1:]
    return (ends <= r_point + 1e-12).astype(float)

def solve_g_star(A: np.ndarray, beta: np.ndarray, lam: float) -> np.ndarray:
    C = A.shape[1]
    AtA = A.T @ A
    rhs = A.T @ beta
    g = np.linalg.solve(AtA + lam * np.eye(C), rhs)

    if np.all(g >= -1e-10):
        return np.maximum(g, 0.0)

    A_aug = np.vstack([A, np.sqrt(lam) * np.eye(C)])
    b_aug = np.concatenate([beta, np.zeros(C)])
    g_nn, _ = nnls(A_aug, b_aug)
    return g_nn

def induced_utilities_from_g(g: np.ndarray, breakpoints: np.ndarray, F, R: int) -> np.ndarray:
    U = np.zeros(R, dtype=float)
    seg_l = breakpoints[:-1]
    seg_r = breakpoints[1:]
    for r in range(1, R + 1):
        cut = float(np.clip(R - r + 1, 0, R))
        val = 0.0
        for c in range(len(g)):
            l = seg_l[c]
            rr = seg_r[c]
            if cut <= l + 1e-12:
                continue
            upper = min(rr, cut)
            if upper > l + 1e-12:
                val += g[c] * (F(np.array([upper]), R)[0] - F(np.array([l]), R)[0])
        U[r - 1] = val
    return U

def normalize_U_per_ij(U_ijr_raw: np.ndarray) -> np.ndarray:
    U = U_ijr_raw.copy()
    I, J, R = U.shape
    for i in range(I):
        for j in range(J):
            s = float(np.sum(U[i, j, :]))
            if s <= EPS:
                raise ValueError(f"Sum of U_ijr is too small for (i={i+1}, j={j+1}).")
            U[i, j, :] = U[i, j, :] / s
    return U


def read_inputs(xlsx_path: str, n_expert: int, n_attribute: int, n_alternative: int):
    R = n_alternative
    df_expert_rank = pd.read_excel(xlsx_path, sheet_name="Expert Ranking", header=None)
    df_attr_rank = pd.read_excel(xlsx_path, sheet_name="Attribute Ranking", header=None)
    df_alt_rank = pd.read_excel(xlsx_path, sheet_name="Alternative Ranking", header=None)
    df_constraint = pd.read_excel(xlsx_path, sheet_name="Constraint", header=None)
    df_ref = pd.read_excel(xlsx_path, sheet_name="Reference Function", header=None)

    t = df_expert_rank.values.reshape(-1).astype(int)
    if t.size != n_expert:
        raise ValueError(f"Expert Ranking size mismatch, expected {n_expert}, got {t.size}")

    s = df_attr_rank.values.astype(int)
    if s.shape != (n_expert, n_attribute):
        raise ValueError(f"Attribute Ranking shape mismatch, expected {(n_expert, n_attribute)}, got {s.shape}")

    alt_raw = df_alt_rank.values.astype(int)
    expected_rows = n_expert * R
    if alt_raw.shape != (expected_rows, n_attribute):
        raise ValueError(f"Alternative Ranking shape mismatch, expected {(expected_rows, n_attribute)}, got {alt_raw.shape}")

    alt_rank = np.zeros((n_expert, n_attribute, R), dtype=int)
    for i in range(n_expert):
        block = alt_raw[i * R : (i + 1) * R, :]
        for j in range(n_attribute):
            alt_rank[i, j, :] = block[:, j]

    ref_type = df_ref.values.astype(int)
    if ref_type.shape != (n_expert, n_attribute):
        raise ValueError(f"Reference Function shape mismatch, expected {(n_expert, n_attribute)}, got {ref_type.shape}")

    cons_raw = df_constraint.values
    expected_rows = n_expert * n_attribute * R
    if cons_raw.shape != (expected_rows, R + 1):
        raise ValueError(f"Constraint shape mismatch, expected {(expected_rows, R+1)}, got {cons_raw.shape}")

    return t, s, alt_rank, ref_type, cons_raw


def run_gopa_rue_core(
    xlsx_path: str,
    n_expert: int,
    n_attribute: int,
    n_alternative: int,
) -> Dict[str, Any]:
    I, J, R = n_expert, n_attribute, n_alternative
    t, s, alt_rank, ref_type, cons_raw = read_inputs(xlsx_path, I, J, R)

    U_ijr_raw = np.zeros((I, J, R), dtype=float)
    for i in range(I):
        for j in range(J):
            start = (i * J + j) * R
            block = cons_raw[start : start + R, :]
            specs = parse_constraints_block(block, R)

            bps = build_breakpoints_from_constraints(specs, R)
            C = len(bps) - 1
            if C <= 0:
                raise ValueError(f"Breakpoints invalid for expert {i+1}, attr {j+1}")

            _, F_fun = get_vF(int(ref_type[i, j]))
            dV = interval_masses(F_fun, bps, R)
            dV = np.maximum(dV, EPS)

            Psi_rows, beta_rows = [], []
            for cs in specs:
                if cs.kind == "LOWER_BOUND":
                    Psi_rows.append(bar_phi_vector(cs.rb, bps))
                    beta_rows.append(cs.beta)
                else:
                    phi_rb = bar_phi_vector(cs.rb, bps)
                    phi_rb2 = bar_phi_vector(cs.rb2, bps)
                    Psi_rows.append(phi_rb - cs.alpha * phi_rb2)
                    beta_rows.append(cs.beta)

            Psi_rows.append(np.ones(C))
            beta_rows.append(1.0)

            Psi = np.vstack(Psi_rows)
            beta_vec = np.array(beta_rows, float)
            A = Psi * dV.reshape(1, -1)

            g_star = solve_g_star(A, beta_vec, LAMBDA_RIDGE)
            U_ijr_raw[i, j, :] = induced_utilities_from_g(g_star, bps, F_fun, R)

    U_ijr = normalize_U_per_ij(U_ijr_raw)

    denom = 0.0
    for i in range(I):
        for j in range(J):
            denom += np.sum((R * U_ijr[i, j, :]) / (max(t[i], EPS) * max(s[i, j], EPS)))
    z_star = 1.0 / max(denom, EPS)

    bar_w = np.zeros_like(U_ijr)
    for i in range(I):
        for j in range(J):
            bar_w[i, j, :] = (R * U_ijr[i, j, :] * z_star) / (max(t[i], EPS) * max(s[i, j], EPS))
            bar_w[i, j, :] = np.maximum(bar_w[i, j, :], 0.0)

    w_ijk = np.zeros((I, J, R), dtype=float)
    for i in range(I):
        for j in range(J):
            for k in range(R):
                r = int(alt_rank[i, j, k])
                w_ijk[i, j, k] = bar_w[i, j, r - 1]

    W_J = w_ijk.sum(axis=(0, 2))
    W_K = w_ijk.sum(axis=(0, 1))
    W_J = W_J / max(float(np.sum(W_J)), EPS)
    W_K = W_K / max(float(np.sum(W_K)), EPS)

    order = np.argsort(-W_K, kind="mergesort")
    rank_K = np.empty(R, dtype=int)
    rank_K[order] = np.arange(1, R + 1)

    return {"w_ijk": w_ijk, "W_J": W_J, "W_K": W_K, "rank_K": rank_K, "z_star": z_star}


def build_dataset_from_gopa_rue(w_ijk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    I, J, K = w_ijk.shape
    w_jk = np.sum(w_ijk, axis=0)            # (J,K)
    w_j = np.sum(w_jk, axis=1)              # (J,)
    w_j = np.maximum(w_j, EPS)
    v_jk = w_jk / w_j[:, None]              # (J,K)
    dataset = v_jk.T                        # (K,J)
    return dataset, v_jk


# =========================
# Wrapper: parse outputs; if need derive rank from scores, use method direction
# =========================
def _to_1d_array(x) -> Optional[np.ndarray]:
    try:
        return np.array(x, dtype=float).reshape(-1)
    except Exception:
        return None

def _to_1d_int(x) -> Optional[np.ndarray]:
    try:
        return np.array(x).reshape(-1).astype(int)
    except Exception:
        return None

def rank_from_scores(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    scores = np.array(scores, dtype=float).reshape(-1)
    order = np.argsort(-scores if higher_is_better else scores, kind="mergesort")
    rk = np.empty_like(order, dtype=int)
    rk[order] = np.arange(1, scores.size + 1)
    return rk

def run_pydecision_method(
    name: str,
    method_func,
    dataset: np.ndarray,
    weights: List[float],
    criterion_type: List[str],
) -> Dict[str, Any]:
    try:
        out = method_func(dataset, weights, criterion_type, graph=False, verbose=False)
    except TypeError:
        try:
            out = method_func(dataset=dataset, weights=weights, criterion_type=criterion_type, graph=False)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "raw": None}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "raw": None}

    K = dataset.shape[0]
    scores = None
    rank = None

    if isinstance(out, (tuple, list)):
        if len(out) == 1:
            a0 = out[0]
            r0 = _to_1d_int(a0)
            s0 = _to_1d_array(a0)
            if r0 is not None and r0.size == K and np.all((r0 >= 1) & (r0 <= K)):
                rank = r0
            else:
                scores = s0
        else:
            a0, a1 = out[0], out[1]
            r0, r1 = _to_1d_int(a0), _to_1d_int(a1)
            s0, s1 = _to_1d_array(a0), _to_1d_array(a1)

            def looks_like_rank(r):
                return (r is not None) and (r.size == K) and np.all((r >= 1) & (r <= K))

            if looks_like_rank(r0) and (s1 is not None and s1.size == K):
                rank, scores = r0, s1
            elif looks_like_rank(r1) and (s0 is not None and s0.size == K):
                rank, scores = r1, s0
            elif looks_like_rank(r0):
                rank, scores = r0, (s1 if (s1 is not None and s1.size == K) else None)
            elif looks_like_rank(r1):
                rank, scores = r1, (s0 if (s0 is not None and s0.size == K) else None)
            else:
                scores = s0 if (s0 is not None and s0.size == K) else None
    else:
        r = _to_1d_int(out)
        if r is not None and r.size == K and np.all((r >= 1) & (r <= K)):
            rank = r
        else:
            scores = _to_1d_array(out)

    # derive rank from scores if needed
    if rank is None and scores is not None and scores.size == K:
        higher_is_better = METHOD_SCORE_DIRECTION.get(name, True)
        rank = rank_from_scores(scores, higher_is_better=higher_is_better)

    if scores is None:
        scores = np.full(K, np.nan, dtype=float)

    if rank is None:
        return {"ok": False, "error": f"Could not parse output format for method {name}.", "raw": out}

    return {"ok": True, "scores": scores, "rank": rank, "raw": out}


def spearman_matrix(rank_vectors: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(rank_vectors.keys())
    S = len(names)
    M = np.zeros((S, S), dtype=float)
    for i in range(S):
        for j in range(S):
            a = rank_vectors[names[i]].astype(float)
            b = rank_vectors[names[j]].astype(float)
            corr, _ = spearmanr(a, b)
            if np.isnan(corr):
                corr = 0.0
            M[i, j] = float(corr)
    return pd.DataFrame(M, index=names, columns=names)


def run_method_comparison(
    xlsx_path: str,
    n_expert: int,
    n_attribute: int,
    n_alternative: int,
    criterion_type: Optional[List[str]] = None,
    out_path: str = "method_comparison.xlsx",
) -> Dict[str, Any]:
    I, J, K = n_expert, n_attribute, n_alternative

    gopa = run_gopa_rue_core(xlsx_path, I, J, K)

    w_ijk = gopa["w_ijk"]
    W_J = gopa["W_J"]
    W_K = gopa["W_K"]
    rank_gopa = gopa["rank_K"]

    dataset, v_jk = build_dataset_from_gopa_rue(w_ijk)
    weights = (W_J / max(float(np.sum(W_J)), EPS)).tolist()

    if criterion_type is None:
        criterion_type = ["max"] * J
    if len(criterion_type) != J:
        raise ValueError(f"criterion_type length mismatch: expected {J}, got {len(criterion_type)}")

    methods = {
        "GOPA-RUE": None,
        "CRADIS": cradis_method,
        "EDAS": edas_method,
        "MACBETH": macbeth_method,
        "MAIRCA": mairca_method,
        "MARCOS": marcos_method,
        "TOPSIS": topsis_method,
    }

    results: Dict[str, Dict[str, Any]] = {}
    results["GOPA-RUE"] = {"ok": True, "scores": W_K.copy(), "rank": rank_gopa.copy(), "raw": None}

    for name, func in methods.items():
        if name == "GOPA-RUE":
            continue
        r = run_pydecision_method(name, func, dataset, weights, criterion_type)
        results[name] = r
        if not r["ok"]:
            print(f"[WARN] {name} failed: {r.get('error')}")

    rows = []
    for mname, r in results.items():
        if not r["ok"]:
            continue
        for k in range(K):
            rows.append({
                "method": mname,
                "alt_k": k + 1,
                "score_or_weight": float(r["scores"][k]) if np.isfinite(r["scores"][k]) else np.nan,
                "rank": int(r["rank"][k]),
            })
    df_long = pd.DataFrame(rows).sort_values(["method", "rank", "alt_k"]).reset_index(drop=True)

    wide_rank = pd.DataFrame({"alt_k": np.arange(1, K + 1, dtype=int)})
    wide_score = pd.DataFrame({"alt_k": np.arange(1, K + 1, dtype=int)})

    rank_vectors = {}
    for mname, r in results.items():
        if not r["ok"]:
            continue
        wide_rank[mname] = r["rank"]
        wide_score[mname] = r["scores"]
        rank_vectors[mname] = r["rank"].copy()

    df_spear = spearman_matrix(rank_vectors)

    df_weights = pd.DataFrame({"criterion_j": np.arange(1, J + 1, dtype=int), "weight_WJ": W_J})
    df_dataset = pd.DataFrame(dataset, columns=[f"c{j}" for j in range(1, J + 1)])
    df_dataset.insert(0, "alt_k", np.arange(1, K + 1, dtype=int))

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_long.to_excel(writer, sheet_name="Compare_Long", index=False)
        wide_rank.to_excel(writer, sheet_name="Ranks_Wide", index=False)
        wide_score.to_excel(writer, sheet_name="Scores_Wide", index=False)
        df_spear.to_excel(writer, sheet_name="Spearman_Rank", index=True)
        df_weights.to_excel(writer, sheet_name="GOPA_WJ_Weights", index=False)
        df_dataset.to_excel(writer, sheet_name="Dataset_vjk", index=False)

        df_vjk = pd.DataFrame(v_jk, index=[f"j={j}" for j in range(1, J + 1)],
                              columns=[f"k={k}" for k in range(1, K + 1)])
        df_vjk.to_excel(writer, sheet_name="v_jk", index=True)

    return {
        "dataset": dataset,
        "weights": weights,
        "results": results,
        "spearman_rank": df_spear,
        "output_path": out_path,
    }


if __name__ == "__main__":
    n_expert = 5
    n_attribute = 6
    n_alternative = 10

    res = run_method_comparison(
        xlsx_path="test.xlsx",
        n_expert=n_expert,
        n_attribute=n_attribute,
        n_alternative=n_alternative,
        criterion_type=None,
        out_path="method_comparison.xlsx",
    )
    print("Done. Saved to:", res["output_path"])
