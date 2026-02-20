import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from scipy.optimize import nnls
from scipy.stats import spearmanr


# =========================
# Tunable parameters
# =========================
LAMBDA_RIDGE = 1e-6
EPS = 1e-12
BETA_ZERO_TOL = 1e-10

# HARA parameters (adjust if needed)
HARA_ALPHA = 2
HARA_BETA = 1.0
HARA_GAMMA = 1.5


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
    """
    Normalized integral:
        F(x) = ∫_0^x v(t) dt / ∫_0^R v(t) dt
    """
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
    """
    Normalizes logistic CDF V(x) to [0,1]:
        F(x) = (V(x)-V(0)) / (V(R)-V(0))
    """
    x = np.clip(x, 0, R)

    def V(t):
        return 1.0 / (1.0 + np.exp(-(t - R/2.0)))

    V0 = V(0.0)
    VR = V(R)
    denom = max(VR - V0, EPS)
    return (V(x) - V0) / denom

def get_vF(ref_type: int):
    # 1=Uniform, 2=HARA, 3=Logistic(S-shape)
    if ref_type == 1:
        return v_uniform, F_uniform
    if ref_type == 2:
        return v_hara, F_hara
    if ref_type == 3:
        return v_logistic, F_logistic
    raise ValueError(f"Unknown reference function type: {ref_type}")


# =========================
# Constraint structure
# =========================
@dataclass
class ConstraintSpec:
    rb: int
    rb2: Optional[int] = None
    alpha: float = 1.0
    beta: float = 0.0
    kind: str = "TWO_POINT"  # TWO_POINT / LOWER_BOUND
    mode: str = ""          # ABS_DIFF / RATIO / LOWER_BOUND


def flip_rank(r: int, R: int) -> int:
    """Rank flipping: r -> R+1-r"""
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
            specs.append(ConstraintSpec(rb=r, beta=degree, kind="LOWER_BOUND", mode="LOWER_BOUND"))
            continue

        r1_raw = int(nz[0] + 1)
        r2_raw = int(nz[1] + 1)
        r1 = flip_rank(r1_raw, R)
        r2 = flip_rank(r2_raw, R)

        # enforce rb < rb2
        if r1 <= r2:
            rb, rb2 = r1, r2
            coeff_rb2 = judge[nz[1]]
        else:
            rb, rb2 = r2, r1
            coeff_rb2 = judge[nz[0]]

        if abs(degree) <= BETA_ZERO_TOL:
            alpha = abs(coeff_rb2)
            if alpha <= EPS:
                raise ValueError("RATIO constraint detected but alpha cannot be inferred.")
            specs.append(ConstraintSpec(rb=rb, rb2=rb2, alpha=alpha, beta=0.0,
                                        kind="TWO_POINT", mode="RATIO"))
        else:
            alpha = abs(coeff_rb2)
            if alpha <= EPS:
                alpha = 1.0
            specs.append(ConstraintSpec(rb=rb, rb2=rb2, alpha=alpha, beta=degree,
                                        kind="TWO_POINT", mode="ABS_DIFF"))

    return specs


# =========================
# Lower layer utilities: solve g* and compute U
# =========================
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
    """
    U_r = ∫_0^{R-r+1} u(x) dx, u(x)=g_c*v(x) on each interval.
    """
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
                raise ValueError(f"Sum of U_ijr is zero for (i={i+1}, j={j+1}).")
            U[i, j, :] = U[i, j, :] / s
    return U

def U_from_reference_only(ref_type: int, R: int) -> np.ndarray:
    """
    No constraints: u(x) ∝ v(x) => U_r ∝ F(R-r+1)
    """
    _, F_fun = get_vF(ref_type)
    U_raw = np.zeros(R, dtype=float)
    for r in range(1, R + 1):
        cut = float(np.clip(R - r + 1, 0, R))
        U_raw[r - 1] = float(F_fun(np.array([cut]), R)[0])
    return U_raw


# =========================
# Read Excel inputs (same as you之前格式)
# =========================
def read_inputs(xlsx_path: str, n_expert: int, n_attribute: int, n_alternative: int):
    R = n_alternative
    df_expert_rank = pd.read_excel(xlsx_path, sheet_name="Expert Ranking", header=None)
    df_attr_rank = pd.read_excel(xlsx_path, sheet_name="Attribute Ranking", header=None)
    df_alt_rank = pd.read_excel(xlsx_path, sheet_name="Alternative Ranking", header=None)
    df_constraint = pd.read_excel(xlsx_path, sheet_name="Constraint", header=None)

    # expert ranks t_i
    t = df_expert_rank.values.reshape(-1).astype(int)
    if t.size != n_expert:
        raise ValueError(f"Expert Ranking size mismatch, expected {n_expert}, got {t.size}")

    # attribute ranks s_ij
    s = df_attr_rank.values.astype(int)
    if s.shape != (n_expert, n_attribute):
        raise ValueError(f"Attribute Ranking shape mismatch, expected {(n_expert, n_attribute)}, got {s.shape}")

    # alt ranking blocks: (n_expert*R) x n_attribute
    alt_raw = df_alt_rank.values.astype(int)
    expected_rows = n_expert * R
    if alt_raw.shape != (expected_rows, n_attribute):
        raise ValueError(f"Alternative Ranking shape mismatch, expected {(expected_rows, n_attribute)}, got {alt_raw.shape}")

    alt_rank = np.zeros((n_expert, n_attribute, R), dtype=int)
    for i in range(n_expert):
        block = alt_raw[i * R : (i + 1) * R, :]
        for j in range(n_attribute):
            alt_rank[i, j, :] = block[:, j]

    # constraints: (n_expert*n_attribute*R) x (R+1)
    cons_raw = df_constraint.values
    expected_rows = n_expert * n_attribute * R
    if cons_raw.shape != (expected_rows, R + 1):
        raise ValueError(f"Constraint shape mismatch, expected {(expected_rows, R+1)}, got {cons_raw.shape}")

    return t, s, alt_rank, cons_raw


# =========================
# Scenario: compute U_ijr (raw & normalized)
# =========================
def compute_U_ijr_for_scenario(
    n_expert: int,
    n_attribute: int,
    n_alternative: int,
    cons_raw: np.ndarray,
    ref_type_scalar: int,   # 1/2/3
    use_constraints: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    I, J, R = n_expert, n_attribute, n_alternative
    U_ijr_raw = np.zeros((I, J, R), dtype=float)
    _, F_fun = get_vF(ref_type_scalar)

    for i in range(I):
        for j in range(J):
            if not use_constraints:
                U_ijr_raw[i, j, :] = U_from_reference_only(ref_type_scalar, R)
                continue

            start = (i * J + j) * R
            block = cons_raw[start : start + R, :]
            specs = parse_constraints_block(block, R)

            bps = build_breakpoints_from_constraints(specs, R)
            C = len(bps) - 1
            if C <= 0:
                raise ValueError(f"Breakpoints invalid for expert {i+1}, attr {j+1}")

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

            # density normalization: e^T ΔV g = 1
            Psi_rows.append(np.ones(C))
            beta_rows.append(1.0)

            Psi = np.vstack(Psi_rows)
            beta_vec = np.array(beta_rows, float)
            A = Psi * dV.reshape(1, -1)

            g_star = solve_g_star(A, beta_vec, LAMBDA_RIDGE)
            U_ijr_raw[i, j, :] = induced_utilities_from_g(g_star, bps, F_fun, R)

    U_ijr = normalize_U_per_ij(U_ijr_raw)
    return U_ijr_raw, U_ijr


# =========================
# GOPA-RUE aggregated (original upper layer style)
# =========================
def gopa_rue_aggregate_WK(
    U_ijr_norm: np.ndarray,   # (I,J,R), normalized per (i,j)
    t: np.ndarray,            # (I,)
    s_attr_rank: np.ndarray,  # (I,J)
    alt_rank: np.ndarray,     # (I,J,R), rank of alternative k under attribute j for expert i
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate across experts:
      z* = 1 / Σ_{i,j} Σ_r ( R*U_{ijr} / (t_i*s_ij) )
      bar_w_{ijr} = ( R*U_{ijr}*z* ) / (t_i*s_ij)
      w_{ijk} = bar_w_{ij, r=alt_rank(i,j,k)}
      W_K = Σ_{i,j} w_{ijk}
      normalize W_K and rank
    """
    I, J, R = U_ijr_norm.shape

    denom = 0.0
    for i in range(I):
        for j in range(J):
            denom += np.sum((R * U_ijr_norm[i, j, :]) / (max(t[i], EPS) * max(s_attr_rank[i, j], EPS)))
    z_star = 1.0 / max(denom, EPS)

    bar_w = np.zeros((I, J, R), dtype=float)
    for i in range(I):
        for j in range(J):
            bar_w[i, j, :] = (R * U_ijr_norm[i, j, :] * z_star) / (max(t[i], EPS) * max(s_attr_rank[i, j], EPS))
            bar_w[i, j, :] = np.maximum(bar_w[i, j, :], 0.0)

    w_ijk = np.zeros((I, J, R), dtype=float)
    for i in range(I):
        for j in range(J):
            for k in range(R):
                r = int(alt_rank[i, j, k])
                if r < 1 or r > R:
                    raise ValueError(f"Alternative rank out of range at i={i+1}, j={j+1}, k={k+1}: {r}")
                w_ijk[i, j, k] = bar_w[i, j, r - 1]

    W_K = np.sum(w_ijk, axis=(0, 1))  # (R,)
    W_K = W_K / max(float(np.sum(W_K)), EPS)

    # rank: 1 best
    order = np.argsort(-W_K, kind="mergesort")
    rank_K = np.empty(R, dtype=int)
    rank_K[order] = np.arange(1, R + 1)
    return W_K, rank_K


# =========================
# OPA baseline aggregated: ROC at expert/attribute/alternative levels
# =========================
def roc_weights_from_ranks(rank_vec: np.ndarray) -> np.ndarray:
    """
    Rank Order Centroid weights, rank=1 best.
    """
    n = rank_vec.size
    roc_by_r = np.zeros(n + 1, dtype=float)
    for r in range(1, n + 1):
        roc_by_r[r] = (1.0 / n) * np.sum(1.0 / np.arange(r, n + 1))
    w = np.array([roc_by_r[int(r)] for r in rank_vec], dtype=float)
    w = np.maximum(w, 0.0)
    w = w / max(float(np.sum(w)), EPS)
    return w

def opa_aggregate_WK(
    t: np.ndarray,            # (I,)
    s_attr_rank: np.ndarray,  # (I,J)
    alt_rank: np.ndarray,     # (I,J,R)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregated OPA baseline using ROC:
      expert weights = ROC(t)
      attribute weights per expert = ROC(s_i,:)
      alternative weights per (i,j) = ROC(alt_rank[i,j,:])
      final W_K = Σ_i w_i Σ_j a_ij * alt_w_ijk
    """
    I, J = s_attr_rank.shape
    R = alt_rank.shape[2]

    w_expert = roc_weights_from_ranks(t.astype(int))  # (I,)

    W_K = np.zeros(R, dtype=float)
    for i in range(I):
        w_attr = roc_weights_from_ranks(s_attr_rank[i, :].astype(int))  # (J,)
        for j in range(J):
            w_alt = roc_weights_from_ranks(alt_rank[i, j, :].astype(int))  # (R,)
            W_K += w_expert[i] * w_attr[j] * w_alt

    W_K = W_K / max(float(np.sum(W_K)), EPS)

    order = np.argsort(-W_K, kind="mergesort")
    rank_K = np.empty(R, dtype=int)
    rank_K[order] = np.arange(1, R + 1)
    return W_K, rank_K


# =========================
# Spearman matrix helper
# =========================
def spearman_matrix(vectors: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(vectors.keys())
    S = len(names)
    M = np.zeros((S, S), dtype=float)

    for a in range(S):
        for b in range(S):
            x = vectors[names[a]]
            y = vectors[names[b]]
            corr, _ = spearmanr(x, y)  # ties handled
            if np.isnan(corr):
                corr = 0.0
            M[a, b] = float(corr)

    return pd.DataFrame(M, index=names, columns=names)


# =========================
# Main: 7 scenarios aggregated + export
# =========================
def run_ablation_aggregate(
    xlsx_path: str,
    n_expert: int,
    n_attribute: int,
    n_alternative: int,
    out_path: str = "ablation_aggregate.xlsx",
) -> Dict[str, object]:
    I, J, R = n_expert, n_attribute, n_alternative
    t, s_attr_rank, alt_rank, cons_raw = read_inputs(xlsx_path, I, J, R)

    # 6 GOPA-RUE scenarios: (name, use_constraints, ref_type)
    scenarios = [
        ("S1_PC+Uniform",   True,  1),
        ("S2_noPC+Uniform", False, 1),
        ("S3_PC+Sshape",    True,  3),
        ("S4_noPC+Sshape",  False, 3),
        ("S5_PC+HARA",      True,  2),
        ("S6_noPC+HARA",    False, 2),
    ]

    results: Dict[str, Dict[str, np.ndarray]] = {}

    # Compute GOPA-RUE aggregated results
    for name, use_constraints, ref_type in scenarios:
        _, U_norm = compute_U_ijr_for_scenario(I, J, R, cons_raw, ref_type, use_constraints)
        W_K, rank_K = gopa_rue_aggregate_WK(U_norm, t, s_attr_rank, alt_rank)
        results[name] = {"W_K": W_K, "rank_K": rank_K}

    # OPA baseline aggregated
    W_opa, rank_opa = opa_aggregate_WK(t, s_attr_rank, alt_rank)
    results["S7_OPA_ROC"] = {"W_K": W_opa, "rank_K": rank_opa}

    scenario_names = list(results.keys())

    # Build one big table: scenario x alternative
    rows = []
    for sc in scenario_names:
        W = results[sc]["W_K"]
        rk = results[sc]["rank_K"]
        for k in range(R):
            rows.append({
                "scenario": sc,
                "alt_k": k + 1,
                "weight": float(W[k]),
                "rank": int(rk[k]),
            })
    df_all = pd.DataFrame(rows).sort_values(["scenario", "rank", "alt_k"]).reset_index(drop=True)

    # Spearman matrices
    rank_vectors = {sc: results[sc]["rank_K"].astype(float) for sc in scenario_names}
    weight_vectors = {sc: results[sc]["W_K"].astype(float) for sc in scenario_names}
    df_spear_rank = spearman_matrix(rank_vectors)
    df_spear_weight = spearman_matrix(weight_vectors)

    # Summary top1
    top1_rows = []
    for sc in scenario_names:
        rk = results[sc]["rank_K"]
        W = results[sc]["W_K"]
        k_best = int(np.argmin(rk)) + 1
        top1_rows.append({
            "scenario": sc,
            "top1_alt": k_best,
            "top1_weight": float(W[k_best - 1]),
        })
    df_top1 = pd.DataFrame(top1_rows).sort_values(["scenario"]).reset_index(drop=True)

    # Export
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="Scenario_WK_Rank", index=False)
        df_top1.to_excel(writer, sheet_name="Summary_Top1", index=False)
        df_spear_rank.to_excel(writer, sheet_name="Spearman_Rank", index=True)
        df_spear_weight.to_excel(writer, sheet_name="Spearman_Weight", index=True)

        # Optional: wide format for easy visual compare (alt_k as rows, scenarios as columns)
        wide_w = pd.DataFrame({"alt_k": np.arange(1, R + 1, dtype=int)})
        wide_r = pd.DataFrame({"alt_k": np.arange(1, R + 1, dtype=int)})
        for sc in scenario_names:
            wide_w[sc] = results[sc]["W_K"]
            wide_r[sc] = results[sc]["rank_K"]
        wide_w.to_excel(writer, sheet_name="WK_Wide", index=False)
        wide_r.to_excel(writer, sheet_name="Rank_Wide", index=False)

    return {
        "results": results,
        "spearman_rank": df_spear_rank,
        "spearman_weight": df_spear_weight,
        "output_path": out_path,
    }


if __name__ == "__main__":
    n_expert = 5
    n_attribute = 6
    n_alternative = 10

    res = run_ablation_aggregate(
        xlsx_path="test.xlsx",
        n_expert=n_expert,
        n_attribute=n_attribute,
        n_alternative=n_alternative,
        out_path="ablation_aggregate.xlsx",
    )
    print("Done. Saved to:", res["output_path"])
