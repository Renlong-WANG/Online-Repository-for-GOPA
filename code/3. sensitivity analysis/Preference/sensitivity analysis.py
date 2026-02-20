import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from scipy.optimize import nnls


# =========================
# Tunable parameters
# =========================
LAMBDA_RIDGE = 1e-6
EPS = 1e-12
BETA_ZERO_TOL = 1e-10

# HARA reference density parameters
HARA_ALPHA = 2
HARA_BETA = 1.0
HARA_GAMMA = 1.5


# =========================
# Reference density v(x) and its CDF-like integral F(x)
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
    # 1=Uniform, 2=HARA, 3=Logistic(S-shape)
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
    kind: str = "TWO_POINT"    # TWO_POINT / LOWER_BOUND
    mode: str = ""             # ABS_DIFF / RATIO / LOWER_BOUND


# =========================
# Rank flipping: r -> R+1-r
# =========================
def flip_rank(r: int, R: int) -> int:
    if r < 1 or r > R:
        raise ValueError(f"Rank out of range: r={r}, R={R}")
    return R + 1 - r


# =========================
# Constraint parsing (mixed modes + rank flip)
# =========================
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
            # RATIO: u_rb - alpha*u_rb2 = 0
            alpha = abs(coeff_rb2)
            if alpha <= EPS:
                raise ValueError("RATIO constraint detected but cannot infer alpha from judgement row.")
            specs.append(ConstraintSpec(rb=rb, rb2=rb2, alpha=alpha, beta=0.0, kind="TWO_POINT", mode="RATIO"))
        else:
            # ABS_DIFF: u_rb - alpha*u_rb2 = beta
            alpha = abs(coeff_rb2)
            if alpha <= EPS:
                alpha = 1.0
            specs.append(ConstraintSpec(rb=rb, rb2=rb2, alpha=alpha, beta=degree, kind="TWO_POINT", mode="ABS_DIFF"))

    return specs


# =========================
# Build A_ij, beta_ij and solve g*
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
                raise ValueError(f"Sum of U_ijr too small for (i={i+1}, j={j+1}).")
            U[i, j, :] = U[i, j, :] / s
    return U


# =========================
# Apply +/-10% perturbations to alpha (RATIO) and beta (ABS_DIFF)
# =========================
def perturb_specs(
    specs: List[ConstraintSpec],
    rng: np.random.Generator,
    pct: float = 0.10,
) -> Tuple[List[ConstraintSpec], List[Dict[str, object]]]:
    """
    Return perturbed specs (deep-copied) and a log list describing perturbations.
    - mode == RATIO: perturb alpha
    - mode == ABS_DIFF: perturb beta
    """
    new_specs: List[ConstraintSpec] = []
    logs: List[Dict[str, object]] = []

    for idx, cs in enumerate(specs):
        cs_new = ConstraintSpec(
            rb=cs.rb, rb2=cs.rb2, alpha=cs.alpha, beta=cs.beta, kind=cs.kind, mode=cs.mode
        )

        if cs.mode == "RATIO":
            delta = rng.uniform(-pct, pct)
            cs_new.alpha = cs.alpha * (1.0 + delta)
            # ensure positive
            cs_new.alpha = float(max(cs_new.alpha, EPS))
            logs.append({
                "constraint_idx": idx,
                "mode": "RATIO",
                "rb": cs.rb,
                "rb2": cs.rb2,
                "alpha_base": cs.alpha,
                "alpha_new": cs_new.alpha,
                "delta": delta,
            })

        elif cs.mode == "ABS_DIFF":
            delta = rng.uniform(-pct, pct)
            cs_new.beta = cs.beta * (1.0 + delta)
            logs.append({
                "constraint_idx": idx,
                "mode": "ABS_DIFF",
                "rb": cs.rb,
                "rb2": cs.rb2,
                "beta_base": cs.beta,
                "beta_new": cs_new.beta,
                "delta": delta,
            })

        # LOWER_BOUND not perturbed by your request
        new_specs.append(cs_new)

    return new_specs, logs


# =========================
# Core run: one trial with possibly perturbed constraints
# =========================
def run_gopa_rue_one_trial(
    t: np.ndarray,
    s: np.ndarray,
    alt_rank: np.ndarray,
    cons_raw: np.ndarray,
    ref_type: np.ndarray,
    n_expert: int,
    n_attribute: int,
    n_alternative: int,
    rng: Optional[np.random.Generator] = None,
    perturb: bool = False,
    perturb_pct: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """
    Returns:
      W_K (R,), rank_K (R,), perturbation_logs (list)
    """
    I, J, R = n_expert, n_attribute, n_alternative
    U_ijr_raw = np.zeros((I, J, R), dtype=float)
    perturb_logs_all: List[Dict[str, object]] = []

    for i in range(I):
        for j in range(J):
            start = (i * J + j) * R
            block = cons_raw[start : start + R, :]
            specs = parse_constraints_block(block, R)

            if perturb and rng is not None:
                specs, logs = perturb_specs(specs, rng=rng, pct=perturb_pct)
                # add (i,j) identity
                for d in logs:
                    d["expert_i"] = i + 1
                    d["attr_j"] = j + 1
                perturb_logs_all.extend(logs)

            bps = build_breakpoints_from_constraints(specs, R)
            C = len(bps) - 1
            if C <= 0:
                raise ValueError(f"Breakpoints invalid for expert {i+1}, attr {j+1}")

            _, F_fun = get_vF(int(ref_type[i, j]))

            dV = interval_masses(F_fun, bps, R)
            dV = np.maximum(dV, EPS)

            Psi_rows = []
            beta_rows = []

            for cs in specs:
                if cs.kind == "LOWER_BOUND":
                    phi_rb = bar_phi_vector(cs.rb, bps)
                    Psi_rows.append(phi_rb)
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

    # Upper level
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

    # Map to alternatives
    w_ijk = np.zeros((I, J, R), dtype=float)
    for i in range(I):
        for j in range(J):
            for k in range(R):
                r = int(alt_rank[i, j, k])
                if r < 1 or r > R:
                    raise ValueError(f"Alternative rank out of range at i={i+1}, j={j+1}, k={k+1}: {r}")
                w_ijk[i, j, k] = bar_w[i, j, r - 1]

    W_K = w_ijk.sum(axis=(0, 1))  # (R,)
    W_K = W_K / max(float(np.sum(W_K)), EPS)

    order = np.argsort(-W_K, kind="mergesort")
    rank_K = np.empty(R, dtype=int)
    rank_K[order] = np.arange(1, R + 1)

    return W_K, rank_K, perturb_logs_all


# =========================
# Sensitivity analysis: multiple trials
# =========================
def run_preference_parameter_sensitivity(
    xlsx_path: str,
    n_expert: int,
    n_attribute: int,
    n_alternative: int,
    n_trials: int = 200,
    seed: int = 123,
    perturb_pct: float = 0.10,
    out_path: str = "sensitivity_output.xlsx",
) -> Dict[str, object]:

    R = n_alternative

    # ---- Read inputs ----
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

    # ---- Baseline (no perturbation) ----
    W_base, rank_base, _ = run_gopa_rue_one_trial(
        t=t, s=s, alt_rank=alt_rank, cons_raw=cons_raw, ref_type=ref_type,
        n_expert=n_expert, n_attribute=n_attribute, n_alternative=n_alternative,
        rng=None, perturb=False
    )

    # ---- Trials ----
    rng = np.random.default_rng(seed)
    W_trials = np.zeros((n_trials, R), dtype=float)
    rank_trials = np.zeros((n_trials, R), dtype=int)
    perturb_logs_rows: List[Dict[str, object]] = []

    for trial in range(n_trials):
        Wk, rk, logs = run_gopa_rue_one_trial(
            t=t, s=s, alt_rank=alt_rank, cons_raw=cons_raw, ref_type=ref_type,
            n_expert=n_expert, n_attribute=n_attribute, n_alternative=n_alternative,
            rng=rng, perturb=True, perturb_pct=perturb_pct
        )
        W_trials[trial, :] = Wk
        rank_trials[trial, :] = rk

        for d in logs:
            d2 = dict(d)
            d2["trial"] = trial + 1
            perturb_logs_rows.append(d2)

    # ---- Build output dataframes ----
    base_df = pd.DataFrame({
        "alt_k": np.arange(1, R + 1, dtype=int),
        "weight": W_base,
        "rank": rank_base
    }).sort_values(["rank", "alt_k"]).reset_index(drop=True)

    long_rows = []
    for trial in range(n_trials):
        for k in range(R):
            long_rows.append({
                "trial": trial + 1,
                "alt_k": k + 1,
                "weight": float(W_trials[trial, k]),
                "rank": int(rank_trials[trial, k]),
            })
    long_df = pd.DataFrame(long_rows).sort_values(["trial", "rank", "alt_k"]).reset_index(drop=True)

    wide_w = pd.DataFrame(W_trials, columns=[f"alt_{k}" for k in range(1, R + 1)])
    wide_w.insert(0, "trial", np.arange(1, n_trials + 1, dtype=int))

    wide_r = pd.DataFrame(rank_trials, columns=[f"alt_{k}" for k in range(1, R + 1)])
    wide_r.insert(0, "trial", np.arange(1, n_trials + 1, dtype=int))

    perturb_df = pd.DataFrame(perturb_logs_rows)
    if not perturb_df.empty:
        # sort for readability
        cols_front = ["trial", "expert_i", "attr_j", "constraint_idx", "mode", "rb", "rb2", "delta"]
        remaining = [c for c in perturb_df.columns if c not in cols_front]
        perturb_df = perturb_df[cols_front + remaining].sort_values(
            ["trial", "expert_i", "attr_j", "constraint_idx"]
        ).reset_index(drop=True)

    # ---- Export ----
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        base_df.to_excel(writer, sheet_name="Base_WK", index=False)
        long_df.to_excel(writer, sheet_name="Perturb_WK_Long", index=False)
        wide_w.to_excel(writer, sheet_name="Perturb_WK_Wide", index=False)
        wide_r.to_excel(writer, sheet_name="Perturb_Rank_Wide", index=False)
        perturb_df.to_excel(writer, sheet_name="AlphaBeta_Perturbations", index=False)

        # quick summary stats on weights (mean/std/min/max)
        stats = pd.DataFrame({
            "alt_k": np.arange(1, R + 1, dtype=int),
            "base_weight": W_base,
            "mean_weight": np.mean(W_trials, axis=0),
            "std_weight": np.std(W_trials, axis=0, ddof=1),
            "min_weight": np.min(W_trials, axis=0),
            "max_weight": np.max(W_trials, axis=0),
        })
        stats.to_excel(writer, sheet_name="Weight_Stats", index=False)

    return {
        "W_base": W_base,
        "rank_base": rank_base,
        "W_trials": W_trials,
        "rank_trials": rank_trials,
        "output_path": out_path,
    }


if __name__ == "__main__":
    n_expert = 5
    n_attribute = 6
    n_alternative = 10

    res = run_preference_parameter_sensitivity(
        xlsx_path="test.xlsx",
        n_expert=n_expert,
        n_attribute=n_attribute,
        n_alternative=n_alternative,
        n_trials=200,          # 你可以改成 1000 等
        seed=123,              # 固定随机种子方便复现
        perturb_pct=0.10,      # ±10%
        out_path="sensitivity_output.xlsx",
    )
    print("Done. Saved to:", res["output_path"])
