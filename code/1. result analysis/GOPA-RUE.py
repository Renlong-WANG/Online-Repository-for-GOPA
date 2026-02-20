import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from scipy.optimize import nnls


# =========================
# Tunable parameters
# =========================
LAMBDA_RIDGE = 1e-6   # Ridge regularization coefficient λ
EPS = 1e-12           # Numerical stability epsilon
BETA_ZERO_TOL = 1e-10 # Tolerance to treat beta as zero (avoid Excel floating noise)

# HARA reference density parameters (defaults; adjust to your experimental setting if needed)
HARA_ALPHA = 2
HARA_BETA = 1.0
HARA_GAMMA = 1.5


# =========================
# Reference density v(x) and its CDF-like integral F(x)
# =========================
def v_uniform(x: np.ndarray, R: float) -> np.ndarray:
    return np.ones_like(x) / R

def F_uniform(x: np.ndarray, R: float) -> np.ndarray:
    return np.clip(x, 0, R) / R

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
    Returns the normalized integral:
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
    Normalizes the logistic utility V(x) to [0, 1]:
        F(x) = (V(x) - V(0)) / (V(R) - V(0))
    """
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

        # Two-point constraint: use the first two non-zero positions
        r1_raw = int(nz[0] + 1)
        r2_raw = int(nz[1] + 1)

        r1 = flip_rank(r1_raw, R)
        r2 = flip_rank(r2_raw, R)

        # Enforce rb < rb2 for consistent construction
        if r1 <= r2:
            rb, rb2 = r1, r2
            coeff_rb2 = judge[nz[1]]
        else:
            rb, rb2 = r2, r1
            coeff_rb2 = judge[nz[0]]

        # Decide ABS_DIFF vs RATIO by whether beta (degree) is zero
        if abs(degree) <= BETA_ZERO_TOL:
            # RATIO: u_rb - alpha * u_rb2 = 0
            alpha = abs(coeff_rb2)
            if alpha <= EPS:
                raise ValueError(
                    "Detected RATIO constraint (beta≈0) but cannot infer alpha. "
                    "Please ensure judgement row contains alpha as a nonzero coefficient."
                )
            specs.append(ConstraintSpec(rb=rb, rb2=rb2, alpha=alpha, beta=0.0, kind="TWO_POINT", mode="RATIO"))
        else:
            # ABS_DIFF: u_rb - alpha * u_rb2 = beta
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
    """
    First solve unconstrained ridge:
        g = (A^T A + λI)^(-1) A^T beta
    If any component is negative, solve NNLS with ridge by augmentation:
        min ||A g - beta||^2 + λ||g||^2,  s.t. g >= 0
    """
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
    Compute induced utilities:
        U_r = ∫_0^{R-r+1} u(x) dx
    where u(x) = g_c * v(x) on interval c.
    Using F to evaluate truncated integrals efficiently.
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


# =========================
# Normalize U_{ijr}: enforce sum_r U = 1 for each (i, j)
# =========================
def normalize_U_per_ij(U_ijr_raw: np.ndarray) -> np.ndarray:
    U = U_ijr_raw.copy()
    I, J, R = U.shape
    for i in range(I):
        for j in range(J):
            s = float(np.sum(U[i, j, :]))
            if s <= EPS:
                raise ValueError(
                    f"Sum of U_ijr is zero (or too small) for (i={i+1}, j={j+1}). Cannot normalize."
                )
            U[i, j, :] = U[i, j, :] / s
    return U


# =========================
# Main pipeline: read Excel -> run GOPA-RUE -> export results
# =========================
def run_gopa_rue(
    xlsx_path: str,
    n_expert: int,
    n_attribute: int,
    n_alternative: int,
    out_path: str = "output.xlsx",
) -> Dict[str, object]:

    R = n_alternative

    # ---- Read inputs ----
    df_expert_rank = pd.read_excel(xlsx_path, sheet_name="Expert Ranking", header=None)
    df_attr_rank = pd.read_excel(xlsx_path, sheet_name="Attribute Ranking", header=None)
    df_alt_rank = pd.read_excel(xlsx_path, sheet_name="Alternative Ranking", header=None)
    df_constraint = pd.read_excel(xlsx_path, sheet_name="Constraint", header=None)
    df_ref = pd.read_excel(xlsx_path, sheet_name="Reference Function", header=None)

    # Expert ranks t_i (1..n_expert)
    t = df_expert_rank.values.reshape(-1).astype(int)
    if t.size != n_expert:
        raise ValueError(f"Expert Ranking size mismatch, expected {n_expert}, got {t.size}")

    # Attribute ranks s_ij (n_expert x n_attribute)
    s = df_attr_rank.values.astype(int)
    if s.shape != (n_expert, n_attribute):
        raise ValueError(f"Attribute Ranking shape mismatch, expected {(n_expert, n_attribute)}, got {s.shape}")

    # Alternative Ranking: (n_expert*R) x n_attribute
    alt_raw = df_alt_rank.values.astype(int)
    expected_rows = n_expert * R
    if alt_raw.shape != (expected_rows, n_attribute):
        raise ValueError(
            f"Alternative Ranking shape mismatch, expected {(expected_rows, n_attribute)}, got {alt_raw.shape}"
        )

    # alt_rank[i, j, k] = rank r (1..R) for alternative k under attribute j, given expert i
    alt_rank = np.zeros((n_expert, n_attribute, R), dtype=int)
    for i in range(n_expert):
        block = alt_raw[i * R : (i + 1) * R, :]
        for j in range(n_attribute):
            alt_rank[i, j, :] = block[:, j]

    # Reference function type per (i, j): 1=Uniform, 2=HARA, 3=Logistic(S-shape)
    ref_type = df_ref.values.astype(int)
    if ref_type.shape != (n_expert, n_attribute):
        raise ValueError(f"Reference Function shape mismatch, expected {(n_expert, n_attribute)}, got {ref_type.shape}")

    # Constraints: (n_expert*n_attribute*R) x (R+1)
    cons_raw = df_constraint.values
    expected_rows = n_expert * n_attribute * R
    if cons_raw.shape != (expected_rows, R + 1):
        raise ValueError(
            f"Constraint shape mismatch, expected {(expected_rows, R+1)}, got {cons_raw.shape}"
        )

    # ---- Lower level: solve g and compute raw U ----
    U_ijr_raw = np.zeros((n_expert, n_attribute, R), dtype=float)
    g_store: Dict[Tuple[int, int], Dict[str, object]] = {}

    for i in range(n_expert):
        for j in range(n_attribute):
            start = (i * n_attribute + j) * R
            block = cons_raw[start : start + R, :]
            specs = parse_constraints_block(block, R)

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

            # Density normalization: e^T ΔV g = 1
            Psi_rows.append(np.ones(C))
            beta_rows.append(1.0)

            Psi = np.vstack(Psi_rows)
            beta_vec = np.array(beta_rows, float)

            A = Psi * dV.reshape(1, -1)

            g_star = solve_g_star(A, beta_vec, LAMBDA_RIDGE)

            U_raw = induced_utilities_from_g(g_star, bps, F_fun, R)
            U_ijr_raw[i, j, :] = U_raw

            g_store[(i, j)] = {
                "breakpoints": bps,
                "dV": dV,
                "g": g_star,
                "ref_type": int(ref_type[i, j]),
                "num_constraints": len(specs),
                "constraint_modes": [c.mode for c in specs],
            }

    # ---- Normalize U_{ijr} per (i, j) to make sum_r U = 1 ----
    U_ijr = normalize_U_per_ij(U_ijr_raw)

    # ---- Upper level: closed-form solution using normalized U ----
    denom = 0.0
    for i in range(n_expert):
        for j in range(n_attribute):
            denom += np.sum((R * U_ijr[i, j, :]) / (t[i] * s[i, j]))

    z_star = 1.0 / max(denom, EPS)

    bar_w = np.zeros_like(U_ijr)
    for i in range(n_expert):
        for j in range(n_attribute):
            bar_w[i, j, :] = (R * U_ijr[i, j, :] * z_star) / (t[i] * s[i, j])
            bar_w[i, j, :] = np.maximum(bar_w[i, j, :], 0.0)

    # ---- Map bar_w_{ijr} to w_{ijk} using alternative rankings ----
    w_ijk = np.zeros((n_expert, n_attribute, R), dtype=float)
    for i in range(n_expert):
        for j in range(n_attribute):
            for k in range(R):
                r = int(alt_rank[i, j, k])
                if r < 1 or r > R:
                    raise ValueError(
                        f"Alternative rank out of range at expert {i+1}, attr {j+1}, alt {k+1}: {r}"
                    )
                w_ijk[i, j, k] = bar_w[i, j, r - 1]

    # ---- Global weights ----
    W_I = w_ijk.sum(axis=(1, 2))  # per expert
    W_J = w_ijk.sum(axis=(0, 2))  # per attribute
    W_K = w_ijk.sum(axis=(0, 1))  # per alternative

    # ---- Export to Excel ----
    idx = []
    Uraw_rows = []
    U_rows = []
    BW_rows = []
    for i in range(n_expert):
        for j in range(n_attribute):
            idx.append((i + 1, j + 1))
            Uraw_rows.append(U_ijr_raw[i, j, :])
            U_rows.append(U_ijr[i, j, :])
            BW_rows.append(bar_w[i, j, :])

    Uraw_df = pd.DataFrame(
        Uraw_rows,
        index=pd.MultiIndex.from_tuples(idx, names=["expert_i", "attr_j"]),
        columns=[f"r={r}" for r in range(1, R + 1)],
    )
    U_df = pd.DataFrame(
        U_rows,
        index=pd.MultiIndex.from_tuples(idx, names=["expert_i", "attr_j"]),
        columns=[f"r={r}" for r in range(1, R + 1)],
    )
    BW_df = pd.DataFrame(
        BW_rows,
        index=pd.MultiIndex.from_tuples(idx, names=["expert_i", "attr_j"]),
        columns=[f"r={r}" for r in range(1, R + 1)],
    )

    W_rows = []
    for i in range(n_expert):
        for j in range(n_attribute):
            W_rows.append(w_ijk[i, j, :])
    w_df = pd.DataFrame(
        W_rows,
        index=pd.MultiIndex.from_tuples(idx, names=["expert_i", "attr_j"]),
        columns=[f"k={k}" for k in range(1, R + 1)],
    )

    WI_df = pd.DataFrame({"W_I": W_I}, index=[f"i={i}" for i in range(1, n_expert + 1)])
    WJ_df = pd.DataFrame({"W_J": W_J}, index=[f"j={j}" for j in range(1, n_attribute + 1)])
    WK_df = pd.DataFrame({"W_K": W_K}, index=[f"k={k}" for k in range(1, R + 1)])
    z_df = pd.DataFrame({"z*": [z_star]})

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        Uraw_df.to_excel(writer, sheet_name="U_ijr_raw")
        U_df.to_excel(writer, sheet_name="U_ijr_norm")
        BW_df.to_excel(writer, sheet_name="bar_w_ijr")
        w_df.to_excel(writer, sheet_name="w_ijk")
        WI_df.to_excel(writer, sheet_name="W_I")
        WJ_df.to_excel(writer, sheet_name="W_J")
        WK_df.to_excel(writer, sheet_name="W_K")
        z_df.to_excel(writer, sheet_name="z_star", index=False)

    return {
        "U_ijr_raw": U_ijr_raw,
        "U_ijr": U_ijr,
        "bar_w_ijr": bar_w,
        "w_ijk": w_ijk,
        "W_I": W_I,
        "W_J": W_J,
        "W_K": W_K,
        "z_star": z_star,
        "debug_g": g_store,
        "output_path": out_path,
    }


if __name__ == "__main__":
    n_expert = 5
    n_attribute = 6
    n_alternative = 10

    res = run_gopa_rue(
        xlsx_path="test.xlsx",
        n_expert=n_expert,
        n_attribute=n_attribute,
        n_alternative=n_alternative,
        out_path="output.xlsx",
    )
    print("Done. z* =", res["z_star"])
    print("Saved to:", res["output_path"])
