"""
Sensitivities-based method (MAR21): delta, vega and curvature capital
requirements for the seven risk classes, under the three correlation
scenarios.

Input: a "sensitivity table" (one row per sensitivity, see
:mod:`frtb.schema`). Sensitivities are expressed in the reporting currency
following MAR21.19-21.25 (PV01 and CS01 divided by 0.0001, equity /
commodity / FX deltas divided by 0.01, vega = vega x implied volatility).
Curvature rows carry CVR_up and CVR_down (MAR21.5), positive = loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import params as P

FACTOR_KEYS = ["bucket", "qualifier", "tenor", "underlying_tenor", "subtype",
               "eu_ets", "covered_aa"]

OTHER_BUCKET = {"CSR_NS": 16, "CSR_SEC_CTP": 16, "CSR_SEC_NCTP": 25, "EQ": 11}
NUMERIC_BUCKETS = {"CSR_NS", "CSR_SEC_CTP", "CSR_SEC_NCTP", "EQ", "COMM"}


# ==========================================================================
# Risk weights
# ==========================================================================


def _girr_specified(ccy: str, regime: P.Regime) -> bool:
    return ccy in P.GIRR_SPECIFIED_CCY or ccy == regime.reporting_ccy


def _fx_specified(ccy: str, regime: P.Regime) -> bool:
    # specified pairs vs USD and first-order crosses: both legs in the list
    return ccy in P.FX_SPECIFIED and regime.reporting_ccy in P.FX_SPECIFIED


def delta_rw(rc: str, f: pd.Series, regime: P.Regime) -> float:
    b = f["bucket"]
    if rc == "GIRR":
        st = f["subtype"]
        rw = P.GIRR_RW_INFLATION if st == "Inflation" else (
            P.GIRR_RW_XCCY if st == "XCCY" else P.GIRR_RW[float(f["tenor"])])
        if regime.girr_sqrt2 and _girr_specified(b, regime):
            rw /= P.SQRT2
        return rw
    if rc == "CSR_NS":
        if int(b) == 8 and bool(f["covered_aa"]) and regime.covered_aa_reduced_rw:
            return P.CSR_NS_RW_COVERED_AA
        return P.CSR_NS_RW[int(b)]
    if rc == "CSR_SEC_CTP":
        return P.CSR_CTP_RW[int(b)]
    if rc == "CSR_SEC_NCTP":
        return P.CSR_SEC_NCTP_RW[int(b)]
    if rc == "EQ":
        return (P.EQ_RW_REPO if f["subtype"] == "Repo" else P.EQ_RW_SPOT)[int(b)]
    if rc == "COMM":
        if regime.eu and bool(f["eu_ets"]):
            return regime.eu_ets_rw
        return P.COMM_RW[int(b)]
    if rc == "FX":
        rw = P.FX_RW
        if regime.fx_sqrt2 and _fx_specified(str(b), regime):
            rw /= P.SQRT2
        return rw
    raise ValueError(rc)


def curvature_rw(rc: str, bucket, regime: P.Regime, eu_ets: bool = False) -> float:
    """Size of the curvature shock (MAR21.98-21.99).

    GIRR / CSR: absolute parallel shift equal to the highest delta risk weight
    of the bucket. Equity / FX / commodity: relative shift equal to the delta
    risk weight.
    """
    if rc == "GIRR":
        rw = max(P.GIRR_RW.values())
        if regime.girr_sqrt2 and _girr_specified(str(bucket), regime):
            rw /= P.SQRT2
        return rw
    if rc == "CSR_NS":
        return P.CSR_NS_RW[int(bucket)]
    if rc == "CSR_SEC_CTP":
        return P.CSR_CTP_RW[int(bucket)]
    if rc == "CSR_SEC_NCTP":
        return P.CSR_SEC_NCTP_RW[int(bucket)]
    if rc == "EQ":
        return P.EQ_RW_SPOT[int(bucket)]
    if rc == "COMM":
        return regime.eu_ets_rw if (regime.eu and eu_ets) else P.COMM_RW[int(bucket)]
    if rc == "FX":
        rw = P.FX_RW
        if regime.fx_sqrt2 and _fx_specified(str(bucket), regime):
            rw /= P.SQRT2
        return rw
    raise ValueError(rc)


# ==========================================================================
# Intra-bucket correlations (medium scenario)
# ==========================================================================


def _same(a: np.ndarray) -> np.ndarray:
    return a[:, None] == a[None, :]


def _tenor_rho(t: np.ndarray, theta: float, floor: float = 0.0) -> np.ndarray:
    tk, tl = t[:, None], t[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.exp(-theta * np.abs(tk - tl) / np.minimum(tk, tl))
    r = np.where(tk == tl, 1.0, r)
    return np.maximum(r, floor)


def _girr_delta_corr(f: pd.DataFrame) -> np.ndarray:
    st = f["subtype"].to_numpy()
    curve = f["qualifier"].to_numpy()
    t = f["tenor"].to_numpy(dtype=float)
    yld = st == "Yield"
    rho_t = _tenor_rho(np.where(yld, t, 1.0), P.GIRR_THETA, P.GIRR_RHO_FLOOR)
    yy = rho_t * np.where(_same(curve), 1.0, P.GIRR_DIFF_CURVE)
    infl = st == "Inflation"
    xccy = st == "XCCY"
    R = np.zeros((len(f), len(f)))
    both_y = yld[:, None] & yld[None, :]
    R[both_y] = yy[both_y]
    both_i = infl[:, None] & infl[None, :]
    R[both_i] = np.where(_same(curve), 1.0, P.GIRR_DIFF_CURVE)[both_i]
    yi = (yld[:, None] & infl[None, :]) | (infl[:, None] & yld[None, :])
    R[yi] = P.GIRR_INFL_YIELD
    # cross-currency basis: 0 with everything else (MAR21.49)
    np.fill_diagonal(R, 1.0)
    return R


def _type_rho_girr(st: np.ndarray) -> np.ndarray:
    """Delta correlation between GIRR factor types, used for vega (MAR21.94)."""
    yld, infl = st == "Yield", st == "Inflation"
    R = np.zeros((len(st), len(st)))
    R[_same(st)] = 1.0
    R[(yld[:, None] & infl[None, :]) | (infl[:, None] & yld[None, :])] = P.GIRR_INFL_YIELD
    return R


def _csr_name_rho(rc: str, bucket: int) -> float:
    if rc == "CSR_SEC_NCTP":
        return P.CSR_SEC_RHO_TRANCHE
    if rc == "CSR_NS" and bucket in (17, 18):
        return P.CSR_NS_RHO_NAME_INDEX
    return P.CSR_NS_RHO_NAME


def _csr_delta_corr(rc: str, bucket: int, f: pd.DataFrame) -> np.ndarray:
    name = f["qualifier"].to_numpy()
    tenor = f["tenor"].to_numpy(dtype=float)
    basis = f["subtype"].to_numpy()
    rho_name = _csr_name_rho(rc, bucket)
    if rc == "CSR_SEC_NCTP":
        rho_tenor, rho_basis = P.CSR_SEC_RHO_TENOR, P.CSR_SEC_RHO_BASIS
    elif rc == "CSR_SEC_CTP":
        rho_tenor, rho_basis = P.CSR_NS_RHO_TENOR, P.CSR_CTP_RHO_BASIS
    else:
        rho_tenor, rho_basis = P.CSR_NS_RHO_TENOR, P.CSR_NS_RHO_BASIS
    return (np.where(_same(name), 1.0, rho_name)
            * np.where(_same(tenor), 1.0, rho_tenor)
            * np.where(_same(basis), 1.0, rho_basis))


def _eq_delta_corr(bucket: int, f: pd.DataFrame) -> np.ndarray:
    name = f["qualifier"].to_numpy()
    st = f["subtype"].to_numpy()
    rho = P.eq_name_rho(bucket)
    same_name, same_type = _same(name), _same(st)
    R = np.where(same_name, np.where(same_type, 1.0, P.EQ_SPOT_REPO_SAME_NAME),
                 rho * np.where(same_type, 1.0, P.EQ_SPOT_REPO_SAME_NAME))
    return R


def _comm_delta_corr(bucket: int, f: pd.DataFrame, regime: P.Regime) -> np.ndarray:
    cty = f["qualifier"].to_numpy()
    tenor = f["tenor"].to_numpy(dtype=float)
    loc = f["subtype"].to_numpy()
    ets = f["eu_ets"].to_numpy(dtype=bool)
    rho_tenor = np.full((len(f), len(f)), P.COMM_RHO_TENOR)
    if regime.eu:
        both_ets = ets[:, None] & ets[None, :]
        rho_tenor = np.where(both_ets, min(P.COMM_RHO_TENOR / regime.eu_ets_tenor_divisor, 1.0),
                             rho_tenor)
    return (np.where(_same(cty), 1.0, P.COMM_RHO_CTY[bucket])
            * np.where(_same(tenor), 1.0, rho_tenor)
            * np.where(_same(loc), 1.0, P.COMM_RHO_BASIS))


def delta_corr(rc: str, bucket, f: pd.DataFrame, regime: P.Regime) -> np.ndarray:
    if rc == "GIRR":
        return _girr_delta_corr(f)
    if rc in ("CSR_NS", "CSR_SEC_CTP", "CSR_SEC_NCTP"):
        return _csr_delta_corr(rc, int(bucket), f)
    if rc == "EQ":
        return _eq_delta_corr(int(bucket), f)
    if rc == "COMM":
        return _comm_delta_corr(int(bucket), f, regime)
    if rc == "FX":
        return np.ones((len(f), len(f)))  # one risk factor per bucket
    raise ValueError(rc)


def vega_corr(rc: str, bucket, f: pd.DataFrame, regime: P.Regime) -> np.ndarray:
    """MAR21.93-21.94 and the FAQ to MAR21.94."""
    t_opt = f["tenor"].to_numpy(dtype=float)
    rho_opt = _tenor_rho(t_opt, P.VEGA_ALPHA)
    if rc == "GIRR":
        st = f["subtype"].to_numpy()
        yld = st == "Yield"
        u = np.where(yld, f["underlying_tenor"].to_numpy(dtype=float), 1.0)
        rho_und = _tenor_rho(u, P.VEGA_ALPHA)
        both_y = yld[:, None] & yld[None, :]
        rho_delta = np.where(both_y, rho_und, _type_rho_girr(st))
        return np.minimum(rho_opt * rho_delta, 1.0)
    name = f["qualifier"].to_numpy()
    if rc in ("CSR_NS", "CSR_SEC_CTP", "CSR_SEC_NCTP"):
        rho_delta = np.where(_same(name), 1.0, _csr_name_rho(rc, int(bucket)))
    elif rc == "EQ":
        rho_delta = np.where(_same(name), 1.0, P.eq_name_rho(int(bucket)))
    elif rc == "COMM":
        rho_delta = np.where(_same(name), 1.0, P.COMM_RHO_CTY[int(bucket)])
    elif rc == "FX":
        rho_delta = np.ones((len(f), len(f)))
    else:
        raise ValueError(rc)
    return np.minimum(rho_opt * rho_delta, 1.0)


def curvature_corr(rc: str, bucket, f: pd.DataFrame) -> np.ndarray:
    """Squared delta correlations on the name / commodity dimension (MAR21.100)."""
    name = f["qualifier"].to_numpy()
    if rc in ("GIRR", "FX"):
        return np.ones((len(f), len(f)))
    if rc in ("CSR_NS", "CSR_SEC_CTP", "CSR_SEC_NCTP"):
        rho = _csr_name_rho(rc, int(bucket))
    elif rc == "EQ":
        rho = P.eq_name_rho(int(bucket))
    elif rc == "COMM":
        rho = P.COMM_RHO_CTY[int(bucket)]
    else:
        raise ValueError(rc)
    return np.where(_same(name), 1.0, rho) ** 2


# ==========================================================================
# Cross-bucket correlations (medium scenario)
# ==========================================================================


def gamma(rc: str, b, c) -> float:
    if b == c:
        return 1.0
    if rc == "GIRR":
        return P.GIRR_GAMMA
    if rc in ("CSR_NS", "CSR_SEC_CTP"):
        return P.csr_ns_gamma(int(b), int(c))
    if rc == "CSR_SEC_NCTP":
        return 0.0
    if rc == "EQ":
        return P.eq_gamma(int(b), int(c))
    if rc == "COMM":
        return P.comm_gamma(int(b), int(c))
    if rc == "FX":
        return P.FX_GAMMA
    raise ValueError(rc)


def gamma_matrix(rc: str, buckets: list) -> np.ndarray:
    n = len(buckets)
    G = np.empty((n, n))
    for i, b in enumerate(buckets):
        for j, c in enumerate(buckets):
            G[i, j] = gamma(rc, b, c)
    return G


# ==========================================================================
# Aggregation helpers
# ==========================================================================


def _apply_scenario(R: np.ndarray, scenario: str) -> np.ndarray:
    out = P.scenario_corr(R, scenario)
    np.fill_diagonal(out, 1.0)
    return out


def _is_other(rc: str, bucket) -> bool:
    ob = OTHER_BUCKET.get(rc)
    return ob is not None and int(bucket) == ob


def _across(K: np.ndarray, S: np.ndarray, G: np.ndarray, psi: np.ndarray | None = None) -> tuple[float, bool]:
    """sqrt(sum K_b^2 + sum_{b!=c} gamma S_b S_c), alternative spec if negative (MAR21.4(5))."""
    off = G.copy()
    np.fill_diagonal(off, 0.0)
    if psi is not None:
        off = off * psi
    q = np.sum(K**2) + S @ off @ S
    alt = False
    if q < 0 and psi is None:
        S2 = np.clip(S, -K, K)
        q = np.sum(K**2) + S2 @ off @ S2
        alt = True
    return float(np.sqrt(max(q, 0.0))), alt


# ==========================================================================
# Delta / vega
# ==========================================================================


def net_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Net sensitivities to the same risk factor across instruments (MAR21.4(2))."""
    g = df.groupby(FACTOR_KEYS, dropna=False, sort=True)["amount"].sum().reset_index()
    return g


def delta_vega_charge(df: pd.DataFrame, rc: str, measure: str, regime: P.Regime):
    """Returns (factor table, bucket table, {scenario: capital})."""
    f = net_factors(df)
    if measure == "Delta":
        f["rw"] = [delta_rw(rc, row, regime) for _, row in f.iterrows()]
    else:
        f["rw"] = [P.vega_rw(rc, row["bucket"]) for _, row in f.iterrows()]
    f["ws"] = f["amount"] * f["rw"]

    buckets = sorted(f["bucket"].unique(), key=lambda x: (str(type(x)), x))
    bucket_rows, caps = [], {}
    corr_cache = {}
    for b in buckets:
        fb = f[f["bucket"] == b]
        corr_cache[b] = (None if _is_other(rc, b) else
                         (delta_corr if measure == "Delta" else vega_corr)(rc, b, fb, regime))
    for scn in P.SCENARIOS:
        K, S, names = [], [], []
        for b in buckets:
            ws = f.loc[f["bucket"] == b, "ws"].to_numpy()
            if _is_other(rc, b):
                kb = float(np.sum(np.abs(ws)))
            else:
                R = _apply_scenario(corr_cache[b], scn)
                kb = float(np.sqrt(max(ws @ R @ ws, 0.0)))
            K.append(kb)
            S.append(float(ws.sum()))
            names.append(b)
            bucket_rows.append({"risk_class": rc, "measure": measure, "scenario": scn,
                                "bucket": b, "K_b": kb, "S_b": float(ws.sum()),
                                "n_factors": len(ws)})
        K, S = np.array(K), np.array(S)
        if rc == "CSR_SEC_NCTP" and P.CSR_SEC_OTHER in [int(x) for x in names]:
            mask = np.array([int(x) != P.CSR_SEC_OTHER for x in names])
            G = _apply_scenario(gamma_matrix(rc, [n for n, m in zip(names, mask) if m]), scn)
            main, alt = _across(K[mask], S[mask], G) if mask.any() else (0.0, False)
            caps[scn] = main + float(K[~mask].sum())  # MAR21.71
        else:
            G = _apply_scenario(gamma_matrix(rc, names), scn)
            caps[scn], alt = _across(K, S, G)
    return f, pd.DataFrame(bucket_rows), caps


# ==========================================================================
# Curvature
# ==========================================================================


def curvature_charge(df: pd.DataFrame, rc: str, regime: P.Regime):
    d = df.copy()
    if rc == "FX" and regime.fx_curvature_scalar and "fx_cross_option" in d:
        scale = np.where(d["fx_cross_option"].astype(bool), 1.0 / P.FX_CURVATURE_SCALAR, 1.0)
        d["cvr_up"] = d["cvr_up"] * scale
        d["cvr_down"] = d["cvr_down"] * scale
    if rc in ("GIRR", "FX"):
        d["qualifier"] = d["bucket"].astype(str)
    f = d.groupby(["bucket", "qualifier"], sort=True)[["cvr_up", "cvr_down"]].sum().reset_index()

    buckets = sorted(f["bucket"].unique(), key=lambda x: (str(type(x)), x))
    bucket_rows, caps = [], {}
    base_corr = {b: curvature_corr(rc, b, f[f["bucket"] == b]) for b in buckets}
    for scn in P.SCENARIOS:
        K, S, names = [], [], []
        for b in buckets:
            fb = f[f["bucket"] == b]
            up, dn = fb["cvr_up"].to_numpy(), fb["cvr_down"].to_numpy()
            if _is_other(rc, b):  # MAR21.56(2), 21.69(2), 21.79(2)
                k_up = float(np.sum(np.maximum(up, 0.0)))
                k_dn = float(np.sum(np.maximum(dn, 0.0)))
            else:
                R = _apply_scenario(base_corr[b], scn)

                def kb(c):
                    neg = c < 0
                    psi = ~(neg[:, None] & neg[None, :])
                    off = R * psi
                    np.fill_diagonal(off, 0.0)
                    q = np.sum(np.maximum(c, 0.0) ** 2) + c @ off @ c
                    return float(np.sqrt(max(q, 0.0)))

                k_up, k_dn = kb(up), kb(dn)
            if k_up > k_dn or (k_up == k_dn and up.sum() > dn.sum()):
                k, s, direction = k_up, float(up.sum()), "Up"
            else:
                k, s, direction = k_dn, float(dn.sum()), "Down"
            K.append(k)
            S.append(s)
            names.append(b)
            bucket_rows.append({"risk_class": rc, "measure": "Curvature", "scenario": scn,
                                "bucket": b, "K_b": k, "S_b": s, "K_up": k_up, "K_down": k_dn,
                                "direction": direction, "n_factors": len(fb)})
        K, S = np.array(K), np.array(S)
        neg = S < 0
        psi = (~(neg[:, None] & neg[None, :])).astype(float)
        if rc == "CSR_SEC_NCTP" and P.CSR_SEC_OTHER in [int(x) for x in names]:
            mask = np.array([int(x) != P.CSR_SEC_OTHER for x in names])
            sel = [n for n, m in zip(names, mask) if m]
            G = _apply_scenario(gamma_matrix(rc, sel) ** 2, scn)
            main = _across(K[mask], S[mask], G, psi[np.ix_(mask, mask)])[0] if mask.any() else 0.0
            caps[scn] = main + float(K[~mask].sum())
        else:
            G = _apply_scenario(gamma_matrix(rc, names) ** 2, scn)
            caps[scn] = _across(K, S, G, psi)[0]
    return f, pd.DataFrame(bucket_rows), caps


# ==========================================================================
# Orchestration
# ==========================================================================


@dataclass
class SBMResult:
    by_class: pd.DataFrame        # risk_class x measure x scenario -> capital
    buckets: pd.DataFrame         # bucket-level K_b, S_b per scenario
    factors: dict                 # (rc, measure) -> factor table (WS, RW, CVR)
    totals: pd.Series             # scenario -> total SBM capital
    binding_scenario: str
    capital: float                # max over scenarios (before any EU multiplier)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c, default in [("qualifier", ""), ("subtype", ""), ("tenor", np.nan),
                       ("underlying_tenor", np.nan), ("amount", 0.0), ("cvr_up", 0.0),
                       ("cvr_down", 0.0), ("eu_ets", False), ("covered_aa", False),
                       ("fx_cross_option", False)]:
        if c not in d:
            d[c] = default
    d["qualifier"] = d["qualifier"].fillna("").astype(str)
    d["subtype"] = d["subtype"].fillna("").astype(str)
    for c in ("eu_ets", "covered_aa", "fx_cross_option"):
        d[c] = d[c].fillna(False).astype(bool)
    for c in ("amount", "cvr_up", "cvr_down"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["tenor"] = pd.to_numeric(d["tenor"], errors="coerce").fillna(-1.0)
    d["underlying_tenor"] = pd.to_numeric(d["underlying_tenor"], errors="coerce").fillna(-1.0)
    girr = d["risk_class"] == "GIRR"
    d.loc[girr & (d["subtype"] == ""), "subtype"] = "Yield"
    eq = d["risk_class"] == "EQ"
    d.loc[eq & (d["subtype"] == ""), "subtype"] = "Spot"
    # bucket typing: integers for numbered buckets, strings otherwise
    d["bucket"] = [int(float(b)) if rc in NUMERIC_BUCKETS else str(b)
                   for rc, b in zip(d["risk_class"], d["bucket"])]
    return d


def compute_sbm(sens: pd.DataFrame, regime: P.Regime) -> SBMResult:
    d = prepare(sens)
    rows, bucket_tables, factors = [], [], {}
    for rc in P.RISK_CLASSES:
        for measure in P.MEASURES:
            sub = d[(d["risk_class"] == rc) & (d["measure"] == measure)]
            if sub.empty:
                continue
            if measure == "Curvature":
                ft, bt, caps = curvature_charge(sub, rc, regime)
            else:
                ft, bt, caps = delta_vega_charge(sub, rc, measure, regime)
            factors[(rc, measure)] = ft
            bucket_tables.append(bt)
            for scn, v in caps.items():
                rows.append({"risk_class": rc, "measure": measure, "scenario": scn, "capital": v})
    by_class = pd.DataFrame(rows, columns=["risk_class", "measure", "scenario", "capital"])
    totals = by_class.groupby("scenario")["capital"].sum().reindex(P.SCENARIOS).fillna(0.0)
    binding = str(totals.idxmax()) if len(by_class) else "Medium"
    buckets = pd.concat(bucket_tables, ignore_index=True) if bucket_tables else pd.DataFrame()
    return SBMResult(by_class, buckets, factors, totals, binding, float(totals.max()))
