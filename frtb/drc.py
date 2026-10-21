"""
Default risk capital requirement (MAR22) and residual risk add-on (MAR23).

DRC input: one row per position with columns
    position_id, drc_class (NonSec | SecNonCTP | CTP), bucket, obligor,
    seniority (Equity | NonSenior | Senior | Covered), rating, notional,
    market_value, maturity, instrument, rw, zero_rw, hedge_maturity, pnl
Signs: long credit exposure (loss on default) => positive notional;
short credit exposure => negative notional (MAR22.10, 22.13).

RRAO input: position_id, rrao_type (Exotic | Other), notional, eu_exempt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import params as P

EQUITY_DERIVATIVES = {"equity_derivative", "equity_option", "equity_future", "equity_forward"}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    defaults = {"seniority": "Senior", "rating": "Unrated", "market_value": np.nan,
                "maturity": 1.0, "instrument": "", "rw": np.nan, "zero_rw": False,
                "hedge_maturity": np.nan, "pnl": np.nan, "obligor": "", "bucket": ""}
    for c, v in defaults.items():
        if c not in d:
            d[c] = v
    d["zero_rw"] = d["zero_rw"].fillna(False).astype(bool)
    for c in ("notional", "market_value", "maturity", "rw", "hedge_maturity", "pnl"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["instrument"] = d["instrument"].fillna("").astype(str)
    return d


def effective_maturity(row: pd.Series, regime: P.Regime) -> float:
    """Maturity used for scaling (MAR22.15-22.18), with DR 2026/1221 Art. 495q."""
    m = row["maturity"]
    if regime.eu:
        if regime.eu_drc_equity_deriv_3m and row["instrument"] in EQUITY_DERIVATIVES:
            m = 0.25                                   # Art. 495q(b)
        elif (regime.eu_drc_cash_hedge_maturity and row["instrument"] == "cash_equity_hedge"
              and pd.notna(row["hedge_maturity"])):
            m = row["hedge_maturity"]                  # Art. 495q(a)
    if pd.isna(m):
        m = 1.0
    return float(min(max(m, 0.25), 1.0))


def gross_jtd(d: pd.DataFrame, regime: P.Regime) -> pd.DataFrame:
    out = d.copy()
    pnl = np.where(out["pnl"].notna(), out["pnl"], out["market_value"].fillna(out["notional"]) - out["notional"])
    nonsec = out["drc_class"] == "NonSec"
    lgd = out["seniority"].map(P.DRC_LGD).fillna(0.75)
    raw = np.where(nonsec, lgd * out["notional"] + pnl,
                   out["market_value"].fillna(out["notional"]))  # MAR22.27, 22.36-22.37
    # direction: notional sign (MAR22.13); for zero-notional instruments
    # (eg bought / sold calls) the sign of the loss-on-default term decides
    is_long = (out["notional"] > 0) | ((out["notional"] == 0) & (raw > 0))
    out["jtd_gross"] = np.where(is_long, np.maximum(raw, 0.0), np.minimum(raw, 0.0))
    out["lgd"] = np.where(nonsec, lgd, np.nan)
    out["pnl_used"] = pnl
    out["m_scale"] = [effective_maturity(r, regime) for _, r in out.iterrows()]
    out["jtd_scaled"] = out["jtd_gross"] * out["m_scale"]
    return out


def _net_by_seniority(g: pd.DataFrame) -> tuple[float, float]:
    """Offset longs and shorts of one obligor; a short may offset a long of
    equal or higher seniority (MAR22.19(1)). Returns (net long, net short)."""
    longs = g[g["jtd_scaled"] > 0].groupby("rank")["jtd_scaled"].sum()
    shorts = -g[g["jtd_scaled"] < 0].groupby("rank")["jtd_scaled"].sum()
    pool, residual_short = 0.0, 0.0
    for r in sorted(set(longs.index) | set(shorts.index), reverse=True):
        pool += longs.get(r, 0.0)
        s = shorts.get(r, 0.0)
        take = min(pool, s)
        pool -= take
        residual_short += s - take
    return pool, -residual_short


def _risk_weight(row: pd.Series, regime: P.Regime) -> float:
    if row["drc_class"] == "NonSec" or (row["drc_class"] == "CTP" and pd.isna(row["rw"])):
        if row["zero_rw"] and (regime.eu and regime.eu_zero_rw_sovereigns):
            return 0.0
        return P.DRC_RW.get(str(row["rating"]), P.DRC_RW["Unrated"])
    return float(row["rw"])


def net_jtd(g: pd.DataFrame, regime: P.Regime) -> pd.DataFrame:
    rows = []
    g = g.copy()
    g["rank"] = g["seniority"].map(P.SENIORITY_RANK).fillna(2)
    for (cls, bucket, obligor), grp in g.groupby(["drc_class", "bucket", "obligor"], sort=True):
        if cls == "NonSec":
            nl, ns = _net_by_seniority(grp)
        else:  # securitisations: offset only identical tranches (obligor = tranche id)
            tot = grp["jtd_scaled"].sum()
            nl, ns = max(tot, 0.0), min(tot, 0.0)
        first = grp.iloc[0]
        rw = _risk_weight(first, regime)
        rows.append({"drc_class": cls, "bucket": bucket, "obligor": obligor,
                     "rating": first["rating"], "rw": rw, "net_long": nl, "net_short": ns})
    return pd.DataFrame(rows)


@dataclass
class DRCResult:
    positions: pd.DataFrame
    net: pd.DataFrame
    buckets: pd.DataFrame
    by_class: pd.Series
    capital: float


def compute_drc(df: pd.DataFrame, regime: P.Regime) -> DRCResult:
    if df is None or df.empty:
        empty = pd.Series({"NonSec": 0.0, "SecNonCTP": 0.0, "CTP": 0.0})
        return DRCResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), empty, 0.0)
    d = gross_jtd(_prepare(df), regime)
    net = net_jtd(d, regime)
    bucket_rows = []
    by_class = {}
    for cls in ["NonSec", "SecNonCTP", "CTP"]:
        n = net[net["drc_class"] == cls]
        if n.empty:
            by_class[cls] = 0.0
            continue
        if cls == "CTP":  # HBR computed across the whole CTP (MAR22.44)
            L, S = n["net_long"].sum(), -n["net_short"].sum()
            hbr_ctp = L / (L + S) if (L + S) > 0 else 0.0
        total_terms = []
        for b, nb in n.groupby("bucket", sort=True):
            L, S = nb["net_long"].sum(), -nb["net_short"].sum()
            hbr = hbr_ctp if cls == "CTP" else (L / (L + S) if (L + S) > 0 else 0.0)
            rwl = float((nb["rw"] * nb["net_long"]).sum())
            rws = float((nb["rw"] * -nb["net_short"]).sum())
            raw = rwl - hbr * rws
            drc_b = raw if cls == "CTP" else max(raw, 0.0)  # no floor for CTP buckets
            bucket_rows.append({"drc_class": cls, "bucket": b, "net_long": L, "net_short": -S,
                                "HBR": hbr, "RW_long": rwl, "RW_short": rws, "DRC_b": drc_b})
            total_terms.append(drc_b)
        if cls == "CTP":  # MAR22.45
            by_class[cls] = max(sum(max(x, 0.0) + 0.5 * min(x, 0.0) for x in total_terms), 0.0)
        else:  # MAR22.26, 22.35
            by_class[cls] = float(sum(total_terms))
    s = pd.Series(by_class)
    return DRCResult(d, net, pd.DataFrame(bucket_rows), s, float(s.sum()))  # MAR22.4


# ==========================================================================
# RRAO
# ==========================================================================


@dataclass
class RRAOResult:
    positions: pd.DataFrame
    by_type: pd.Series
    capital: float


def compute_rrao(df: pd.DataFrame, regime: P.Regime) -> RRAOResult:
    if df is None or df.empty:
        return RRAOResult(pd.DataFrame(), pd.Series({"Exotic": 0.0, "Other": 0.0}), 0.0)
    d = df.copy()
    if "eu_exempt" not in d:
        d["eu_exempt"] = False
    d["eu_exempt"] = d["eu_exempt"].fillna(False).astype(bool)
    d["rw"] = d["rrao_type"].map(P.RRAO_RW)
    d["multiplier"] = np.where(regime.eu & regime.eu_rrao_exemptions & d["eu_exempt"], 0.0, 1.0)
    d["rrao"] = d["notional"].abs() * d["rw"] * d["multiplier"]  # MAR23.8 gross notional
    by_type = d.groupby("rrao_type")["rrao"].sum().reindex(["Exotic", "Other"]).fillna(0.0)
    return RRAOResult(d, by_type, float(d["rrao"].sum()))
