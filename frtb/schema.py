"""
Input file schemas, validation and tenor mapping.

Sensitivity file (one row per sensitivity)
------------------------------------------
trade_id         optional, free text
risk_class       GIRR | CSR_NS | CSR_SEC_NCTP | CSR_SEC_CTP | EQ | COMM | FX
measure          Delta | Vega | Curvature
bucket           GIRR: currency (EUR); FX delta/curvature: currency (USD);
                 FX vega: currency pair (USD/EUR); others: bucket number
qualifier        curve (GIRR), issuer (CSR_NS / CTP), tranche (non-CTP),
                 equity name, commodity name, currency
tenor            delta tenor (GIRR, CSR, COMM) or option maturity (vega), years
underlying_tenor GIRR vega only: residual maturity of the underlying, years
subtype          GIRR: Yield | Inflation | XCCY ; CSR: Bond | CDS ;
                 EQ: Spot | Repo ; COMM: delivery location
amount           delta / vega sensitivity in reporting currency (MAR21.19-25)
cvr_up, cvr_down curvature rows: CVR+ and CVR- (MAR21.5), positive = loss
eu_ets           COMM only: true for EU ETS carbon allowances (CRR3)
covered_aa       CSR_NS bucket 8 only: covered bond rated AA- or better
fx_cross_option  FX curvature only: option not referencing the reporting ccy
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from . import params as P

SENS_COLUMNS = ["trade_id", "risk_class", "measure", "bucket", "qualifier", "tenor",
                "underlying_tenor", "subtype", "amount", "cvr_up", "cvr_down", "eu_ets",
                "covered_aa", "fx_cross_option"]
DRC_COLUMNS = ["position_id", "drc_class", "bucket", "obligor", "seniority", "rating",
               "notional", "market_value", "pnl", "maturity", "instrument", "rw", "zero_rw",
               "hedge_maturity"]
RRAO_COLUMNS = ["position_id", "description", "rrao_type", "notional", "eu_exempt"]

_BOOL = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False,
         "y": True, "n": False, "": False, "nan": False}


def read_table(file) -> pd.DataFrame:
    if hasattr(file, "read"):
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
        return pd.read_csv(io.StringIO(content), sep=None, engine="python")
    return pd.read_csv(file, sep=None, engine="python")


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(_BOOL).fillna(False)


def normalise_sensitivities(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    for c in SENS_COLUMNS:
        if c not in d:
            d[c] = np.nan
    for c in ("eu_ets", "covered_aa", "fx_cross_option"):
        d[c] = _to_bool(d[c])
    for c in ("tenor", "underlying_tenor", "amount", "cvr_up", "cvr_down"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("risk_class", "measure", "subtype", "qualifier", "trade_id"):
        d[c] = d[c].fillna("").astype(str).str.strip()
    d["risk_class"] = d["risk_class"].str.upper()
    d["measure"] = d["measure"].str.capitalize()
    d["bucket"] = d["bucket"].astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    return d[SENS_COLUMNS]


def _grid_for(rc: str, measure: str):
    if measure == "Vega":
        return P.VEGA_TENORS
    if measure == "Delta":
        return {"GIRR": P.GIRR_TENORS, "CSR_NS": P.CSR_TENORS, "CSR_SEC_NCTP": P.CSR_TENORS,
                "CSR_SEC_CTP": P.CSR_TENORS, "COMM": P.COMM_TENORS}.get(rc)
    return None


def map_tenors(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Split sensitivities at non-regulatory tenors onto the two adjacent
    prescribed tenors by linear interpolation (MAR21.8 footnote 5, FAQ2)."""
    out, n_mapped = [], 0
    for _, r in df.iterrows():
        grid = _grid_for(r["risk_class"], r["measure"])
        needs_tenor = grid is not None and not (r["risk_class"] == "GIRR" and r["subtype"] in ("Inflation", "XCCY"))
        t = r["tenor"]
        if not needs_tenor or pd.isna(t) or float(t) in grid:
            out.append(r)
            continue
        n_mapped += 1
        t = float(t)
        if t <= grid[0] or t >= grid[-1]:
            rr = r.copy()
            rr["tenor"] = grid[0] if t <= grid[0] else grid[-1]
            out.append(rr)
            continue
        j = int(np.searchsorted(grid, t))
        lo, hi = grid[j - 1], grid[j]
        w = (t - lo) / (hi - lo)
        for tt, ww in ((lo, 1 - w), (hi, w)):
            rr = r.copy()
            rr["tenor"] = tt
            rr["amount"] = r["amount"] * ww
            out.append(rr)
    return pd.DataFrame(out).reset_index(drop=True), n_mapped


_CCY = re.compile(r"^[A-Z]{3}$")
_PAIR = re.compile(r"^[A-Z]{3}/[A-Z]{3}$")


def validate_sensitivities(df: pd.DataFrame, reporting_ccy: str) -> list[str]:
    errors = []
    bucket_ranges = {"CSR_NS": range(1, 19), "CSR_SEC_CTP": range(1, 17),
                     "CSR_SEC_NCTP": range(1, 26), "EQ": range(1, 14), "COMM": range(1, 12)}
    for i, r in df.iterrows():
        line = i + 2  # header + 1-based
        rc, ms, b, st = r["risk_class"], r["measure"], r["bucket"], r["subtype"]
        if rc not in P.RISK_CLASSES:
            errors.append(f"Row {line}: unknown risk_class '{rc}'")
            continue
        if ms not in P.MEASURES:
            errors.append(f"Row {line}: unknown measure '{ms}'")
            continue
        if rc in bucket_ranges:
            try:
                if int(float(b)) not in bucket_ranges[rc]:
                    raise ValueError
            except ValueError:
                errors.append(f"Row {line}: invalid bucket '{b}' for {rc}")
        elif rc == "GIRR" and not _CCY.match(b):
            errors.append(f"Row {line}: GIRR bucket must be a currency code, got '{b}'")
        elif rc == "FX":
            if ms == "Vega" and not _PAIR.match(b):
                errors.append(f"Row {line}: FX vega bucket must be a currency pair (eg USD/EUR)")
            if ms != "Vega" and (not _CCY.match(b) or b == reporting_ccy):
                errors.append(f"Row {line}: FX bucket must be a non-reporting currency, got '{b}'")
        if ms == "Curvature":
            if pd.isna(r["cvr_up"]) or pd.isna(r["cvr_down"]):
                errors.append(f"Row {line}: curvature rows need cvr_up and cvr_down")
            continue
        if pd.isna(r["amount"]):
            errors.append(f"Row {line}: missing amount")
        grid = _grid_for(rc, ms)
        if rc == "GIRR" and ms == "Delta" and st not in ("", "Yield", "Inflation", "XCCY"):
            errors.append(f"Row {line}: GIRR subtype must be Yield, Inflation or XCCY")
        if rc.startswith("CSR") and ms == "Delta" and st not in ("Bond", "CDS"):
            errors.append(f"Row {line}: CSR subtype must be Bond or CDS")
        if rc == "EQ" and st not in ("", "Spot", "Repo"):
            errors.append(f"Row {line}: EQ subtype must be Spot or Repo")
        if rc == "EQ" and ms == "Vega" and st == "Repo":
            errors.append(f"Row {line}: no vega for equity repo rates (MAR21.12)")
        infl = rc == "GIRR" and st in ("Inflation", "XCCY")
        if grid is not None and not (infl and ms == "Delta"):
            if pd.isna(r["tenor"]):
                errors.append(f"Row {line}: missing tenor")
            elif float(r["tenor"]) not in grid:
                errors.append(f"Row {line}: tenor {r['tenor']} not on the prescribed grid {grid}")
        if rc == "GIRR" and ms == "Vega" and st in ("", "Yield"):
            if pd.isna(r["underlying_tenor"]) or float(r["underlying_tenor"]) not in P.VEGA_TENORS:
                errors.append(f"Row {line}: GIRR vega needs underlying_tenor in {P.VEGA_TENORS}")
        if len(errors) > 50:
            errors.append("… further errors truncated")
            break
    return errors


def normalise_drc(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    for c in DRC_COLUMNS:
        if c not in d:
            d[c] = np.nan
    d["zero_rw"] = _to_bool(d["zero_rw"])
    for c in ("notional", "market_value", "pnl", "maturity", "rw", "hedge_maturity"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("position_id", "drc_class", "bucket", "obligor", "seniority", "rating", "instrument"):
        d[c] = d[c].fillna("").astype(str).str.strip()
    return d[DRC_COLUMNS]


def validate_drc(df: pd.DataFrame) -> list[str]:
    errors = []
    for i, r in df.iterrows():
        line = i + 2
        if r["drc_class"] not in ("NonSec", "SecNonCTP", "CTP"):
            errors.append(f"Row {line}: drc_class must be NonSec, SecNonCTP or CTP")
            continue
        if pd.isna(r["notional"]):
            errors.append(f"Row {line}: missing notional")
        if r["drc_class"] == "NonSec":
            if r["seniority"] not in P.DRC_LGD:
                errors.append(f"Row {line}: seniority must be one of {list(P.DRC_LGD)}")
            if r["rating"] not in P.DRC_RW:
                errors.append(f"Row {line}: rating must be one of {list(P.DRC_RW)}")
        elif r["drc_class"] == "SecNonCTP" and pd.isna(r["rw"]):
            errors.append(f"Row {line}: securitisation positions need a risk weight (rw)")
    return errors


def normalise_rrao(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]
    for c in RRAO_COLUMNS:
        if c not in d:
            d[c] = np.nan
    d["eu_exempt"] = _to_bool(d["eu_exempt"])
    d["notional"] = pd.to_numeric(d["notional"], errors="coerce")
    d["rrao_type"] = d["rrao_type"].fillna("").astype(str).str.strip().str.capitalize()
    return d[RRAO_COLUMNS]


def validate_rrao(df: pd.DataFrame) -> list[str]:
    return [f"Row {i + 2}: rrao_type must be Exotic or Other"
            for i, r in df.iterrows() if r["rrao_type"] not in P.RRAO_RW]
