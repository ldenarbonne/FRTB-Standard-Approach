"""
Regulatory parameters of the FRTB standardised approach.

Sources
-------
* BCBS, Basel Framework, MAR21 (sensitivities-based method), MAR22 (default
  risk capital) and MAR23 (residual risk add-on), version effective
  1 January 2023 incorporating FAQs up to 23 March 2026. Paragraph
  references are given next to each parameter.
* EU: Regulation (EU) 2024/1623 (CRR3) and Commission Delegated Regulation
  (EU) 2026/1221 of 4 June 2026 (transitional provisions, Articles 495i to
  495v of the CRR), applicable from 1 January 2027 to 31 December 2029.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

SQRT2 = math.sqrt(2.0)

RISK_CLASSES = ["GIRR", "CSR_NS", "CSR_SEC_NCTP", "CSR_SEC_CTP", "EQ", "COMM", "FX"]
RISK_CLASS_LABEL = {
    "GIRR": "General interest rate risk",
    "CSR_NS": "Credit spread – non-securitisation",
    "CSR_SEC_NCTP": "Credit spread – securitisation (non-CTP)",
    "CSR_SEC_CTP": "Credit spread – securitisation (CTP)",
    "EQ": "Equity",
    "COMM": "Commodity",
    "FX": "Foreign exchange",
}
MEASURES = ["Delta", "Vega", "Curvature"]

# --------------------------------------------------------------------------
# GIRR (MAR21.8, 21.41-21.50)
# --------------------------------------------------------------------------
GIRR_TENORS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]
GIRR_RW = dict(zip(GIRR_TENORS, [0.017, 0.017, 0.016, 0.013, 0.012,
                                 0.011, 0.011, 0.011, 0.011, 0.011]))   # Table 1
GIRR_RW_INFLATION = 0.016                                            # 21.43
GIRR_RW_XCCY = 0.016                                                 # 21.43
GIRR_SPECIFIED_CCY = {"EUR", "USD", "GBP", "AUD", "JPY", "SEK", "CAD"}  # 21.44 fn 14
GIRR_THETA = 0.03                                                    # 21.46 fn 15
GIRR_RHO_FLOOR = 0.40
GIRR_DIFF_CURVE = 0.999                                              # 21.45 / 21.47
GIRR_INFL_YIELD = 0.40                                               # 21.48
GIRR_XCCY = 0.0                                                      # 21.49
GIRR_GAMMA = 0.50                                                    # 21.50

# --------------------------------------------------------------------------
# CSR non-securitisation (MAR21.51-21.57)
# --------------------------------------------------------------------------
CSR_TENORS = [0.5, 1.0, 3.0, 5.0, 10.0]
CSR_NS_RW = {1: 0.005, 2: 0.010, 3: 0.050, 4: 0.030, 5: 0.030, 6: 0.020,
             7: 0.015, 8: 0.025, 9: 0.020, 10: 0.040, 11: 0.120, 12: 0.070,
             13: 0.085, 14: 0.055, 15: 0.050, 16: 0.120, 17: 0.015, 18: 0.050}  # Table 4
CSR_NS_RW_COVERED_AA = 0.015                                         # 21.53 fn 19
CSR_NS_BUCKET_LABEL = {
    1: "IG – Sovereigns, central banks, MDBs",
    2: "IG – Local government, public administration",
    3: "IG – Financials",
    4: "IG – Basic materials, energy, industrials",
    5: "IG – Consumer goods and services, transport",
    6: "IG – Technology, telecommunications",
    7: "IG – Health care, utilities, professional",
    8: "IG – Covered bonds",
    9: "HY/NR – Sovereigns, central banks, MDBs",
    10: "HY/NR – Local government, public administration",
    11: "HY/NR – Financials",
    12: "HY/NR – Basic materials, energy, industrials",
    13: "HY/NR – Consumer goods and services, transport",
    14: "HY/NR – Technology, telecommunications",
    15: "HY/NR – Health care, utilities, professional",
    16: "Other sector",
    17: "IG indices",
    18: "HY indices",
}
CSR_NS_RHO_NAME = 0.35        # 21.54
CSR_NS_RHO_NAME_INDEX = 0.80  # 21.55 (buckets 17-18)
CSR_NS_RHO_TENOR = 0.65
CSR_NS_RHO_BASIS = 0.999
CSR_NS_OTHER = 16             # 21.56

# Sector correlations across buckets, Table 5 (MAR21.57). Keys are "sector
# groups": 1 (1/9), 2 (2/10) ... 7 (7/15), 8, 16, 17, 18.
_CSR_SECTOR_UPPER = {
    (1, 2): 0.75, (1, 3): 0.10, (1, 4): 0.20, (1, 5): 0.25, (1, 6): 0.20, (1, 7): 0.15, (1, 8): 0.10,
    (2, 3): 0.05, (2, 4): 0.15, (2, 5): 0.20, (2, 6): 0.15, (2, 7): 0.10, (2, 8): 0.10,
    (3, 4): 0.05, (3, 5): 0.15, (3, 6): 0.20, (3, 7): 0.05, (3, 8): 0.20,
    (4, 5): 0.20, (4, 6): 0.25, (4, 7): 0.05, (4, 8): 0.05,
    (5, 6): 0.25, (5, 7): 0.05, (5, 8): 0.15,
    (6, 7): 0.05, (6, 8): 0.20,
    (7, 8): 0.05,
    (17, 18): 0.75,
}
for _g in [1, 2, 3, 4, 5, 6, 7, 8]:
    _CSR_SECTOR_UPPER[(_g, 16)] = 0.0
    _CSR_SECTOR_UPPER[(_g, 17)] = 0.45
    _CSR_SECTOR_UPPER[(_g, 18)] = 0.45
_CSR_SECTOR_UPPER[(16, 17)] = 0.0
_CSR_SECTOR_UPPER[(16, 18)] = 0.0


def csr_sector_group(bucket: int) -> int:
    if 1 <= bucket <= 7:
        return bucket
    if 9 <= bucket <= 15:
        return bucket - 8
    return bucket  # 8, 16, 17, 18


def csr_ns_gamma(b: int, c: int) -> float:
    """Cross-bucket correlation gamma_bc = gamma_rating * gamma_sector (MAR21.57)."""
    if b == c:
        return 1.0
    rating = 1.0
    if 1 <= b <= 15 and 1 <= c <= 15:
        ig_b, ig_c = b <= 8, c <= 8
        if ig_b != ig_c:
            rating = 0.50
    gb, gc = csr_sector_group(b), csr_sector_group(c)
    if gb == gc:
        sector = 1.0
    else:
        key = (min(gb, gc), max(gb, gc))
        sector = _CSR_SECTOR_UPPER[key]
    return rating * sector


# --------------------------------------------------------------------------
# CSR securitisation CTP (MAR21.58-21.61)
# --------------------------------------------------------------------------
CSR_CTP_RW = {1: 0.04, 2: 0.04, 3: 0.08, 4: 0.05, 5: 0.04, 6: 0.03, 7: 0.02, 8: 0.06,
              9: 0.13, 10: 0.13, 11: 0.16, 12: 0.10, 13: 0.12, 14: 0.12, 15: 0.12,
              16: 0.13}                                              # Table 6
CSR_CTP_RHO_BASIS = 0.99                                             # 21.60

# --------------------------------------------------------------------------
# CSR securitisation non-CTP (MAR21.62-21.71)
# --------------------------------------------------------------------------
_SEC_SENIOR_RW = {1: 0.009, 2: 0.015, 3: 0.020, 4: 0.020, 5: 0.008, 6: 0.012, 7: 0.012,
                  8: 0.014}                                          # Table 8
CSR_SEC_NCTP_RW = dict(_SEC_SENIOR_RW)
CSR_SEC_NCTP_RW.update({b + 8: rw * 1.25 for b, rw in _SEC_SENIOR_RW.items()})   # 21.65
CSR_SEC_NCTP_RW.update({b + 16: rw * 1.75 for b, rw in _SEC_SENIOR_RW.items()})  # 21.66
CSR_SEC_NCTP_RW[25] = 0.035                                                       # 21.67
_SEC_SECTORS = ["RMBS – Prime", "RMBS – Mid-prime", "RMBS – Sub-prime", "CMBS",
                "ABS – Student loans", "ABS – Credit cards", "ABS – Auto", "CLO non-CTP"]
CSR_SEC_NCTP_BUCKET_LABEL = {}
for _i, _s in enumerate(_SEC_SECTORS, start=1):
    CSR_SEC_NCTP_BUCKET_LABEL[_i] = f"Senior IG – {_s}"
    CSR_SEC_NCTP_BUCKET_LABEL[_i + 8] = f"Non-senior IG – {_s}"
    CSR_SEC_NCTP_BUCKET_LABEL[_i + 16] = f"HY/NR – {_s}"
CSR_SEC_NCTP_BUCKET_LABEL[25] = "Other sector"
CSR_SEC_RHO_TRANCHE = 0.40                                           # 21.68
CSR_SEC_RHO_TENOR = 0.80
CSR_SEC_RHO_BASIS = 0.999
CSR_SEC_OTHER = 25                                                   # 21.69-21.71

# --------------------------------------------------------------------------
# Equity (MAR21.72-21.80)
# --------------------------------------------------------------------------
EQ_RW_SPOT = {1: 0.55, 2: 0.60, 3: 0.45, 4: 0.55, 5: 0.30, 6: 0.35, 7: 0.40, 8: 0.50,
              9: 0.70, 10: 0.50, 11: 0.70, 12: 0.15, 13: 0.25}       # Table 10
EQ_RW_REPO = {b: rw / 100 for b, rw in EQ_RW_SPOT.items()}
EQ_BUCKET_LABEL = {
    1: "Large EM – Consumer, transport, healthcare, utilities",
    2: "Large EM – Telecoms, industrials",
    3: "Large EM – Materials, energy, agriculture",
    4: "Large EM – Financials, real estate, technology",
    5: "Large AE – Consumer, transport, healthcare, utilities",
    6: "Large AE – Telecoms, industrials",
    7: "Large AE – Materials, energy, agriculture",
    8: "Large AE – Financials, real estate, technology",
    9: "Small EM – All sectors",
    10: "Small AE – All sectors",
    11: "Other sector",
    12: "Large AE indices",
    13: "Other indices",
}
EQ_SPOT_REPO_SAME_NAME = 0.999                                       # 21.78(1)


def eq_name_rho(bucket: int) -> float:
    """Correlation between two different names in the same bucket (MAR21.78(2))."""
    if 1 <= bucket <= 4:
        return 0.15
    if 5 <= bucket <= 8:
        return 0.25
    if bucket == 9:
        return 0.075
    if bucket == 10:
        return 0.125
    if bucket in (12, 13):
        return 0.80
    raise ValueError(f"No intra-bucket correlation for equity bucket {bucket}")


def eq_gamma(b: int, c: int) -> float:
    """MAR21.80."""
    if b == c:
        return 1.0
    if b == 11 or c == 11:
        return 0.0
    if b <= 10 and c <= 10:
        return 0.15
    if {b, c} == {12, 13}:
        return 0.75
    return 0.45


EQ_OTHER = 11

# --------------------------------------------------------------------------
# Commodity (MAR21.81-21.85)
# --------------------------------------------------------------------------
COMM_TENORS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]  # 21.13
COMM_RW = {1: 0.30, 2: 0.35, 3: 0.60, 4: 0.80, 5: 0.40, 6: 0.45, 7: 0.20, 8: 0.35,
           9: 0.25, 10: 0.35, 11: 0.50}                              # Table 11
COMM_BUCKET_LABEL = {
    1: "Energy – solid combustibles", 2: "Energy – liquid combustibles",
    3: "Energy – electricity and carbon trading", 4: "Freight",
    5: "Metals – non-precious", 6: "Gaseous combustibles",
    7: "Precious metals (incl. gold)", 8: "Grains and oilseed",
    9: "Livestock and dairy", 10: "Softs and other agriculturals",
    11: "Other commodity",
}
COMM_RHO_CTY = {1: 0.55, 2: 0.95, 3: 0.40, 4: 0.80, 5: 0.60, 6: 0.65, 7: 0.55, 8: 0.45,
                9: 0.15, 10: 0.40, 11: 0.15}                         # Table 12
COMM_RHO_TENOR = 0.99                                                # 21.83
COMM_RHO_BASIS = 0.999


def comm_gamma(b: int, c: int) -> float:
    """MAR21.85."""
    if b == c:
        return 1.0
    return 0.0 if (b == 11 or c == 11) else 0.20


# --------------------------------------------------------------------------
# FX (MAR21.86-21.89)
# --------------------------------------------------------------------------
FX_RW = 0.15
FX_SPECIFIED = {"USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "MXN", "CNY", "NZD", "RUB",
                "HKD", "SGD", "TRY", "KRW", "SEK", "ZAR", "INR", "NOK", "BRL"}  # 21.88 fn 24-25
FX_GAMMA = 0.60
FX_CURVATURE_SCALAR = 1.5                                            # 21.98

# --------------------------------------------------------------------------
# Vega (MAR21.90-21.95)
# --------------------------------------------------------------------------
VEGA_TENORS = [0.5, 1.0, 3.0, 5.0, 10.0]
VEGA_ALPHA = 0.01                                                    # 21.93
VEGA_RW_SIGMA = 0.55                                                 # Table 13 fn 26
VEGA_LH = {"GIRR": 60, "CSR_NS": 120, "CSR_SEC_CTP": 120, "CSR_SEC_NCTP": 120,
           "EQ_LARGE": 20, "EQ_SMALL": 60, "COMM": 120, "FX": 40}


def vega_rw(risk_class: str, bucket: int | str | None = None) -> float:
    if risk_class == "EQ":
        key = "EQ_SMALL" if int(bucket) in (9, 10, 11) else "EQ_LARGE"
    else:
        key = risk_class
    return min(VEGA_RW_SIGMA * math.sqrt(VEGA_LH[key] / 10.0), 1.0)


# --------------------------------------------------------------------------
# Default risk capital (MAR22)
# --------------------------------------------------------------------------
DRC_RW = {"AAA": 0.005, "AA": 0.02, "A": 0.03, "BBB": 0.06, "BB": 0.15, "B": 0.30,
          "CCC": 0.50, "Unrated": 0.15, "Defaulted": 1.00}           # 22.24 Table 2
DRC_LGD = {"Equity": 1.00, "NonSenior": 1.00, "Senior": 0.75, "Covered": 0.25}  # 22.12
SENIORITY_RANK = {"Equity": 0, "NonSenior": 1, "Senior": 2, "Covered": 3}
DRC_NONSEC_BUCKETS = ["Corporates", "Sovereigns", "Local governments"]  # 22.22

# --------------------------------------------------------------------------
# RRAO (MAR23.8)
# --------------------------------------------------------------------------
RRAO_RW = {"Exotic": 0.010, "Other": 0.001}

# --------------------------------------------------------------------------
# Correlation scenarios (MAR21.6)
# --------------------------------------------------------------------------
SCENARIOS = ["Low", "Medium", "High"]


def scenario_corr(rho, scenario: str):
    """Apply the low / medium / high correlation scenario (works on arrays)."""
    import numpy as np

    rho = np.asarray(rho, dtype=float)
    if scenario == "Medium":
        return rho
    if scenario == "High":
        return np.minimum(1.25 * rho, 1.0)
    if scenario == "Low":
        return np.maximum(2.0 * rho - 1.0, 0.75 * rho)
    raise ValueError(scenario)


# --------------------------------------------------------------------------
# Regime configuration
# --------------------------------------------------------------------------


@dataclass
class Regime:
    """Calculation options. ``eu`` switches on the EU transitional provisions."""

    name: str = "BCBS"
    reporting_ccy: str = "EUR"
    girr_sqrt2: bool = True           # MAR21.44 bank discretion
    fx_sqrt2: bool = True             # MAR21.88 bank discretion
    covered_aa_reduced_rw: bool = True  # MAR21.53 fn 19 discretion
    fx_curvature_scalar: bool = True  # MAR21.98 discretion
    csr_curvature_floor: bool = True  # FAQ2 to MAR21.99
    # --- EU specific -------------------------------------------------------
    eu: bool = False
    eu_ets_rw: float = 0.40           # CRR3 Art. 325as: EU ETS carbon trading
    eu_ets_tenor_divisor: float = 0.991  # DR 2026/1221 Art. 495r
    eu_sbm_multiplier: float = 0.90   # DR 2026/1221 Art. 495s(1)
    eu_rrao_exemptions: bool = True   # DR 2026/1221 Art. 495p
    eu_drc_cash_hedge_maturity: bool = True   # DR 2026/1221 Art. 495q(a)
    eu_drc_equity_deriv_3m: bool = False      # DR 2026/1221 Art. 495q(b)
    eu_zero_rw_sovereigns: bool = True   # CRR3 Art. 325y / MAR22.7 discretion
    eu_bank_multiplier: float = 1.0   # DR 2026/1221 Art. 495v (bank-specific, <= 1)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def bcbs(cls, **kw) -> "Regime":
        return cls(name="BCBS (MAR21-23)", eu=False, **kw)

    @classmethod
    def eu_transitional(cls, **kw) -> "Regime":
        return cls(name="EU – CRR3 + DR 2026/1221", eu=True, **kw)
