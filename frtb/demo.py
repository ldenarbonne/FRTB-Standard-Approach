"""
Demonstration market and multi-asset trading book (EUR reporting currency).

All market levels and names are fictitious and illustrative; they are chosen
to produce realistic sensitivity profiles across the seven risk classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P
from .pricing import (CDS, Bond, CommCurve, CommodityFuture, CommodityOption, Curve,
                      EquityForward, EquityOption, EquityPosition, FXForward, FXOption,
                      InflationSwap, IRSwap, Market, SpreadCurve, Swaption,
                      portfolio_sensitivities)

T = P.GIRR_TENORS
CS = P.CSR_TENORS
CT = P.COMM_TENORS


def demo_market() -> Market:
    m = Market(reporting_ccy="EUR")
    m.ir = {
        "EUR-ESTR": Curve("EUR", T, [0.0195, 0.0198, 0.0203, 0.0212, 0.0222, 0.0240, 0.0268, 0.0282, 0.0286, 0.0280]),
        "EUR-EURIBOR6M": Curve("EUR", T, [0.0215, 0.0220, 0.0226, 0.0234, 0.0244, 0.0262, 0.0290, 0.0303, 0.0306, 0.0299]),
        "USD-SOFR": Curve("USD", T, [0.0370, 0.0362, 0.0350, 0.0338, 0.0335, 0.0340, 0.0360, 0.0375, 0.0383, 0.0380]),
        "GBP-SONIA": Curve("GBP", T, [0.0360, 0.0352, 0.0340, 0.0330, 0.0328, 0.0335, 0.0365, 0.0385, 0.0395, 0.0392]),
        "JPY-TONA": Curve("JPY", T, [0.0060, 0.0063, 0.0070, 0.0082, 0.0095, 0.0115, 0.0160, 0.0195, 0.0220, 0.0245]),
    }
    m.inflation = {"EUR-HICPxT": ("EUR", 0.0205)}
    m.ir_normal_vol = {"EUR": 0.0085, "USD": 0.0105}
    # credit curves (bond and CDS basis), spreads in decimal
    m.credit = {
        ("EnergyCo", "Bond"): SpreadCurve(CS, [0.0060, 0.0068, 0.0090, 0.0110, 0.0135]),
        ("EnergyCo", "CDS"): SpreadCurve(CS, [0.0055, 0.0062, 0.0085, 0.0105, 0.0128]),
        ("BankCo", "Bond"): SpreadCurve(CS, [0.0045, 0.0050, 0.0068, 0.0082, 0.0100]),
        ("BankCo", "CDS"): SpreadCurve(CS, [0.0042, 0.0048, 0.0065, 0.0080, 0.0098]),
        ("TelcoHY", "Bond"): SpreadCurve(CS, [0.0180, 0.0195, 0.0240, 0.0275, 0.0310]),
        ("SovereignA", "Bond"): SpreadCurve(CS, [0.0040, 0.0048, 0.0070, 0.0090, 0.0120]),
        ("CoveredBankCo", "Bond"): SpreadCurve(CS, [0.0012, 0.0015, 0.0022, 0.0030, 0.0040]),
        ("RMBS-NL-2026-A1", "Bond"): SpreadCurve(CS, [0.0050, 0.0055, 0.0065, 0.0075, 0.0090]),
        ("ITRAXX-EUR-S46", "CDS"): SpreadCurve(CS, [0.0030, 0.0036, 0.0048, 0.0060, 0.0080]),
    }
    m.credit_meta = {
        "EnergyCo": dict(risk_class="CSR_NS", bucket=4, rating="BBB", seniority="Senior", drc_bucket="Corporates"),
        "BankCo": dict(risk_class="CSR_NS", bucket=3, rating="A", seniority="Senior", drc_bucket="Corporates"),
        "TelcoHY": dict(risk_class="CSR_NS", bucket=14, rating="BB", seniority="Senior", drc_bucket="Corporates"),
        "SovereignA": dict(risk_class="CSR_NS", bucket=1, rating="BBB", seniority="Senior", drc_bucket="Sovereigns", zero_rw=True),
        "CoveredBankCo": dict(risk_class="CSR_NS", bucket=8, rating="AA", seniority="Covered", drc_bucket="Corporates", covered_aa=True),
        "RMBS-NL-2026-A1": dict(risk_class="CSR_SEC_NCTP", bucket=1, rating="AAA", seniority="Senior",
                                 drc_bucket="RMBS | Europe", drc_rw=0.016),
        # stylised CTP: index tranche and its index hedge, CSR mapped to bucket 16 (FAQ to MAR21.34)
        "ITRAXX-EUR-S46": dict(risk_class="CSR_SEC_CTP", bucket=16, rating="A", drc_bucket="iTraxx Europe IG",
                                csr_qualifier="ITRAXX-EUR-S46", drc_rw=0.08, tranche_id="ITRAXX-EUR-S46 3-6%"),
    }
    m.eq_spot = {"TechAE": 180.0, "ConsumerAE": 95.0, "BankAE": 42.0, "MiningEM": 23.0,
                 "SmallCapAE": 14.0, "EU-LARGECAP-IDX": 5200.0}
    m.eq_vol = {"TechAE": 0.28, "ConsumerAE": 0.22, "BankAE": 0.30, "MiningEM": 0.38,
                "SmallCapAE": 0.40, "EU-LARGECAP-IDX": 0.17}
    m.eq_repo = {"TechAE": 0.0040, "ConsumerAE": 0.0035, "BankAE": 0.0045,
                 "MiningEM": 0.0080, "SmallCapAE": 0.0090, "EU-LARGECAP-IDX": 0.0030}
    m.eq_bucket = {"TechAE": 8, "ConsumerAE": 5, "BankAE": 8, "MiningEM": 3,
                   "SmallCapAE": 10, "EU-LARGECAP-IDX": 12}
    m.fx = {"USD": 0.865, "GBP": 1.150, "JPY": 0.00575}
    m.fx_vol = {"USD/EUR": 0.075, "USD/JPY": 0.095}
    m.comm = {
        "Brent": CommCurve(2, "ICE Rotterdam", CT, [78, 77.6, 77.1, 76.2, 74.8, 73.9, 72.5, 71.0, 70.5, 70.2, 70.0]),
        "WTI": CommCurve(2, "NYMEX Cushing", CT, [74, 73.5, 73.0, 72.0, 70.6, 69.8, 68.6, 67.5, 67.0, 66.8, 66.5]),
        "Gold": CommCurve(7, "LBMA London", CT, [2650, 2665, 2680, 2710, 2770, 2830, 2950, 3250, 3550, 3850, 4400]),
        "EUA": CommCurve(3, "ICE Endex", CT, [72, 72.5, 73.0, 74.4, 77.0, 79.6, 85.0, 99.0, 115, 130, 160], eu_ets=True),
        "TTF-Gas": CommCurve(6, "TTF Netherlands", CT, [38, 40, 41, 36, 31, 28, 26, 25, 25, 25, 25]),
        "DE-Power-Base": CommCurve(3, "EEX Germany", CT, [85, 88, 90, 86, 80, 76, 72, 70, 70, 70, 70]),
    }
    m.comm_vol = {"TTF-Gas": 0.55, "Brent": 0.32}
    return m


def demo_instruments() -> list:
    return [
        # ---- rates ------------------------------------------------------------
        IRSwap("IRS-EUR-10Y", "EUR", 50e6, 0.0270, 10, "EUR-ESTR", "EUR-EURIBOR6M",
               description="EUR payer swap 10Y vs 6M Euribor (ESTR discounting)"),
        IRSwap("IRS-USD-5Y", "USD", -40e6, 0.0335, 5, "USD-SOFR", "USD-SOFR",
               description="USD receiver swap 5Y vs SOFR"),
        IRSwap("IRS-GBP-3Y", "GBP", 30e6, 0.0330, 3, "GBP-SONIA", "GBP-SONIA",
               description="GBP payer swap 3Y vs SONIA"),
        Swaption("SWPT-EUR-1Yx5Y", "EUR", 100e6, 0.0255, 1.0, 5.0, "EUR-ESTR",
                 description="Long EUR payer swaption 1Y x 5Y"),
        Swaption("SWPT-EUR-2Yx10Y-SOLD", "EUR", -100e6, 0.0290, 2.0, 10.0, "EUR-ESTR",
                 description="Short EUR payer swaption 2Y x 10Y"),
        InflationSwap("INFL-EUR-7Y", "EUR", 25e6, 0.0200, 7, "EUR-ESTR", "EUR-HICPxT",
                      description="EUR zero-coupon inflation swap 7Y (receive HICP)"),
        # ---- credit -----------------------------------------------------------
        Bond("BOND-ENERGYCO-5Y", "EUR", 20e6, 0.0375, 5, "EUR-ESTR", "EnergyCo",
             description="EnergyCo senior bond 5Y, BBB (CSR bucket 4)"),
        Bond("BOND-BANKCO-3Y", "EUR", 15e6, 0.0300, 3, "EUR-ESTR", "BankCo",
             description="BankCo senior bond 3Y, A (CSR bucket 3)"),
        Bond("BOND-TELCOHY-7Y", "EUR", 10e6, 0.0550, 7, "EUR-ESTR", "TelcoHY",
             description="TelcoHY senior bond 7Y, BB (CSR bucket 14)"),
        Bond("BOND-SOVA-3Y", "EUR", 30e6, 0.0290, 3, "EUR-ESTR", "SovereignA",
             description="SovereignA government bond 3Y (CSR bucket 1)"),
        Bond("COVERED-BANKCO-5Y", "EUR", 10e6, 0.0260, 5, "EUR-ESTR", "CoveredBankCo",
             description="Covered bond 5Y, AA (CSR bucket 8)"),
        CDS("CDS-BANKCO-5Y-BUY", "EUR", -20e6, 0.0100, 5, "EUR-ESTR", "BankCo",
            description="CDS protection bought on BankCo 5Y"),
        CDS("CDS-ENERGYCO-5Y-SELL", "EUR", 10e6, 0.0100, 5, "EUR-ESTR", "EnergyCo",
            description="CDS protection sold on EnergyCo 5Y"),
        # ---- securitisation ---------------------------------------------------
        Bond("RMBS-NL-A1", "EUR", 12e6, 0.0310, 4, "EUR-ESTR", "RMBS-NL-2026-A1",
             description="Senior AAA Dutch RMBS tranche (non-CTP bucket 1)"),
        CDS("CTP-ITRAXX-3-6", "EUR", 8e6, 0.0100, 5, "EUR-ESTR", "ITRAXX-EUR-S46", leverage=3.0,
            description="Stylised iTraxx 3-6% tranche, protection sold (CTP)"),
        CDS("CTP-ITRAXX-IDX-HEDGE", "EUR", -20e6, 0.0100, 5, "EUR-ESTR", "ITRAXX-EUR-S46",
            description="iTraxx Europe index protection bought (CTP hedge)"),
        # ---- equity -----------------------------------------------------------
        EquityPosition("EQ-TECHAE", "EUR", "TechAE", 40_000, description="Long TechAE shares (bucket 8)"),
        EquityPosition("EQ-BANKAE-SHORT", "EUR", "BankAE", -60_000, description="Short BankAE shares (bucket 8)"),
        EquityPosition("EQ-CONSUMERAE", "EUR", "ConsumerAE", 20_000,
                       description="Long ConsumerAE shares hedging the 6M forward (bucket 5)",
                       instrument_type="cash_equity_hedge", hedge_maturity=0.5),
        EquityPosition("EQ-MININGEM", "USD", "MiningEM", 80_000, description="Long MiningEM shares, USD (bucket 3)"),
        EquityPosition("EQ-SMALLCAP", "EUR", "SmallCapAE", 100_000, description="Long SmallCapAE shares (bucket 10)"),
        EquityForward("EQFWD-CONSUMERAE-6M", "EUR", "ConsumerAE", -20_000, 95.5, 0.5, "EUR-ESTR",
                      description="Short 6M forward on ConsumerAE (repo exposure)"),
        EquityOption("EQOPT-BANKAE-P-SOLD", "EUR", "BankAE", -150_000, 40.0, 0.5, "EUR-ESTR", False,
                     description="Short 6M put on BankAE"),
        EquityOption("EQOPT-TECHAE-C", "EUR", "TechAE", 25_000, 190.0, 1.0, "EUR-ESTR", True,
                     description="Long 1Y call on TechAE"),
        EquityOption("EQOPT-IDX-P", "EUR", "EU-LARGECAP-IDX", 2_000, 4900.0, 0.5, "EUR-ESTR", False,
                     description="Long 6M put on EU large-cap index (bucket 12)"),
        # ---- FX ---------------------------------------------------------------
        FXForward("FXFWD-USDJPY-1Y", "USD", 10e6, "JPY", 10e6 * 151.0, 1.0, "USD-SOFR", "JPY-TONA",
                  description="Buy USD / sell JPY 1Y forward"),
        FXOption("FXOPT-USDEUR-C", "USD", "EUR", 15e6, 0.88, 0.5, "EUR-ESTR", "USD-SOFR", True,
                 description="Long 6M USD call / EUR put"),
        FXOption("FXOPT-USDJPY-C-SOLD", "USD", "JPY", -8e6, 152.0, 0.5, "JPY-TONA", "USD-SOFR", True,
                 description="Short 6M USD call / JPY put (cross not referencing EUR)"),
        # ---- commodities ------------------------------------------------------
        CommodityFuture("FUT-BRENT-1Y", "USD", "Brent", 100_000, 76.0, 1.0, "USD-SOFR",
                        description="Long Brent 1Y futures (100k bbl)"),
        CommodityFuture("FUT-WTI-1Y", "USD", "WTI", -80_000, 72.0, 1.0, "USD-SOFR",
                        description="Short WTI 1Y futures (80k bbl)"),
        CommodityFuture("FUT-GOLD-6M", "USD", "Gold", 2_000, 2650.0, 0.5, "USD-SOFR",
                        description="Long gold 6M (2,000 oz)"),
        CommodityFuture("FUT-EUA-DEC", "EUR", "EUA", 150_000, 73.0, 1.0, "EUR-ESTR",
                        description="Long EU allowances Dec+1 (EU ETS)"),
        CommodityFuture("FWD-POWER-DE-1Y", "EUR", "DE-Power-Base", -40_000, 84.0, 1.0, "EUR-ESTR",
                        description="Short German baseload power 1Y"),
        CommodityOption("OPT-TTF-C-SOLD", "EUR", "TTF-Gas", -300_000, 42.0, 0.5, "EUR-ESTR", True,
                        description="Short 6M call on TTF gas"),
        CommodityOption("OPT-BRENT-P", "USD", "Brent", 100_000, 70.0, 0.5, "USD-SOFR", False,
                        description="Long 6M put on Brent (hedge of the futures)"),
    ]


def demo_rrao() -> pd.DataFrame:
    """Illustrative instruments subject to the residual risk add-on (MAR23)."""
    return pd.DataFrame([
        dict(position_id="BARRIER-EQ-KO", description="Knock-out call on TechAE",
             rrao_type="Other", notional=5e6, eu_exempt=False),
        dict(position_id="VARSWAP-IDX", description="Variance swap on EU large-cap index",
             rrao_type="Exotic", notional=3e6, eu_exempt=True),
        dict(position_id="BERM-SWPT-EUR", description="Bermudan swaption EUR 10NC2",
             rrao_type="Other", notional=40e6, eu_exempt=True),
        dict(position_id="CMS-SPREAD-OPT", description="EUR CMS 10Y-2Y spread option",
             rrao_type="Other", notional=25e6, eu_exempt=True),
        dict(position_id="WEATHER-HDD", description="Heating-degree-day weather swap",
             rrao_type="Exotic", notional=2e6, eu_exempt=False),
        dict(position_id="CTP-ITRAXX-3-6", description="CTP tranche (MAR23.4(2))",
             rrao_type="Other", notional=8e6, eu_exempt=False),
    ])


def demo_portfolio(regime: P.Regime | None = None):
    regime = regime or P.Regime.bcbs()
    m = demo_market()
    inst = demo_instruments()
    sens, drc, trades = portfolio_sensitivities(inst, m, regime)
    # the cash equity that hedges the TechAE call keeps its own maturity unless
    # the EU Art. 495q option is used (handled in the DRC module)
    return sens, drc, demo_rrao(), trades
