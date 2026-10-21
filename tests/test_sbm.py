"""Hand-checked tests of the sensitivities-based method (MAR21)."""
import math

import numpy as np
import pandas as pd
import pytest

from frtb import params as P
from frtb import sbm

BCBS = P.Regime.bcbs()
NO_SQRT2 = P.Regime.bcbs(girr_sqrt2=False, fx_sqrt2=False)


def run(rows, regime=BCBS):
    return sbm.compute_sbm(pd.DataFrame(rows), regime)


def cap(res, rc, measure, scenario="Medium"):
    t = res.by_class
    return float(t[(t.risk_class == rc) & (t.measure == measure) & (t.scenario == scenario)].capital.iloc[0])


# ---------------------------------------------------------------- parameters

def test_girr_correlation_table_matches_formula():
    # MAR21.46 Table 2: 1y vs 5y = 88.7%, 0.25y vs 30y floored at 40%
    f = pd.DataFrame({"subtype": ["Yield"] * 3, "qualifier": ["C"] * 3, "tenor": [1.0, 5.0, 30.0]})
    R = sbm._girr_delta_corr(f)
    assert R[0, 1] == pytest.approx(0.887, abs=5e-4)
    f2 = pd.DataFrame({"subtype": ["Yield"] * 2, "qualifier": ["C"] * 2, "tenor": [0.25, 30.0]})
    assert sbm._girr_delta_corr(f2)[0, 1] == pytest.approx(0.40)


def test_girr_curve_inflation_xccy_correlations():
    f = pd.DataFrame({"subtype": ["Yield", "Yield", "Inflation", "XCCY"],
                      "qualifier": ["EUR-ESTR", "EUR-EURIBOR6M", "EUR-HICP", "EUR-XCCY"],
                      "tenor": [5.0, 5.0, -1.0, -1.0]})
    R = sbm._girr_delta_corr(f)
    assert R[0, 1] == pytest.approx(0.999)      # MAR21.45
    assert R[0, 2] == pytest.approx(0.40)       # MAR21.48
    assert R[0, 3] == 0.0 and R[2, 3] == 0.0    # MAR21.49


@pytest.mark.parametrize("b,c,expected", [
    (1, 9, 0.50),          # same sector, different rating
    (3, 12, 0.5 * 0.05),   # financials vs basic materials, IG vs HY
    (4, 6, 0.25),
    (8, 3, 0.20),          # covered bonds vs financials (both IG)
    (17, 18, 0.75),
    (16, 5, 0.0),
    (2, 17, 0.45),
])
def test_csr_cross_bucket_gamma(b, c, expected):
    assert P.csr_ns_gamma(b, c) == pytest.approx(expected)


def test_sec_nctp_risk_weights():
    assert P.CSR_SEC_NCTP_RW[9] == pytest.approx(0.009 * 1.25)
    assert P.CSR_SEC_NCTP_RW[17] == pytest.approx(0.009 * 1.75)
    assert P.CSR_SEC_NCTP_RW[25] == 0.035


def test_vega_risk_weights():
    assert P.vega_rw("EQ", 5) == pytest.approx(0.7778, abs=1e-4)
    assert P.vega_rw("EQ", 9) == 1.0
    assert P.vega_rw("GIRR") == 1.0 and P.vega_rw("FX") == 1.0


def test_scenarios():
    assert P.scenario_corr(0.5, "High") == pytest.approx(0.625)
    assert P.scenario_corr(0.5, "Low") == pytest.approx(0.375)   # max(0, 0.375)
    assert P.scenario_corr(0.9, "Low") == pytest.approx(0.8)     # max(0.8, 0.675)
    assert P.scenario_corr(0.9, "High") == 1.0


# ---------------------------------------------------------------- delta

def test_girr_two_currencies_by_hand():
    res = run([
        dict(risk_class="GIRR", measure="Delta", bucket="EUR", qualifier="C", tenor=1, amount=1e6),
        dict(risk_class="GIRR", measure="Delta", bucket="EUR", qualifier="C", tenor=5, amount=-5e5),
        dict(risk_class="GIRR", measure="Delta", bucket="USD", qualifier="U", tenor=10, amount=3e5),
    ])
    ws1, ws2 = 1e6 * 0.016 / math.sqrt(2), -5e5 * 0.011 / math.sqrt(2)
    rho = math.exp(-0.03 * 4)
    k_eur = math.sqrt(ws1**2 + ws2**2 + 2 * rho * ws1 * ws2)
    k_usd = 3e5 * 0.011 / math.sqrt(2)
    expected = math.sqrt(k_eur**2 + k_usd**2 + 2 * 0.5 * (ws1 + ws2) * k_usd)
    assert cap(res, "GIRR", "Delta") == pytest.approx(expected, rel=1e-10)
    # high scenario
    rho_h, g_h = min(1.25 * rho, 1), 0.625
    k_eur_h = math.sqrt(ws1**2 + ws2**2 + 2 * rho_h * ws1 * ws2)
    exp_h = math.sqrt(k_eur_h**2 + k_usd**2 + 2 * g_h * (ws1 + ws2) * k_usd)
    assert cap(res, "GIRR", "Delta", "High") == pytest.approx(exp_h, rel=1e-10)


def test_netting_same_risk_factor():
    res = run([
        dict(risk_class="GIRR", measure="Delta", bucket="EUR", qualifier="C", tenor=2, amount=1e6),
        dict(risk_class="GIRR", measure="Delta", bucket="EUR", qualifier="C", tenor=2, amount=-1e6),
    ])
    assert cap(res, "GIRR", "Delta") == 0.0


def test_sqrt2_discretion_and_unspecified_currency():
    rows = [dict(risk_class="GIRR", measure="Delta", bucket="PLN", qualifier="C", tenor=1, amount=1e6)]
    assert cap(run(rows), "GIRR", "Delta") == pytest.approx(1e6 * 0.016)      # PLN not specified
    rows[0]["bucket"] = "EUR"
    assert cap(run(rows), "GIRR", "Delta") == pytest.approx(1e6 * 0.016 / math.sqrt(2))
    assert cap(run(rows, NO_SQRT2), "GIRR", "Delta") == pytest.approx(1e6 * 0.016)


def test_csr_intra_bucket_by_hand():
    res = run([
        dict(risk_class="CSR_NS", measure="Delta", bucket=4, qualifier="A", tenor=5, subtype="Bond", amount=1e6),
        dict(risk_class="CSR_NS", measure="Delta", bucket=4, qualifier="B", tenor=10, subtype="CDS", amount=-6e5),
    ])
    w1, w2 = 1e6 * 0.03, -6e5 * 0.03
    rho = 0.35 * 0.65 * 0.999
    assert cap(res, "CSR_NS", "Delta") == pytest.approx(math.sqrt(w1**2 + w2**2 + 2 * rho * w1 * w2))


def test_csr_other_bucket_absolute_sum():
    res = run([
        dict(risk_class="CSR_NS", measure="Delta", bucket=16, qualifier="A", tenor=5, subtype="Bond", amount=1e6),
        dict(risk_class="CSR_NS", measure="Delta", bucket=16, qualifier="B", tenor=5, subtype="Bond", amount=-1e6),
    ])
    assert cap(res, "CSR_NS", "Delta") == pytest.approx(2 * 1e6 * 0.12)


def test_sec_nctp_bucket25_added_without_diversification():
    res = run([
        dict(risk_class="CSR_SEC_NCTP", measure="Delta", bucket=1, qualifier="T1", tenor=5, subtype="Bond", amount=1e6),
        dict(risk_class="CSR_SEC_NCTP", measure="Delta", bucket=4, qualifier="T2", tenor=5, subtype="Bond", amount=1e6),
        dict(risk_class="CSR_SEC_NCTP", measure="Delta", bucket=25, qualifier="T3", tenor=5, subtype="Bond", amount=-1e6),
    ])
    expected = math.sqrt((1e6 * 0.009) ** 2 + (1e6 * 0.020) ** 2) + 1e6 * 0.035  # gamma = 0
    assert cap(res, "CSR_SEC_NCTP", "Delta") == pytest.approx(expected)


def test_equity_spot_repo_and_names():
    res = run([
        dict(risk_class="EQ", measure="Delta", bucket=5, qualifier="X", subtype="Spot", amount=1e6),
        dict(risk_class="EQ", measure="Delta", bucket=5, qualifier="X", subtype="Repo", amount=2e7),
        dict(risk_class="EQ", measure="Delta", bucket=5, qualifier="Y", subtype="Spot", amount=-5e5),
    ])
    w = np.array([1e6 * 0.30, 2e7 * 0.003, -5e5 * 0.30])
    R = np.array([[1, 0.999, 0.25], [0.999, 1, 0.25 * 0.999], [0.25, 0.25 * 0.999, 1]])
    assert cap(res, "EQ", "Delta") == pytest.approx(math.sqrt(w @ R @ w))


def test_equity_cross_bucket_gamma():
    assert P.eq_gamma(1, 7) == 0.15
    assert P.eq_gamma(12, 13) == 0.75
    assert P.eq_gamma(5, 12) == 0.45
    assert P.eq_gamma(11, 3) == 0.0


def test_commodity_intra_bucket_and_eu_ets():
    rows = [
        dict(risk_class="COMM", measure="Delta", bucket=2, qualifier="Brent", tenor=1, subtype="ICE", amount=1e6),
        dict(risk_class="COMM", measure="Delta", bucket=2, qualifier="WTI", tenor=2, subtype="NYMEX", amount=-8e5),
    ]
    w1, w2 = 1e6 * 0.35, -8e5 * 0.35
    rho = 0.95 * 0.99 * 0.999
    assert cap(run(rows), "COMM", "Delta") == pytest.approx(math.sqrt(w1**2 + w2**2 + 2 * rho * w1 * w2))
    ets = [dict(risk_class="COMM", measure="Delta", bucket=3, qualifier="EUA", tenor=1, subtype="ICE",
                amount=1e6, eu_ets=True),
           dict(risk_class="COMM", measure="Delta", bucket=3, qualifier="EUA", tenor=2, subtype="ICE",
                amount=-1e6, eu_ets=True)]
    eu = P.Regime.eu_transitional()
    # BCBS: 60%, tenor correlation 99%; EU: 40%, tenor correlation 99%/99.1%
    assert cap(run(ets), "COMM", "Delta") == pytest.approx(0.6e6 * math.sqrt(2 - 2 * 0.99))
    assert cap(run(ets, eu), "COMM", "Delta") == pytest.approx(0.4e6 * math.sqrt(2 - 2 * 0.99 / 0.991))


def test_fx_specified_pairs():
    rows = [dict(risk_class="FX", measure="Delta", bucket="USD", qualifier="USD", amount=1e6),
            dict(risk_class="FX", measure="Delta", bucket="PLN", qualifier="PLN", amount=1e6)]
    w_usd, w_pln = 1e6 * 0.15 / math.sqrt(2), 1e6 * 0.15
    expected = math.sqrt(w_usd**2 + w_pln**2 + 2 * 0.6 * w_usd * w_pln)
    assert cap(run(rows), "FX", "Delta") == pytest.approx(expected)


def test_alternative_specification_when_negative():
    # opposite risks in two currencies, each bucket internally hedged
    res = run([
        dict(risk_class="GIRR", measure="Delta", bucket="PLN", qualifier="C", tenor=1, amount=1e6),
        dict(risk_class="GIRR", measure="Delta", bucket="PLN", qualifier="C", tenor=30, amount=1e6),
        dict(risk_class="GIRR", measure="Delta", bucket="HUF", qualifier="C", tenor=1, amount=-1e6),
        dict(risk_class="GIRR", measure="Delta", bucket="HUF", qualifier="C", tenor=30, amount=-1e6),
    ], NO_SQRT2)
    w1, w30 = 1e6 * 0.016, 1e6 * 0.011
    rho = math.exp(-0.03 * 29 / 1)
    rho = max(rho, 0.4)
    k = math.sqrt(w1**2 + w30**2 + 2 * rho * w1 * w30)
    s = w1 + w30
    q = 2 * k**2 - 2 * 0.5 * s * s
    assert q > 0  # here the standard formula is positive; check it is used
    assert cap(res, "GIRR", "Delta") == pytest.approx(math.sqrt(q))
    # force a negative sum with low K: identical factors make K_b = |S_b|
    res2 = run([
        dict(risk_class="FX", measure="Delta", bucket="PLN", qualifier="PLN", amount=1e6),
        dict(risk_class="FX", measure="Delta", bucket="HUF", qualifier="HUF", amount=-1e6),
    ])
    k = 1e6 * 0.15
    assert cap(res2, "FX", "Delta") == pytest.approx(math.sqrt(2 * k**2 - 2 * 0.6 * k * k))


# ---------------------------------------------------------------- vega

def test_girr_vega_correlation_by_hand():
    res = run([
        dict(risk_class="GIRR", measure="Vega", bucket="EUR", qualifier="C", tenor=1, underlying_tenor=5, amount=1e6),
        dict(risk_class="GIRR", measure="Vega", bucket="EUR", qualifier="C", tenor=3, underlying_tenor=10, amount=5e5),
    ])
    rho = math.exp(-0.01 * 2 / 1) * math.exp(-0.01 * 5 / 5)
    assert cap(res, "GIRR", "Vega") == pytest.approx(math.sqrt(1e12 + 2.5e11 + 2 * rho * 5e11))


def test_equity_vega_large_cap_rw():
    res = run([dict(risk_class="EQ", measure="Vega", bucket=8, qualifier="X", tenor=1, amount=1e6)])
    assert cap(res, "EQ", "Vega") == pytest.approx(1e6 * 0.55 * math.sqrt(2))


# ---------------------------------------------------------------- curvature

def test_curvature_long_convexity_gives_zero():
    res = run([dict(risk_class="EQ", measure="Curvature", bucket=8, qualifier="X", cvr_up=-1e5, cvr_down=-2e5)])
    assert cap(res, "EQ", "Curvature") == 0.0


def test_curvature_bucket_selection_and_psi():
    rows = [dict(risk_class="EQ", measure="Curvature", bucket=8, qualifier="X", cvr_up=3e5, cvr_down=1e5),
            dict(risk_class="EQ", measure="Curvature", bucket=8, qualifier="Y", cvr_up=-1e5, cvr_down=2e5)]
    res = run(rows)
    rho2 = 0.25**2
    k_up = math.sqrt(9e10 + 2 * rho2 * 3e5 * -1e5)
    k_dn = math.sqrt(1e10 + 4e10 + 2 * rho2 * 1e5 * 2e5)
    assert cap(res, "EQ", "Curvature") == pytest.approx(max(k_up, k_dn))


def test_curvature_cross_bucket_uses_gamma_squared():
    rows = [dict(risk_class="GIRR", measure="Curvature", bucket="EUR", cvr_up=1e5, cvr_down=5e4),
            dict(risk_class="GIRR", measure="Curvature", bucket="USD", cvr_up=2e5, cvr_down=1e4)]
    res = run(rows)
    expected = math.sqrt(1e10 + 4e10 + 2 * 0.25 * 1e5 * 2e5)
    assert cap(res, "GIRR", "Curvature") == pytest.approx(expected)
    # high scenario: gamma^2 = 0.25 -> 0.3125
    exp_h = math.sqrt(1e10 + 4e10 + 2 * 0.3125 * 1e5 * 2e5)
    assert cap(res, "GIRR", "Curvature", "High") == pytest.approx(exp_h)


def test_fx_curvature_scalar():
    rows = [dict(risk_class="FX", measure="Curvature", bucket="JPY", cvr_up=3e5, cvr_down=1e5,
                 fx_cross_option=True)]
    assert cap(run(rows), "FX", "Curvature") == pytest.approx(2e5)
    assert cap(run(rows, P.Regime.bcbs(fx_curvature_scalar=False)), "FX", "Curvature") == pytest.approx(3e5)


def test_total_is_max_over_scenarios():
    res = run([
        dict(risk_class="EQ", measure="Delta", bucket=8, qualifier="A", subtype="Spot", amount=1e6),
        dict(risk_class="EQ", measure="Delta", bucket=8, qualifier="B", subtype="Spot", amount=-1e6),
    ])
    assert res.capital == pytest.approx(res.totals.max())
    assert res.binding_scenario == "Low"  # hedged position: lower correlation hurts
