"""Hand-checked tests of the default risk charge (MAR22) and RRAO (MAR23)."""
import pandas as pd
import pytest

from frtb import params as P
from frtb.drc import compute_drc, compute_rrao

BCBS = P.Regime.bcbs()
EU = P.Regime.eu_transitional()


def pos(**kw):
    base = dict(drc_class="NonSec", bucket="Corporates", obligor="X", seniority="Senior",
                rating="BBB", maturity=5.0, instrument="bond")
    base.update(kw)
    return base


def test_long_bond_jtd_and_capital():
    # JTD = 75% x 100 + (98 - 100) = 73 ; DRC = 6% x 73
    r = compute_drc(pd.DataFrame([pos(notional=100.0, market_value=98.0)]), BCBS)
    assert r.positions["jtd_gross"].iloc[0] == pytest.approx(73.0)
    assert r.capital == pytest.approx(0.06 * 73.0)


def test_maturity_scaling_and_floor():
    r = compute_drc(pd.DataFrame([pos(notional=100.0, market_value=100.0, maturity=0.5),
                                  pos(obligor="Y", notional=100.0, market_value=100.0, maturity=0.1)]), BCBS)
    assert list(r.positions["m_scale"]) == [0.5, 0.25]


def test_seniority_offsetting_rules():
    # short equity may offset a long bond...
    a = compute_drc(pd.DataFrame([pos(notional=100.0, market_value=100.0),
                                  pos(seniority="Equity", notional=-50.0, market_value=-50.0, instrument="cash_equity")]), BCBS)
    assert a.net["net_long"].iloc[0] == pytest.approx(75.0 - 50.0)
    # ...but a short bond may not offset a long equity
    b = compute_drc(pd.DataFrame([pos(seniority="Equity", notional=100.0, market_value=100.0, instrument="cash_equity"),
                                  pos(notional=-100.0, market_value=-100.0)]), BCBS)
    assert b.net["net_long"].iloc[0] == pytest.approx(100.0)
    assert b.net["net_short"].iloc[0] == pytest.approx(-75.0)


def test_hedge_benefit_ratio():
    rows = [pos(obligor="A", notional=100.0, market_value=100.0, rating="BBB"),
            pos(obligor="B", notional=-100.0, market_value=-100.0, rating="A")]
    r = compute_drc(pd.DataFrame(rows), BCBS)
    hbr = 75 / (75 + 75)
    assert r.buckets["HBR"].iloc[0] == pytest.approx(hbr)
    assert r.capital == pytest.approx(0.06 * 75 - hbr * 0.03 * 75)


def test_no_hedging_across_buckets():
    rows = [pos(obligor="A", notional=100.0, market_value=100.0),
            pos(obligor="S", bucket="Sovereigns", notional=-100.0, market_value=-100.0)]
    r = compute_drc(pd.DataFrame(rows), BCBS)
    assert r.capital == pytest.approx(0.06 * 75)


def test_ctp_negative_bucket_and_half_rule():
    rows = [dict(drc_class="CTP", bucket="IDX1", obligor="T1", notional=100.0, market_value=100.0, rw=0.10, maturity=5),
            dict(drc_class="CTP", bucket="IDX2", obligor="T2", notional=-300.0, market_value=-300.0, rw=0.10, maturity=5)]
    r = compute_drc(pd.DataFrame(rows), BCBS)
    hbr = 100 / 400
    drc1 = 0.10 * 100
    drc2 = -hbr * 0.10 * 300
    assert r.by_class["CTP"] == pytest.approx(max(drc1 + 0.5 * drc2, 0))


def test_eu_zero_rw_and_equity_hedge_maturity():
    rows = [pos(obligor="SOV", bucket="Sovereigns", notional=100.0, market_value=100.0, zero_rw=True)]
    assert compute_drc(pd.DataFrame(rows), BCBS).capital > 0
    assert compute_drc(pd.DataFrame(rows), EU).capital == 0.0
    hedge = [pos(obligor="E", seniority="Equity", notional=100.0, market_value=100.0, maturity=1.0,
                 instrument="cash_equity_hedge", hedge_maturity=0.5),
             pos(obligor="E", seniority="Equity", notional=-100.0, market_value=-100.0, maturity=0.5,
                 instrument="equity_forward")]
    assert compute_drc(pd.DataFrame(hedge), BCBS).capital == pytest.approx(0.06 * 50)  # broken hedge, BBB
    assert compute_drc(pd.DataFrame(hedge), EU).capital == pytest.approx(0.0)          # Art. 495q(a)


def test_sold_call_is_short_bought_call_is_long():
    rows = [pos(obligor="C", seniority="Equity", notional=0.0, market_value=10.0, pnl=10.0, instrument="equity_option"),
            pos(obligor="D", seniority="Equity", notional=0.0, market_value=-10.0, pnl=-10.0, instrument="equity_option")]
    r = compute_drc(pd.DataFrame(rows), BCBS)
    j = r.positions.set_index("obligor")["jtd_gross"]
    assert j["C"] == pytest.approx(10.0) and j["D"] == pytest.approx(-10.0)


def test_rrao_and_eu_exemption():
    df = pd.DataFrame([dict(position_id="a", rrao_type="Exotic", notional=1e6, eu_exempt=False),
                       dict(position_id="b", rrao_type="Other", notional=-2e6, eu_exempt=True)])
    assert compute_rrao(df, BCBS).capital == pytest.approx(1e4 + 2e3)
    assert compute_rrao(df, EU).capital == pytest.approx(1e4)
