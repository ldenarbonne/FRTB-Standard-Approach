"""Pricing / sensitivity generation, input files and app smoke tests."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frtb import demo, engine, pricing, schema
from frtb import params as P

BCBS = P.Regime.bcbs()


@pytest.fixture(scope="module")
def market():
    return demo.demo_market()


def test_equity_delta_equals_market_value(market):
    inst = pricing.EquityPosition("E", "EUR", "TechAE", 1000)
    rows = pricing.delta_sensitivities(inst, market)
    assert rows[0]["amount"] == pytest.approx(1000 * market.eq_spot["TechAE"])


def test_swap_pv01_sum_matches_parallel_shift(market):
    inst = pricing.IRSwap("S", "EUR", 10e6, 0.027, 10, "EUR-ESTR", "EUR-EURIBOR6M")
    rows = pricing.delta_sensitivities(inst, market)
    total = sum(r["amount"] for r in rows if r["risk_class"] == "GIRR")
    import copy
    m2 = copy.deepcopy(market)
    for n in ("EUR-ESTR", "EUR-EURIBOR6M"):
        m2.ir[n].zeros = [z + 1e-4 for z in m2.ir[n].zeros]
    parallel = (inst.pv(m2) - inst.pv(market)) / 1e-4
    assert total == pytest.approx(parallel, rel=1e-3)


def test_vega_mapping_conserves_sensitivity(market):
    inst = pricing.Swaption("W", "EUR", 1e7, 0.025, 2.0, 7.0, "EUR-ESTR")
    rows = pricing.vega_sensitivities(inst, market)
    assert {r["tenor"] for r in rows} == {1.0, 3.0}
    assert {r["underlying_tenor"] for r in rows} == {5.0, 10.0}
    v0 = inst.pv(market)
    bumped = inst.vega_factors(market)[0]["bump"](market, 1e-4)
    expected = (inst.pv(bumped) - v0) / 1e-4 * market.ir_normal_vol["EUR"]
    assert sum(r["amount"] for r in rows) == pytest.approx(expected)


def test_long_option_has_negative_cvr(market):
    inst = pricing.EquityOption("O", "EUR", "TechAE", 1000, 180, 1.0, "EUR-ESTR")
    d = pricing.delta_sensitivities(inst, market)
    c = pricing.curvature_sensitivities(inst, market, BCBS, d)
    eq = [r for r in c if r["risk_class"] == "EQ"][0]
    assert eq["cvr_up"] < 0 and eq["cvr_down"] < 0


def test_linear_instruments_have_no_curvature(market):
    inst = pricing.CommodityFuture("F", "USD", "Brent", 1000, 75, 1.0, "USD-SOFR")
    assert pricing.curvature_sensitivities(inst, market, BCBS, []) == []


def test_fx_forward_deltas(market):
    inst = pricing.FXForward("X", "USD", 1e6, "JPY", 1.5e8, 1.0, "USD-SOFR", "JPY-TONA")
    rows = {r["bucket"]: r["amount"] for r in pricing.delta_sensitivities(inst, market) if r["risk_class"] == "FX"}
    usd_leg = 1e6 * market.ir["USD-SOFR"].df(1.0) * market.fx["USD"]
    assert rows["USD"] == pytest.approx(usd_leg, rel=1e-6)
    assert rows["JPY"] < 0


def test_csv_roundtrip_reproduces_capital():
    sens, drc, rrao, _ = demo.demo_portfolio(BCBS)
    direct = engine.compute_frtb(sens, drc, rrao, BCBS).total
    s2 = schema.normalise_sensitivities(schema.read_table(io.StringIO(sens.to_csv(index=False))))
    d2 = schema.normalise_drc(schema.read_table(io.StringIO(drc.to_csv(index=False))))
    r2 = schema.normalise_rrao(schema.read_table(io.StringIO(rrao.to_csv(index=False))))
    assert schema.validate_sensitivities(s2, "EUR") == []
    assert schema.validate_drc(d2) == []
    assert engine.compute_frtb(s2, d2, r2, BCBS).total == pytest.approx(direct, rel=1e-9)


def test_tenor_mapping_splits_linearly():
    df = schema.normalise_sensitivities(pd.DataFrame([
        dict(risk_class="GIRR", measure="Delta", bucket="EUR", qualifier="C", tenor=7, amount=100.0)]))
    out, n = schema.map_tenors(df)
    assert n == 1
    assert dict(zip(out["tenor"], out["amount"])) == pytest.approx({5.0: 60.0, 10.0: 40.0})


def test_validation_catches_errors():
    df = schema.normalise_sensitivities(pd.DataFrame([
        dict(risk_class="CSR_NS", measure="Delta", bucket=19, qualifier="A", tenor=5, subtype="Bond", amount=1),
        dict(risk_class="GIRR", measure="Delta", bucket="EUR", qualifier="C", tenor=7, amount=1),
        dict(risk_class="FX", measure="Delta", bucket="EUR", qualifier="EUR", amount=1),
        dict(risk_class="EQ", measure="Vega", bucket=5, qualifier="X", tenor=1, subtype="Repo", amount=1),
    ]))
    errs = schema.validate_sensitivities(df, "EUR")
    assert len(errs) == 4


def test_eu_total_below_bcbs_on_demo():
    sens, drc, rrao, _ = demo.demo_portfolio(BCBS)
    b = engine.compute_frtb(sens, drc, rrao, BCBS)
    e = engine.compute_frtb(sens, drc, rrao, P.Regime.eu_transitional())
    assert e.total < b.total
    assert e.sbm_after_multiplier == pytest.approx(0.9 * e.sbm.capital)


def test_app_runs():
    from streamlit.testing.v1 import AppTest
    app = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app, default_timeout=300).run()
    assert not at.exception and not at.error
    at.radio[0].set_value("EU (CRR3 + DR 2026/1221)").run()
    assert not at.exception
