"""
FRTB Standardised Approach Calculator
SBM (delta, vega, curvature) + DRC + RRAO, BCBS MAR21-23 with EU variants.

Author: Laurent Motais de Narbonne
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frtb import demo, engine, schema
from frtb import params as P

st.set_page_config(page_title="FRTB SA Calculator", page_icon="🏦", layout="wide")

MEASURE_COLOR = {"Delta": "#2a78d6", "Vega": "#eb6834", "Curvature": "#1baf7a"}
COMP_COLOR = {"SBM": "#2a78d6", "DRC": "#eb6834", "RRAO": "#1baf7a"}
SHORT = {"GIRR": "GIRR", "CSR_NS": "CSR non-sec", "CSR_SEC_NCTP": "CSR sec non-CTP",
         "CSR_SEC_CTP": "CSR CTP", "EQ": "Equity", "COMM": "Commodity", "FX": "FX"}


def money(x: float, ccy: str) -> str:
    return f"{x:,.0f} {ccy}"


def strb(df: pd.DataFrame) -> pd.DataFrame:
    """Cast bucket columns to text (they mix numbers and currency codes)."""
    d = df.copy()
    if "bucket" in d.columns:
        d["bucket"] = d["bucket"].astype(str)
    return d


# --------------------------------------------------------------------------
# Sidebar: regime and options
# --------------------------------------------------------------------------

st.sidebar.title("🏦 FRTB SA Calculator")
st.sidebar.caption("Sensitivities-based method, default risk charge and residual risk add-on")

regime_choice = st.sidebar.radio(
    "Regulatory regime",
    ["BCBS (MAR21-23)", "EU (CRR3 + DR 2026/1221)"],
    help="EU applies the transitional provisions of Delegated Regulation (EU) 2026/1221 "
         "(1 January 2027 – 31 December 2029) and the CRR3 EU ETS carbon risk weight.",
)
is_eu = regime_choice.startswith("EU")

with st.sidebar.expander("Bank discretions (MAR21)"):
    girr_sqrt2 = st.checkbox("GIRR: divide risk weights by √2 for specified currencies (21.44)", True)
    fx_sqrt2 = st.checkbox("FX: divide risk weight by √2 for specified pairs (21.88)", True)
    covered_aa = st.checkbox("CSR: 1.5% risk weight for covered bonds rated AA- or better (21.53)", True)
    fx_scalar = st.checkbox("FX curvature: divide by 1.5 for options not referencing the reporting ccy (21.98)", True)

eu_kw = {}
if is_eu:
    with st.sidebar.expander("EU transitional provisions", expanded=True):
        st.caption("Delegated Regulation (EU) 2026/1221, CRR Articles 495p-495v")
        eu_kw["eu_sbm_multiplier"] = 0.90 if st.checkbox("SBM phase-in multiplier 0.9 (Art. 495s)", True) else 1.0
        eu_kw["eu_rrao_exemptions"] = st.checkbox("RRAO multiplier 0 for eligible instruments (Art. 495p)", True)
        eu_kw["eu_drc_cash_hedge_maturity"] = st.checkbox(
            "DRC: cash equity hedges take the derivative's maturity (Art. 495q(a))", True)
        eu_kw["eu_drc_equity_deriv_3m"] = st.checkbox(
            "DRC: 3-month maturity for equity derivatives (Art. 495q(b))", False)
        eu_kw["eu_zero_rw_sovereigns"] = st.checkbox(
            "DRC: 0% for exposures with a 0% credit-risk weight (MAR22.7 / CRR3)", True)
        eu_kw["eu_ets_rw"] = st.number_input("EU ETS carbon risk weight (CRR3 Art. 325as)", 0.0, 1.0, 0.40, 0.05)
        eu_kw["eu_bank_multiplier"] = st.slider(
            "Bank-specific multiplier (Art. 495v, ≤ 1)", 0.50, 1.00, 1.00, 0.01,
            help="Optional, calibrated quarterly by the bank as the ratio of Basel 2.5 to FRTB "
                 "own funds requirements. Leave at 1.00 if not applied.")

st.sidebar.subheader("Portfolio")
source = st.sidebar.radio("Input", ["Demo portfolio (priced in the tool)", "Upload sensitivity files"])

common = dict(girr_sqrt2=girr_sqrt2, fx_sqrt2=fx_sqrt2, covered_aa_reduced_rw=covered_aa,
              fx_curvature_scalar=fx_scalar)


def make_regime(eu: bool, ccy: str) -> P.Regime:
    if eu:
        return P.Regime.eu_transitional(reporting_ccy=ccy, **common, **eu_kw)
    return P.Regime.bcbs(reporting_ccy=ccy, **common)


@st.cache_data(show_spinner="Pricing the demo portfolio and bumping risk factors…")
def cached_demo(regime_fields: tuple):
    reg = P.Regime(**dict(regime_fields))
    return demo.demo_portfolio(reg)


def regime_key(reg: P.Regime) -> tuple:
    d = dataclasses.asdict(reg)
    d.pop("notes", None)
    return tuple(sorted(d.items()))


trades = None
if source.startswith("Demo"):
    ccy = "EUR"
    regime = make_regime(is_eu, ccy)
    sens, drc_pos, rrao_pos, trades = cached_demo(regime_key(regime))
    sens_bcbs = sens  # curvature shocks identical across regimes for the demo book
else:
    ccy = st.sidebar.text_input("Reporting currency", "EUR", max_chars=3).upper()
    regime = make_regime(is_eu, ccy)
    up_s = st.sidebar.file_uploader("Sensitivities (CSV, required)", type=["csv", "txt"])
    up_d = st.sidebar.file_uploader("DRC positions (CSV, optional)", type=["csv", "txt"])
    up_r = st.sidebar.file_uploader("RRAO positions (CSV, optional)", type=["csv", "txt"])
    auto_map = st.sidebar.checkbox("Map non-standard tenors by linear interpolation", True)
    if up_s is None:
        st.title("FRTB Standardised Approach Calculator")
        st.info("Upload a sensitivity file in the sidebar, or switch to the demo portfolio. "
                "Templates (pre-filled with the demo book) are available in the "
                "**Portfolio & inputs** tab of the demo, and the file format is described in "
                "the **Methodology** tab.")
        st.stop()
    try:
        sens = schema.normalise_sensitivities(schema.read_table(up_s))
        n_mapped = 0
        if auto_map:
            sens, n_mapped = schema.map_tenors(sens)
        errs = schema.validate_sensitivities(sens, ccy)
        drc_pos = schema.normalise_drc(schema.read_table(up_d)) if up_d else pd.DataFrame()
        errs += [f"DRC – {e}" for e in (schema.validate_drc(drc_pos) if len(drc_pos) else [])]
        rrao_pos = schema.normalise_rrao(schema.read_table(up_r)) if up_r else pd.DataFrame()
        errs += [f"RRAO – {e}" for e in (schema.validate_rrao(rrao_pos) if len(rrao_pos) else [])]
    except Exception as exc:
        st.error(f"Could not read the files: {exc}")
        st.stop()
    if errs:
        st.error(f"{len(errs)} validation issue(s) – please correct the files:")
        st.code("\n".join(errs[:60]))
        st.stop()
    if n_mapped:
        st.sidebar.success(f"{n_mapped} sensitivities mapped to prescribed tenors.")
    sens_bcbs = sens

# --------------------------------------------------------------------------
# Computation (selected regime + the other one for comparison)
# --------------------------------------------------------------------------

res = engine.compute_frtb(sens, drc_pos, rrao_pos, regime)
other = make_regime(not is_eu, ccy)
res_other = engine.compute_frtb(sens_bcbs, drc_pos, rrao_pos, other)

st.title("FRTB Standardised Approach Calculator")
st.caption(f"{regime.name} · reporting currency {ccy} · "
           f"{len(sens):,} sensitivities · {len(drc_pos)} DRC positions · {len(rrao_pos)} RRAO positions")
if source.startswith("Demo"):
    st.info("Demonstration book of 35 fictitious trades across the seven risk classes, priced in "
            "the tool with stylised models; sensitivities are obtained by bump-and-reprice "
            "following MAR21.19-21.25.", icon="ℹ️")

tabs = st.tabs(["Summary", "Sensitivities-based method", "Default risk", "Residual risk",
                "Portfolio & inputs", "Methodology"])

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

with tabs[0]:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Own funds requirement", money(res.total, ccy))
    k2.metric("SBM", money(res.sbm_after_multiplier, ccy),
              help=f"Binding correlation scenario: {res.sbm.binding_scenario}")
    k3.metric("DRC", money(res.drc.capital, ccy))
    k4.metric("RRAO", money(res.rrao.capital, ccy))
    k5.metric("RWA (× 12.5)", money(res.rwa, ccy))

    c1, c2 = st.columns([3, 2])
    with c1:
        bc = res.sbm.by_class[res.sbm.by_class["scenario"] == res.sbm.binding_scenario]
        fig = go.Figure()
        for msr in P.MEASURES:
            sub = bc[bc["measure"] == msr].set_index("risk_class").reindex(P.RISK_CLASSES).fillna(0)
            fig.add_trace(go.Bar(y=[SHORT[r] for r in sub.index], x=sub["capital"], name=msr,
                                 orientation="h", marker_color=MEASURE_COLOR[msr],
                                 hovertemplate="%{y} – " + msr + ": %{x:,.0f}<extra></extra>"))
        fig.update_layout(barmode="stack", title=f"SBM capital by risk class ({res.sbm.binding_scenario} "
                          "correlation scenario, before any multiplier)",
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.08),
                          yaxis=dict(autorange="reversed"))
        fig.update_xaxes(title=ccy, tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Correlation scenarios (MAR21.6)**")
        sc = res.sbm.totals.to_frame("SBM capital")
        sc["Binding"] = ["◀ max" if s == res.sbm.binding_scenario else "" for s in sc.index]
        st.dataframe(sc.style.format({"SBM capital": "{:,.0f}"}))
        st.markdown("**Capital build-up**")
        st.dataframe(res.summary().style.format("{:,.0f}"))

    st.subheader("Regime comparison")
    bcbs_res, eu_res = (res_other, res) if is_eu else (res, res_other)
    comp = pd.DataFrame({
        "BCBS (MAR21-23)": [bcbs_res.sbm_after_multiplier, bcbs_res.drc.capital, bcbs_res.rrao.capital, bcbs_res.total],
        "EU (CRR3 + DR 2026/1221)": [eu_res.sbm_after_multiplier, eu_res.drc.capital, eu_res.rrao.capital, eu_res.total],
    }, index=["SBM (after multiplier)", "DRC", "RRAO", "Total"])
    comp["EU vs BCBS"] = comp.iloc[:, 1] / comp.iloc[:, 0].replace(0, np.nan) - 1
    st.dataframe(comp.style.format({"BCBS (MAR21-23)": "{:,.0f}", "EU (CRR3 + DR 2026/1221)": "{:,.0f}",
                                    "EU vs BCBS": "{:+.1%}"}))
    st.caption("The EU column uses the options selected in the sidebar (or their defaults when the "
               "BCBS regime is selected). Differences come from the 0.9 SBM phase-in, the EU ETS "
               "carbon risk weight and tenor correlation, the RRAO exemptions and the DRC options.")

# --------------------------------------------------------------------------
# SBM drill-down
# --------------------------------------------------------------------------

with tabs[1]:
    piv = res.sbm.by_class.pivot_table(index=["risk_class", "measure"], columns="scenario",
                                       values="capital").reindex(columns=P.SCENARIOS)
    piv.index = [f"{SHORT[a]} – {b}" for a, b in piv.index]
    st.markdown("**Capital by risk class, measure and correlation scenario**")
    st.dataframe(piv.style.format("{:,.0f}").highlight_max(axis=None, props="font-weight:600"))

    avail = [rc for rc in P.RISK_CLASSES if (res.sbm.by_class["risk_class"] == rc).any()]
    c1, c2, c3 = st.columns(3)
    rc_sel = c1.selectbox("Risk class", avail, format_func=lambda r: P.RISK_CLASS_LABEL[r])
    msr_avail = [m for m in P.MEASURES if (rc_sel, m) in res.sbm.factors]
    msr_sel = c2.selectbox("Measure", msr_avail)
    scn_sel = c3.selectbox("Scenario", P.SCENARIOS, index=P.SCENARIOS.index(res.sbm.binding_scenario))

    bt = res.sbm.buckets
    bt = bt[(bt["risk_class"] == rc_sel) & (bt["measure"] == msr_sel) & (bt["scenario"] == scn_sel)]
    cols = ["bucket", "K_b", "S_b", "n_factors"] + (["K_up", "K_down", "direction"] if msr_sel == "Curvature" else [])
    st.markdown("**Bucket level (K_b and S_b)**")
    st.dataframe(strb(bt[cols]).set_index("bucket").style.format(
        {"K_b": "{:,.0f}", "S_b": "{:,.0f}", "K_up": "{:,.0f}", "K_down": "{:,.0f}"}))

    ft = strb(res.sbm.factors[(rc_sel, msr_sel)])
    st.markdown("**Risk factors** (net sensitivities after offsetting, medium-scenario inputs)")
    if msr_sel == "Curvature":
        st.dataframe(ft.style.format({"cvr_up": "{:,.0f}", "cvr_down": "{:,.0f}"}))
    else:
        show = ft.drop(columns=[c for c in ("eu_ets", "covered_aa") if c in ft and not ft[c].any()])
        show = show.replace({"tenor": {-1.0: np.nan}, "underlying_tenor": {-1.0: np.nan}})
        st.dataframe(show.style.format({"amount": "{:,.0f}", "rw": "{:.4%}", "ws": "{:,.0f}",
                                        "tenor": "{:g}", "underlying_tenor": "{:g}"}, na_rep=""))
        agg = ft.groupby("bucket")["ws"].sum()
        fig = go.Figure(go.Bar(x=[str(b) for b in agg.index], y=agg.values,
                               marker_color=MEASURE_COLOR[msr_sel],
                               hovertemplate="Bucket %{x}: %{y:,.0f}<extra></extra>"))
        fig.update_layout(title=f"Net weighted sensitivity by bucket – {SHORT[rc_sel]} {msr_sel}",
                          margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(title="Bucket", type="category")
        fig.update_yaxes(title=f"Σ WS ({ccy})", tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
# DRC
# --------------------------------------------------------------------------

with tabs[2]:
    if res.drc.positions.empty:
        st.info("No DRC positions supplied.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Non-securitisation", money(res.drc.by_class.get("NonSec", 0), ccy))
        d2.metric("Securitisation non-CTP", money(res.drc.by_class.get("SecNonCTP", 0), ccy))
        d3.metric("Correlation trading portfolio", money(res.drc.by_class.get("CTP", 0), ccy))
        st.markdown("**Buckets (hedge benefit ratio and weighted JTD)**")
        st.dataframe(res.drc.buckets.set_index(["drc_class", "bucket"]).style.format(
            {"net_long": "{:,.0f}", "net_short": "{:,.0f}", "HBR": "{:.1%}", "RW_long": "{:,.0f}",
             "RW_short": "{:,.0f}", "DRC_b": "{:,.0f}"}))
        st.markdown("**Net JTD by obligor** (offsetting subject to seniority, MAR22.19)")
        st.dataframe(res.drc.net.set_index(["drc_class", "bucket", "obligor"]).style.format(
            {"rw": "{:.1%}", "net_long": "{:,.0f}", "net_short": "{:,.0f}"}))
        with st.expander("Gross JTD by position"):
            cols = ["position_id", "drc_class", "bucket", "obligor", "seniority", "rating", "instrument",
                    "notional", "pnl_used", "lgd", "jtd_gross", "m_scale", "jtd_scaled"]
            st.dataframe(res.drc.positions[cols].set_index("position_id").style.format(
                {"notional": "{:,.0f}", "pnl_used": "{:,.0f}", "lgd": "{:.0%}", "jtd_gross": "{:,.0f}",
                 "m_scale": "{:.2f}", "jtd_scaled": "{:,.0f}"}, na_rep=""))
        st.caption("Gross JTD = max(LGD × notional + P&L, 0) for long and min(…, 0) for short exposures; "
                   "positions under one year are scaled by maturity (floor 3 months). DRC_b = max(Σ RW·JTD_long "
                   "− HBR · Σ RW·|JTD_short|, 0); for the CTP the HBR is computed across all indices, "
                   "buckets may be negative and DRC = max(Σ [max(DRC_b,0) + 0.5·min(DRC_b,0)], 0).")

# --------------------------------------------------------------------------
# RRAO
# --------------------------------------------------------------------------

with tabs[3]:
    if res.rrao.positions.empty:
        st.info("No RRAO positions supplied.")
    else:
        r1, r2 = st.columns(2)
        r1.metric("Exotic underlyings (1.0%)", money(res.rrao.by_type.get("Exotic", 0), ccy))
        r2.metric("Other residual risks (0.1%)", money(res.rrao.by_type.get("Other", 0), ccy))
        cols = [c for c in ["position_id", "description", "rrao_type", "notional", "rw", "eu_exempt",
                            "multiplier", "rrao"] if c in res.rrao.positions]
        st.dataframe(res.rrao.positions[cols].set_index("position_id").style.format(
            {"notional": "{:,.0f}", "rw": "{:.1%}", "multiplier": "{:.0f}", "rrao": "{:,.0f}"}))
        st.caption("RRAO = Σ gross notional × risk weight (MAR23.8). Under the EU regime, a multiplier of 0 "
                   "applies until 31 December 2029 to instruments that attract the add-on only because they "
                   "reference future realised volatility, are options exercisable on a finite set of dates, "
                   "or are options on a spread between two CMS rates in the same currency (Art. 495p).")

# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

with tabs[4]:
    if trades is not None:
        st.markdown("**Demo trades**")
        st.dataframe(trades.set_index("trade_id").style.format({"pv": "{:,.0f}"}))
    st.markdown("**Sensitivity table** – use it as a template for your own files")
    st.dataframe(strb(sens), height=300)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download sensitivities (CSV)", sens.to_csv(index=False).encode(),
                       "frtb_sensitivities.csv", "text/csv")
    if len(drc_pos):
        c2.download_button("Download DRC positions (CSV)", drc_pos.to_csv(index=False).encode(),
                           "frtb_drc_positions.csv", "text/csv")
    if len(rrao_pos):
        c3.download_button("Download RRAO positions (CSV)", rrao_pos.to_csv(index=False).encode(),
                           "frtb_rrao_positions.csv", "text/csv")

# --------------------------------------------------------------------------
# Methodology
# --------------------------------------------------------------------------

with tabs[5]:
    st.markdown(r"""
### Structure (MAR20.4)
$$K_{SA} = \underbrace{\max_{s \in \{L,M,H\}} \sum_{\text{class}} \left(\Delta_s + \mathcal{V}_s + \mathcal{C}_s\right)}_{\text{SBM}} + \text{DRC} + \text{RRAO}$$

### Delta and vega (MAR21.4)
$WS_k = RW_k \, s_k$, then within a bucket
$K_b = \sqrt{\max\left(0, \sum_k WS_k^2 + \sum_{k \ne l} \rho_{kl} WS_k WS_l\right)}$ and across buckets
$\sqrt{\sum_b K_b^2 + \sum_{b \ne c} \gamma_{bc} S_b S_c}$ with $S_b = \sum_k WS_k$; if the sum is negative,
$S_b$ is capped and floored at $\pm K_b$ (alternative specification). "Other" buckets (CSR 16, securitisation 25,
equity 11) use $\sum |WS_k|$; securitisation bucket 25 is added without diversification.

### Curvature (MAR21.5)
$CVR_k^{\pm} = -\sum_i \left[V_i(x_k^{RW\pm}) - V_i(x_k) \mp RW_k \, s_{ik}\right]$,
$K_b^{\pm} = \sqrt{\max\left(0, \sum_k \max(CVR_k^{\pm},0)^2 + \sum_{k\ne l} \rho_{kl}^2 CVR_k CVR_l \,\psi\right)}$,
$K_b = \max(K_b^+, K_b^-)$, aggregated across buckets with $\gamma_{bc}^2$ and $\psi(S_b,S_c)$. The across-bucket
quantity is floored at zero.

### Correlation scenarios (MAR21.6)
High: $\min(1.25\rho, 1)$; low: $\max(2\rho - 1, 0.75\rho)$; applied to $\rho$ and $\gamma$ (to the squared
values for curvature, MAR21.100-101). The SBM requirement is the largest of the three totals.

### EU variants implemented
| Provision | Effect in the tool |
|---|---|
| DR 2026/1221 Art. 495s(1) | SBM result × 0.9 until 31/12/2029 |
| DR 2026/1221 Art. 495p | RRAO × 0 for realised-volatility, finite-exercise and CMS-spread options |
| DR 2026/1221 Art. 495q | DRC maturity options for equity hedges and equity derivatives |
| DR 2026/1221 Art. 495r | EU ETS commodity tenor correlation divided by 99.1% |
| DR 2026/1221 Art. 495v | Optional bank-specific multiplier ≤ 1 (user input) |
| CRR3 Art. 325as | 40% risk weight for EU ETS carbon allowances (kept in bucket 3 here) |
| MAR22.7 / CRR3 | 0% default risk weight for exposures with a 0% credit-risk weight |

### File formats
**Sensitivities** – `trade_id, risk_class, measure, bucket, qualifier, tenor, underlying_tenor, subtype, amount,
cvr_up, cvr_down, eu_ets, covered_aa, fx_cross_option`.
Risk classes: `GIRR, CSR_NS, CSR_SEC_NCTP, CSR_SEC_CTP, EQ, COMM, FX`; measures: `Delta, Vega, Curvature`.
Sensitivities follow MAR21.19-21.25 (PV01 / CS01 divided by 0.0001, equity / commodity / FX deltas divided
by 0.01, vega × implied volatility), in the reporting currency.

**DRC** – `position_id, drc_class (NonSec | SecNonCTP | CTP), bucket, obligor, seniority (Equity | NonSenior |
Senior | Covered), rating, notional (+ long / − short), market_value, pnl, maturity, instrument, rw, zero_rw,
hedge_maturity`.

**RRAO** – `position_id, description, rrao_type (Exotic | Other), notional, eu_exempt`.

### Limitations
* The demo pricers are stylised (single-curve swaptions, flat volatilities, reduced-form credit).
* Securitisation and CTP tranche default risk weights are user inputs (banking-book securitisation framework).
* No look-through for index DRC; index positions are excluded from DRC in the demo.
* CRR3 renumbers commodity buckets to separate EU ETS and non-EU ETS carbon; here the EU ETS risk weight is
  applied within bucket 3.
* Educational tool: it does not replace a regulatory calculation or supervisory reporting.
""")

st.divider()
st.caption(
    "Built by [Laurent Motais de Narbonne](https://www.linkedin.com/in/laurent-motais-de-narbonne/) · "
    "[Source code](https://github.com/ldenarbonne/frtb-sa-calculator) · "
    "Based on BCBS MAR21-23 (version effective 1 January 2023) and Commission Delegated Regulation (EU) "
    "2026/1221. Educational tool, not regulatory advice."
)
