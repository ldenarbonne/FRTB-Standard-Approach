"""
Stylised pricing library and bump-and-reprice sensitivity generator.

Produces MAR21-compliant sensitivities (delta, vega, curvature CVR) for a
demonstration portfolio, so that the capital engine can be fed either from
a user file or from instruments priced in the tool.

Models are deliberately simple (flat or piecewise-linear curves, Black,
Bachelier and Garman-Kohlhagen formulas, reduced-form credit) and are meant
to generate realistic sensitivity profiles, not production valuations.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from . import params as P
from .sbm import curvature_rw

BP = 1e-4


# ==========================================================================
# Market data
# ==========================================================================


@dataclass
class Curve:
    ccy: str
    tenors: list[float]
    zeros: list[float]          # continuously compounded zero rates

    def zero(self, t: float) -> float:
        return float(np.interp(t, self.tenors, self.zeros))

    def df(self, t: float) -> float:
        return math.exp(-self.zero(t) * t)


@dataclass
class SpreadCurve:
    tenors: list[float]
    spreads: list[float]

    def s(self, t: float) -> float:
        return max(float(np.interp(t, self.tenors, self.spreads)), 0.0)


@dataclass
class CommCurve:
    bucket: int
    location: str
    tenors: list[float]
    prices: list[float]
    eu_ets: bool = False

    def price(self, t: float) -> float:
        return float(np.interp(t, self.tenors, self.prices))


@dataclass
class Market:
    reporting_ccy: str = "EUR"
    ir: dict = field(default_factory=dict)          # curve name -> Curve
    inflation: dict = field(default_factory=dict)   # curve name -> (ccy, flat rate)
    ir_normal_vol: dict = field(default_factory=dict)  # ccy -> normal vol
    credit: dict = field(default_factory=dict)      # (name, basis) -> SpreadCurve
    credit_meta: dict = field(default_factory=dict)  # name -> dict(risk_class, bucket, ...)
    eq_spot: dict = field(default_factory=dict)
    eq_vol: dict = field(default_factory=dict)
    eq_repo: dict = field(default_factory=dict)
    eq_bucket: dict = field(default_factory=dict)
    fx: dict = field(default_factory=dict)          # ccy -> reporting ccy per unit
    fx_vol: dict = field(default_factory=dict)      # "USD/EUR" -> vol
    comm: dict = field(default_factory=dict)        # name -> CommCurve
    comm_vol: dict = field(default_factory=dict)

    def fx_rate(self, ccy: str) -> float:
        return 1.0 if ccy == self.reporting_ccy else self.fx[ccy]


def _bump_pillar(values: list[float], tenors: list[float], k: int, h: float, relative=False) -> list[float]:
    v = list(values)
    v[k] = v[k] * (1 + h) if relative else v[k] + h
    return v


def _interp_weights(x: float, grid: list[float]) -> dict[float, float]:
    """Linear interpolation weights of x on a grid (flat extrapolation)."""
    if x <= grid[0]:
        return {grid[0]: 1.0}
    if x >= grid[-1]:
        return {grid[-1]: 1.0}
    j = int(np.searchsorted(grid, x))
    lo, hi = grid[j - 1], grid[j]
    w = (x - lo) / (hi - lo)
    return {lo: 1 - w, hi: w} if w not in (0.0, 1.0) else {hi if w == 1.0 else lo: 1.0}


# ==========================================================================
# Instruments
# ==========================================================================


class Instrument:
    """Base class: value(m) returns the value in instrument currency."""

    optional = False             # subject to vega / curvature (MAR21.2)

    def value(self, m: Market) -> float:
        raise NotImplementedError

    def pv(self, m: Market) -> float:
        return self.value(m) * m.fx_rate(self.ccy)

    # dependencies used to restrict bumping
    def ir_curves(self) -> list[str]:
        return []

    def inflation_curves(self) -> list[str]:
        return []

    def credit_names(self) -> list[tuple[str, str]]:
        return []

    def equities(self) -> list[str]:
        return []

    def fx_ccys(self) -> list[str]:
        return [self.ccy]

    def commodities(self) -> list[str]:
        return []

    def vega_factors(self, m: Market) -> list[dict]:
        """List of dicts: risk_class, bucket, qualifier, expiry, underlying_tenor,
        vol_key, bump(m, h) -> Market, vol."""
        return []

    def drc_rows(self, m: Market) -> list[dict]:
        return []


@dataclass
class IRSwap(Instrument):
    trade_id: str
    ccy: str
    notional: float           # > 0 payer (pay fixed), < 0 receiver
    fixed_rate: float
    maturity: float
    disc: str
    proj: str
    fixed_freq: float = 1.0
    float_freq: float = 0.5
    description: str = "Interest rate swap"

    def value(self, m):
        d, p = m.ir[self.disc], m.ir[self.proj]
        fixed = sum(self.fixed_freq * d.df(t) for t in np.arange(self.fixed_freq, self.maturity + 1e-9, self.fixed_freq))
        flt = 0.0
        for t in np.arange(self.float_freq, self.maturity + 1e-9, self.float_freq):
            t0 = t - self.float_freq
            fwd = (p.df(t0) / p.df(t) - 1) / self.float_freq
            flt += self.float_freq * fwd * d.df(t)
        return self.notional * (flt - self.fixed_rate * fixed)

    def ir_curves(self):
        return sorted({self.disc, self.proj})


@dataclass
class Swaption(Instrument):
    trade_id: str
    ccy: str
    notional: float           # > 0 long payer swaption
    strike: float
    expiry: float
    tenor: float
    curve: str
    description: str = "Payer swaption (Bachelier)"
    optional: bool = True

    def _fwd_annuity(self, m):
        c = m.ir[self.curve]
        times = np.arange(self.expiry + 1, self.expiry + self.tenor + 1e-9, 1.0)
        ann = sum(c.df(t) for t in times)
        fwd = (c.df(self.expiry) - c.df(self.expiry + self.tenor)) / ann
        return fwd, ann

    def value(self, m):
        f, a = self._fwd_annuity(m)
        s = m.ir_normal_vol[self.ccy] * math.sqrt(self.expiry)
        d = (f - self.strike) / s
        return self.notional * a * ((f - self.strike) * norm.cdf(d) + s * norm.pdf(d))

    def ir_curves(self):
        return [self.curve]

    def vega_factors(self, m):
        def bump(mk, h):
            mk = copy.deepcopy(mk)
            mk.ir_normal_vol[self.ccy] += h
            return mk
        return [dict(risk_class="GIRR", bucket=self.ccy, qualifier=self.curve, subtype="Yield",
                     expiry=self.expiry, underlying_tenor=self.tenor, bump=bump,
                     vol=m.ir_normal_vol[self.ccy], h=BP)]


@dataclass
class InflationSwap(Instrument):
    trade_id: str
    ccy: str
    notional: float           # > 0 receive inflation
    fixed_rate: float
    maturity: float
    curve: str
    inflation_curve: str
    description: str = "Zero-coupon inflation swap"

    def value(self, m):
        i = m.inflation[self.inflation_curve][1]
        return self.notional * ((1 + i) ** self.maturity - (1 + self.fixed_rate) ** self.maturity) * m.ir[self.curve].df(self.maturity)

    def ir_curves(self):
        return [self.curve]

    def inflation_curves(self):
        return [self.inflation_curve]


@dataclass
class Bond(Instrument):
    """Fixed-coupon bond discounted at risk-free + credit spread (bond basis).
    Also used for securitisation tranches (stylised)."""
    trade_id: str
    ccy: str
    notional: float           # face value, > 0 long
    coupon: float
    maturity: float
    curve: str
    issuer: str
    description: str = "Fixed-rate bond"

    def value(self, m):
        c, sc = m.ir[self.curve], m.credit[(self.issuer, "Bond")]
        times = list(np.arange(1.0, self.maturity + 1e-9, 1.0))
        if not times or abs(times[-1] - self.maturity) > 1e-9:
            times.append(self.maturity)
        pv = 0.0
        prev = 0.0
        for t in times:
            df = math.exp(-(c.zero(t) + sc.s(t)) * t)
            pv += self.coupon * (t - prev) * df
            prev = t
        pv += math.exp(-(c.zero(self.maturity) + sc.s(self.maturity)) * self.maturity)
        return self.notional * pv

    def ir_curves(self):
        return [self.curve]

    def credit_names(self):
        return [(self.issuer, "Bond")]

    def drc_rows(self, m):
        meta = m.credit_meta[self.issuer]
        cls = {"CSR_NS": "NonSec", "CSR_SEC_NCTP": "SecNonCTP", "CSR_SEC_CTP": "CTP"}[meta["risk_class"]]
        mv = self.pv(m)
        face = self.notional * m.fx_rate(self.ccy)
        row = dict(position_id=self.trade_id, drc_class=cls, bucket=meta["drc_bucket"],
                   obligor=self.issuer, seniority=meta.get("seniority", "Senior"),
                   rating=meta.get("rating", "Unrated"), notional=face, market_value=mv,
                   maturity=self.maturity, instrument="bond", rw=meta.get("drc_rw", np.nan),
                   zero_rw=meta.get("zero_rw", False))
        return [row]


@dataclass
class CDS(Instrument):
    """Credit default swap, reduced-form with flat hazard per maturity.
    notional > 0 = protection sold (long credit), < 0 = protection bought."""
    trade_id: str
    ccy: str
    notional: float
    coupon: float
    maturity: float
    curve: str
    name: str
    recovery: float = 0.4
    leverage: float = 1.0     # > 1 for a stylised CTP tranche
    description: str = "Credit default swap"

    def value(self, m):
        c, sc = m.ir[self.curve], m.credit[(self.name, "CDS")]
        s = sc.s(self.maturity)
        lam = s / (1 - self.recovery)
        rpv01 = sum(0.25 * c.df(t) * math.exp(-lam * t)
                    for t in np.arange(0.25, self.maturity + 1e-9, 0.25))
        # protection seller receives coupon, pays (spread-equivalent) protection
        return self.leverage * self.notional * (self.coupon - s) * rpv01

    def ir_curves(self):
        return [self.curve]

    def credit_names(self):
        return [(self.name, "CDS")]

    def drc_rows(self, m):
        meta = m.credit_meta[self.name]
        cls = {"CSR_NS": "NonSec", "CSR_SEC_NCTP": "SecNonCTP", "CSR_SEC_CTP": "CTP"}[meta["risk_class"]]
        mtm = self.pv(m)
        nominal = self.notional * m.fx_rate(self.ccy)
        if cls == "NonSec":
            # MAR22.14 Table 1: bond-equivalent market value = notional + MtM, P&L = MtM
            return [dict(position_id=self.trade_id, drc_class=cls, bucket=meta["drc_bucket"],
                         obligor=self.name, seniority="Senior", rating=meta.get("rating", "Unrated"),
                         notional=nominal, market_value=nominal + mtm, maturity=self.maturity,
                         instrument="cds", zero_rw=meta.get("zero_rw", False))]
        # securitisation / CTP: JTD = market value of the (bond-equivalent) exposure
        tranche = self.leverage != 1
        return [dict(position_id=self.trade_id, drc_class=cls, bucket=meta["drc_bucket"],
                     obligor=meta.get("tranche_id", self.name) if tranche else self.name,
                     seniority="Senior",
                     rating=meta.get("rating", "Unrated"), notional=nominal,
                     market_value=nominal + mtm, maturity=self.maturity,
                     instrument="cds_tranche" if self.leverage != 1 else "cds_index",
                     rw=meta.get("drc_rw", np.nan) if self.leverage != 1 else np.nan)]


@dataclass
class EquityPosition(Instrument):
    trade_id: str
    ccy: str
    name: str
    shares: float
    description: str = "Cash equity"
    drc_maturity: float = 1.0
    instrument_type: str = "cash_equity"
    hedge_maturity: float = float("nan")

    def value(self, m):
        return self.shares * m.eq_spot[self.name]

    def equities(self):
        return [self.name]

    def drc_rows(self, m):
        if m.eq_bucket[self.name] in (12, 13):
            return []  # index: look-through required for DRC (MAR22.5), not modelled
        mv = self.pv(m)
        return [dict(position_id=self.trade_id, drc_class="NonSec", bucket="Corporates",
                     obligor=self.name, seniority="Equity", rating=m.credit_meta.get(self.name, {}).get("rating", "Unrated"),
                     notional=mv, market_value=mv, maturity=self.drc_maturity,
                     instrument=self.instrument_type, hedge_maturity=self.hedge_maturity)]


@dataclass
class EquityForward(Instrument):
    trade_id: str
    ccy: str
    name: str
    shares: float
    strike: float
    maturity: float
    curve: str
    description: str = "Equity forward"

    def value(self, m):
        df = m.ir[self.curve].df(self.maturity)
        fwd = m.eq_spot[self.name] * math.exp(-m.eq_repo[self.name] * self.maturity) / df
        return self.shares * (fwd - self.strike) * df

    def ir_curves(self):
        return [self.curve]

    def equities(self):
        return [self.name]

    def drc_rows(self, m):
        if m.eq_bucket[self.name] in (12, 13):
            return []
        df = m.ir[self.curve].df(self.maturity)
        notional = self.shares * self.strike * df * m.fx_rate(self.ccy)
        mv = self.pv(m)
        # long forward: loses (notional - MtM) on default; JTD = notional + P&L with P&L = MtM
        return [dict(position_id=self.trade_id, drc_class="NonSec", bucket="Corporates",
                     obligor=self.name, seniority="Equity", rating="Unrated",
                     notional=notional, market_value=notional + mv, maturity=self.maturity,
                     instrument="equity_forward")]


@dataclass
class EquityOption(Instrument):
    trade_id: str
    ccy: str
    name: str
    quantity: float           # > 0 long
    strike: float
    expiry: float
    curve: str
    call: bool = True
    description: str = "Equity option (Black-Scholes)"
    optional: bool = True

    def value(self, m):
        S, K, T = m.eq_spot[self.name], self.strike, self.expiry
        r = m.ir[self.curve].zero(T)
        q = m.eq_repo.get(self.name, 0.0)
        v = m.eq_vol[self.name]
        d1 = (math.log(S / K) + (r - q + 0.5 * v * v) * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)
        if self.call:
            px = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            px = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        return self.quantity * px

    def ir_curves(self):
        return [self.curve]

    def equities(self):
        return [self.name]

    def vega_factors(self, m):
        def bump(mk, h):
            mk = copy.deepcopy(mk)
            mk.eq_vol[self.name] += h
            return mk
        return [dict(risk_class="EQ", bucket=m.eq_bucket[self.name], qualifier=self.name,
                     subtype="Spot", expiry=self.expiry, underlying_tenor=np.nan, bump=bump,
                     vol=m.eq_vol[self.name], h=0.01)]

    def drc_rows(self, m):
        if m.eq_bucket[self.name] in (12, 13):
            return []  # index options: look-through required (MAR22.5), not modelled
        mv = self.pv(m)                       # signed (negative for sold options)
        fx = m.fx_rate(self.ccy)
        base = dict(position_id=self.trade_id, drc_class="NonSec", bucket="Corporates",
                    obligor=self.name, seniority="Equity", rating="Unrated",
                    maturity=self.expiry, instrument="equity_option")
        if self.call:
            # default extinguishes the call: notional 0, P&L = value lost / gained
            return [dict(base, notional=0.0, market_value=mv, pnl=mv)]
        strike_amt = abs(self.quantity) * self.strike * fx
        if self.quantity < 0:   # sold put: long credit exposure (MAR22.10(2))
            return [dict(base, notional=strike_amt, market_value=strike_amt - abs(mv),
                         pnl=-abs(mv))]
        # bought put: short credit exposure, gains strike minus premium on default
        return [dict(base, notional=-strike_amt, market_value=-strike_amt + mv, pnl=mv)]


@dataclass
class FXForward(Instrument):
    """Receive notional_buy in ccy_buy, pay notional_sell in ccy_sell at T."""
    trade_id: str
    ccy_buy: str
    notional_buy: float
    ccy_sell: str
    notional_sell: float
    maturity: float
    curve_buy: str
    curve_sell: str
    description: str = "FX forward"
    ccy: str = "EUR"

    def value(self, m):  # in reporting currency directly
        b = self.notional_buy * m.ir[self.curve_buy].df(self.maturity) * m.fx_rate(self.ccy_buy)
        s = self.notional_sell * m.ir[self.curve_sell].df(self.maturity) * m.fx_rate(self.ccy_sell)
        return b - s

    def pv(self, m):
        return self.value(m)

    def ir_curves(self):
        return [self.curve_buy, self.curve_sell]

    def fx_ccys(self):
        return [c for c in (self.ccy_buy, self.ccy_sell)]


@dataclass
class FXOption(Instrument):
    """Garman-Kohlhagen option on the rate X = FOR/DOM (units of DOM per FOR).
    notional in FOR; value converted to the reporting currency."""
    trade_id: str
    foreign: str
    domestic: str
    notional: float
    strike: float
    expiry: float
    curve_dom: str
    curve_for: str
    call: bool = True
    description: str = "FX option (Garman-Kohlhagen)"
    optional: bool = True
    ccy: str = "EUR"

    def spot(self, m):
        return m.fx_rate(self.foreign) / m.fx_rate(self.domestic)

    def pair(self):
        return f"{self.foreign}/{self.domestic}"

    def value(self, m):
        S, K, T = self.spot(m), self.strike, self.expiry
        rd = m.ir[self.curve_dom].zero(T)
        rf = m.ir[self.curve_for].zero(T)
        v = m.fx_vol[self.pair()]
        d1 = (math.log(S / K) + (rd - rf + 0.5 * v * v) * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)
        if self.call:
            px = S * math.exp(-rf * T) * norm.cdf(d1) - K * math.exp(-rd * T) * norm.cdf(d2)
        else:
            px = K * math.exp(-rd * T) * norm.cdf(-d2) - S * math.exp(-rf * T) * norm.cdf(-d1)
        return self.notional * px * m.fx_rate(self.domestic)

    def pv(self, m):
        return self.value(m)

    def fx_ccys(self):
        return [c for c in (self.foreign, self.domestic)]

    def ir_curves(self):
        return [self.curve_dom, self.curve_for]

    def references_reporting(self, m):
        return m.reporting_ccy in (self.foreign, self.domestic)

    def vega_factors(self, m):
        def bump(mk, h):
            mk = copy.deepcopy(mk)
            mk.fx_vol[self.pair()] += h
            return mk
        return [dict(risk_class="FX", bucket=self.pair(), qualifier=self.pair(), subtype="",
                     expiry=self.expiry, underlying_tenor=np.nan, bump=bump,
                     vol=m.fx_vol[self.pair()], h=0.01)]


@dataclass
class CommodityFuture(Instrument):
    trade_id: str
    ccy: str
    name: str
    quantity: float
    strike: float
    maturity: float
    curve: str
    description: str = "Commodity future / forward"

    def value(self, m):
        return self.quantity * (m.comm[self.name].price(self.maturity) - self.strike) * m.ir[self.curve].df(self.maturity)

    def ir_curves(self):
        return [self.curve]

    def commodities(self):
        return [self.name]


@dataclass
class CommodityOption(Instrument):
    trade_id: str
    ccy: str
    name: str
    quantity: float
    strike: float
    expiry: float
    curve: str
    call: bool = True
    description: str = "Commodity option (Black-76)"
    optional: bool = True

    def value(self, m):
        F, K, T = m.comm[self.name].price(self.expiry), self.strike, self.expiry
        v = m.comm_vol[self.name]
        df = m.ir[self.curve].df(T)
        d1 = (math.log(F / K) + 0.5 * v * v * T) / (v * math.sqrt(T))
        d2 = d1 - v * math.sqrt(T)
        px = F * norm.cdf(d1) - K * norm.cdf(d2) if self.call else K * norm.cdf(-d2) - F * norm.cdf(-d1)
        return self.quantity * df * px

    def ir_curves(self):
        return [self.curve]

    def commodities(self):
        return [self.name]

    def vega_factors(self, m):
        def bump(mk, h):
            mk = copy.deepcopy(mk)
            mk.comm_vol[self.name] += h
            return mk
        c = m.comm[self.name]
        return [dict(risk_class="COMM", bucket=c.bucket, qualifier=self.name, subtype=c.location,
                     expiry=self.expiry, underlying_tenor=np.nan, bump=bump,
                     vol=m.comm_vol[self.name], h=0.01, eu_ets=c.eu_ets)]


# ==========================================================================
# Sensitivity generation
# ==========================================================================


def _row(inst, rc, measure, bucket, qualifier, tenor, amount, subtype="", underlying=np.nan, **kw):
    r = dict(trade_id=inst.trade_id, risk_class=rc, measure=measure, bucket=bucket,
             qualifier=qualifier, tenor=tenor, underlying_tenor=underlying, subtype=subtype,
             amount=amount, cvr_up=0.0, cvr_down=0.0, eu_ets=False, covered_aa=False,
             fx_cross_option=False)
    r.update(kw)
    return r


def delta_sensitivities(inst: Instrument, m: Market) -> list[dict]:
    rows = []
    v0 = inst.pv(m)
    # GIRR yield curves (MAR21.19)
    for name in inst.ir_curves():
        c = m.ir[name]
        for k, t in enumerate(c.tenors):
            mk = copy.deepcopy(m)
            mk.ir[name].zeros = _bump_pillar(c.zeros, c.tenors, k, BP)
            s = (inst.pv(mk) - v0) / BP
            if abs(s) > 1e-6:
                rows.append(_row(inst, "GIRR", "Delta", c.ccy, name, t, s, "Yield"))
    for name in inst.inflation_curves():
        ccy, rate = m.inflation[name]
        mk = copy.deepcopy(m)
        mk.inflation[name] = (ccy, rate + BP)
        s = (inst.pv(mk) - v0) / BP
        rows.append(_row(inst, "GIRR", "Delta", ccy, name, np.nan, s, "Inflation"))
    # CSR (MAR21.20)
    for key in inst.credit_names():
        name, basis = key
        sc = m.credit[key]
        meta = m.credit_meta[name]
        for k, t in enumerate(sc.tenors):
            mk = copy.deepcopy(m)
            mk.credit[key].spreads = _bump_pillar(sc.spreads, sc.tenors, k, BP)
            s = (inst.pv(mk) - v0) / BP
            if abs(s) > 1e-6:
                rows.append(_row(inst, meta["risk_class"], "Delta", meta["bucket"],
                                 meta.get("csr_qualifier", name), t, s, basis,
                                 covered_aa=meta.get("covered_aa", False)))
    # Equity spot and repo (MAR21.21-21.22)
    for name in inst.equities():
        mk = copy.deepcopy(m)
        mk.eq_spot[name] *= 1.01
        rows.append(_row(inst, "EQ", "Delta", m.eq_bucket[name], name, np.nan,
                         (inst.pv(mk) - v0) / 0.01, "Spot"))
        if name in m.eq_repo and not isinstance(inst, EquityPosition):
            mk = copy.deepcopy(m)
            mk.eq_repo[name] += BP
            s = (inst.pv(mk) - v0) / BP
            if abs(s) > 1e-6:
                rows.append(_row(inst, "EQ", "Delta", m.eq_bucket[name], name, np.nan, s, "Repo"))
    # Commodity (MAR21.23)
    for name in inst.commodities():
        c = m.comm[name]
        for k, t in enumerate(c.tenors):
            mk = copy.deepcopy(m)
            mk.comm[name].prices = _bump_pillar(c.prices, c.tenors, k, 0.01, relative=True)
            s = (inst.pv(mk) - v0) / 0.01
            if abs(s) > 1e-6:
                rows.append(_row(inst, "COMM", "Delta", c.bucket, name, t, s, c.location,
                                 eu_ets=c.eu_ets))
    # FX (MAR21.24): every non-reporting currency the instrument depends on
    for ccy in sorted(set(inst.fx_ccys())):
        if ccy == m.reporting_ccy:
            continue
        mk = copy.deepcopy(m)
        mk.fx[ccy] *= 1.01
        s = (inst.pv(mk) - v0) / 0.01
        if abs(s) > 1e-6:
            rows.append(_row(inst, "FX", "Delta", ccy, ccy, np.nan, s))
    return rows


def vega_sensitivities(inst: Instrument, m: Market) -> list[dict]:
    """vega x implied vol, mapped to option maturity (and underlying) tenors (MAR21.25)."""
    rows = []
    v0 = inst.pv(m)
    for vf in inst.vega_factors(m):
        mk = vf["bump"](m, vf["h"])
        sens = (inst.pv(mk) - v0) / vf["h"] * vf["vol"]
        w_opt = _interp_weights(vf["expiry"], P.VEGA_TENORS)
        if vf["risk_class"] == "GIRR":
            w_und = _interp_weights(vf["underlying_tenor"], P.VEGA_TENORS)
        else:
            w_und = {np.nan: 1.0}
        for to, wo in w_opt.items():
            for tu, wu in w_und.items():
                rows.append(_row(inst, vf["risk_class"], "Vega", vf["bucket"], vf["qualifier"],
                                 to, sens * wo * wu, vf["subtype"], tu,
                                 eu_ets=vf.get("eu_ets", False)))
    return rows


def curvature_sensitivities(inst: Instrument, m: Market, regime: P.Regime,
                            deltas: list[dict]) -> list[dict]:
    """CVR_up / CVR_down per curvature risk factor (MAR21.5), in reporting ccy."""
    if not inst.optional:
        return []
    rows = []
    v0 = inst.pv(m)

    def cvr(bumped_up, bumped_dn, rw, s):
        up = -(inst.pv(bumped_up) - v0 - rw * s)
        dn = -(inst.pv(bumped_dn) - v0 + rw * s)
        return up, dn

    # GIRR: all curves of a currency shifted in parallel (absolute shift)
    ccys = sorted({m.ir[n].ccy for n in inst.ir_curves()})
    for ccy in ccys:
        rw = curvature_rw("GIRR", ccy, regime)
        s = sum(d["amount"] for d in deltas if d["risk_class"] == "GIRR" and d["bucket"] == ccy
                and d["subtype"] == "Yield")
        mu, md = copy.deepcopy(m), copy.deepcopy(m)
        for n, c in m.ir.items():
            if c.ccy == ccy:
                mu.ir[n].zeros = [z + rw for z in c.zeros]
                md.ir[n].zeros = [z - rw for z in c.zeros]
        up, dn = cvr(mu, md, rw, s)
        rows.append(_row(inst, "GIRR", "Curvature", ccy, ccy, np.nan, 0.0, cvr_up=up, cvr_down=dn))
    # Equity spot (relative shift)
    for name in inst.equities():
        b = m.eq_bucket[name]
        rw = curvature_rw("EQ", b, regime)
        s = sum(d["amount"] for d in deltas if d["risk_class"] == "EQ" and d["qualifier"] == name
                and d["subtype"] == "Spot")
        mu, md = copy.deepcopy(m), copy.deepcopy(m)
        mu.eq_spot[name] *= 1 + rw
        md.eq_spot[name] *= 1 - rw
        up, dn = cvr(mu, md, rw, s)
        rows.append(_row(inst, "EQ", "Curvature", b, name, np.nan, 0.0, cvr_up=up, cvr_down=dn))
    # Commodity: whole curve, relative shift (FAQ1 to MAR21.99)
    for name in inst.commodities():
        c = m.comm[name]
        rw = curvature_rw("COMM", c.bucket, regime, c.eu_ets)
        s = sum(d["amount"] for d in deltas if d["risk_class"] == "COMM" and d["qualifier"] == name)
        mu, md = copy.deepcopy(m), copy.deepcopy(m)
        mu.comm[name].prices = [p * (1 + rw) for p in c.prices]
        md.comm[name].prices = [p * (1 - rw) for p in c.prices]
        up, dn = cvr(mu, md, rw, s)
        rows.append(_row(inst, "COMM", "Curvature", c.bucket, name, np.nan, 0.0, cvr_up=up,
                         cvr_down=dn, eu_ets=c.eu_ets))
    # FX: each non-reporting currency (relative shift)
    cross = isinstance(inst, FXOption) and not inst.references_reporting(m)
    for ccy in sorted(set(inst.fx_ccys())):
        if ccy == m.reporting_ccy:
            continue
        rw = curvature_rw("FX", ccy, regime)
        s = sum(d["amount"] for d in deltas if d["risk_class"] == "FX" and d["bucket"] == ccy)
        mu, md = copy.deepcopy(m), copy.deepcopy(m)
        mu.fx[ccy] *= 1 + rw
        md.fx[ccy] *= 1 - rw
        up, dn = cvr(mu, md, rw, s)
        rows.append(_row(inst, "FX", "Curvature", ccy, ccy, np.nan, 0.0, cvr_up=up, cvr_down=dn,
                         fx_cross_option=cross))
    return rows


def portfolio_sensitivities(instruments: list[Instrument], m: Market, regime: P.Regime):
    """Sensitivity table, DRC positions and a trade summary for a portfolio."""
    import pandas as pd

    sens, drc, trades = [], [], []
    for inst in instruments:
        d = delta_sensitivities(inst, m)
        v = vega_sensitivities(inst, m)
        c = curvature_sensitivities(inst, m, regime, d)
        sens += d + v + c
        drc += inst.drc_rows(m)
        trades.append(dict(trade_id=inst.trade_id, description=inst.description,
                           type=type(inst).__name__, ccy=getattr(inst, "ccy", ""),
                           pv=inst.pv(m), optionality=inst.optional))
    return pd.DataFrame(sens), pd.DataFrame(drc), pd.DataFrame(trades)
