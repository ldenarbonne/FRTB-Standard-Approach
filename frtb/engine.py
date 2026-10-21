"""Total FRTB standardised approach capital requirement (MAR20.4)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import params as P
from .drc import DRCResult, RRAOResult, compute_drc, compute_rrao
from .sbm import SBMResult, compute_sbm


@dataclass
class FRTBResult:
    regime: P.Regime
    sbm: SBMResult
    drc: DRCResult
    rrao: RRAOResult
    sbm_after_multiplier: float
    total_before_bank_multiplier: float
    total: float
    rwa: float

    def summary(self) -> pd.DataFrame:
        rows = [("Sensitivities-based method", self.sbm.capital)]
        if self.regime.eu and self.regime.eu_sbm_multiplier != 1.0:
            rows.append((f"SBM after EU phase-in multiplier ×{self.regime.eu_sbm_multiplier:.2f}",
                         self.sbm_after_multiplier))
        rows += [("Default risk capital", self.drc.capital),
                 ("Residual risk add-on", self.rrao.capital),
                 ("Total own funds requirement", self.total_before_bank_multiplier)]
        if self.regime.eu and self.regime.eu_bank_multiplier != 1.0:
            rows.append((f"Total after bank-specific multiplier ×{self.regime.eu_bank_multiplier:.3f}",
                         self.total))
        rows.append(("Risk-weighted assets (× 12.5)", self.rwa))
        return pd.DataFrame(rows, columns=["Component", "Amount"]).set_index("Component")


def compute_frtb(sens: pd.DataFrame, drc_positions: pd.DataFrame | None,
                 rrao_positions: pd.DataFrame | None, regime: P.Regime) -> FRTBResult:
    sbm = compute_sbm(sens, regime)
    drc = compute_drc(drc_positions, regime)
    rrao = compute_rrao(rrao_positions, regime)
    sbm_adj = sbm.capital * (regime.eu_sbm_multiplier if regime.eu else 1.0)
    total = sbm_adj + drc.capital + rrao.capital
    bank_mult = min(regime.eu_bank_multiplier, 1.0) if regime.eu else 1.0
    final = total * bank_mult
    return FRTBResult(regime, sbm, drc, rrao, sbm_adj, total, final, 12.5 * final)
