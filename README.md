# FRTB Standardised Approach Calculator

An interactive Streamlit application that computes the **market risk own funds requirement under the FRTB standardised approach**:

* the **sensitivities-based method** (delta, vega and curvature) for all **seven risk classes**, under the three correlation scenarios;
* the **default risk capital** requirement (non-securitisations, securitisations non-CTP, correlation trading portfolio);
* the **residual risk add-on**.

Parameters follow the **Basel Framework MAR21-23** (version effective 1 January 2023, FAQs up to March 2026). A switch applies the **EU variants**: CRR3 and the transitional provisions of **Commission Delegated Regulation (EU) 2026/1221**, applicable from 1 January 2027 to 31 December 2029.

**Live app:** `https://<your-app-name>.streamlit.app` *(to be completed after deployment)*

![tests](https://github.com/ldenarbonne/frtb-sa-calculator/actions/workflows/tests.yml/badge.svg)

---

## 1. Two ways to feed the engine

| Input | Description |
|---|---|
| **Demo portfolio** | 35 fictitious trades priced in the tool: swaps, swaptions, an inflation swap, corporate, sovereign and covered bonds, CDS, a securitisation tranche, a stylised CTP tranche with its index hedge, equities, equity forward and options, FX forward and options, commodity futures and options (Brent, WTI, gold, EU ETS allowances, power, TTF gas). Sensitivities are obtained by **bump-and-reprice** following MAR21.19-21.25, and CVR by the prescribed up/down shocks (MAR21.5). |
| **Your own files** | Sensitivities, DRC positions and RRAO positions as CSV (templates in `templates/`, format in the *Methodology* tab). Non-standard tenors can be mapped automatically to the prescribed grid by linear interpolation (MAR21.8 footnote 5). Every row is validated before calculation. |

## 2. What the engine covers

| Component | Coverage | References |
|---|---|---|
| GIRR | Delta per curve and tenor (10 tenors), inflation and cross-currency basis; vega by option × underlying maturity; curvature per currency | MAR21.8, 21.41-21.50, 21.93 |
| CSR non-securitisation | 18 buckets, bond/CDS basis, sector × rating cross-bucket correlations, "other sector" bucket | MAR21.51-21.57 |
| CSR securitisation (CTP) | 16 buckets, 99% basis correlation | MAR21.58-21.61 |
| CSR securitisation (non-CTP) | 25 buckets (senior IG, non-senior IG × 1.25, HY × 1.75), bucket 25 added without diversification | MAR21.62-21.71 |
| Equity | 13 buckets, spot and repo, large / small cap vega liquidity horizons | MAR21.72-21.80, 21.92 |
| Commodity | 11 buckets, 11 tenors, delivery-location basis | MAR21.81-21.85 |
| FX | Specified currency pairs and first-order crosses (√2), curvature scalar 1.5 | MAR21.86-21.89, 21.98 |
| Correlation scenarios | Low / medium / high on ρ and γ (squared values for curvature), maximum taken | MAR21.6-21.7, 21.100-21.101 |
| DRC | Gross JTD (LGD, notional, P&L), maturity scaling with 3-month floor, offsetting by seniority, hedge benefit ratio, CTP 50% rule | MAR22 |
| RRAO | 1.0% exotic underlyings, 0.1% other residual risks | MAR23 |

## 3. EU variants

| Provision | Effect in the tool |
|---|---|
| DR 2026/1221, Art. 495s(1) | SBM result multiplied by 0.9 until 31/12/2029 |
| DR 2026/1221, Art. 495p | RRAO multiplier of 0 for realised-volatility, finite-exercise and CMS-spread options |
| DR 2026/1221, Art. 495q(a) and (b) | DRC maturity options for cash equity hedges and equity derivatives (selectable separately) |
| DR 2026/1221, Art. 495r | Tenor correlation for EU ETS carbon divided by 99.1% |
| DR 2026/1221, Art. 495v | Optional bank-specific multiplier ≤ 1 (user input) |
| CRR3, Art. 325as | 40% risk weight for EU ETS carbon allowances (60% for non-EU ETS carbon) |
| MAR22.7 / CRR3 | 0% default risk weight for exposures with a 0% credit-risk weight |

The application shows the BCBS and EU figures side by side.

## 4. Illustrative results (demo book, EUR)

| Component | BCBS | EU |
|---|---|---|
| SBM (binding scenario: low correlations) | 32.4 m | 30.8 m |
| SBM after 0.9 phase-in | – | 27.7 m |
| DRC | 6.2 m | 4.8 m |
| RRAO | 0.13 m | 0.03 m |
| **Total own funds requirement** | **38.8 m** | **32.5 m** |
| RWA (× 12.5) | 485 m | 406 m |

The EU figure is 16% lower. The main drivers are:
* the 0.9 phase-in on the SBM;
* the zero DRC risk weight for the sovereign bond;
* the 40% EU ETS risk weight on the carbon future;
* the RRAO exemptions;
* the DRC maturity alignment of the equity hedge.

## 5. Project structure

```
frtb-sa-calculator/
├── app.py                 # Streamlit front-end
├── frtb/
│   ├── params.py          # All regulatory parameters with paragraph references
│   ├── sbm.py             # Delta, vega, curvature aggregation, correlation scenarios
│   ├── drc.py             # Default risk charge and residual risk add-on
│   ├── engine.py          # Total own funds requirement
│   ├── pricing.py         # Stylised pricers and bump-and-reprice sensitivity generator
│   ├── demo.py            # Demonstration market and trading book
│   └── schema.py          # CSV formats, validation, tenor mapping
├── templates/             # Pre-filled CSV templates (demo book)
├── tests/                 # 50 pytest tests, most checked against hand calculations
└── .github/workflows/     # CI on Python 3.11 and 3.12
```

## 6. Run locally

```bash
git clone https://github.com/ldenarbonne/frtb-sa-calculator.git
cd frtb-sa-calculator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py
```

## 7. Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (public).
2. On [share.streamlit.io](https://share.streamlit.io), click **Create app**, choose the repository, branch `main` and main file `app.py`; select Python 3.12 in *Advanced settings*.
3. Paste the app URL at the top of this README.

## 8. Limitations

* Demo pricers are stylised: single-curve swaptions, flat volatilities and reduced-form credit. They generate realistic sensitivity profiles, not production valuations.
* Default risk weights for securitisation and CTP tranches are user inputs derived from the banking-book securitisation framework.
* No look-through of index positions for DRC; index positions are excluded from DRC in the demo.
* CRR3 renumbers the commodity buckets to separate EU ETS and non-EU ETS carbon; the tool applies the EU ETS risk weight within bucket 3.
* The curvature across-bucket quantity is floored at zero (no alternative specification is prescribed for curvature).
* Educational tool: it does not replace a regulatory calculation or supervisory reporting.

## 9. References

* Basel Committee on Banking Supervision, *Basel Framework*, MAR20-MAR23 (version effective 1 January 2023).
* Regulation (EU) 2024/1623 (CRR3) amending Regulation (EU) No 575/2013.
* Commission Delegated Regulation (EU) 2026/1221 of 4 June 2026 (CRR Articles 495i to 495v).

---

**Author:** [Laurent Motais de Narbonne](https://www.linkedin.com/in/laurent-motais-de-narbonne/), Senior Risk Manager. Former Calypso trainer (Market Risk and FRTB). MIT licence.
