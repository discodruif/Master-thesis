# Thesis Proposal Outline (Aligned with Lead Proposal)

**Title:** Machine Learning for Cross-Sectional Option Return Prediction: Out-of-Sample Evidence and the Limits of Implementability
**Author:** Maurits van Eck
**ANR:** 2062644
**Program:** MSc Finance, Tilburg University
**Date:** February 2026
**Status:** Lead Proposal — full version in `drafts/proposals/FINAL-thesis-proposal.md`

---

## Structure: Tilburg 5-Part Format

| Part | Content | Word limit |
|---|---|---|
| **Part 1** | Research Question | ~200 words |
| **Part 2** | Literature Review | MAX 1 page |
| **Part 3** | Research Plan (incl. regression equations) | ~500 words |
| **Part 4** | Data Sources (incl. descriptive statistics, confirm access) | ~400 words |
| **Part 5** | References (APA) | — |

---

## Part 1: Research Question

**Main RQ:** *To what extent do nonlinear ML models improve out-of-sample prediction of cross-sectional option returns beyond linear benchmarks, and does the resulting long-short alpha remain economically significant after realistic transaction costs?*

**H1 (Predictability):** Nonlinear ML (XGBoost, MLP) achieves higher OOS R-squared than OLS/Lasso for weekly option returns, driven by volatility-risk-premium proxies.

**H2 (Implementability):** The long-short portfolio's net-of-cost Sharpe ratio is significantly > 0, with break-even spread exceeding the median observed half-spread.

---

## Part 2: Literature Review

| Theme | Key papers |
|---|---|
| Option return predictability | Goyal & Saretto (2009, *JFE*); Cao & Han (2013, *JF*); Christoffersen et al. (2013, *RFS*) |
| ML in asset pricing | Gu, Kelly, & Xiu (2020, *RFS*); Bali et al. (2023, *RFS*) |
| Transaction costs | Muravyev (2016, *JF*); Novy-Marx & Velikov (2016, *RFS*) |
| Multiple testing | Harvey, Liu, & Zhu (2016, *RFS*) |

**Contributions:** (i) Transaction cost boundary; (ii) Risk-premium vs. mispricing decomposition; (iii) Multiple-testing-aware inference.

---

## Part 3: Research Plan

- **Returns:** Raw + delta-hedged (5-day holding period)
- **Features:** Option-level (IV, moneyness, TTM, spread, Greeks) + IV surface (skew, term structure) + underlying (momentum, RV, VIX)
- **Models:** OLS, Lasso, XGBoost, RF, MLP — expanding window
- **Evaluation:** OOS R-squared, Diebold-Mariano tests, decile long-short portfolios
- **Cost model:** TC = lambda * half-spread; lambda in {0.5, 1.0, 1.5, 2.0}; break-even lambda*
- **Regressions:** (A) Fama-MacBeth, (B) ML spanning test, (C) risk-premium decomposition, (D) factor alpha
- **Robustness:** VIX regimes, moneyness buckets, holding period variation, feature ablation

---

## Part 4: Data

- **OptionMetrics IvyDB US via WRDS (access verified)**
- SPX options, Jan 2010 - Dec 2024; OOS: Jan 2013 - Dec 2024
- Filters: Bid > 0, Mid >= $0.50, moneyness [0.85, 1.15], TTM [14, 180], volume >= 50, OI >= 500
- Expected sample: ~2.5M option-week observations
- Supplementary: FF factors, VIX, realized volatility

---

## Part 5: References (12 papers, all top-journal)

Bali et al. (2023, *RFS*) | Black & Scholes (1973, *JPE*) | Cao & Han (2013, *JFE*) | Christoffersen et al. (2013, *RFS*) | Diebold & Mariano (1995, *JBES*) | Goyal & Saretto (2009, *JFE*) | Gu, Kelly, & Xiu (2020, *RFS*) | Harvey et al. (2016, *RFS*) | Muravyev (2016, *JF*) | Newey & West (1987, *Econometrica*) | Novy-Marx & Velikov (2016, *RFS*) | Petersen (2009, *RFS*)

Full APA citations in `drafts/proposals/FINAL-thesis-proposal.md`.
