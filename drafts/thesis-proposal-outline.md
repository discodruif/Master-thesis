# Thesis Proposal Outline (Aligned with Lead Proposal)

**Title:** Machine Learning for Cross-Sectional Option Return Prediction: Out-of-Sample Evidence and the Limits of Implementability
**Author:** Maurits van Eck
**ANR:** 2062644
**Program:** MSc Finance, Tilburg University
**Date:** February 2026
**Status:** Lead Proposal — full version in `drafts/proposals/2026-02-16-proposal-3.md`

---

## Part 1: Research Question

Option returns exhibit predictable cross-sectional variation linked to implied volatility, skew, and underlying momentum. Bali et al. (2023, *RFS*) show that ML exploits these patterns, but the critical question is whether alpha survives option market frictions.

**Main RQ:** *To what extent do nonlinear ML models improve out-of-sample prediction of cross-sectional option returns beyond linear benchmarks, and does the resulting long-short alpha remain economically significant after realistic transaction costs?*

**H1 (Predictability):** Nonlinear ML (XGBoost, MLP) achieves higher OOS R-squared than OLS/Lasso for weekly option returns, driven by volatility-risk-premium proxies (IV level, skew, term structure slope).

**H2 (Implementability):** The long-short portfolio's net-of-cost Sharpe ratio is significantly > 0, with break-even spread exceeding the median observed half-spread.

**Key insight:** H1 can hold while H2 fails — implying ML captures real predictability that is unimplementable, informing option market efficiency.

---

## Part 2: Literature Review

| Theme | Key papers | Role in thesis |
|---|---|---|
| Option return predictability | Goyal & Saretto (2009, *JFE*); Cao & Han (2013, *JF*); Christoffersen, Heston, & Jacobs (2013, *RFS*) | Establishes that IV-realized vol gap, idiosyncratic vol, and surface shape predict option returns |
| ML in asset pricing | Gu, Kelly, & Xiu (2020, *RFS*); Bali et al. (2023, *RFS*) | Methodological framework (rolling splits, portfolio sorts) and direct precedent for ML in options |
| Transaction costs & implementability | Muravyev (2016, *JF*); Novy-Marx & Velikov (2016, *RFS*) | Option spreads are wide; equity anomalies often vanish after costs — gap: no systematic cost analysis for ML option strategies |
| Multiple testing | Harvey, Liu, & Zhu (2016, *RFS*) | Justifies |t| > 3.0 threshold for alpha significance |

**Contribution:** (i) Transaction cost boundary for ML option signals; (ii) Risk-premium vs. mispricing decomposition via delta-hedged returns; (iii) Multiple-testing-aware inference.

---

## Part 3: Research Plan

### Return measures
- **Raw:** r(i,t,t+h) = [Mid(i,t+h) - Mid(i,t)] / Mid(i,t)
- **Delta-hedged:** r_dh = r - Delta * r_underlying (isolates non-directional component)

### Feature groups (winsorized 1/99 pctile, cross-sectionally standardized)
1. **Option-level:** IV, ln(K/F), TTM, normalized spread, log volume, log OI, delta, gamma, vega
2. **IV surface:** ATM level, risk reversal (skew), butterfly (curvature), term structure slope
3. **Underlying:** 5d/21d/63d returns, realized vol, VIX level, VIX change

### Models
OLS, Lasso, XGBoost, Random Forest, MLP — expanding window (train up to t-4, validate t-3 to t-1, predict t)

### Evaluation framework
1. **Statistical:** OOS R-squared, Diebold-Mariano test vs. OLS
2. **Economic:** Decile long-short portfolios, Sharpe ratio gross and net of costs
3. **Cost model:** TC = lambda * half-spread; lambda in {0.5, 1.0, 1.5, 2.0}; break-even lambda*
4. **Risk adjustment:** Factor regression alpha (MKT, SMB, HML, MOM, dVIX) with HAC standard errors

### Regression tests
- **(A) Fama-MacBeth:** r = a + b'X + e (weekly cross-sections, NW standard errors)
- **(B) Spanning:** r = a + phi * ML_signal + b'X + e (double-clustered SE per Petersen 2009)
- **(C) Decomposition:** r_dh = a + psi * ML_signal + gamma * VRP + delta * Skew + controls + u
- **(D) Factor alpha:** r_net_LS = alpha + factor loadings + epsilon (HAC SE, |t| > 3.0 threshold)

### Robustness
VIX regime splits | moneyness buckets | holding period variation (1/5/10/21d) | buy-at-ask/sell-at-bid | retrain frequency | feature ablation

---

## Part 4: Data

**OptionMetrics IvyDB US via WRDS (access verified)**
- SPX options, Jan 2010 - Dec 2024 (15 years; OOS window: Jan 2013 - Dec 2024)
- Variables: bid/ask/mid, K, TTM, IV, Greeks, volume, OI from optionm.opprcd and optionm.vsurfd
- Supplementary: Fama-French factors, VIX, realized volatility

**Filters:** Bid > 0, Mid >= $0.50, 0.85 <= K/F <= 1.15, 14 <= TTM <= 180 days, volume >= 50, OI >= 500, winsorized 1/99 pctile. Expected sample: ~2.5M option-week observations.

**Descriptive statistics:** See full proposal for expected ranges.

---

## Part 5: References

Bali, T. G., et al. (2023). *Review of Financial Studies, 36*(9).
Black, F., & Scholes, M. (1973). *Journal of Political Economy, 81*(3).
Cao, J., & Han, B. (2013). *Journal of Financial Economics, 108*(1).
Christoffersen, P., Heston, S., & Jacobs, K. (2013). *Review of Financial Studies, 26*(8).
Diebold, F. X., & Mariano, R. S. (1995). *Journal of Business & Economic Statistics, 13*(3).
Goyal, A., & Saretto, A. (2009). *Journal of Financial Economics, 94*(2).
Gu, S., Kelly, B., & Xiu, D. (2020). *Review of Financial Studies, 33*(5).
Harvey, C. R., Liu, Y., & Zhu, H. (2016). *Review of Financial Studies, 29*(1).
Muravyev, D. (2016). *The Journal of Finance, 71*(2).
Newey, W. K., & West, K. D. (1987). *Econometrica, 55*(3).
Novy-Marx, R., & Velikov, M. (2016). *Review of Financial Studies, 29*(1).
Petersen, M. A. (2009). *Review of Financial Studies, 22*(1).

Full APA citations in `drafts/proposals/2026-02-16-proposal-3.md`.

---

## Timeline

| Month | Deliverable |
|---|---|
| Mar 2026 | Data pipeline, filters, descriptive stats (Table 1) |
| Apr 2026 | Model training, OOS R-squared, DM tests (Table 2) |
| May 2026 | Portfolio construction, net returns, factor alphas (Tables 3-4) |
| Jun 2026 | Robustness, SHAP, spanning/decomposition regressions (Tables 5-7) |
| Jul 2026 | Writing, revision, supervisor feedback |

---

## Risks & Mitigation

| Risk | Mitigation |
|---|---|
| Overfitting | Expanding window, early stopping, linear benchmarks, ablation |
| Multiple testing | Harvey et al. threshold (|t| > 3.0), report all specs |
| Look-ahead bias | All features from data at *t*; lagged filters |
| Transaction costs underestimated | Multiple lambda levels, break-even spread, worst-case execution |
| Regime shifts | VIX subsample analysis, report crisis drawdowns |
