# Thesis Proposal — MSc Finance, Tilburg University

## Title Page

**Title:** Machine Learning for Cross-Sectional Option Return Prediction: Out-of-Sample Evidence and the Limits of Implementability

**Name:** Maurits van Eck
**ANR:** 2062644
**Program:** MSc Finance, Tilburg University
**Date:** February 2026

---

## Part 1 — Research Question (~200 words)

The cross-section of option returns exhibits predictable variation linked to implied volatility, moneyness, and underlying momentum (Goyal & Saretto, 2009; Cao & Han, 2013). Bali, Beckmeyer, Moerke, and Weigert (2023) show that machine learning can exploit these patterns to generate economically large long-short option portfolio returns. However, a critical open question is whether such returns survive the transaction costs that characterize option markets — where bid-ask spreads routinely consume 5-15% of option value for out-of-the-money contracts.

**Main Research Question:**
*To what extent do nonlinear machine learning models improve out-of-sample prediction of cross-sectional option returns beyond linear benchmarks, and does the resulting long-short alpha remain economically significant after realistic transaction costs?*

**Sub-hypotheses:**

**H1 (Predictability):** Nonlinear ML models (gradient boosted trees, neural networks) achieve higher out-of-sample R-squared in predicting weekly option returns than OLS and Lasso, with the improvement concentrated in volatility risk premium proxies (IV level, skew, term structure slope).

**H2 (Implementability):** The long-short portfolio formed on ML predictions generates a Sharpe ratio significantly greater than zero after transaction costs modeled at observed bid-ask spreads, with the break-even cost exceeding the median observed half-spread in the tradable universe.

These hypotheses are jointly testable: H1 can hold while H2 fails, which would indicate that ML captures genuine predictability that is nonetheless unimplementable — itself economically informative about option market efficiency.

---

## Part 2 — Literature Review (MAX 1 page)

The Black-Scholes (1973) framework implies that option characteristics should not predict returns beyond compensation for systematic risk. Empirically, this is rejected. Goyal and Saretto (2009) document that the implied-realized volatility gap predicts delta-hedged option returns. Cao and Han (2013) show that high idiosyncratic volatility predicts low delta-hedged returns, consistent with hedging pressure from end-users. Christoffersen, Heston, and Jacobs (2013) demonstrate that IV surface shape carries information about time-varying risk premia. These findings suggest two channels: (i) **time-varying risk premia** and (ii) **limits to arbitrage**. My thesis tests which dominates, because risk-premium-driven predictability should survive transaction costs, while limits-to-arbitrage predictability may not.

Gu, Kelly, and Xiu (2020) establish the methodological standard for ML in cross-sectional return prediction: rolling time-series splits, linear benchmarks, and portfolio-based evaluation. **Bali et al. (2023)** apply this framework to option returns, finding that ML-based long-short strategies earn substantial returns. My thesis extends their work by (i) focusing explicitly on the **transaction cost boundary** and (ii) decomposing predictability into risk-premium vs. mispricing components using delta-hedged returns.

Option markets are characterized by wide spreads, especially for OTM contracts (Muravyev, 2016). Novy-Marx and Velikov (2016) show that trading costs eliminate many equity anomaly profits. No comparable analysis exists for ML-based option strategies — this is the gap my thesis fills.

**Contributions:** (i) I quantify the transaction cost boundary by computing break-even spreads across moneyness/maturity buckets. (ii) I test risk premia vs. mispricing by comparing raw and delta-hedged returns across VIX regimes. (iii) I address multiple testing (Harvey, Liu, & Zhu, 2016) by applying a |t| > 3.0 threshold.

---

## Part 3 — Research Plan (~500 words; include regression equations)

### Return definitions
For each option *i* on date *t*, I compute two return measures (holding period *h* = 5 trading days):

> **r(i,t,t+h) = [Mid(i,t+h) - Mid(i,t)] / Mid(i,t)**

> **r_dh(i,t,t+h) = r(i,t,t+h) - Delta(i,t) * r_underlying(t,t+h)**

The delta-hedged return isolates non-directional (volatility risk premium) predictability from directional exposure.

### Feature set and ML models
Predictors include option-level characteristics (IV, log-moneyness ln(K/F), TTM, normalized spread, volume, Greeks), IV surface features (ATM level, skew, term structure slope), and underlying characteristics (momentum, realized volatility, VIX). All features are winsorized at 1st/99th percentiles monthly and cross-sectionally standardized.

I estimate five models: OLS, Lasso, XGBoost, Random Forest, and a Multilayer Perceptron (MLP). Training uses an expanding window: train up to month *t-4*, validate on *t-3* to *t-1*, predict month *t*. I report out-of-sample R-squared and **Diebold-Mariano tests** (Diebold & Mariano, 1995) comparing each ML model to OLS.

### Portfolio construction and cost analysis
Each week, I sort options into decile portfolios based on predicted returns. The long-short (LS) portfolio goes long the top decile, short the bottom decile (equal-weighted). Eligibility: Mid >= $0.50, volume >= 50, OI >= 500, moneyness 0.85-1.15, TTM 14-180 days. Transaction costs are modeled as:

> **TC(i,t) = lambda * [Ask(i,t) - Bid(i,t)] / 2**

with lambda in {0.5, 1.0, 1.5, 2.0}. I compute the **break-even lambda*** where net Sharpe = 0.

### Regression equations

**(A) Fama-MacBeth predictive regression** (weekly cross-sections, Newey-West SE):
> **r(i,t,t+h) = a(t) + b(t)' * X(i,t) + e(i,t)**

**(B) ML spanning test** (double-clustered SE per Petersen, 2009):
> **r(i,t,t+h) = a(t) + phi(t) * s(i,t) + b(t)' * X(i,t) + e(i,t)**

where s(i,t) is the out-of-sample ML prediction. Significant phi indicates ML information beyond linear characteristics.

**(C) Risk-premium vs. mispricing decomposition:**
> **r_dh(i,t,t+h) = a + psi * s(i,t) + gamma * VRP(t) + delta * Skew(t) + controls + u(i,t)**

If psi remains significant after controlling for aggregate risk-premium proxies, ML captures option-level mispricing.

**(D) Factor-model alpha** (Newey-West HAC, 4 lags; |t| > 3.0 per Harvey et al., 2016):
> **r_net_LS(t) = alpha + b1*MKT(t) + b2*SMB(t) + b3*HML(t) + b4*MOM(t) + b5*dVIX(t) + epsilon(t)**

### Robustness and risk mitigation
VIX regime splits (high vs. low); moneyness buckets (ATM, OTM puts, OTM calls); holding period variation (1, 5, 10, 21 days); conservative execution (buy at ask, sell at bid); retrain frequency (monthly vs. quarterly); feature ablation. **Overfitting** is mitigated by expanding-window training, early stopping, and linear benchmarks. **Look-ahead bias** is prevented by computing all features from data at *t* only.

---

## Part 4 — Data Sources (~400 words; include descriptive statistics table; confirm access)

### Primary data
**OptionMetrics IvyDB US via WRDS (access verified through Tilburg University's WRDS subscription).** End-of-day SPX option data, January 2010 through December 2024 (15 years). Training burn-in: 2010-2012; out-of-sample evaluation: January 2013 to December 2024 (12 years).

Variables: bid, ask, mid price; strike (K); expiration/TTM; implied volatility; delta, gamma, vega; volume; open interest; option type (C/P). Underlying: SPX index close. Risk-free rate: OptionMetrics zero-coupon curve.

### Supplementary data
Fama-French factors + Momentum (Kenneth French Data Library via WRDS); CBOE VIX index; realized volatility (trailing 21-day annualized standard deviation of daily SPX log-returns).

### Sample filters
Remove: Bid <= 0; Mid < $0.50; moneyness outside [0.85, 1.15]; TTM < 14 or > 180 days; duplicates. Winsorize all features and returns at 1st/99th percentiles monthly. Apply stricter liquidity filters (volume >= 50, OI >= 500) for the tradable universe. Expected sample: ~2.5 million option-week observations.

### Descriptive statistics (expected ranges from SPX options literature)

| Variable | Mean | Std | P5 | P50 | P95 | N (approx.) |
|---|---:|---:|---:|---:|---:|---:|
| Raw option return (weekly, %) | -1.2 | 28.5 | -45.0 | -3.8 | 42.0 | ~2,500,000 |
| Delta-hedged return (weekly, %) | -0.8 | 12.4 | -22.0 | -1.0 | 16.5 | ~2,500,000 |
| Implied volatility | 0.19 | 0.09 | 0.10 | 0.17 | 0.38 | ~2,500,000 |
| Bid-ask spread (% of mid) | 8.5 | 12.3 | 0.5 | 4.2 | 32.0 | ~2,500,000 |
| Volume (contracts/day) | 285 | 2,100 | 2 | 35 | 850 | ~2,500,000 |
| Log-moneyness ln(K/F) | -0.02 | 0.10 | -0.15 | -0.01 | 0.12 | ~2,500,000 |
| Time to maturity (days) | 58 | 45 | 15 | 42 | 150 | ~2,500,000 |

*Note: Expected ranges based on published SPX options studies. Final values computed from actual filtered sample.*

---

## Part 5 — References (APA)

Bali, T. G., Beckmeyer, H., Moerke, M., & Weigert, F. (2023). Option return predictability with machine learning and big data. *Review of Financial Studies, 36*(9), 3548-3600.

Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy, 81*(3), 637-654.

Cao, J., & Han, B. (2013). Cross section of option returns and idiosyncratic stock volatility. *Journal of Financial Economics, 108*(1), 231-249.

Christoffersen, P., Heston, S., & Jacobs, K. (2013). Capturing option anomalies with a variance-dependent pricing kernel. *Review of Financial Studies, 26*(8), 1963-2006.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*(3), 253-263.

Goyal, A., & Saretto, A. (2009). Cross-section of option returns and stock volatility. *Journal of Financial Economics, 94*(2), 310-326.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies, 33*(5), 2223-2273.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies, 29*(1), 5-68.

Muravyev, D. (2016). Order flow and expected option returns. *The Journal of Finance, 71*(2), 673-708.

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica, 55*(3), 703-708.

Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their trading costs. *Review of Financial Studies, 29*(1), 104-147.

Petersen, M. A. (2009). Estimating standard errors in finance panel data sets: Comparing approaches. *Review of Financial Studies, 22*(1), 435-480.
