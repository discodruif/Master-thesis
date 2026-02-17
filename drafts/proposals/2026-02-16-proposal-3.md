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

**H1 (Predictability):** Nonlinear ML models (gradient boosted trees, neural networks) achieve higher out-of-sample R-squared in predicting weekly option returns than OLS and Lasso, with the improvement concentrated in option characteristics that proxy for volatility risk premia (IV level, skew, term structure slope).

**H2 (Implementability):** The long-short portfolio formed on ML predictions generates a Sharpe ratio significantly greater than zero after transaction costs modeled at observed bid-ask spreads, with the break-even cost exceeding the median observed half-spread in the tradable universe.

These hypotheses are jointly testable: H1 can hold while H2 fails, which would indicate that ML captures genuine predictability that is nonetheless unimplementable — a finding that is itself economically informative about the efficiency of option markets.

---

## Part 2 — Literature Review (MAX 1 page)

### Option return predictability and risk premia
The Black-Scholes (1973) framework implies that, absent frictions, option characteristics should not predict returns beyond compensation for systematic risk. Empirically, this is rejected. Goyal and Saretto (2009) document that the gap between implied and realized volatility — a proxy for the variance risk premium — predicts delta-hedged option returns in the cross-section. Cao and Han (2013) show that high idiosyncratic volatility predicts low delta-hedged returns, consistent with hedging pressure from end-users depressing option prices. Christoffersen, Heston, and Jacobs (2013) demonstrate that the shape of the IV surface carries information about time-varying risk premia, linking surface dynamics to expected returns through a formal stochastic volatility model.

These findings suggest two economic channels for option return predictability: (i) **time-varying risk premia** (investors demand compensation for bearing volatility, jump, and correlation risk) and (ii) **limits to arbitrage** (transaction costs, margin constraints, and model uncertainty prevent full price correction). My thesis directly tests which channel dominates, because risk-premium-driven predictability should be robust to transaction costs (it compensates a real risk), while limits-to-arbitrage-driven predictability may vanish once costs are included.

### Machine learning in asset pricing
Gu, Kelly, and Xiu (2020) establish the methodological standard for ML in cross-sectional return prediction: rolling time-series splits, comparison to linear benchmarks, and economic evaluation via portfolio sorts. They show that tree-based models and neural networks capture nonlinear interactions among firm characteristics that linear models miss. Bali et al. (2023) apply this framework to option returns using a comprehensive set of option-level and underlying characteristics, finding that ML-based long-short strategies earn substantial returns. My thesis extends their work by (i) focusing explicitly on the transaction cost boundary — the point at which alpha disappears — and (ii) decomposing predictability into risk-premium vs. mispricing components using delta-hedged returns.

### Transaction costs and implementability
Option market microstructure is characterized by wide spreads, especially for short-dated OTM contracts (Muravyev, 2016). The implementability of any option trading strategy depends critically on whether signals concentrate in liquid contracts or in the illiquid tails where spreads erode returns. Novy-Marx and Velikov (2016) demonstrate for equity strategies that accounting for trading costs can eliminate or substantially reduce anomaly profits. No comparable systematic analysis exists for ML-based option strategies, which is the gap this thesis fills.

### Contribution
This thesis makes three contributions. First, I quantify the **transaction cost boundary** for ML option signals by computing break-even spreads across moneyness and maturity buckets. Second, I test whether predictability reflects **risk premia or mispricing** by comparing results for raw vs. delta-hedged returns and by conditioning on VIX regimes. Third, I address **multiple testing concerns** (Harvey, Liu, & Zhu, 2016) by reporting adjusted t-statistics and by evaluating the full decile spread rather than cherry-picked subsamples.

---

## Part 3 — Research Plan (~500 words)

### 3.1 Return definitions

For each option contract *i* on date *t*, I define two return measures over holding period *h* (baseline: 5 trading days):

**Raw option return:**
> **r(i, t, t+h) = [ Mid(i, t+h) - Mid(i, t) ] / Mid(i, t)**

**Delta-hedged return** (isolates non-directional component):
> **r_dh(i, t, t+h) = r(i, t, t+h) - Delta(i, t) * r_underlying(t, t+h)**

where Mid is the bid-ask midpoint and Delta is the Black-Scholes delta from OptionMetrics. Using both measures allows me to test whether ML signals capture directional (underlying) exposure or option-specific predictability linked to volatility risk premia.

### 3.2 Feature set

The predictor vector X(i,t) contains three groups:

| Group | Variables |
|---|---|
| **Option-level** | IV, log-moneyness ln(K/F), days to expiry, normalized bid-ask spread, log volume, log open interest, delta, gamma, vega |
| **IV surface** | ATM IV level, 25-delta risk reversal (skew), 25-delta butterfly (curvature), term structure slope (90d IV minus 30d IV) |
| **Underlying** | 5-day, 21-day, 63-day past returns (momentum/reversal), 21-day realized volatility, VIX level, VIX change |

All features are winsorized at the 1st and 99th percentiles each month to limit the influence of outliers. Features are standardized (zero mean, unit variance) within each cross-section to ensure comparability across time.

### 3.3 ML models and training protocol

| Model | Specification |
|---|---|
| **OLS** | Linear benchmark; all features enter linearly |
| **Lasso** | L1-penalized linear; tests whether feature selection alone improves prediction |
| **Gradient Boosted Trees (XGBoost)** | max_depth in {3, 5, 7}, learning_rate in {0.01, 0.05, 0.1}, n_estimators up to 1000 with early stopping |
| **Random Forest** | max_depth in {5, 10, None}, n_estimators = 500 |
| **Multilayer Perceptron** | 2 hidden layers (64, 32), ReLU activation, dropout = 0.2, Adam optimizer, early stopping on validation loss |

**Training protocol.** I use an expanding window: train on all data up to month *t - 4*, validate on months *t - 3* to *t - 1* for hyperparameter selection, and predict month *t*. This ensures strictly out-of-sample predictions and mimics real-time implementability. Predictions are generated monthly; portfolios are rebalanced weekly within each month using the same model.

**Forecast evaluation.** I report out-of-sample R-squared:
> **R2_oos = 1 - SUM[ (r_i - r-hat_i)^2 ] / SUM[ (r_i - r-bar)^2 ]**

where r-bar is the cross-sectional mean return. I also apply the **Diebold-Mariano test** to compare squared prediction errors between each ML model and the OLS benchmark, using Newey-West standard errors with lag h to account for autocorrelation in forecast errors.

### 3.4 Portfolio construction and transaction cost analysis

**Portfolio formation.** Each week, I sort options into decile portfolios based on predicted return r-hat(i,t). The long-short (LS) portfolio goes long the top decile and short the bottom decile, equal-weighted within each leg.

**Eligibility filters (tradable universe):**
- Bid > 0 and Mid >= $0.50 (avoids penny options with extreme percentage spreads)
- Volume >= 50 contracts/day and Open interest >= 500
- Moneyness: 0.85 <= K/F <= 1.15
- Time to maturity: 14 to 180 calendar days

**Transaction cost model.** Each trade incurs a cost equal to a fraction of the observed bid-ask spread:
> **TC(i, t) = lambda * [ Ask(i, t) - Bid(i, t) ] / 2**

where lambda = 1 is the baseline (full half-spread) and lambda in {0.5, 1.0, 1.5, 2.0} provides sensitivity analysis. Net portfolio return:
> **r_net_LS(t) = r_gross_LS(t) - (1/N) * SUM[i in trades] TC(i, t) / Mid(i, t)**

**Break-even spread.** For each portfolio, I compute the cost level lambda* at which the annualized net Sharpe ratio equals zero. This single number summarizes implementability: if lambda* > 1, the strategy survives full half-spread costs.

### 3.5 Regression-based tests

**(A) Cross-sectional predictive regression (Fama-MacBeth style):**

Each week *t*, estimate:
> **r(i, t, t+h) = a(t) + b(t)' * X(i, t) + e(i, t)**

Report time-series averages of b(t) with Newey-West standard errors (4 lags). This provides a linear benchmark and identifies which characteristics have standalone predictive power. Standard errors are computed using the Newey-West (1987) procedure to account for autocorrelation in the weekly coefficient estimates.

**(B) ML signal spanning test:**
> **r(i, t, t+h) = a(t) + phi(t) * s(i, t) + b(t)' * X(i, t) + e(i, t)**

where s(i,t) is the ML predicted return (from the out-of-sample prediction). A significant time-series average of phi indicates that ML contains information beyond the linear combination of characteristics. I double-cluster standard errors by option and week following Petersen (2009) to address both cross-sectional and time-series correlation.

**(C) Risk-premium vs. mispricing decomposition:**
> **r_dh(i, t, t+h) = a + psi * s(i, t) + gamma * VRP(t) + delta * Skew(t) + controls + u(i, t)**

where VRP(t) = IV(t) - RV(t) is the variance risk premium and Skew(t) is the option-implied skewness. If psi remains significant after controlling for aggregate risk-premium proxies, the ML signal captures option-level mispricing rather than time-varying compensation for systematic risk.

**(D) Factor-model alpha:**
> **r_net_LS(t) = alpha + b1 * MKT(t) + b2 * SMB(t) + b3 * HML(t) + b4 * MOM(t) + b5 * dVIX(t) + epsilon(t)**

The intercept alpha is the risk-adjusted net return. I test alpha = 0 using HAC standard errors (Newey-West, 4 lags) and report the associated t-statistic. Following Harvey et al. (2016), I apply a threshold of |t| > 3.0 rather than the conventional 1.96 to account for multiple testing across model specifications and subsamples.

### 3.6 Robustness

| Test | Purpose |
|---|---|
| **VIX regime split** (above/below median) | Tests whether alpha concentrates in high-vol periods (limits-to-arbitrage prediction) or is stable (risk-premium prediction) |
| **Moneyness buckets** (ATM / OTM puts / OTM calls) | Identifies where ML signal is strongest; OTM puts have widest spreads and most skew |
| **Holding period** (1, 5, 10, 21 days) | Quantifies turnover-alpha tradeoff; longer horizons reduce costs but may dilute signal |
| **Conservative execution** (buy at ask, sell at bid) | Worst-case implementability bound |
| **Retrain frequency** (monthly vs. quarterly) | Tests model stability and practical retraining burden |
| **Feature ablation** | Removes feature groups one at a time to identify key predictive drivers |

---

## Part 4 — Data Sources (~400 words)

### 4.1 Primary data

**OptionMetrics IvyDB US via WRDS (access verified).** I use end-of-day option data for S&P 500 index options (SPX), January 2010 through December 2024. This 15-year sample spans the European debt crisis (2011-2012), the low-volatility environment (2014-2019), the COVID crash (March 2020), the meme-stock/retail-options episode (2021), and the rate-hiking cycle (2022-2024). The training burn-in consumes the first 36 months (2010-2012), producing out-of-sample predictions from January 2013 through December 2024 — a 12-year evaluation window.

**Variables extracted from IvyDB:**

| Variable | IvyDB Table | Description |
|---|---|---|
| Bid, Ask, Mid price | optionm.opprcd | End-of-day option quotes |
| Strike (K) | optionm.opprcd | Contract strike price |
| Expiration date, TTM | optionm.opprcd | Time to maturity in calendar days |
| Implied volatility | optionm.vsurfd / opprcd | Black-Scholes IV from OptionMetrics |
| Delta, Gamma, Vega | optionm.opprcd | OptionMetrics-computed Greeks |
| Volume, Open interest | optionm.opprcd | Daily trading activity |
| Option type (C/P) | optionm.opprcd | Call or put indicator |
| SPX close | optionm.secprd | Underlying index level |
| Forward price (F) | Computed | F = S * exp(r * tau), using zero-curve |
| Risk-free rate | optionm.zerocd | Zero-coupon yield curve |

### 4.2 Supplementary data

- **Fama-French factors + Momentum:** Kenneth French Data Library via WRDS (daily MKT, SMB, HML, MOM).
- **VIX:** CBOE VIX index (daily close), from WRDS or direct download.
- **Realized volatility:** Computed as annualized standard deviation of daily SPX log-returns over trailing 21-day window.

### 4.3 Sample construction and filters

| Filter | Rule | Rationale |
|---|---|---|
| Zero or negative bid | Remove if Bid <= 0 | Unreliable quotes |
| Extreme moneyness | Keep 0.85 <= K/F <= 1.15 | Removes deep ITM/OTM with unreliable IV |
| Short-dated | Remove if TTM < 14 days | Avoid expiration effects and gamma risk |
| Long-dated | Remove if TTM > 180 days | Illiquid, wide spreads, few observations |
| Penny options | Remove if Mid < $0.50 | Percentage spreads are extreme |
| Duplicates | Keep last observation per option-date | Standard cleaning |
| Winsorization | 1st/99th percentile per month for all features and returns | Limits outlier influence |

After filters, the expected sample size is approximately 2-3 million option-week observations.

### 4.4 Descriptive statistics (expected ranges from SPX options literature)

| Variable | Mean | Std | P5 | P50 | P95 | N (approx.) |
|---|---:|---:|---:|---:|---:|---:|
| Raw option return (weekly, %) | -1.2 | 28.5 | -45.0 | -3.8 | 42.0 | ~2,500,000 |
| Delta-hedged return (weekly, %) | -0.8 | 12.4 | -22.0 | -1.0 | 16.5 | ~2,500,000 |
| Implied volatility | 0.19 | 0.09 | 0.10 | 0.17 | 0.38 | ~2,500,000 |
| Bid-ask spread (% of mid) | 8.5 | 12.3 | 0.5 | 4.2 | 32.0 | ~2,500,000 |
| Volume (contracts/day) | 285 | 2,100 | 2 | 35 | 850 | ~2,500,000 |
| Open interest | 4,200 | 12,500 | 50 | 1,200 | 18,000 | ~2,500,000 |
| Log-moneyness ln(K/F) | -0.02 | 0.10 | -0.15 | -0.01 | 0.12 | ~2,500,000 |
| Time to maturity (days) | 58 | 45 | 15 | 42 | 150 | ~2,500,000 |

*Note: Values are expected ranges based on published SPX options studies. Final statistics will be computed from the actual filtered sample.*

---

## Part 5 — Risks, Limitations, and Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| **Overfitting** | ML models fit noise, OOS performance collapses | Strict expanding-window protocol; validation-based early stopping; comparison to linear benchmarks; feature ablation |
| **Multiple testing** | Testing many models and subsamples inflates false discovery rate | Apply Harvey et al. (2016) adjusted t-statistic threshold (|t| > 3.0); report all specifications, not cherry-picked results |
| **Look-ahead bias** | Using future information in features or filters | All features computed from data available at *t*; filters applied using lagged volume/OI; forward price uses contemporaneous zero-curve only |
| **Regime shifts** | Model trained on low-vol data performs poorly in crisis | Expanding window includes all past regimes; explicit VIX-regime subsample analysis; report drawdowns during COVID crash |
| **Transaction costs underestimated** | Real execution costs exceed half-spread assumption | Report results at multiple cost levels (lambda = 0.5 to 2.0); compute break-even spread; worst-case buy-at-ask/sell-at-bid bound |
| **Data limitations** | WRDS download limits; intraday data unavailable | Focus on daily end-of-day data (standard in literature); sample period adjustable; SPX is the most liquid option market |
| **Survivorship/selection bias** | SPX index options are not subject to delisting, but contract filters may introduce selection | Document filter impact by comparing filtered vs. unfiltered descriptive statistics |

---

## Part 6 — References (APA)

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

---

## Part 7 — Feasibility and Timeline

| Period | Deliverable |
|---|---|
| **Month 1** (Mar 2026) | Data pipeline: extract IvyDB SPX panel via WRDS, apply filters, compute returns and features. Verify sample against published descriptive statistics. Produce Table 1 (descriptive stats). |
| **Month 2** (Apr 2026) | Model training: implement expanding-window pipeline for all five models. Compute OOS R-squared and Diebold-Mariano tests. Produce Table 2 (forecast comparison). |
| **Month 3** (May 2026) | Portfolio analysis: construct decile portfolios, compute gross and net returns under all cost scenarios. Estimate factor regressions. Produce Tables 3-4 (portfolio returns, factor alphas). |
| **Month 4** (Jun 2026) | Robustness and interpretation: VIX regimes, moneyness buckets, feature ablation, SHAP importance. Estimate spanning and decomposition regressions. Produce Tables 5-7 and figures. |
| **Month 5** (Jul 2026) | Writing, revision, and buffer for supervisor feedback. |

**Computational requirements:** All models are estimable on a standard laptop (16GB RAM). XGBoost and Random Forest are the most computationally intensive; estimated training time per rolling window is < 5 minutes. Full pipeline (12 years x 12 months x 5 models) requires approximately 30-40 hours of total computation, easily parallelizable.
