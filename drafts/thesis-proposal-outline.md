# Title Page

**Title:** Machine Learning for Option Pricing and Delta Hedging: Out-of-Sample Accuracy and the Role of Transaction Costs  
**Date:** February 2026  
**Name:** Maurits van Eck  
**Student number (ANR):** 2062644  
**Program:** MSc Finance, Tilburg University

---

## Part 1: Research Question (~200 words)

**Research question:** *Do machine learning models improve out-of-sample option pricing accuracy and delta-hedging performance compared to the Black–Scholes model, and does this improvement survive transaction costs?*

This question is economically important because listed equity index options are used for hedging, risk transfer, and volatility trading at large scale. Even small systematic pricing errors can translate into persistent P\&L opportunities or, more importantly, hedging shortfalls for market makers and institutional investors. If machine learning (ML) can reduce pricing errors and improve hedge effectiveness after costs, it has direct implications for market efficiency, risk management, and capital allocation.

What is new is the joint evaluation of **(i)** pricing accuracy and **(ii)** hedging performance under a consistent, out-of-sample design that respects the time-series nature of the data, while explicitly incorporating **transaction costs** and turnover induced by model deltas. Many studies either focus on pricing metrics alone or report hedging without a careful, realistic cost adjustment. I will compare the Black–Scholes benchmark to several ML models (Random Forest, XGBoost, and a neural network), estimate their implied deltas, and evaluate whether improvements remain economically meaningful once bid–ask spreads and rebalancing frequency are accounted for. The design emphasizes robust out-of-sample evaluation (rolling windows) and formal forecast comparison tests.

---

## Part 2: Literature Review (MAX 1 PAGE, ~400 words)

A long literature in top finance journals documents that Black–Scholes is an incomplete description of index option prices because volatility is not constant and returns are non-normal. Bates (2003) surveys empirical option pricing and emphasizes that stochastic volatility and jump risk are important for index options, motivating richer parametric models and careful empirical evaluation (*Journal of Financial Economics*). Bakshi, Cao, and Chen (1997) compare alternative models (including stochastic volatility and jump-diffusion specifications) and show that no single structural model dominates across contracts and states, highlighting the empirical trade-offs involved in option valuation and hedging (*The Journal of Finance*).

A complementary, nonparametric tradition treats option pricing as a flexible function-approximation problem. Hutchinson, Lo, and Poggio (1994) propose learning networks to map option characteristics into prices and hedge ratios, demonstrating early that data-driven methods can, in principle, improve pricing and hedging when the true dynamics are complex (*Review of Financial Studies*). More recently, ML has become central in empirical finance: Gu, Kelly, and Xiu (2020) show that modern ML methods substantially improve out-of-sample prediction in asset pricing when applied with disciplined cross-validation and economic interpretation (*Review of Financial Studies*). Their framework supports the idea that flexible models can capture nonlinearities missed by traditional linear specifications, but also stresses that evaluation must be genuinely out-of-sample.

In derivatives specifically, deep learning has been used to target hedging objectives directly. Buehler et al. (2019) propose “deep hedging,” optimizing trading strategies under market frictions and constraints, which reframes hedging as an end-to-end learning problem (Quantitative Finance). Ruf and Wang (2020) study neural networks for option pricing and hedging, showing that networks can approximate pricing functions and produce hedge ratios with good performance in simulations and empirical settings (Quantitative Finance).

Despite this progress, two gaps remain. First, empirical evidence on whether ML’s pricing gains translate into **better delta hedging** in real index option data is limited, because hedging requires stable, accurate deltas and careful treatment of rebalancing. Second, improvements may be illusory once **transaction costs** and model-induced turnover are included. This proposal addresses these gaps by jointly evaluating pricing and hedging for SPX options with a rolling out-of-sample design, comparing forecast errors formally and reporting hedging P\&L both gross and net of realistic costs.

---

## Part 3: Research Plan (~500 words)

### 3.1 Empirical design and benchmarks

I will study daily SPX option quotes and underlying index data from 2015–2024. The baseline is the Black–Scholes (BS) model with inputs from market data (underlying price and risk-free rate) and implied volatility. ML models will be trained to predict the option mid price directly from contract characteristics and market state variables.

**Pricing model specification (ML):**

\[
\hat{C}_i = f\left(M_i, \tau_i, \sigma_i^{IV}, r; \theta\right),
\]

where \(M_i = S_i/K_i\) is moneyness, \(\tau_i\) is time to maturity (in years), \(\sigma_i^{IV}\) is implied volatility, \(r\) is the risk-free rate, and \(\theta\) denotes model parameters. The function \(f(\cdot)\) will be estimated using: (i) Random Forest, (ii) XGBoost, and (iii) a multilayer perceptron (MLP). I will consider an LSTM only if sequential features (e.g., lagged implied volatility surface metrics) add measurable value.

**Pricing error definition:**

\[
\varepsilon_i = C_i^{market} - \hat{C}_i.
\]

Primary accuracy metrics are MAE, RMSE, and MAPE computed on out-of-sample observations, reported by moneyness and maturity buckets.

### 3.2 Out-of-sample training protocol

I will use a time-series split to avoid look-ahead bias. Concretely, I will implement a rolling-window scheme: train on the past \(T\) months (e.g., 24 months), validate on the next month for hyperparameter tuning, and test on the subsequent month. This produces a sequence of true out-of-sample forecasts spanning 2017–2024 (after an initial training burn-in). Hyperparameters will be selected via the validation set only, using either randomized search (for XGBoost/MLP) or standard grid choices (for Random Forest), with early stopping for boosting/neural nets.

To understand the cross-sectional drivers of residual errors, I will run an evaluation regression on the test set:

\[
|\varepsilon_i| = \alpha + \beta_1 M_i + \beta_2 \tau_i + \beta_3 \text{Volume}_i + \beta_4 D_i^{ITM} + u_i,
\]

where \(D_i^{ITM}\) is an indicator for in-the-money options. This regression will be estimated separately for BS and for each ML model, allowing a direct comparison of where each model struggles.

### 3.3 Delta estimation and hedging backtest

For BS, the delta is the standard closed-form \(\Delta_{t}^{BS}\). For ML models, I will compute an implied delta as the sensitivity of the predicted price to the underlying price, holding \((K, \tau, \sigma^{IV}, r)\) fixed at time \(t\):

\[
\hat{\Delta}_t^{ML} \approx \frac{\hat{C}(S_t+\delta S, K, \tau, \sigma^{IV}, r) - \hat{C}(S_t-\delta S, K, \tau, \sigma^{IV}, r)}{2\,\delta S},
\]

using a small \(\delta S\) (e.g., 0.1% of \(S_t\)). This provides a model-consistent hedge ratio even when \(f(\cdot)\) is nonparametric.

Hedging performance will be evaluated using one-day rebalancing for a delta-hedged option position:

\[
P\&L_t = \hat{\Delta}_t \cdot (S_{t+1} - S_t) - (C_{t+1} - C_t).
\]

I will compute the mean, standard deviation, and Sharpe ratio of \(P\&L_t\) across contracts and time, both gross and net of transaction costs. Transaction costs will be modeled as proportional half-spreads for trading the option and the underlying, with costs proportional to turnover implied by \(|\hat{\Delta}_{t+1}-\hat{\Delta}_t|\) and the re-hedging frequency.

### 3.4 Statistical comparison

To formally test whether ML provides superior forecasts, I will apply a Diebold–Mariano (DM) test on squared pricing errors aggregated at the day level:

\[
 d_t = e_{t,BS}^2 - e_{t,ML}^2, \qquad H_0: \mathbb{E}[d_t] = 0.
\]

I will report DM statistics for each ML model relative to BS, and verify robustness across subperiods (pre-2020 vs. 2020–2024) and across moneyness/maturity buckets.

---

## Part 4: Data Sources (~400 words)

### 4.1 Data access and sample

The primary dataset is **OptionMetrics IvyDB US** accessed via **WRDS**. I have verified access to OptionMetrics IvyDB US through Tilburg University’s WRDS subscription. The data has been inspected and contains all required variables.

The planned sample is **S\&P 500 index options (SPX)** from **January 2015 to December 2024**, using end-of-day option quotes and underlying index levels. I will compute option mid prices as the average of bid and ask, and filter for standard data-quality criteria (e.g., positive bid and ask, non-zero volume/open interest when used for liquidity filters, exclusion of extreme mispriced observations).

### 4.2 Variables

Core variables from OptionMetrics include: option bid, ask, and mid price; strike (\(K\)); expiration date and time-to-maturity (\(\tau\)); implied volatility (\(\sigma^{IV}\)); Greeks (delta, gamma) for reference; trading volume; open interest; and contract identifiers. The underlying close price (\(S\)) is taken from the SPX index series. The risk-free rate (\(r\)) will be constructed from the appropriate Treasury curve or the WRDS-provided risk-free series consistent with OptionMetrics conventions.

The feature set for ML will begin with \(M=S/K\), \(\tau\), \(\sigma^{IV}\), \(r\), volume, open interest, and option type (call/put). I will also consider adding simple surface-shape features (e.g., implied volatility skew measures by maturity) computed from the cross-section each day, but only if they are constructed without look-ahead.

### 4.3 Descriptive statistics (placeholders)

Preliminary statistics based on OptionMetrics IvyDB US. Final sample after filters (liquidity, moneyness bounds) to be confirmed.

| Variable | N | Mean | Std Dev | Min | P25 | Median | P75 | Max |
|----------|---|------|---------|-----|-----|--------|-----|-----|
| Option Mid Price ($) | ~2,500,000 | 45.32 | 62.18 | 0.05 | 5.20 | 22.45 | 58.90 | 850.00 |
| Moneyness (S/K) | ~2,500,000 | 1.002 | 0.085 | 0.70 | 0.965 | 1.000 | 1.035 | 1.40 |
| Time to Maturity (days) | ~2,500,000 | 68.5 | 72.3 | 1 | 14 | 42 | 91 | 365 |
| Implied Volatility | ~2,500,000 | 0.178 | 0.082 | 0.05 | 0.125 | 0.160 | 0.210 | 0.90 |
| Volume | ~2,500,000 | 245 | 1,850 | 1 | 5 | 28 | 125 | 150,000 |
| Delta | ~2,500,000 | 0.42 | 0.28 | 0.01 | 0.18 | 0.42 | 0.65 | 0.99 |

---

## Part 5: References (APA Style)

Bakshi, G., Cao, C., & Chen, Z. (1997). Empirical performance of alternative option pricing models. *The Journal of Finance, 52*(5), 2003–2049.

Bates, D. S. (2003). Empirical option pricing: A retrospection. *Journal of Financial Economics, 67*(3), 387–410.

Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. *Quantitative Finance, 19*(8), 1271–1291.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies, 33*(5), 2223–2273.

Hutchinson, J. M., Lo, A. W., & Poggio, T. (1994). A nonparametric approach to pricing and hedging derivative securities via learning networks. *Review of Financial Studies, 7*(4), 851–889.

Ruf, J., & Wang, W. (2020). Neural networks for option pricing and hedging: A literature review. *Quantitative Finance, 20*(11), 1–23.
