<!-- Tilburg University MSc Finance — Master Thesis Proposal (max 4 pages text; references & tables excluded) -->

# Title page

**Title:** Machine Learning for Option Pricing and Delta-Hedging: Out-of-Sample Accuracy, Robustness, and Transaction Costs  
**Date:** 13 February 2026  
**Student:** Maurits van Eck  
**Student number (ANR):** 2062644  

---

\newpage

# Part 1 — Research Question

**Main research question.**  
*To what extent do machine learning (ML) models improve out-of-sample option pricing and delta-hedging performance relative to Black–Scholes-type benchmarks, and does any improvement remain economically meaningful after accounting for transaction costs and varying volatility regimes?*

**Economic importance.**  
Options are central to risk transfer, price discovery, and risk management. Pricing errors translate into systematic misvaluation (for market makers) and biased risk measurement (for hedgers). Even when an option is “priced” via an implied volatility (IV), practitioners rely on a stable, no-arbitrage-consistent mapping from option characteristics to prices/greeks for quoting, hedging, and stress testing. Persistent pricing/hedging errors matter economically because they:

1. **Create hedging losses:** a misspecified delta (or effective delta) generates predictable delta-hedged P&L; small per-trade errors can cumulate at scale.
2. **Distort risk and capital:** inaccurate greeks change margin/capital requirements and can amplify losses in volatile periods.
3. **Affect market quality:** better pricing/hedging models can reduce spreads and inventory risk.

**Novel contribution / “not pure replication”.**  
Most ML option pricing studies focus on in-sample fit or pure pricing metrics. This thesis adds three elements aimed at economic relevance and novelty:

- **Joint evaluation of pricing *and* hedging:** models are judged by out-of-sample pricing error *and* realized delta-hedged P&L, where the hedge ratio is produced by the same model.
- **Regime robustness:** performance is evaluated across volatility regimes (e.g., calm vs stress periods) and across moneyness/maturity buckets, rather than a single pooled test.
- **Transaction-cost-aware hedging:** delta-hedging is assessed net of plausible bid–ask/turnover costs; improvements must survive implementation frictions.

Deliverable: a set of benchmark ML models (e.g., gradient boosting and neural networks; potentially a Transformer-style architecture for time-varying surfaces) compared to Black–Scholes and standard practitioner baselines.

---

\newpage

# Part 2 — Literature Review (≤ 1 page)

Classical option pricing links option values to an underlying diffusion under no-arbitrage. Black and Scholes (1973) and Merton (1973) provide the foundational framework, but empirical option prices exhibit volatility smiles and term structures inconsistent with constant volatility. A large literature extends the model class and documents economically meaningful deviations from Black–Scholes. For example, Bakshi, Cao, and Chen (1997, *Journal of Finance*) evaluate alternative structural models and show that richer dynamics can reduce mispricing. More broadly, option-implied information contains risk premia and predictive content; Bakshi, Kapadia, and Madan (2003, *Review of Financial Studies*) show variance risk premia embedded in option prices are substantial and state-dependent. In the same vein, volatility dynamics and implied-volatility-surface restrictions are central for pricing and risk management (e.g., Christoffersen, Jacobs, Ornthanalai, & Wang, 2008, *Journal of Financial Economics*; Carr & Wu, 2016, *Journal of Financial Economics*).

Recent work in finance documents that flexible, high-dimensional prediction methods can outperform linear specifications when relationships are nonlinear and involve interactions. In asset pricing, Gu, Kelly, and Xiu (2020, *Journal of Finance*) show that ML methods materially improve return prediction and can uncover nonlinearities missed by classical models, while emphasizing rigorous out-of-sample evaluation and economic metrics. Relatedly, deep learning methods have been proposed for factor modeling and high-dimensional forecasting in top outlets (e.g., Feng, He, Polson, & Xu, 2023, *Journal of Financial and Quantitative Analysis*), supporting the broader premise that ML is useful when the state space is large.

In derivatives, ML has been used to approximate pricing functions, smooth implied volatility surfaces, and (in some strands) learn hedging policies end-to-end (e.g., Buehler, Gonon, Teichmann, & Wood, 2019, *Quantitative Finance*). However, **two gaps** motivate this thesis. First, much of the ML option pricing evidence is based on pricing errors alone, while a trading desk ultimately cares about **hedging outcomes** (greeks and realized P&L) and implementation frictions. Second, it remains unclear whether ML improvements persist **across regimes** and **out-of-sample** when the evaluation design mimics production constraints (walk-forward training, limited look-ahead, stable hyperparameters).

This thesis targets these gaps by comparing ML models to Black–Scholes-type benchmarks using both pricing errors and delta-hedged P&L net of transaction costs, and by explicitly testing stability across moneyness/maturity buckets and volatility regimes.

---

\newpage

# Part 3 — Research Plan

## 3.1 Empirical setting and sample design

- **Universe:** S&P 500 index options (SPX) and/or a large liquid subset of S&P 500 equity options (robustness). I will start with SPX because of liquidity and standardized contract features.
- **Sample period:** planned 2015–2024 (extended if feasible).
- **Filtering:** standard OptionMetrics filters (exclude obvious data errors; remove options with extremely low prices; require non-missing IV and greeks; handle early closes/holidays). Contracts are aligned by option identifier, trade date, expiration, strike, and call/put flag.
- **Train/validation/test:** walk-forward scheme (e.g., train on years t−k…t−1, validate on year t, test on year t+1), repeated through the sample to avoid look-ahead bias.

## 3.2 Pricing models to be estimated

### Benchmarks
1. **Black–Scholes (BS) with implied volatility:** compute BS prices using market IV (when available) and compare mapping stability; also evaluate BS with historical volatility as a stricter benchmark.
2. **Surface-based baseline:** a simple parametric or spline-based implied-volatility-surface fit (per date), then price via BS using fitted IV.

### Machine learning models
- **Gradient boosting regression (GBRT / XGBoost-style):** strong tabular baseline; fast and interpretable via feature importance.
- **Neural network (MLP):** learns nonlinear interactions in (moneyness, maturity, IV level, etc.).
- **(Optional extension) Sequence model / Transformer-like architecture:** if the “state” includes lagged surface information, evaluate whether attention mechanisms improve stability through time.

**Target variable:** option midquote price (or normalized price, e.g., price/spot).  
**Inputs (illustrative):**
\[
X_{i,t} = \big(\log(S_t/K_i),\; \tau_{i,t},\; r_t,\; q_t,\; \sigma^{IV}_{i,t},\; \text{option type},\; \text{surface features},\; \text{liquidity controls}\big)
\]
where \(S_t\) is the underlying level, \(K_i\) strike, \(\tau\) time-to-maturity, \(r_t\) risk-free rate, \(q_t\) dividend yield (or index equivalent), and liquidity controls include volume/open interest and bid–ask spreads.

### Main pricing specification (supervised learning)
For each contract \(i\) on date \(t\):
\[
C^{\text{mid}}_{i,t} = f_\theta(X_{i,t}) + \varepsilon_{i,t}
\]
where \(f_\theta(\cdot)\) is an ML model (GBRT/NN/Transformer) trained to minimize an out-of-sample objective (e.g., Huber loss or squared error). Model performance is evaluated by RMSE/MAE and by errors scaled by option vega or price.

### Model-implied delta and hedging test
For differentiable models (NN/Transformer), delta is obtained by automatic differentiation of \(\hat C\) w.r.t. \(S\). For non-differentiable models (GBRT), delta is estimated via a stable numerical derivative:
\[
\hat\Delta^{\text{ML}}_{i,t} \approx \frac{\hat C(S_t(1+h)) - \hat C(S_t(1-h))}{2h S_t}
\]

Define a one-day (or intraday, if feasible) delta-hedged P&L for a long option position:
\[
\pi_{i,t+1} = \big(C^{\text{mid}}_{i,t+1}-C^{\text{mid}}_{i,t}\big) - \hat\Delta_{i,t}\,\big(S_{t+1}-S_t\big) - \text{TC}_{i,t+1}
\]
where \(\text{TC}\) is a transaction-cost proxy driven by hedge turnover and bid–ask spreads (e.g., half-spread on underlying times \(|\Delta_{t+1}-\Delta_t|\)). Outcomes include mean \(\pi\), volatility, downside risk, and tail metrics, across buckets.

## 3.3 Hypotheses and statistical tests

**H1 (pricing):** ML models reduce out-of-sample pricing errors relative to BS and a simple surface baseline.

**H2 (hedging):** ML-implied deltas reduce the dispersion of delta-hedged P&L, and any improvement remains after transaction costs.

**H3 (where ML helps):** improvements are larger in regions where parametric assumptions are weakest: short maturities, deep OTM options, and stress regimes.

### Cross-sectional error regression (explanatory analysis)
To understand where models fail, estimate:
\[
|\varepsilon_{i,t}| = \alpha + \beta_1\,|\log(S_t/K_i)| + \beta_2\,\tau_{i,t} + \beta_3\,\text{Spread}_{i,t} + \beta_4\,\text{VIX}_t + \gamma_{\text{date}} + u_{i,t}
\]
where \(|\varepsilon_{i,t}| = |C^{\text{mid}}_{i,t} - \hat C_{i,t}|\) for each model, and \(\gamma_{\text{date}}\) are date fixed effects (or regime indicators). Standard errors will be clustered by date.

## 3.4 Model validation and robustness

- **No-arbitrage sanity checks:** monotonicity in strike and maturity (where appropriate), non-negativity, and call–put parity deviations. If needed, incorporate constraints via post-processing or constrained learning.
- **Bucketed evaluation:** by moneyness (ITM/ATM/OTM), maturity (short/medium/long), and volatility regimes (e.g., VIX terciles).
- **Alternative targets:** implied volatility as the target, then price via BS, to compare stability vs direct price learning.

---

\newpage

# Part 4 — Data Sources

## 4.1 Primary data

**OptionMetrics IvyDB US via WRDS (Tilburg University subscription): yes, I have access via Tilburg University.**

Planned WRDS sources (IvyDB US):
- **Option prices and contract characteristics:** *OptionMetrics IvyDB US* option price files (e.g., OPPRCD-style daily option price/quote data with bid, ask, mid, volume, open interest; identifiers; strike; expiration; call/put).
- **Underlying data:** corresponding underlying index/equity price files (close/open/high/low/volume; corporate actions where relevant).
- **Greeks and implied volatility:** OptionMetrics-provided IV and greeks (delta, gamma, vega) when available, used both as features and as benchmark inputs.

Auxiliary data:
- **Risk-free rates:** e.g., Treasury yields (WRDS FRED or CRSP/Compustat sources).
- **Market volatility regime proxy:** VIX level (CBOE / WRDS if available).

## 4.2 Data inspection status and descriptive statistics (placeholder)

I attempted to programmatically download and inspect OptionMetrics data from this execution environment, but WRDS requires interactive authentication and the WRDS Python client is not available here (no pip). **Action item:** run the WRDS query using Tilburg/WRDS login and export a first extract (SPX options, 2015–2024) to replace the placeholder statistics below before final submission.

**Table 1. Illustrative descriptive statistics (to be replaced with WRDS extract).**

| Variable | Definition | N | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| $C^{mid}$ | Option midquote (USD) | 12,500,000 | 4.25 | 6.80 | 0.01 | 250.00 |
| $S$ | Underlying level (SPX) | 12,500,000 | 3,200 | 720 | 1,850 | 4,800 |
| $K$ | Strike | 12,500,000 | 3,230 | 760 | 1,500 | 6,000 |
| $\tau$ | Time to maturity (years) | 12,500,000 | 0.18 | 0.23 | 0.003 | 2.00 |
| $\sigma^{IV}$ | Implied volatility | 12,500,000 | 0.22 | 0.09 | 0.03 | 1.50 |
| $\Delta$ | Option delta | 12,500,000 | 0.47 | 0.32 | 0.00 | 1.00 |
| Spread | Bid–ask spread (USD) | 12,500,000 | 0.10 | 0.15 | 0.00 | 5.00 |
| Volume | Contracts traded | 12,500,000 | 75 | 210 | 0 | 25,000 |
| OI | Open interest | 12,500,000 | 1,250 | 3,900 | 0 | 250,000 |
| $\log(S/K)$ | Log-moneyness | 12,500,000 | -0.01 | 0.18 | -1.20 | 0.90 |

The final proposal submission will replace Table 1 with statistics computed from the downloaded WRDS extract, including exact sample period and cross-sectional coverage (SPX only vs SPX + equities).

---

# Part 5 — References (APA)

Bakshi, G., Cao, C., & Chen, Z. (1997). Empirical performance of alternative option pricing models. *The Journal of Finance, 52*(5), 2003–2049.

Bakshi, G., Kapadia, N., & Madan, D. (2003). Stock return characteristics, skew laws, and the differential pricing of individual equity options. *Review of Financial Studies, 16*(1), 101–143.

Bertsimas, D., & Kallus, N. (2020). From predictive to prescriptive analytics. *Management Science, 66*(3), 1025–1044.

Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy, 81*(3), 637–654.

Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. *Quantitative Finance, 19*(8), 1271–1291.

Carr, P., & Wu, L. (2016). Analyzing volatility risk and risk premium in option contracts: A new theory. *Journal of Financial Economics, 120*(1), 1–20.

Christoffersen, P., Jacobs, K., Ornthanalai, C., & Wang, Y. (2008). Option valuation with long-run and short-run volatility components. *Journal of Financial Economics, 90*(3), 272–297.

Feng, G., He, J., Polson, N., & Xu, J. (2023). Deep learning in characteristics-sorted factor models. *Journal of Financial and Quantitative Analysis, 58*(5), 1716–1750.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Journal of Finance, 75*(6), 3319–3370.

Merton, R. C. (1973). Theory of rational option pricing. *The Bell Journal of Economics and Management Science, 4*(1), 141–183.
