# Master Thesis Proposal (Updated) — 2026-02-13

## Working Title
**Machine Learning for Options Pricing and ML-Informed Algorithmic Trading: Volatility Surfaces, Early-Exercise Features, and Hedging Performance**

## Refined Research Question (gap-driven)
**Primary RQ:**
How can modern ML models (deep learning + gradient boosting + physics/finance-informed constraints) be used to **(i)** learn arbitrage-aware option pricing maps and implied-volatility surfaces and **(ii)** improve **out-of-sample hedging and trading performance** compared with classical models?

**Sub-questions:**
1. **Surface/price learning:** Which model classes (e.g., deep smoothing/vol-surface networks, gated networks, PINNs, boosted trees) best approximate option prices / implied volatility surfaces under realistic market noise?
2. **Early exercise:** To what extent do deep optimal-stopping / recurrent methods improve American-style option pricing accuracy and exercise-boundary estimation versus traditional benchmarks?
3. **Hedging utility:** Do ML-based prices and Greeks translate into superior hedging P&L (after transaction costs) versus Black–Scholes / stochastic-vol baselines?
4. **Trading utility:** Can volatility-surface forecasts or option-implied representations improve downstream trading decisions (supervised signals and/or RL trading agents) without overfitting?

**Key literature gaps motivating the RQ:**
- Many papers demonstrate strong in-sample pricing accuracy, but fewer evaluate **end-to-end economic value** (hedging P&L, transaction costs, stability under regime shifts).
- Volatility surface learning is powerful (deep smoothing, rough-vol calibration surrogates) but needs clearer comparison of **arbitrage constraints, calibration speed, and downstream performance**.
- American-style options remain challenging at scale; recent deep learning methods offer promise, but robust, reproducible comparisons across products/horizons are still limited.

---
## Literature Review Summary (grouped by theme)

### Theme A — ML for option pricing (function approximation)
- **Network architecture comparisons for option pricing (2023, arXiv)** show that architecture choice materially impacts error and training efficiency.
- **Traditional ML baselines (RF/XGBoost/CatBoost)** appear competitive on certain datasets, motivating careful baseline selection and evaluation beyond neural nets.
- **Physics-informed neural networks for option pricing** highlight the value of embedding known structure (e.g., PDE residuals / constraints) to improve generalization.

### Theme B — Implied volatility surface modeling, smoothing, and calibration
- **Deep smoothing of IV surfaces (NeurIPS 2020)** proposes data-driven smoothing methods suited for noisy, sparse option grids—useful as a stable representation.
- **Deep learning volatility & calibration in (rough) volatility models (Quantitative Finance 2021)** demonstrates neural surrogates for fast pricing and calibration across model families.
- **Forecasting IV surfaces (Journal of Futures Markets 2022)** and **representation learning from IV surfaces (SSRN 2023)** link surfaces to predictive signals, motivating a combined pricing + forecasting + trading pipeline.
- **Gated DNNs for IV surfaces (arXiv 2019)** provide architecture-level inductive biases tailored to surface dynamics.

### Theme C — American options, optimal stopping, and hedging
- **Pricing and hedging American-style options with deep learning (JRFM 2020)** provides a scalable deep optimal-stopping approach with hedging evaluation.
- **Recent work on accelerated American pricing (SSRN 2023)** and **deep recurrent networks for high-dimensional American options (Quantitative Finance 2023)** stresses computational feasibility, path dependence, and joint pricing/hedging criteria.

### Theme D — ML/RL for algorithmic trading and market microstructure
- **Deep RL for trading (2019, arXiv)** is a foundational reference for reward design, risk scaling, and evaluation pitfalls.
- **Deep Q-learning for commodity futures trading (ESWA 2024)** provides a practical RL trading system design on futures data.
- **Deep limit order book forecasting (2024)** connects microstructure modeling to short-horizon price moves, relevant for signal generation and execution-aware evaluation.

### Theme E — Supporting methods and broader context
- **GANs in finance (2021)** motivate synthetic data augmentation and stress-testing.
- **Federated learning in financial systems** informs privacy-aware training setups (optional extension).
- **Bayesian derivatives pricing (high volatility)** motivates uncertainty quantification and calibration confidence intervals.

---
## Proposed Methodology (concrete and testable)

### 1) Data
- **Options data:** liquid equity index options (e.g., SPX) or a highly liquid equity options universe; extract option chains across maturities/strikes.
- **Underlying + rates/dividends:** underlying price, realized volatility proxies, term structure inputs.
- **Preprocessing:** construct implied vols, standardized moneyness, time-to-maturity, ensure data cleaning (stale quotes, arbitrage checks).

### 2) Baselines
- Black–Scholes (with historical/IV inputs) and at least one stochastic volatility benchmark (e.g., Heston) where feasible.
- Classical ML baselines: XGBoost / Random Forest for prices or implied vols.

### 3) ML models to implement/compare
- **IV-surface models:** deep smoothing/NN surface parameterizations; gated architectures; optional monotonicity/convexity penalties.
- **Direct option price models:** feed-forward nets + regularization; physics/finance-informed constraints (PDE residual or no-arbitrage penalties).
- **American options:** deep optimal stopping / continuation-value networks; recurrent architectures for path dependence.

### 4) Evaluation metrics (beyond RMSE)
- **Pricing accuracy:** RMSE/MAE on prices and implied vols across buckets (moneyness, maturity).
- **No-arbitrage diagnostics:** calendar spread / butterfly violations; monotonicity constraints; surface smoothness.
- **Calibration/inference speed:** runtime comparisons for calibration/surface generation.
- **Hedging backtests:** delta-hedging / vega-hedging P&L distributions, turnover, transaction costs.
- **Trading backtests (optional but targeted):**
  - Supervised signals using IV-surface forecasts/embeddings (e.g., predicting volatility risk premium proxies).
  - RL agent (e.g., DQN-style) using option-implied state; compare to rule-based baselines.

### 5) Robustness & reproducibility
- Train/validation/test split by time (avoid leakage), regime-based analysis, ablations on features and constraints.
- Sensitivity analysis for transaction costs, liquidity filters, and re-hedging frequency.

---
## Expected Results / Contributions
- A **structured comparison** of ML approaches for option pricing and IV surface modeling with explicit no-arbitrage and economic-value evaluation.
- Evidence on whether improved pricing metrics **translate into** better hedging P&L and/or trading outcomes.
- Practical guidance on model design choices (architecture, constraints, features) and on evaluation pitfalls in financial ML.

---
## Updated Timeline (realistic, milestone-based)
- **Feb 13–Feb 20:** finalize literature review synthesis; lock dataset choice; define evaluation protocol.
- **Feb 21–Mar 06:** data ingestion/cleaning; implement baselines (BS, Heston if feasible; XGBoost/RF).
- **Mar 07–Mar 27:** implement IV-surface models + direct pricing networks; run pricing/no-arb evaluations.
- **Mar 28–Apr 17:** implement American option deep learning (optimal stopping / recurrent); evaluate.
- **Apr 18–May 09:** hedging backtests; cost/turnover robustness.
- **May 10–May 31:** optional trading experiments (supervised + RL) if time permits.
- **Jun 01–Jun 20:** write-up, results consolidation, revisions, final submission preparation.
