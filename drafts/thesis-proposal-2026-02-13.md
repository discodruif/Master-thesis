# Master Thesis Proposal

## Working Title
**"Robust Machine Learning for Option Pricing & Hedging (and How It Translates into Tradable Signals)"**

## 1) Motivation
Option pricing sits at the intersection of financial theory (no-arbitrage, stochastic volatility, PDE/BSDE formulations) and real-world market frictions (microstructure noise, liquidity, transaction costs, regime shifts). Recent literature shows that ML can match or outperform classical parametric models (e.g., Black–Scholes / Heston) in pricing accuracy, and deep learning can scale to high-dimensional American-style problems where traditional PDE methods are infeasible.

However, the literature still has practical gaps:
- Many papers focus on pricing error but do not evaluate *hedging performance* (P&L) under realistic costs.
- Model comparisons are often not standardized (different datasets, metrics, and splits), making it hard to conclude what works best out-of-sample.
- New model classes (Transformers for sequences, signature models for path-dependence, KAN-based networks) are emerging but are not yet consistently benchmarked against established baselines.

## 2) Refined Research Question
**Primary RQ:**
> To what extent do modern ML models (tree ensembles, deep networks, and sequence/path models) improve *out-of-sample option pricing and hedging performance* relative to classical option pricing benchmarks, and can the resulting mispricing estimates be converted into *economically meaningful* trading signals after transaction costs?

**Sub-questions:**
1. Which model families deliver the best trade-off between accuracy, stability across regimes, and computational efficiency?
2. Does improved pricing accuracy translate into improved hedging performance (e.g., delta-hedged P&L, tail risk)?
3. Can model-implied mispricing be used to construct robust delta-neutral trading strategies that survive realistic frictions?

## 3) Literature Review Summary (Grouped by Theme)

### A) ML option pricing (supervised learning baselines)
- **Regression/Ensembles/Boosting:** Multiple studies compare ML regressors (RF, XGBoost/CatBoost, NN) with Black–Scholes/Heston and report improved fit under volatile conditions. These motivate a strong baseline set (linear/GBM/tree ensemble).
- **Architecture sensitivity:** Work on option pricing network architectures highlights that performance depends materially on design choices and regularization—so the thesis should standardize training and validation protocols.

### B) American options & high-dimensional pricing via deep learning
- **Deep learning for American-style options:** Becker–Cheridito–Jentzen (2020) demonstrates deep learning approaches that scale better than classical LSM in higher dimensions.
- **Deep BSDE methods:** Chen & Wan (2021) show pricing and hedging of American options through BSDE-inspired deep networks, a key bridge between theory and ML.
- **Acceleration & practicality:** Anderson & Ulrych (2023) emphasize speed/latency and implementation considerations—important if the thesis also includes tradable signal construction.

### C) Hedging as a learning problem
- **Deep Hedging:** Buehler et al. formulate hedging as a sequential learning problem, enabling end-to-end learned hedging strategies under frictions. This shifts evaluation from “pricing RMSE” to “economic hedging outcomes.”
- **Newer architectures:** Very recent work (e.g., KANHedge, 2026) suggests architectural improvements within deep-BSDE hedgers that could materially affect hedging performance.

### D) Sequence/path-dependent models (Transformers, signatures)
- **Transformers for option pricing:** Recent arXiv work explores Transformer variants (e.g., Informer) to capture long-range dependencies in financial sequences, potentially improving robustness for time-varying dynamics.
- **Deep signature approach for non-Markovian volatility:** Signature-based deep models target memory and path dependence, where standard Markovian models struggle.

### E) Algorithmic trading & reinforcement learning (execution → evaluation)
- **Benchmarking RL in trading:** FinRL provides widely used RL environments and evaluation practices, reducing “backtest overfitting” risk through more standardized pipelines.
- **Multi-agent RL for HFT:** Recent work highlights stability, interaction effects, and evaluation pitfalls for RL in microstructure settings.

**Synthesis:** The literature supports that ML can improve option pricing, but the key gap is translating accuracy gains into *hedging* and *economic value* under frictions. A thesis contribution can be a clean benchmark study that evaluates both pricing and hedging outcomes, and then tests whether mispricing signals remain profitable after costs.

## 4) Proposed Methodology

### 4.1 Data
Planned datasets (choose based on availability via Tilburg access):
- Equity options dataset (e.g., OptionMetrics / exchange dataset) with bid/ask, maturities, strikes, volumes, implied vol.
- Underlying returns, realized volatility measures, rates/dividends.

Preprocessing:
- Standardize moneyness, time-to-maturity, and other features.
- Filter illiquid quotes; use mid prices but track bid/ask to model transaction costs.
- Time-based splits (train/validation/test by date) to avoid leakage.

### 4.2 Models to Benchmark
**Classical baselines:** Black–Scholes (with implied vol), Heston (calibrated), and potentially COS/FFT-style benchmarks where feasible.

**ML baselines:**
- Linear regression / splines (sanity check)
- Random Forest / XGBoost
- MLP (feed-forward NN)

**Advanced models (one or two, depending on scope):**
- Transformer (Informer-style) for sequence-aware features
- Deep-BSDE hedging model OR a constrained model (e.g., physics-informed / no-arbitrage regularization)

### 4.3 Evaluation
**Pricing metrics:** RMSE/MAE by moneyness and maturity buckets; stability across regimes; calibration error relative to implied vol surface.

**Hedging metrics:**
- Delta-hedged P&L distributions (mean, volatility, drawdown, tail risk)
- Transaction cost–adjusted hedging performance
- Stress tests (high-volatility periods)

**Economic value / trading signal test (optional but targeted):**
- Construct delta-neutral mispricing portfolios: go long “underpriced” options, short “overpriced,” hedged with underlying.
- Include bid/ask spreads, slippage, and position limits.
- Evaluate net Sharpe, turnover, and robustness across periods.

### 4.4 Reproducibility
- Fixed data splits and a single evaluation harness.
- Ablations: feature sets, model size, and regularization.
- Clear reporting of compute/time cost for each method.

## 5) Expected Results & Contributions
- A rigorous, apples-to-apples comparison of classical vs ML option pricing models on the same dataset.
- Evidence on whether pricing accuracy improvements translate into better hedging performance.
- A practical assessment of whether ML-derived mispricing signals remain profitable after realistic costs.
- A structured thesis narrative that connects option pricing theory (PDE/BSDE/no-arbitrage) with modern ML architectures.

## 6) Updated Timeline (UTC)
- **Feb 13–19:** Expand literature review; finalize dataset choice + data access.
- **Feb 20–25:** Implement data pipeline + baseline models; establish evaluation harness.
- **Feb 26–Mar 3:** Train/validate ML models; run robustness checks.
- **Mar 4–Mar 10:** Hedging and trading-signal backtests; transaction cost modeling.
- **Mar 11–Mar 20:** Write-up (methods, results, discussion); finalize figures/tables.
- **Mar 21–Mar 25:** Revision cycle; finalize submission package.
