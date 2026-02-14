# Thesis Proposal Outline (Best Candidate)

## Title Page
**Title:** Arbitrage-Aware Deep Learning for Implied Volatility Surface Nowcasting and Forecasting: Evidence from OptionMetrics IvyDB  
**Date:** February 2026  
**Name:** Maurits van Eck  
**Student number (ANR):** 2062644  
**Program:** MSc Finance, Tilburg University  

---

## Part 1 — Research Question (~200 words)
Options are quoted and risk-managed through the implied volatility (IV) surface. In practice, however, the observed cross-section is sparse and noisy and arrives in irregular “point clouds” that vary day to day. Dealers therefore require stable procedures to nowcast a smooth, arbitrage-consistent IV surface from available quotes, and risk managers need reliable surface factors (level, skew, term structure) for forecasting and hedging.

**Research question:** *Do arbitrage-aware deep learning surface models (neural operators / constrained networks) produce implied volatility surfaces that (i) exhibit fewer static arbitrage violations, (ii) improve out-of-sample forecasting of IV factor dynamics, and (iii) improve downstream pricing and hedging outcomes relative to classical parametric benchmarks (SVI-style fits)?*

Economically, better surface estimation reduces model risk and hedging costs for market makers and institutional investors. If the surface forecasts improve, it also provides a clean channel for economically meaningful volatility trading signals.

---

## Part 2 — Literature Review (MAX 1 PAGE)
Black and Scholes (1973) and Merton (1973) provide the no-arbitrage foundation, but observed option prices imply volatility smiles and term structure that contradict constant volatility. Bakshi, Cao, and Chen (1997, *Review of Financial Studies*) demonstrate that richer dynamics (stochastic volatility and jumps) materially improve pricing across strikes and maturities.

In practice, the IV surface must be smooth and satisfy static no-arbitrage (e.g., call prices decreasing and convex in strike). Dealer parameterizations such as SVI provide a pragmatic solution but can be unstable under uneven cross-sections or stressed markets.

Recent machine learning work targets surface estimation directly. Operator deep smoothing (Wiedemann, Jacquier, & Gonon, 2024) proposes a neural-operator mapping from irregular quote sets to a full surface, emphasizing robustness to subsampling and adherence to no-arbitrage constraints. Related ML approaches embed financial structure via constraints (PINNs) or payoff boundary conditions (constrained deep learning for pricing/hedging). Since option-implied variables relate to risk premia and expected returns (e.g., Goyal & Saretto, 2009, *Review of Financial Studies*), improved measurement and forecasting of IV factors is economically meaningful.

---

## Part 3 — Research Plan (~500 words; key equations included)
### 3.1 Surface estimation problem
At each day \(t\) observe \(\mathcal{D}_t=\{(k_{j,t},\tau_{j,t},\sigma^{IV}_{j,t},w_{j,t})\}_{j=1}^{n_t}\), where \(k=\ln(K/S)\) and \(w\) is a quality weight (e.g., inverse spread).

Estimate a fitted surface \(\widehat{\sigma}^{(m)}_t(k,\tau)\) on a fixed grid \(\mathcal{G}\).

### 3.2 Models
- **Benchmark:** parametric surface (SVI-style) estimated per day with smoothness/constraint penalties.
- **Deep learning surface:** set-to-surface (operator / constrained network)
\[
\widehat{\sigma}^{DL}_t(\mathcal{G}) = f_\theta(\mathcal{D}_t),
\]
trained via
\[
\min_{\theta} \sum_t \sum_{j=1}^{n_t} w_{j,t}\big(\sigma^{IV}_{j,t}-\widehat{\sigma}^{DL}_t(k_{j,t},\tau_{j,t})\big)^2 + \lambda\,\mathcal{P}(\widehat{\sigma}^{DL}_t),
\]
where \(\mathcal{P}(\cdot)\) penalizes discretized static-arbitrage violations.

### 3.3 Static arbitrage evaluation
Compute per-day arbitrage score
\[
A_t^{(m)} = \frac{1}{|\mathcal{G}|}\sum_{(k,\tau)\in\mathcal{G}} \mathbb{1}(\text{violation at }(k,\tau)).
\]
Test whether \(A_t^{DL}\) is lower than \(A_t^{SVI}\) on average.

### 3.4 Forecasting IV factors
Extract factors \(F_t=(Level_t,Skew_t,Term_t)\) from each fitted surface and estimate
\[
F_{t+h} = \alpha_h + \beta_h F_t^{(m)} + \Gamma_h W_t + \varepsilon_{t+h},
\]
with controls \(W_t\) (VIX, realized volatility, rates). Compare out-of-sample RMSE and \(R^2\) across models.

### 3.5 Downstream pricing & hedging
Price hold-out options via Black–Scholes using \(\widehat{\sigma}^{(m)}\) and compute deltas from the surface. Evaluate delta-hedged P&L with costs:
\[
\Pi_{i,t+1}^{(m)} = P_{i,t+1}-P_{i,t} - \widehat{\Delta}^{(m)}_{i,t}(S_{t+1}-S_t) - c\,S_t\,|\widehat{\Delta}^{(m)}_{i,t}-\widehat{\Delta}^{(m)}_{i,t-1}|.
\]

---

## Part 4 — Data Sources (~400 words)
**Primary dataset:** OptionMetrics IvyDB US via WRDS. **Access is verified** through Tilburg University’s WRDS subscription.

**Universe:** SPX options (baseline) and optionally S&P 100 equity options for external validity. Apply liquidity filters (positive bid, spread thresholds, minimum open interest).

**Auxiliary data:** CRSP (underlyings/dividends), Treasury yields (risk-free), VIX (regime splits), realized volatility from underlying returns.

**Descriptive statistics table (to be filled from IvyDB):**

| Variable | Mean | Std | P25 | Median | P75 |
|---|---:|---:|---:|---:|---:|
| Quotes per day (n_t) |  |  |  |  |  |
| ATM 30D IV (%) |  |  |  |  |  |
| 25D skew (IV pts) |  |  |  |  |  |
| Term slope (90D–30D, IV pts) |  |  |  |  |  |
| Median bid–ask spread (IV pts) |  |  |  |  |  |
| Arbitrage score A_t (SVI) |  |  |  |  |  |
| Arbitrage score A_t (DL) |  |  |  |  |  |

---

## Part 5 — References (APA)
Bakshi, G., Cao, C., & Chen, Z. (1997). Empirical performance of alternative option pricing models. *The Review of Financial Studies, 10*(4), 1115–1162.

Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy, 81*(3), 637–654.

Goyal, A., & Saretto, A. (2009). Cross-section of option returns and volatility. *The Review of Financial Studies, 22*(12), 5025–5057.

Merton, R. C. (1973). Theory of rational option pricing. *The Bell Journal of Economics and Management Science, 4*(1), 141–183.

Wiedemann, R., Jacquier, A., & Gonon, L. (2024). Operator deep smoothing for implied volatility. *arXiv preprint* arXiv:2406.11520.
