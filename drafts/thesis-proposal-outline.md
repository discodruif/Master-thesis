# Thesis Proposal Outline (Selected Best Proposal)

**Lead Proposal Title:** No‑Arbitrage Constrained Machine Learning for Implied Volatility Surfaces: Pricing and Hedging Implications in SPX Options  
**Author:** Maurits van Eck  
**ANR:** 2062644  
**Program:** MSc Finance, Tilburg University  
**Date:** February 16, 2026  
**Status:** Lead Proposal (selected from 5 alternatives)

---

## Part 1: Research Question

Option quotes are sparse and noisy, and unconstrained ML surface estimators can violate static no‑arbitrage restrictions (monotonicity/convexity in strike and calendar monotonicity). These violations may distort downstream pricing and hedging. The thesis asks:

**Main RQ:** *Does enforcing no‑arbitrage constraints in machine‑learning models of the implied volatility surface improve (i) out‑of‑sample option pricing accuracy and (ii) transaction‑cost‑adjusted hedging performance relative to unconstrained ML and classical parametric benchmarks?*

Economic contribution: connect *statistical surface fit* to *economic outcomes* (hedging losses net of transaction costs, tail risk), providing implementable evidence for derivatives desks and risk managers.

---

## Part 2: Literature Review (1 page target)

- **Classical foundations:** Black & Scholes (1973); stochastic volatility as a key empirical extension (Heston, 1993).
- **Deep hedging / economic evaluation:** Buehler et al. (2019, *Quantitative Finance*) frames hedging as direct optimization under discrete trading and frictions.
- **Constraint-aware IV surface learning:** Hoshisashi, Phelan & Barucca (2024) propose Whack‑a‑mole Learning to balance multi‑objective deep calibration while satisfying PDE/no‑arbitrage constraints.
- **IVS information in hedging:** François et al. (2024) show that implied-volatility-surface feedback improves RL hedging for S&P 500 options, particularly with transaction costs.
- **Surface dynamics and scenario realism:** Choudhary, Jaimungal & Bergeron (2023, FuNVol) combine functional PCA and neural SDEs to model implied-vol dynamics.

Positioning: the thesis tests whether **arbitrage‑free ML surfaces** deliver economic gains beyond improved fit.

---

## Part 3: Research Plan (must include equations)

### Stage A — Surface estimation (daily)
Estimate IVS \(\sigma(K,T)\) using three approaches:
1) parametric benchmark (e.g., SVI/Heston-style proxies), 2) unconstrained ML, 3) constrained ML with explicit static no‑arbitrage penalties.

Key static constraints in price space for calls \(C(K,T)\):
- \(\partial C/\partial K \le 0\) (monotonicity)
- \(\partial^2 C/\partial K^2 \ge 0\) (butterfly / convexity)
- \(\partial C/\partial T \ge 0\) (calendar)

### Stage B — Pricing accuracy (out-of-sample)
\[
|PE_{i,t}^{(m)}| = \gamma_m + \delta_1 \text{BidAsk}_{i,t} + \delta_2 \text{Moneyness}_{i,t} + \delta_3 \text{TTM}_{i,t} + u_{i,t}
\]
Compare \(\gamma_m\) across models \(m\) in rolling time splits.

### Stage C — Hedging performance (net of transaction costs)
Compute discrete delta‑hedging P&L using deltas from each surface; evaluate mean, P95, and CVaR of hedging losses.

Hedging loss regression:
\[
HE_{i,t}^{(m)} = \alpha_m + \beta_1 \lvert\Delta\sigma_{t}\rvert + \beta_2 \text{TC}_{i,t} + \beta_3 \text{VIX}_{t} + \beta_4 \text{Moneyness}_{i,t} + \beta_5 \text{TTM}_{i,t} + \varepsilon_{i,t}
\]
Focus: do constrained surfaces reduce hedging losses especially in high‑VIX regimes?

---

## Part 4: Data Sources (must include descriptive stats table)

**Primary dataset:** **OptionMetrics IvyDB US via WRDS (access verified).**

- SPX option end‑of‑day quotes (bid/ask/mid), implied vol, contract terms (K, T), liquidity measures.
- Underlying index series and inputs for forward/discounting (risk‑free rates).

**Descriptive statistics table (to be filled in thesis):**

| Variable | Mean | Std | P5 | P50 | P95 | N |
|---|---:|---:|---:|---:|---:|---:|
| Implied volatility (IV) |  |  |  |  |  |  |
| Log-moneyness ln(K/F) |  |  |  |  |  |  |
| Time-to-maturity (days) |  |  |  |  |  |  |
| Bid-ask spread |  |  |  |  |  |  |
| Option mid price |  |  |  |  |  |  |
| Underlying daily return |  |  |  |  |  |  |

---

## Part 5: References (APA)

Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy, 81*(3), 637–654.

Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. *Quantitative Finance, 19*(8), 1271–1291.

Choudhary, V., Jaimungal, S., & Bergeron, M. (2023). FuNVol: A multi-asset implied volatility market simulator using functional principal components and neural SDEs. *Working paper*.

François, P., Gauthier, G., Godin, F., & Pérez Mendoza, C. O. (2024). Enhancing deep hedging of options with implied volatility surface feedback information. *arXiv*.

Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *Review of Financial Studies, 6*(2), 327–343.

Hoshisashi, K., Phelan, C. E., & Barucca, P. (2024). Whack-a-mole learning: Physics-informed deep calibration for implied volatility surface. *Working paper/Proceedings*.
