# Tilburg MSc Finance — Thesis Proposal Outline (Maurits van Eck, ANR 2062644)

## Title
Machine Learning for Option Pricing and Delta-Hedging: Out-of-Sample Accuracy, Robustness, and Transaction Costs

## Part 1 — Research Question
- RQ: Do ML models improve out-of-sample option pricing and delta-hedging performance vs Black–Scholes-type benchmarks, and do gains survive transaction costs and different volatility regimes?
- Contribution: joint pricing + hedging evaluation, regime robustness, transaction-cost-aware backtest.

## Part 2 — Literature Review (≤ 1 page)
- Classical: Black–Scholes; extensions and empirical mispricing (Bakshi et al., JF; Christoffersen et al., JFE; Carr & Wu, JFE; Bakshi et al., RFS).
- ML in finance: Gu et al. (JF) + deep learning factor models (Feng et al., JFQA).
- Gap: option-ML literature often focuses on pricing errors; limited evidence on hedging P&L net of costs and regime stability.

## Part 3 — Research Plan
- Data: OptionMetrics IvyDB US (WRDS), SPX (primary), 2015–2024.
- Models:
  - Benchmarks: BS (IV-based and historical-vol-based), simple fitted IV surface.
  - ML: GBRT (XGBoost-style), MLP NN; optional Transformer/sequence model.
- Main supervised model: C_mid = f_theta(X) + eps.
- Delta extraction: autodiff (NN) or numerical derivative (trees).
- Hedging test: delta-hedged P&L with transaction cost proxy.
- Statistical tests: pricing RMSE/MAE; hedging P&L distribution; bucketed results; cross-sectional regression of |errors| on moneyness, maturity, spreads, VIX.

## Part 4 — Data Sources
- Confirm WRDS/OptionMetrics access: yes (Tilburg subscription).
- Required: descriptive stats table from first WRDS extract (placeholder until authenticated download).

## Part 5 — References (APA)
- Include all cited papers, prioritizing top finance journals + core ML evaluation reference (Management Science).
