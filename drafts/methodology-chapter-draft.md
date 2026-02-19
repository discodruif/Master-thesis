# Chapter 3: Methodology

**Thesis:** Machine Learning for Option Mispricing: Predicting Cross-Sectional Option Returns and Evaluating Net-of-Cost Trading Strategies  
**Author:** Maurits van Eck (ANR: 2062644)  
**Program:** MSc Finance, Tilburg University  
**Draft Date:** February 19, 2026

---

## 3.1 Research Design Overview

This chapter describes the empirical methodology for investigating whether machine learning models can predict cross-sectional option returns out-of-sample and generate economically meaningful long–short trading strategies after realistic transaction costs. The design follows a four-stage pipeline: (i) data construction and feature engineering, (ii) model estimation with rolling out-of-sample evaluation, (iii) portfolio formation and economic evaluation, and (iv) robustness analysis and economic interpretation.

The methodological framework is anchored in three key contributions from the literature. First, I follow the ML asset pricing paradigm of Gu, Kelly, and Xiu (2020), who establish that genuine out-of-sample evaluation with rolling time splits is essential for credible ML prediction claims in finance. Second, I extend the option return prediction framework of Bali, Beckmeyer, Moerke, and Weigert (2023), who demonstrate that ML with large option characteristic sets can predict the cross-section of option returns—the central empirical anchor for this thesis. Third, I incorporate the transaction cost realism emphasized in the limits-to-arbitrage literature (Muravyev, 2016; De Fontnouvelle, Fishe, & Harris, 2003), ensuring that reported performance reflects implementable profits.

Figure 3.1 (to be included) presents a schematic overview of the pipeline from raw OptionMetrics data through to net-of-cost portfolio performance.

---

## 3.2 Data and Sample Construction

### 3.2.1 Primary Data Source

The primary dataset is **OptionMetrics IvyDB US**, accessed via the Wharton Research Data Services (WRDS) platform (access verified). IvyDB provides end-of-day option quotes, implied volatilities, and Greeks for all US-listed equity and index options. I focus on **S&P 500 index options (SPX)** over the sample period **January 2010 to December 2024** (15 years, approximately 3,750 trading days). This sample captures multiple distinct volatility regimes:

- European debt crisis (2011–2012)
- Low-volatility environment (2013–2019)
- COVID-19 crash and recovery (March–December 2020)
- Post-COVID rate-hiking cycle (2022–2024)

The regime diversity is methodologically important because ML model performance may be regime-dependent (a concern raised by Fan & Sirignano, 2024, in their survey of ML derivative pricing methods).

### 3.2.2 Variables Extracted

For each option contract $i$ observed on date $t$, I extract the following variables from IvyDB:

| Variable | Description | IvyDB Field |
|---|---|---|
| $P^{bid}_{i,t}$, $P^{ask}_{i,t}$ | Best bid and ask prices | `best_bid`, `best_offer` |
| $P^{mid}_{i,t}$ | Mid price: $(P^{bid} + P^{ask})/2$ | Computed |
| $K_i$ | Strike price | `strike_price` |
| $T_i$ | Expiration date | `exdate` |
| $\tau_{i,t}$ | Time to maturity (calendar days) | $T_i - t$ |
| $\sigma^{IV}_{i,t}$ | Implied volatility | `impl_volatility` |
| $\Delta_{i,t}$ | Option delta | `delta` |
| $\Gamma_{i,t}$ | Option gamma | `gamma` |
| $\mathcal{V}_{i,t}$ | Option vega | `vega` |
| $\Theta_{i,t}$ | Option theta | `theta` |
| $V_{i,t}$ | Daily volume (contracts) | `volume` |
| $OI_{i,t}$ | Open interest | `open_interest` |
| $\text{CP}_i$ | Call/put indicator | `cp_flag` |
| $S_t$ | Underlying index level | `spot_price` or SPX close |
| $F_{t,\tau}$ | Forward price (for moneyness) | `forward_price` |

### 3.2.3 Sample Filters

I apply standard filters following Bali et al. (2023) and the broader options literature to ensure data quality and economic relevance:

1. **Positive bid:** exclude options with $P^{bid}_{i,t} = 0$ (no tradable quote)
2. **Maturity bounds:** retain options with $7 \leq \tau_{i,t} \leq 365$ calendar days (removes near-expiration noise and illiquid long-dated contracts)
3. **Moneyness bounds:** retain options with log-moneyness $m_{i,t} = \ln(K_i / F_{t,\tau}) \in [-0.20, +0.20]$, where $F_{t,\tau}$ is the forward price (excludes deep ITM/OTM options with extreme leverage and illiquidity)
4. **Duplicate/erroneous quotes:** remove records with missing implied volatility, negative Greeks where sign should be fixed, or inconsistent bid-ask relationships ($P^{bid} > P^{ask}$)
5. **Minimum liquidity (broad sample):** daily volume $\geq 10$ contracts and open interest $\geq 100$
6. **Minimum liquidity (tradable universe):** for portfolio construction, impose stricter filters of daily volume $\geq 50$ contracts and open interest $\geq 500$

The distinction between the broad sample (used for model training) and the tradable universe (used for portfolio evaluation) follows Bali et al. (2023) and ensures that reported trading profits are implementable. I document the impact of each filter on sample size.

### 3.2.4 Supplementary Data

| Dataset | Source | Purpose |
|---|---|---|
| SPX index levels | OptionMetrics / CRSP | Underlying returns, realized volatility |
| Risk-free rate | WRDS Treasury curve | Return computation, forward prices |
| VIX index | CBOE (via WRDS) | Regime classification |
| Fama–French factors | Kenneth French Data Library | Risk adjustment |
| Realized volatility | Computed from daily SPX returns | Feature construction |

---

## 3.3 Variable Construction

### 3.3.1 Option Return Definitions

The dependent variable is the holding-period return on option $i$ from date $t$ to date $t+h$, where the baseline holding period is $h = 5$ trading days (one week):

$$R^{opt}_{i,t \to t+h} = \frac{P^{mid}_{i,t+h} - P^{mid}_{i,t}}{P^{mid}_{i,t}} \tag{3.1}$$

Following the convention in Goyal and Saretto (2009) and Cao and Han (2013), I also compute **delta-hedged returns** to isolate the option return component not attributable to directional moves in the underlying:

$$R^{dh}_{i,t \to t+h} = \frac{\left(P^{mid}_{i,t+h} - P^{mid}_{i,t}\right) - \Delta_{i,t} \cdot \left(S_{t+h} - S_t\right)}{P^{mid}_{i,t}} \tag{3.2}$$

where $\Delta_{i,t}$ is the OptionMetrics-provided Black–Scholes delta. The delta-hedged return captures exposure to volatility risk, jump risk, and higher-order Greeks while removing first-order directional exposure. This decomposition allows testing whether ML signals reflect volatility risk premia, mispricing, or both.

As robustness, I consider alternative holding periods of $h \in \{1, 5, 20\}$ trading days, creating a spectrum from high-frequency (daily rebalancing) to lower-turnover (monthly) strategies. The trade-off between signal decay and transaction cost accumulation across horizons is an empirically important question.

### 3.3.2 Feature Vector Construction

The feature vector $\mathbf{X}_{i,t} \in \mathbb{R}^p$ for each option-date observation combines option-level characteristics, implied volatility surface features, and underlying characteristics. The construction follows the "kitchen sink" approach of Gu, Kelly, and Xiu (2020) and Bali et al. (2023), letting the ML models determine which features carry predictive power.

#### Option-Level Features

| Feature | Definition | Motivation |
|---|---|---|
| Log-moneyness | $m_{i,t} = \ln(K_i / F_{t,\tau})$ | Controls for leverage; well-known predictor of option returns |
| Time to maturity | $\tau_{i,t}$ (calendar days, log-transformed) | Theta decay rate; affects option return distributions |
| Implied volatility | $\sigma^{IV}_{i,t}$ | Core pricing input; IV level predicts option returns (Goyal & Saretto, 2009) |
| IV deviation | $\sigma^{IV}_{i,t} - \sigma^{RV}_{t,w}$ | Deviation of implied from realized vol; central mispricing signal |
| Delta | $|\Delta_{i,t}|$ | Moneyness proxy; affects return sensitivity |
| Gamma | $\Gamma_{i,t} \cdot S_t$ | Dollar gamma; measures convexity exposure |
| Vega | $\mathcal{V}_{i,t} / P^{mid}_{i,t}$ | Vega-to-price ratio; volatility sensitivity per dollar invested |
| Theta | $\Theta_{i,t} / P^{mid}_{i,t}$ | Time decay rate per dollar invested |
| Bid-ask spread (relative) | $(P^{ask} - P^{bid}) / P^{mid}$ | Illiquidity proxy; limits-to-arbitrage measure |
| Log volume | $\ln(1 + V_{i,t})$ | Trading activity; demand pressure |
| Log open interest | $\ln(1 + OI_{i,t})$ | Position concentration; information content |
| Put indicator | $\mathbb{1}\{\text{CP}_i = P\}$ | Put-call asymmetry in return predictability |

#### Implied Volatility Surface Features

Following Kelly, Kuznetsov, Malamud et al. (2023) and Ackerer, Tagasovska, and Vatter (2020), I extract cross-sectional features from the IV surface to capture market-wide option pricing dynamics:

| Feature | Definition | Motivation |
|---|---|---|
| ATM IV level | $\sigma^{ATM}_t$ (IV at $|\Delta| \approx 0.50$) | Overall volatility pricing level |
| IV skew | $\sigma^{25\Delta P}_t - \sigma^{ATM}_t$ | Put skew; tail risk pricing |
| IV term slope | $\sigma^{ATM,90d}_t - \sigma^{ATM,30d}_t$ | Term structure shape; mean-reversion signal |
| IV butterfly | $\frac{1}{2}(\sigma^{25\Delta P}_t + \sigma^{25\Delta C}_t) - \sigma^{ATM}_t$ | Smile curvature; tail risk premium symmetry |
| VIX level | $\text{VIX}_t$ | Market-wide fear gauge |
| VIX change | $\Delta \text{VIX}_t = \text{VIX}_t - \text{VIX}_{t-5}$ | Recent volatility dynamics |

These surface-level features encode forward-looking information from the options market. As demonstrated by François, Gauthier, Godin, and Mendoza (2024) and Medvedev and Wang (2022), IV surface dynamics contain predictive information for hedging and pricing outcomes.

#### Underlying Characteristics

| Feature | Definition | Motivation |
|---|---|---|
| Past return (5d) | $r_{t-5,t}^{SPX}$ | Short-term momentum/reversal |
| Past return (21d) | $r_{t-21,t}^{SPX}$ | Medium-term momentum |
| Realized volatility (21d) | $\sigma^{RV}_{t,21d}$ | Historical vol benchmark for IV deviation |
| Realized volatility (63d) | $\sigma^{RV}_{t,63d}$ | Longer-horizon vol estimate |
| Realized skewness (63d) | $\text{Skew}^{RV}_{t,63d}$ | Asymmetry of underlying returns |

In total, the raw feature vector has $p \approx 20$–$25$ dimensions. All features are constructed using only information available at time $t$, strictly avoiding look-ahead bias. Continuous features are standardized cross-sectionally (rank-transformed to uniform $[0,1]$ within each date $t$) following Gu, Kelly, and Xiu (2020), which improves ML model stability and reduces the influence of outliers.

---

## 3.4 Machine Learning Models

I estimate four model classes of increasing complexity, providing a systematic comparison from linear to nonlinear methods. The model selection is motivated by the empirical asset pricing ML literature (Gu et al., 2020; Ivașcu, 2021; Fan & Sirignano, 2024) and designed to identify the marginal contribution of nonlinearity and interaction effects.

### 3.4.1 Linear Benchmark: Elastic Net

The linear benchmark is an elastic net regression combining $\ell_1$ (Lasso) and $\ell_2$ (Ridge) penalties:

$$\hat{\beta} = \arg\min_{\beta} \left\{ \frac{1}{N_t} \sum_{i=1}^{N_t} \left(R_{i,t+h} - \beta_0 - \mathbf{X}_{i,t}'\beta\right)^2 + \lambda \left[\alpha \|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2 \right] \right\} \tag{3.3}$$

where $\lambda > 0$ is the regularization strength and $\alpha \in [0,1]$ controls the $\ell_1$/$\ell_2$ mix. The elastic net nests OLS ($\lambda = 0$), Ridge ($\alpha = 0$), and Lasso ($\alpha = 1$). By including this model, I quantify the baseline predictability achievable with linear methods and measure the incremental gains from nonlinear ML models.

**Hyperparameters:** $\lambda$ is selected via the validation set from a logarithmic grid of 50 values; $\alpha \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$.

### 3.4.2 Gradient Boosted Trees (XGBoost)

XGBoost (Chen & Guestrin, 2016) builds an additive ensemble of decision trees via gradient boosting:

$$\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{X}_i), \quad f_k \in \mathcal{F} \tag{3.4}$$

where each $f_k$ is a regression tree and $\mathcal{F}$ is the space of CART trees. Trees are added sequentially, with each new tree fitted to the negative gradient (residuals) of the loss function with respect to the current ensemble predictions.

XGBoost has several properties that make it well-suited for option return prediction. It handles nonlinear relationships and feature interactions natively, is robust to outliers through gradient-based fitting, and includes built-in regularization ($\ell_1$ and $\ell_2$ penalties on leaf weights, maximum depth constraints). In the derivatives literature, XGBoost has demonstrated strong performance for option pricing (arXiv, 2024) and comparative studies show it can outperform both parametric models and simpler ML methods (arXiv, 2025).

**Hyperparameters** (tuned via validation set):

| Parameter | Search Range | Description |
|---|---|---|
| `n_estimators` | $\{100, 200, 500, 1000\}$ | Number of boosting rounds (with early stopping) |
| `max_depth` | $\{3, 4, 5, 6, 8\}$ | Maximum tree depth |
| `learning_rate` | $\{0.01, 0.03, 0.05, 0.1\}$ | Shrinkage rate |
| `subsample` | $\{0.7, 0.8, 0.9, 1.0\}$ | Row subsampling fraction |
| `colsample_bytree` | $\{0.6, 0.7, 0.8, 1.0\}$ | Column subsampling fraction |
| `reg_alpha` ($\ell_1$) | $\{0, 0.01, 0.1, 1.0\}$ | Lasso regularization on weights |
| `reg_lambda` ($\ell_2$) | $\{0.1, 1.0, 5.0, 10.0\}$ | Ridge regularization on weights |
| `min_child_weight` | $\{5, 10, 20, 50\}$ | Minimum leaf weight (prevents small leaves) |

Early stopping is applied based on validation loss with a patience of 20 rounds.

### 3.4.3 Random Forest

Random Forest (Breiman, 2001) constructs an ensemble of decorrelated decision trees via bagging (bootstrap aggregation) and random feature subsampling:

$$\hat{y}_i = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{X}_i) \tag{3.5}$$

where each tree $T_b$ is grown on a bootstrap sample with a random subset of $\lfloor \sqrt{p} \rfloor$ or $\lfloor p/3 \rfloor$ features considered at each split. Random Forest provides a complementary ensemble benchmark to XGBoost: it has lower variance through averaging but does not adaptively correct residuals. Ivașcu (2021) shows that Random Forests can match or exceed Black–Scholes pricing accuracy on real market data, and ensemble methods demonstrate robust performance in volatile markets (arXiv, 2023; 2025).

**Hyperparameters:**

| Parameter | Search Range |
|---|---|
| `n_estimators` | $\{200, 500, 1000\}$ |
| `max_depth` | $\{5, 10, 15, \text{None}\}$ |
| `max_features` | $\{\sqrt{p}, p/3, 0.8p\}$ |
| `min_samples_leaf` | $\{10, 20, 50\}$ |

### 3.4.4 Multilayer Perceptron (MLP)

The neural network model is a feed-forward multilayer perceptron:

$$\hat{y}_i = W^{(L)} \sigma\!\left(W^{(L-1)} \sigma\!\left(\cdots \sigma\!\left(W^{(1)} \mathbf{X}_i + b^{(1)}\right) \cdots\right) + b^{(L-1)}\right) + b^{(L)} \tag{3.6}$$

where $\sigma(\cdot)$ is the ReLU activation function, $L$ is the number of layers, and $\{W^{(l)}, b^{(l)}\}_{l=1}^{L}$ are learnable parameters. Neural networks are the most flexible function approximators in the model set and can capture complex nonlinear interactions.

The use of neural networks for option pricing has a long history dating to Hutchinson, Lo, and Poggio (1994) and has been extensively developed with modern architectures including gated networks (Yang, Zheng, & Hospedales, 2016), deep learning for calibration (Horvath, Muguruza, & Tomas, 2021), and physics-informed neural networks (Liu, Borovykh, Grzelak, & Oosterlee, 2024). While the thesis focuses on option *return* prediction rather than pricing per se, the MLP architecture captures the same function approximation advantages.

**Architecture and regularization:**

| Parameter | Search Range |
|---|---|
| Hidden layers | $\{2, 3, 4\}$ |
| Hidden units per layer | $\{32, 64, 128, 256\}$ |
| Activation | ReLU |
| Dropout rate | $\{0.0, 0.1, 0.2, 0.3\}$ |
| Batch normalization | $\{\text{True, False}\}$ |
| Learning rate | $\{10^{-4}, 5 \times 10^{-4}, 10^{-3}\}$ |
| Optimizer | Adam (Kingma & Ba, 2015) |
| Batch size | $\{256, 512, 1024\}$ |
| Weight decay ($\ell_2$) | $\{0, 10^{-5}, 10^{-4}\}$ |
| Early stopping patience | 10 epochs (based on validation loss) |
| Maximum epochs | 200 |

Training uses mini-batch stochastic gradient descent via the Adam optimizer. The loss function is mean squared error (MSE):

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left(R_{i,t+h} - \hat{R}_{i,t+h}\right)^2 \tag{3.7}$$

### 3.4.5 Model Selection Rationale

The four-model design provides a principled comparison:

| Model | Role | Key Property |
|---|---|---|
| Elastic Net | Linear benchmark | Quantifies baseline linear predictability |
| XGBoost | Nonlinear benchmark | Captures interactions, robust to outliers |
| Random Forest | Ensemble benchmark | Averaged trees, lower variance |
| MLP | Neural network | Maximum flexibility, captures smooth nonlinearities |

This progression from linear to nonlinear mirrors Gu, Kelly, and Xiu (2020), enabling me to decompose total predictability into contributions from (i) linear predictor effects, (ii) nonlinear effects and interactions (tree ensembles), and (iii) smooth nonlinear mappings (neural networks).

---

## 3.5 Training Protocol: Rolling Out-of-Sample Evaluation

### 3.5.1 Time-Series Cross-Validation Design

To prevent look-ahead bias—the most critical methodological concern in financial ML (Gu et al., 2020; Bali et al., 2023)—I use an **expanding window** training design with a validation buffer:

$$
\underbrace{[1, \ldots, t_{\text{train}}]}_{\text{Training}} \quad \underbrace{[t_{\text{train}}+1, \ldots, t_{\text{val}}]}_{\text{Validation}} \quad \underbrace{[t_{\text{val}}+1, \ldots, t_{\text{test}}]}_{\text{Test (OOS)}}
$$

**Baseline specification:**
- **Training window:** expanding, starting with a minimum of 36 months (approximately 756 trading days)
- **Validation window:** 3 months (approximately 63 trading days) immediately following the training window
- **Test window:** 1 month (approximately 21 trading days) immediately following the validation window
- **Step size:** the window advances by 1 month at each iteration

This produces approximately $15 \times 12 - 36 - 3 = 141$ out-of-sample monthly prediction periods (from mid-2013 through December 2024). The validation window serves for hyperparameter selection (XGBoost early stopping, neural network architecture, regularization parameters), while the test window provides the genuine out-of-sample predictions used for all economic evaluations.

**Rationale for expanding (vs. rolling) window:** An expanding window uses all available historical data, which is preferable when the underlying data-generating process is stable or when data efficiency matters (as with option panels). As a robustness check, I also implement a **fixed rolling window** of 60 months to examine whether more recent data subsumes older patterns—relevant if option market microstructure or trading technology has evolved (a concern raised by the evolution from floor trading to electronic markets over the sample period).

### 3.5.2 Hyperparameter Selection

Hyperparameters are selected via **randomized search** on the validation set for each rolling window iteration. I sample 50 hyperparameter configurations for XGBoost and Random Forest, and 30 configurations for the MLP (which is more expensive to train). The configuration minimizing validation MSE is selected for the test-period predictions.

For the elastic net, I use efficient coordinate descent with the full $\lambda$ path, selecting the optimal $(\lambda, \alpha)$ pair via validation MSE.

### 3.5.3 Prediction Generation

At each test date $t$, the selected model produces predicted option returns $\hat{R}_{i,t \to t+h}$ for all options $i$ in the tradable universe. These predictions are stored and used for portfolio construction (Section 3.6). Importantly, no information from $t+1$ onward is used in constructing $\hat{R}_{i,t}$.

---

## 3.6 Portfolio Construction and Economic Evaluation

### 3.6.1 Portfolio Formation

At each weekly rebalancing date $t$, I sort options in the tradable universe into **decile portfolios** based on predicted returns $\hat{R}_{i,t \to t+h}$:

- **Long portfolio:** top decile (highest predicted returns)
- **Short portfolio:** bottom decile (lowest predicted returns)
- **Long–short portfolio:** long minus short

Within each decile, positions are **equally weighted**:

$$R^{LS}_t = \frac{1}{N_L}\sum_{i \in D_{10}} R_{i,t \to t+h} - \frac{1}{N_S}\sum_{i \in D_1} R_{i,t \to t+h} \tag{3.8}$$

where $D_{10}$ and $D_1$ denote the top and bottom deciles, and $N_L$, $N_S$ are the number of options in each. Equal weighting avoids concentration in a few contracts and is standard in the option return predictability literature (Bali et al., 2023).

**Portfolio constraints:** In addition to the tradable universe filters (Section 3.2.3), I impose:
- Maximum position: no single option exceeds 5% of portfolio notional
- Minimum diversification: at least 15 options per leg
- Same-expiration constraint: avoid concentration in a single expiration date

### 3.6.2 Transaction Cost Model

Options markets are characterized by wide bid–ask spreads, particularly for out-of-the-money contracts (De Fontnouvelle, Fishe, & Harris, 2003; Muravyev, 2016). To evaluate implementable performance, I model transaction costs at three levels of conservatism:

| Scenario | Cost per Transaction | Description |
|---|---|---|
| **Optimistic** | $\frac{1}{2}(P^{ask} - P^{bid})$ on entry and exit | Half-spread; assumes ability to trade near mid |
| **Baseline** | $(P^{ask} - P^{bid})$ on entry and exit | Full spread; conservative but standard |
| **Worst-case** | $(P^{ask} - P^{bid}) \times 1.10$ on entry and exit | Full spread + 10% slippage markup |

Net returns are computed as:

$$R^{net}_{p,t} = R^{gross}_{p,t} - \kappa_{p,t} \cdot \text{TO}_{p,t} \tag{3.9}$$

where $\kappa_{p,t}$ is the portfolio-weighted average transaction cost and $\text{TO}_{p,t}$ is the portfolio turnover at time $t$. Turnover is defined as the fraction of portfolio value that changes at each rebalancing:

$$\text{TO}_{p,t} = \sum_{i} |w_{i,t} - w_{i,t-h}^{+}| \tag{3.10}$$

where $w_{i,t}$ is the target weight and $w_{i,t-h}^{+}$ is the drifted weight from the previous period.

I also compute the **break-even spread**: the average transaction cost level at which the strategy's net annualized Sharpe ratio equals zero:

$$\kappa^{BE} = \frac{\bar{R}^{gross}_{LS}}{\overline{\text{TO}}} \tag{3.11}$$

This metric provides an intuitive measure of strategy implementability: if the break-even spread exceeds observed bid–ask spreads in the tradable universe, the strategy is economically viable even under conservative cost assumptions.

### 3.6.3 Performance Metrics

I evaluate portfolio performance using the following metrics:

| Metric | Definition | Purpose |
|---|---|---|
| Annualized mean return | $\bar{R} \times (252/h)$ | Average profitability |
| Annualized volatility | $\hat{\sigma}_R \times \sqrt{252/h}$ | Risk |
| Sharpe ratio | $\text{SR} = \bar{R} / \hat{\sigma}_R \times \sqrt{252/h}$ | Risk-adjusted return |
| Maximum drawdown | $\text{MDD} = \max_{s \leq t} \left(\text{CumRet}_s - \text{CumRet}_t\right) / \text{CumRet}_s$ | Worst peak-to-trough loss |
| Skewness | $\hat{\gamma}_3$ of return distribution | Tail asymmetry |
| Kurtosis | $\hat{\gamma}_4$ of return distribution | Tail heaviness |
| CVaR (5%) | $\mathbb{E}[R \mid R \leq \text{VaR}_{5\%}]$ | Expected shortfall; tail risk |
| Hit rate | Fraction of periods with $R > 0$ | Consistency |

These metrics are reported for both gross and net (after transaction costs) returns across all three cost scenarios.

---

## 3.7 Statistical Tests and Inference

### 3.7.1 Out-of-Sample Prediction Accuracy

I evaluate the statistical accuracy of ML predictions using the out-of-sample $R^2$ (Campbell & Thompson, 2008):

$$R^2_{OOS} = 1 - \frac{\sum_{t \in \text{OOS}} \sum_{i} (R_{i,t+h} - \hat{R}_{i,t+h})^2}{\sum_{t \in \text{OOS}} \sum_{i} (R_{i,t+h} - \bar{R}_{t})^2} \tag{3.12}$$

where $\bar{R}_t$ is the historical mean return (the benchmark forecast). A positive $R^2_{OOS}$ indicates that the ML model outperforms the naïve historical mean. Following Gu, Kelly, and Xiu (2020), even small positive values ($R^2_{OOS} > 0$) can generate economically meaningful trading profits in asset pricing contexts.

Statistical significance of $R^2_{OOS} > 0$ is assessed using the Clark and West (2007) test, which adjusts for the estimation noise inherent in comparing nested forecasts.

### 3.7.2 Spanning Regression

To test whether ML signals contain incremental information beyond standard option predictors, I estimate:

$$R^{opt}_{i,t \to t+h} = \alpha + \phi \, \hat{R}^{ML}_{i,t} + \boldsymbol{\beta}' \mathbf{Z}_{i,t} + \epsilon_{i,t} \tag{3.13}$$

where $\hat{R}^{ML}_{i,t}$ is the ML predicted return and $\mathbf{Z}_{i,t}$ includes the conventional predictors (IV level, IV-RV deviation, moneyness, time to maturity, bid-ask spread, volume). A significant coefficient $\phi > 0$ indicates that the ML signal provides information beyond what is captured by these standard predictors individually.

Standard errors are clustered by date to account for cross-sectional dependence within each period (Petersen, 2009).

### 3.7.3 Risk Factor Regression

To assess whether long–short portfolio returns represent compensation for known risk factors or genuine alpha, I estimate:

$$R^{net}_{LS,t} = \alpha + \beta_1 \text{MKT}_t + \beta_2 \Delta\text{VIX}_t + \beta_3 \text{Mom}_t + \beta_4 \text{HML}_t + \beta_5 \text{SMB}_t + \varepsilon_t \tag{3.14}$$

where MKT is the market excess return, $\Delta\text{VIX}$ captures volatility risk (Ang, Hodrick, Xing, & Zhang, 2006), and Mom, HML, SMB are the Fama–French momentum and value/size factors. The intercept $\alpha$ provides the **abnormal return** after controlling for systematic risk exposures.

I use Newey–West (1987) standard errors with automatic lag selection to correct for heteroskedasticity and autocorrelation in the time series of portfolio returns.

### 3.7.4 Mispricing vs. Risk Premium Decomposition

A central question is whether ML-predicted returns reflect genuine mispricing or compensation for bearing risk. To investigate, I estimate:

$$R^{dh}_{i,t \to t+h} = \alpha + \psi \, M_{i,t} + \boldsymbol{\eta}' \mathbf{W}_{i,t} + u_{i,t} \tag{3.15}$$

where:
- $R^{dh}_{i,t \to t+h}$ is the delta-hedged return (Equation 3.2), isolating non-directional option return
- $M_{i,t} = (\sigma^{IV}_{i,t} - \hat{\sigma}^{IV}_{i,t}) / \sigma^{IV}_{i,t}$ is the mispricing measure, defined as the deviation of observed IV from a benchmark IV surface fitted via SVI parametrization (Gatheral, 2004) or a cubic spline across moneyness
- $\mathbf{W}_{i,t}$ includes controls for gamma exposure ($\Gamma \cdot S^2$), vega exposure ($\mathcal{V}$), theta ($\Theta$), and option illiquidity (relative bid-ask spread)

Using delta-hedged returns as the dependent variable isolates returns beyond directional risk. Controlling for higher-order Greeks addresses whether the signal reflects compensation for non-directional risk exposures or genuine mispricing. A significant $\psi$ after controlling for risk exposures is consistent with limits-to-arbitrage explanations (Goyal & Saretto, 2009).

---

## 3.8 Model Interpretability

### 3.8.1 Feature Importance

For tree-based models (XGBoost, Random Forest), I report **gain-based feature importance**: the average improvement in the loss function across all splits using a given feature. This provides a ranking of which features contribute most to predictive accuracy.

### 3.8.2 SHAP Values

To provide a unified, model-agnostic interpretation across all four model classes, I compute **SHAP values** (SHapley Additive exPlanations; Lundberg & Lee, 2017). For each prediction, SHAP decomposes the output into additive contributions from each feature:

$$\hat{R}_{i,t} = \phi_0 + \sum_{j=1}^{p} \phi_j(\mathbf{X}_{i,t}) \tag{3.16}$$

where $\phi_j$ is the Shapley value for feature $j$. This framework satisfies desirable axiomatic properties (efficiency, symmetry, dummy, additivity) and allows me to identify:

1. **Which features drive predictability** across the full sample
2. **Whether feature importance is regime-dependent** (e.g., do different features matter in high-VIX vs. low-VIX periods?)
3. **Economic interpretation**: connecting ML signals to known option pricing factors such as the volatility risk premium, skew premium, and liquidity

The SHAP analysis connects ML predictions to economic mechanisms, addressing the "black box" criticism that is particularly relevant in academic finance research.

### 3.8.3 Partial Dependence Plots

For the top features identified by SHAP, I construct **partial dependence plots (PDPs)** to visualize the marginal effect of each feature on predicted returns, holding other features constant. This reveals the functional form of the nonlinear relationships captured by ML models—for example, whether the relationship between IV deviation and predicted return is monotonic, concave, or exhibits threshold effects.

---

## 3.9 Robustness Analysis

### 3.9.1 Volatility Regime Analysis

I split the sample at the **median VIX level** into low-volatility and high-volatility subsamples and re-evaluate all performance metrics separately. Under the limits-to-arbitrage hypothesis (Shleifer & Vishny, 1997), ML predictability should be stronger in high-volatility periods when arbitrage constraints bind more tightly—wider spreads, higher margin requirements, and greater inventory risk discourage sophisticated traders from correcting mispricings.

### 3.9.2 Moneyness and Maturity Buckets

I evaluate model performance and portfolio returns separately for:
- **ATM options:** $|m_{i,t}| < 0.03$
- **OTM puts:** $m_{i,t} < -0.03$ and CP $= P$
- **OTM calls:** $m_{i,t} > 0.03$ and CP $= C$
- **Near-term:** $\tau < 30$ days
- **Medium-term:** $30 \leq \tau \leq 90$ days
- **Longer-term:** $\tau > 90$ days

This decomposition reveals where ML predictability concentrates. The literature suggests OTM puts carry the richest information about tail risk premia and are most subject to demand-based pricing (Bates, 2003; Bakshi, Cao, & Chen, 1997).

### 3.9.3 Holding Period Variation

Comparing $h \in \{1, 5, 20\}$ trading days quantifies the turnover–alpha trade-off. Shorter horizons may capture faster-decaying signals but incur higher transaction costs; longer horizons reduce costs but may dilute predictive signals. I report the break-even spread for each horizon to identify the optimal frequency.

### 3.9.4 Conservative Execution

As the strictest implementability check, I compute returns using **buy-at-ask / sell-at-bid** execution prices rather than mid prices:

$$R^{cons}_{i,t \to t+h} = \frac{P^{bid}_{i,t+h} - P^{ask}_{i,t}}{P^{ask}_{i,t}} \tag{3.17}$$

for long positions (reversed for shorts). This provides a lower bound on achievable returns, incorporating the full impact of the bid-ask spread.

### 3.9.5 Model Stability

I test whether prediction accuracy degrades when the retraining frequency drops from **monthly to quarterly** (retraining every 3 months rather than every month). Stable models suggest that the underlying predictive relationships are persistent; degradation suggests regime sensitivity and the need for frequent retraining—an operationally important consideration.

### 3.9.6 Alternative Model Specifications

As additional robustness checks:
- **Feature subset analysis:** restrict features to (a) option-level only, (b) option + surface, (c) all features, to identify incremental value of each feature group
- **Alternative loss function:** train with Huber loss (robust to outliers) instead of MSE, and test whether this improves tail predictions
- **Ensemble of ML models:** average predictions across XGBoost, Random Forest, and MLP to test whether model combination reduces noise (motivated by the ensemble literature; arXiv, 2025)

---

## 3.10 Implementation Details

### 3.10.1 Software and Computational Environment

All analyses are implemented in **Python 3.10+** using the following libraries:
- `pandas`, `numpy` for data manipulation
- `scikit-learn` for elastic net and random forest
- `xgboost` for gradient boosted trees
- `pytorch` for neural network training
- `shap` for model interpretation
- `wrds` for data extraction from WRDS

### 3.10.2 Reproducibility

To ensure reproducibility, I:
1. Fix random seeds for all stochastic components (bootstrap sampling, weight initialization, data shuffling)
2. Document all hyperparameter search spaces and selection criteria
3. Store trained models and out-of-sample predictions for each rolling window
4. Report results for the *full set* of hyperparameter configurations, not only the best (to avoid selection bias in presentation)

### 3.10.3 Computational Requirements

The rolling-window design requires training approximately 141 models per model class (one per test month). For XGBoost and Random Forest, each training run takes approximately 1–5 minutes on a standard laptop (8-core CPU, 16GB RAM). For the MLP, GPU acceleration (single NVIDIA GPU) reduces training time from approximately 30 minutes to 2–5 minutes per window. Total computation is estimated at 40–60 hours across all models and robustness specifications—manageable within the thesis timeline.

---

## 3.11 Summary of Methodological Contributions

The methodology makes several specific contributions relative to the existing literature:

1. **Transparent, replicable pipeline:** Unlike Bali et al. (2023), who use a proprietary feature construction process, I provide complete documentation of all data filters, feature definitions, and model specifications, enabling replication with standard WRDS access.

2. **Multi-level transaction cost evaluation:** Most ML option pricing studies ignore transaction costs entirely (a gap noted in Section 10 of the literature review). This thesis provides break-even spread analysis and three-tier cost scenarios, connecting statistical predictability to economic implementability.

3. **Economic interpretation:** The combination of SHAP analysis, mispricing decomposition (Equation 3.15), and regime-conditional evaluation connects ML predictions to underlying economic mechanisms—volatility risk premia, limits to arbitrage, and demand-based pricing—rather than treating the models as black boxes.

4. **Comprehensive robustness:** The multi-dimensional robustness analysis (regime, moneyness, maturity, horizon, execution, retraining frequency) addresses the concern that ML results in finance are often fragile and sensitive to specification choices (a concern emphasized by Harvey, Liu, & Zhu, 2016, regarding data-snooping in factor discovery).

---

## References

Ackerer, D., Tagasovska, N., & Vatter, T. (2020). Deep smoothing of the implied volatility surface. *Advances in Neural Information Processing Systems (NeurIPS)*.

Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance, 61*(1), 259–299.

Bakshi, G., Cao, C., & Chen, Z. (1997). Empirical performance of alternative option pricing models. *Journal of Finance, 52*(5), 2003–2049.

Bali, T. G., Beckmeyer, H., Moerke, M., & Weigert, F. (2023). Option return predictability with machine learning and big data. *Review of Financial Studies, 36*(9), 3548–3600.

Bates, D. S. (2003). Empirical option pricing: A retrospection. *Journal of Financial Economics, 67*(3), 387–410.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample: Can anything beat the historical average? *Review of Financial Studies, 21*(4), 1509–1531.

Cao, J., & Han, B. (2013). Cross section of option returns and idiosyncratic stock volatility. *Journal of Financial Economics, 108*(1), 231–249.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics, 138*(1), 291–311.

De Fontnouvelle, P., Fishe, R. P. H., & Harris, J. H. (2003). The behavior of bid-ask spreads and volume in options markets. *Journal of Finance, 58*(6), 2437–2463.

Fan, L., & Sirignano, J. (2024). Machine learning methods for pricing financial derivatives. *arXiv preprint*.

François, P., Gauthier, G., Godin, F., & Mendoza, C. O. P. (2024). Enhancing deep hedging of options with implied volatility surface feedback information. *arXiv preprint*.

Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. *Global Derivatives and Risk Management*.

Goyal, A., & Saretto, A. (2009). Cross-section of option returns and stock volatility. *Journal of Financial Economics, 94*(2), 310–326.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies, 33*(5), 2223–2273.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *Review of Financial Studies, 29*(1), 5–68.

Horvath, B., Muguruza, A., & Tomas, M. (2021). Deep learning volatility. *Quantitative Finance, 21*(1).

Hutchinson, J. M., Lo, A. W., & Poggio, T. (1994). A nonparametric approach to pricing and hedging derivative securities via learning networks. *Review of Financial Studies, 7*(4), 851–889.

Ivașcu, C. F. (2021). Option pricing using machine learning. *Expert Systems with Applications, 163*, 113799.

Kelly, B. T., Kuznetsov, B., Malamud, S., et al. (2023). Deep learning from implied volatility surfaces. *SSRN*.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

Liu, S., Borovykh, A., Grzelak, L. A., & Oosterlee, C. W. (2024). Option pricing with physics-informed neural networks. *arXiv preprint*.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*.

Medvedev, N., & Wang, Z. (2022). Multistep forecast of the implied volatility surface using deep learning. *Journal of Futures Markets*.

Muravyev, D. (2016). Order flow and expected option returns. *Journal of Finance, 71*(2), 673–708.

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica, 55*(3), 703–708.

Petersen, M. A. (2009). Estimating standard errors in finance panel data sets: Comparing approaches. *Review of Financial Studies, 22*(1), 435–480.

Shleifer, A., & Vishny, R. W. (1997). The limits of arbitrage. *Journal of Finance, 52*(1), 35–55.

Yang, Y., Zheng, Y., & Hospedales, T. M. (2016). Gated neural networks for option pricing: Rationality by design. *Semantic Scholar*.

---

*Draft compiled: February 19, 2026. Subject to revision based on supervisor feedback.*
