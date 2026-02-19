# Literature Review: Machine Learning Methods in Options Pricing

**Author:** Maurits van Eck (ANR: 2062644)  
**Program:** MSc Finance, Tilburg University  
**Compiled:** February 19, 2026  
**Focus:** ML methods for option pricing, hedging, and implied volatility surface modeling

---

## 1. Introduction

The pricing of financial derivatives—particularly options—has been a central problem in quantitative finance since the seminal work of Black and Scholes (1973). While the Black–Scholes (BS) model provides an elegant closed-form solution under the assumptions of constant volatility and frictionless markets, decades of empirical evidence have demonstrated systematic deviations. Implied volatility varies across strike and maturity (the "smile" and "skew"), reflecting stochastic volatility, jump risk premia, and market microstructure effects (Heston, 1993; Bakshi, Cao, & Chen, 1997; Bates, 2003).

Machine learning (ML) has emerged as a powerful complement—and, in some contexts, a replacement—for traditional parametric pricing models. This literature review synthesizes recent advances across five interconnected domains: (i) classical ML approaches to option pricing, (ii) deep learning architectures for pricing and calibration, (iii) physics-informed and structure-aware neural networks, (iv) implied volatility surface modeling, and (v) ML-based hedging and trading strategies. The review draws on over 60 papers spanning 2016–2026 to provide a comprehensive assessment of the state of the art.

---

## 2. Foundations: From Black–Scholes to Machine Learning

### 2.1 Limitations of Parametric Models

The BS model's assumptions—constant volatility, log-normal returns, and continuous hedging—are violated in practice. Bakshi, Cao, and Chen (1997) compare stochastic volatility and jump-diffusion specifications, showing that no single structural model dominates across contracts and market states (*The Journal of Finance*). Bates (2003) surveys empirical option pricing and emphasizes that stochastic volatility and jump risk are essential for index options, motivating both richer parametric models and data-driven alternatives (*Journal of Financial Economics*).

### 2.2 Early Data-Driven Approaches

The idea of treating option pricing as a function-approximation problem dates to Hutchinson, Lo, and Poggio (1994), who demonstrated that learning networks can map option characteristics into prices and hedge ratios, improving pricing and hedging when true dynamics are complex (*Review of Financial Studies*). This nonparametric tradition laid the groundwork for modern ML applications.

### 2.3 ML in Empirical Finance

Gu, Kelly, and Xiu (2020) establish that modern ML methods—neural networks, tree ensembles, and regularized regressions—substantially improve out-of-sample prediction in asset pricing relative to linear benchmarks (*Review of Financial Studies*). Their framework, emphasizing genuine out-of-sample evaluation with rolling time splits and economic interpretation via feature importance, is directly transferable to option markets where the feature space is richer (Greeks, IV surface shape, term structure).

---

## 3. Classical ML Models for Option Pricing

### 3.1 Tree-Based Ensemble Methods

**XGBoost** has been applied to options pricing with notable success. Studies demonstrate XGBoost outperforming traditional stochastic models on both synthetic and real market data, with significant predictive accuracy improvements for derivative pricing and hedging strategies (arXiv, 2024). However, hyperparameter tuning remains a challenge for financial datasets with regime shifts.

**Random Forest** models have been applied to enhance predictive accuracy in market trend predictions and options pricing. Ensemble methods and model optimization improve long-term profitability in trading decisions, with random forests offering lower variance than single-tree methods (arXiv, 2023). Ivașcu (2021) provides a systematic comparison of ML methods—neural networks, random forests, and SVMs—for option pricing, demonstrating that ML models can match or exceed BS pricing accuracy on real market data (*Expert Systems with Applications*, 109 citations).

### 3.2 Integrated Multi-Model Frameworks

Recent work proposes integrated ML frameworks combining multiple model classes. Gonzalez and Wei (2025) compare SVMs, Random Forests, and LSTMs within a unified options pricing framework, demonstrating improved modeling accuracy under volatile market conditions (arXiv). The literature on ensemble methods for stock and cryptocurrency trading evaluates ensemble reinforcement learning strategies for high-dimensional financial markets, including portfolio management and option pricing (arXiv, 2025).

### 3.3 Comparative Studies

A key question is whether ML can systematically outperform traditional models. Recent investigations comparing neural networks, random forests, and CatBoost against BS and Heston models demonstrate cases where ML outperforms for both synthetic and real market option data, emphasizing accuracy and adaptability in volatile markets (arXiv, 2025). Fan and Sirignano (2024) provide a comprehensive recent review of ML approaches to derivative pricing, covering neural networks, deep learning, and physics-informed methods with focus on practical implementation (arXiv).

---

## 4. Deep Learning for Option Pricing and Calibration

### 4.1 Feed-Forward and Gated Neural Networks

The exploration of network architectures for option pricing has shown the efficacy of generalized highway network architectures for BS and Heston models, evaluated on training time and mean squared error (arXiv, 2023). Yang, Zheng, and Hospedales (2016) propose a **gated neural network** for option pricing that encodes economically motivated structure—"rationality by design"—rather than treating pricing as a pure black box (*Semantic Scholar*). Their architectural inductive biases reduce data requirements and improve out-of-sample stability.

Zheng, Yang, and Chen (2019) extend this with **gated deep neural networks** specifically tailored to learning the mapping from option inputs to implied volatility surfaces, reporting improved fit and generalization by incorporating inductive biases for surface dynamics (arXiv).

### 4.2 Recurrent and Sequence Models

**LSTM networks** have been extensively applied to financial time series and option pricing. Studies demonstrate superior predictive performance over traditional statistical methodologies, with ensemble LSTM models capturing temporal dependencies in volatile financial data (arXiv, 2022). Advanced LSTM frameworks employ comprehensive strategies to mitigate overfitting and improve generalization for stock volatility analysis, directly related to option pricing inputs (arXiv, 2025).

Na and Wan (2023) introduce **deep recurrent architectures** to handle path-dependence and high-dimensional state representations in American option pricing, providing joint pricing-and-hedging evaluation with emphasis on out-of-sample performance (*Quantitative Finance*).

### 4.3 Transformer and Attention-Based Architectures

**Attention mechanisms** have been applied to capture temporal dependencies in financial time series, outperforming LSTM and GRU baselines on stock return prediction tasks while providing improved interpretability through attention weights (Sun, Wang, & An, 2023, arXiv). The **Informer architecture**—a Transformer variant tailored for long-sequence forecasting—has been applied to option pricing, demonstrating that attention-based sequence models capture temporal dependencies relevant for pricing dynamics (arXiv, 2025). **Causal transformer models** have also shown improved predictive accuracy for sequential financial data compared to baseline models (arXiv, 2025).

### 4.4 Convolutional and Hybrid Architectures

**CNNs** have been applied to financial time series prediction by integrating convolutional layers with recurrent networks for hybrid modeling across stock and options data (arXiv, 2022). The **DeepLOB** model links limit order book microstructural assessments with mid-price forecasting using deep learning, demonstrating superior forecasting performance (arXiv, 2024).

### 4.5 Deep Learning for Calibration and Fast Pricing

**Horvath, Muguruza, and Tomas (2021)** use neural networks as fast surrogates for pricing and calibration across stochastic and rough volatility model families, representing implied volatilities on grids to enable effective learning (*Quantitative Finance*). De Spiegeleer, Madan, Reyners, and Schoutens (2018) provide a comprehensive framework for applying ML to derivative pricing, hedging, and model calibration, demonstrating significant speed improvements over traditional numerical methods while maintaining accuracy (*Quantitative Finance*, 237 citations).

**DeepSVM** (2025) replaces numerical pricing with neural operators that generalize across parameter regimes, enabling near real-time option pricing and model calibration using physics-informed deep operator networks (arXiv).

### 4.6 American Option Pricing

American-style options pose particular challenges due to early exercise. ML algorithms combined with Monte Carlo simulations show effectiveness in pricing American options (arXiv, 2024). Becker, Cheridito, and Jentzen (2020) use deep learning to approximate continuation values and hedging strategies for American-style derivatives, showing strong scalability to higher dimensions relative to classical LSM approaches (*Journal of Risk and Financial Management*). Anderson and Ulrych (2023) propose neural architectures to accelerate American option valuation versus standard LSM/MC implementations, reporting speed/accuracy trade-offs relevant for practical deployment (*SSRN*).

Chen and Wan (2021) solve high-dimensional American option pricing/hedging via **deep BSDE methods**, demonstrating scalability and accuracy improvements where grid/PDE methods break down (*Quantitative Finance*). Guo, Langrené, and Wu (2023) use neural networks to compute both lower and upper bounds for American option prices without nested Monte Carlo, producing hedging strategies alongside bounds (arXiv). Sun, Huang, Yang, and Zhang (2024) use model-informed neural networks with jump-diffusion structure combined with transfer learning to improve American option pricing under data scarcity (*Semantic Scholar*).

### 4.7 Novel Architectures

**KANHedge** (2026) replaces standard MLP components in deep-BSDE hedgers with **Kolmogorov–Arnold Networks** (KANs), reporting improved hedging performance for high-dimensional options versus MLP-based baselines (arXiv). This represents the frontier of architectural innovation for derivative pricing.

---

## 5. Physics-Informed and Structure-Aware Neural Networks

### 5.1 Physics-Informed Neural Networks (PINNs)

A significant recent development is the embedding of financial model constraints directly into neural network training. **Liu, Borovykh, Grzelak, and Oosterlee (2024)** embed Black–Scholes PDE constraints into neural network training for option pricing, achieving higher accuracy with less training data compared to standard deep learning approaches and demonstrating generalization across European and American options (arXiv).

**Barrier option pricing** using extended PINNs (ePINNs) shows 69% improvement over standard PINNs in mean absolute error on CSI 300ETF options (2019–2025), demonstrating superior accuracy for exotic derivatives (*Expert Systems with Applications*, 2025).

**Hoshisashi, Phelan, and Barucca (2024)** propose *Whack-a-mole Learning* to balance multiple calibration objectives in deep IV surface fitting, enforcing PDE structure and no-arbitrage inequality constraints while fitting sparse option quote grids (*Semantic Scholar*).

### 5.2 BSDE-Based Approaches

Deep learning-based BSDE solvers have been applied to the Libor Market Model for Bermudan swaption pricing and hedging, demonstrating feasibility of neural BSDE solvers for high-dimensional derivative pricing problems (arXiv, 2018). This extends ML pricing beyond equity options to interest-rate derivatives, supporting the broader claim that deep learning can tackle high-dimensional pricing/hedging where classical PDE/MC methods struggle.

### 5.3 Neural Stochastic Differential Equations

**Arbitrage-Free Neural-SDE Market Models** (2021) combine flexible function approximation with continuous-time dynamics satisfying no-arbitrage conditions, enabling more realistic path-wise behavior and a route to model implied-vol dynamics with financial consistency constraints (*Semantic Scholar*).

### 5.4 Signature-Based Methods

**Deep signature approaches** (2025) apply signature-based deep learning to represent non-Markovian path information for option pricing, targeting settings where latent volatility processes have memory—exactly where classical Markovian models struggle (arXiv).

### 5.5 Constrained Deep Learning

**Baradel (2025)** develops a single-network approach where the network represents the option price and its gradient defines the hedging strategy, trained to satisfy a self-financing constraint. This addresses practical issues from non-smooth payoffs and proposes constrained architectures embedding terminal payoff conditions, demonstrating improved P&L distributions in realistic incomplete markets (arXiv).

---

## 6. Implied Volatility Surface Modeling

### 6.1 Deep Smoothing and Interpolation

**Ackerer, Tagasovska, and Vatter (2020)** propose deep-learning-based approaches to smooth implied volatility surfaces while respecting structural constraints, demonstrating improved stability versus ad-hoc smoothing for downstream pricing and risk (*NeurIPS*). **Operator Deep Smoothing** (Wiedemann, Jacquier, & Gonon, 2024) introduces a neural-operator approach to map sparse/noisy option observations to smooth IV surfaces, enforcing no-arbitrage constraints with robustness to subsampling and outliers (arXiv).

### 6.2 Surface Forecasting

**Medvedev and Wang (2022)** develop deep learning models for multi-step-ahead forecasting of the implied volatility surface, evaluating forecasting accuracy across horizons for trading and risk management (*Journal of Futures Markets*). **Kelly, Kuznetsov, Malamud et al. (2023)** treat IV surfaces as informative state variables and apply deep learning to extract predictive structure, showing that nonlinear representations improve forecasting relative to simpler summaries (*SSRN*).

### 6.3 Generative Models for IV Surfaces

Several generative modeling approaches have been proposed:

- **VAE-based generation** (Wang, Liu, & Vuik, 2025): Synthesizes IV surfaces with explicit control over interpretable shape features (skew/smile, term-structure patterns) in a low-dimensional latent space (arXiv).
- **Arbitrage-Free VAE generation** (Ning, Jaimungal, Zhang, & Bergeron, 2021): Uses VAE-style modeling to learn distributions over IV surfaces respecting no-arbitrage constraints (*SIAM Journal on Financial Mathematics*).
- **Diffusion models** (Jin & Agarwal, 2025): Introduces conditional diffusion (DDPM-style) generative models to forecast entire IV surfaces, targeting improved stability versus GANs while generating arbitrage-free surfaces (arXiv).
- **Simulation of arbitrage-free surfaces** (Cont & Vuletić, 2023): Develops a simulation framework enforcing static arbitrage constraints for risk management and stress testing (*SSRN*).

### 6.4 Arbitrage-Free Surface Construction

**SANOS** (Buehler, Horvath, Kratsios, Limmer, & Saqur, 2026) proposes numerically efficient non-parametric construction of option price surfaces that are smooth and strictly static-arbitrage-free across strikes and maturities—a critical building block for any ML model consuming implied volatilities (arXiv). **Zhang, Li, and Zhang (2021)** propose a two-step framework separating unconstrained statistical/ML prediction from no-arbitrage post-processing, motivating evaluation beyond point error to include frequency and magnitude of arbitrage violations (*Quantitative Finance*).

### 6.5 Novel Architectures for Surface Modeling

- **Meta-Learning Neural Process** with SABR-induced priors (2025): Frames IV surface construction as a meta-learning task, learning across multiple trading days to improve surface predictions while enforcing financial structure (arXiv).
- **Hexagon-Net** (Liang, 2025): Introduces a graph attention network architecture for IV surface forecasting using heterogeneous cross-view alignment across strikes and maturities (*Semantic Scholar*).
- **FuNVol** (Choudhary, Jaimungal, & Bergeron, 2023): Builds multi-asset IV dynamics simulators combining functional PCA factors and neural SDEs for realistic cross-asset IV co-movements (*Semantic Scholar*).

---

## 7. ML-Based Hedging Strategies

### 7.1 Deep Hedging

The **deep hedging** framework of **Buehler, Gonon, Teichmann, and Wood (2019)** frames hedging as a sequential decision problem solvable with deep neural networks, optimizing hedging strategies under market frictions and constraints. This canonical reference connects to utility/risk-based loss functions and establishes the language for ML-based derivative risk management (*Quantitative Finance*).

**Ruf and Wang (2020)** study neural networks for option pricing and hedging in a comprehensive literature review, showing that networks can approximate pricing functions and produce hedge ratios with good performance (*Quantitative Finance*).

### 7.2 Reinforcement Learning for Hedging

The application of RL to hedging has accelerated significantly:

- **Cao, Chen, Hull, and Poulos (2021)**: Apply deep RL to learn optimal hedging strategies for derivative portfolios incorporating transaction costs and market frictions, demonstrating RL superiority over delta hedging under realistic conditions (arXiv, 162 citations).
- **Du, Jin, Kolm, Ritter, Wang, and Zhang (2020)**: Formulate option replication/hedging with discrete rebalancing and nonlinear transaction costs as a deep RL control problem (*Journal of Financial Data Science*).
- **Vittori, Trapletti, and Restelli (2020)**: Apply risk-averse RL to option hedging, emphasizing tail-risk control and connecting RL objective design to hedging outcomes (*AI in Finance*).
- **François, Gauthier, Godin, and Mendoza (2024)**: Learn dynamic hedging policies with deep policy-gradient RL, incorporating forward-looking IV surface information as state features, reporting outperformance versus practitioner delta hedging with transaction costs (arXiv).
- **Lucius, Koch, Starling, Zhu, Urena, and Hu (2025)**: Build a leak-free RL environment for hedging equity index option exposures with transaction costs and position limits, using IV term structure and macro variables as state (arXiv).
- **Gamma and vega hedging using distributional RL** (2022): Optimizes the full distribution of hedging outcomes rather than only expected P&L, reducing tail losses relative to heuristic rules (*Semantic Scholar*).
- **Pan (2025)**: Combines quantile regression with curriculum learning for risk-sensitive option hedging, fusing historical data to improve out-of-sample performance (*Semantic Scholar*).

### 7.3 Model-Free and Practical Deep Hedging

**Brugière (2025)** proposes model-free deep hedging that works with limited training data and explicitly incorporates transaction costs (*Semantic Scholar*). Additional work focuses on computational and robustness improvements for practical deployment, emphasizing stability and speed required for extensive backtests (arXiv, 2025).

---

## 8. Trading Strategy Applications

### 8.1 Option Return Predictability

**Bali, Beckmeyer, Moerke, and Weigert (2023)** demonstrate that ML with large information sets can predict the cross-section of option returns, generating economically significant long–short profits (*Review of Financial Studies*). This is a central reference for ML-based option trading strategies. Huang, Wang, and Xiao (2025) study option return predictability using ML-based multifactor signals in Chinese markets, linking predictive signals to mispricing dynamics (*Journal of Futures Markets*).

### 8.2 End-to-End Trading Approaches

**Tan (2024)** presents an end-to-end deep learning approach mapping option market data directly to trading signals without specifying a structural pricing model, backtested on a decade of equity options (S&P 100) with improvements in risk-adjusted performance. Turnover regularization is shown to materially matter under high transaction costs (arXiv).

### 8.3 Reinforcement Learning for Trading

**Lim and Zohren (2019)** introduce deep RL algorithms for trading continuous futures contracts, exploring both discrete and continuous action spaces with volatility scaling (arXiv). **Liu, Yang, Chen et al. (2020)** provide the FinRL library—a reproducible DRL framework with market environments and benchmark agents for trading (*arXiv*). **Massahi and Mahootchi (2024)** design a deep Q-learning trading agent for commodity futures, evaluating against baselines with reward shaping and risk controls (*Expert Systems with Applications*). Multi-agent RL for high-frequency trading (Wei, Wang, Pu, & Wu, 2024) highlights interaction effects and stability issues absent in single-agent formulations (*Semantic Scholar*).

### 8.4 Cryptocurrency Options

**Brini and Lenz (2024)** apply supervised ML regression to option pricing in a highly volatile crypto-derivatives setting, highlighting feature and volatility handling as key drivers of pricing performance (*Semantic Scholar*). This provides an out-of-sample stress test for robustness and generalization of ML option pricing models.

---

## 9. Auxiliary ML Methods in Finance

### 9.1 Generative Models

**GANs in finance** (2021) provide a comprehensive overview of applications to financial data, testing established GAN architectures on time series and demonstrating practical applications for synthetic data generation in algorithmic trading and risk management (arXiv).

### 9.2 Portfolio Optimization

Neural network-based sensitivity approximations have been proposed for portfolio optimization, using hierarchical clustering of sensitivity matrix dynamics for diversification and risk management (arXiv, 2022).

### 9.3 Alternative Pricing Approaches

**Fuzzy-random option pricing** (Andrés-Sánchez, 2025) provides an alternative methodology beyond traditional BS/Heston frameworks, offering a theoretical foundation for uncertainty modeling applied to European interbank market caplet options (*Axioms, MDPI*).

---

## 10. Key Gaps and Research Opportunities

Based on this comprehensive review, several research gaps emerge:

1. **Joint pricing and hedging evaluation**: Most studies evaluate pricing accuracy or hedging performance separately. Few provide joint evaluation under a consistent out-of-sample design (a gap noted in the thesis proposal by Van Eck, 2026).

2. **Transaction cost realism**: Many ML pricing improvements may be illusory once transaction costs, turnover, and rebalancing frequency are accounted for. Explicit cost-adjusted evaluation remains rare in empirical studies.

3. **Robustness across regimes**: Limited evidence exists on whether ML pricing/hedging gains persist across volatility regimes (pre-2020 calm markets vs. COVID crash vs. post-COVID rate-hiking cycle).

4. **Interpretability**: While SHAP values and feature importance are increasingly used, the economic mechanism connecting ML signals to risk premia versus mispricing remains underexplored in option markets.

5. **Scalability to production**: Speed/accuracy trade-offs for real-time deployment in production trading systems require further investigation, particularly for deep learning approaches.

---

## 11. Summary Statistics

| Category | Paper Count | Key Methods |
|---|---:|---|
| Classical ML (RF, XGBoost, SVM) | 8 | Random Forest, XGBoost, CatBoost, SVM, ensemble methods |
| Deep learning (MLP, LSTM, CNN) | 12 | Feed-forward networks, LSTM, CNN, gated architectures |
| Transformer/Attention | 4 | Informer, causal transformers, attention mechanisms |
| Physics-informed (PINN, BSDE) | 8 | PINNs, deep BSDE solvers, neural SDEs, KANs |
| IV surface modeling | 12 | VAE, diffusion models, deep smoothing, graph networks |
| RL-based hedging | 10 | Deep hedging, policy gradient, Q-learning, distributional RL |
| Trading strategies | 6 | Option return prediction, end-to-end DL, multi-agent RL |
| Surveys/foundations | 5 | Comprehensive reviews, empirical asset pricing |
| **Total unique papers reviewed** | **~65** | |

---

## 12. References

Ackerer, D., Tagasovska, N., & Vatter, T. (2020). Deep smoothing of the implied volatility surface. *Advances in Neural Information Processing Systems (NeurIPS)*.

Anderson, D., & Ulrych, U. (2023). Accelerated American option pricing with deep neural networks. *Quantitative Finance and Economics*.

Andrés-Sánchez, J. (2025). A systematic overview of fuzzy-random option pricing in discrete time. *Axioms, 14*(1), 52.

Bakshi, G., Cao, C., & Chen, Z. (1997). Empirical performance of alternative option pricing models. *The Journal of Finance, 52*(5), 2003–2049.

Bali, T. G., Beckmeyer, H., Moerke, M., & Weigert, F. (2023). Option return predictability with machine learning and big data. *Review of Financial Studies, 36*(9), 3548–3600.

Baradel, N. (2025). Constrained deep learning for pricing and hedging European options in incomplete markets. *arXiv preprint*.

Bates, D. S. (2003). Empirical option pricing: A retrospection. *Journal of Financial Economics, 67*(3), 387–410.

Becker, S., Cheridito, P., & Jentzen, A. (2020). Pricing and hedging American-style options with deep learning. *Journal of Risk and Financial Management, 13*(7), 158.

Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy, 81*(3), 637–654.

Brini, A., & Lenz, J. (2024). Pricing cryptocurrency options with machine learning regression. *Semantic Scholar*.

Brugière, P. (2025). Model-free deep hedging with transaction costs and light data requirements. *arXiv preprint*.

Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. *Quantitative Finance, 19*(8), 1271–1291.

Buehler, H., Horvath, B., Kratsios, A., Limmer, Y., & Saqur, R. (2026). SANOS: Smooth strictly arbitrage-free non-parametric option surfaces. *arXiv preprint*.

Cao, J., Chen, J., Hull, J., & Poulos, Z. (2021). Deep hedging of derivatives using reinforcement learning. *arXiv preprint*.

Chen, Y., & Wan, J. W. L. (2021). Deep neural network framework based on backward stochastic differential equations for pricing and hedging American options in high dimensions. *Quantitative Finance, 21*(1).

Choudhary, V., Jaimungal, S., & Bergeron, M. (2023). FuNVol: A multi-asset implied volatility market simulator. *Semantic Scholar*.

Cont, R., & Vuletić, M. (2023). Simulation of arbitrage-free implied volatility surfaces. *SSRN*.

De Spiegeleer, J., Madan, D. B., Reyners, S., & Schoutens, W. (2018). Machine learning for quantitative finance. *Quantitative Finance, 18*(10).

Ding, L., Lu, E., & Cheung, K. (2025). Deep learning option pricing with market implied volatility surfaces. *arXiv preprint*.

Du, J., Jin, M., Kolm, P., Ritter, G., Wang, Y., & Zhang, B. (2020). Deep reinforcement learning for option replication and hedging. *Journal of Financial Data Science*.

Fan, L., & Sirignano, J. (2024). Machine learning methods for pricing financial derivatives. *arXiv preprint*.

François, P., Gauthier, G., Godin, F., & Mendoza, C. O. P. (2024). Enhancing deep hedging of options with implied volatility surface feedback information. *arXiv preprint*.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies, 33*(5), 2223–2273.

Guo, I., Langrené, N., & Wu, J. (2023). Simultaneous upper and lower bounds of American-style option prices with hedging via neural networks. *arXiv preprint*.

Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies, 6*(2), 327–343.

Horvath, B., Muguruza, A., & Tomas, M. (2021). Deep learning volatility. *Quantitative Finance, 21*(1).

Hoshisashi, K., Phelan, C. E., & Barucca, P. (2024). Whack-a-mole learning: Physics-informed deep calibration for implied volatility surface. *Semantic Scholar*.

Hutchinson, J. M., Lo, A. W., & Poggio, T. (1994). A nonparametric approach to pricing and hedging derivative securities via learning networks. *Review of Financial Studies, 7*(4), 851–889.

Ivașcu, C. F. (2021). Option pricing using machine learning. *Expert Systems with Applications, 163*, 113799.

Jin, C., & Agarwal, A. (2025). Forecasting implied volatility surface with generative diffusion models. *arXiv preprint*.

Kelly, B. T., Kuznetsov, B., Malamud, S., et al. (2023). Deep learning from implied volatility surfaces. *SSRN*.

Liang, K. (2025). Hexagon-Net: Heterogeneous cross-view aligned graph attention networks for implied volatility surface prediction. *Semantic Scholar*.

Lim, B., & Zohren, S. (2019). Deep reinforcement learning for trading. *arXiv preprint*.

Liu, S., Borovykh, A., Grzelak, L. A., & Oosterlee, C. W. (2024). Option pricing with physics-informed neural networks. *arXiv preprint*.

Liu, X.-Y., Yang, H., Chen, Q., et al. (2020). FinRL: A deep reinforcement learning library for automated stock trading. *arXiv preprint*.

Lucius, T., Koch, C., Starling, J., Zhu, J., Urena, M., & Hu, C. (2025). Deep hedging with reinforcement learning: A practical framework. *arXiv preprint*.

Massahi, M., & Mahootchi, M. (2024). A deep Q-learning based algorithmic trading system. *Expert Systems with Applications*.

Medvedev, N., & Wang, Z. (2022). Multistep forecast of the implied volatility surface using deep learning. *Journal of Futures Markets*.

Na, A. S., & Wan, J. W. L. (2023). Efficient pricing and hedging of high-dimensional American options using deep recurrent networks. *Quantitative Finance*.

Ning, B., Jaimungal, S., Zhang, X., & Bergeron, M. (2021). Arbitrage-free implied volatility surface generation with variational autoencoders. *SIAM Journal on Financial Mathematics*.

Pan, Q. (2025). Reinforcement learning for option hedging using quantile regression and curriculum learning. *Semantic Scholar*.

Ruf, J., & Wang, W. (2020). Neural networks for option pricing and hedging: A literature review. *Quantitative Finance, 20*(11).

Sun, Q., Huang, H., Yang, X., & Zhang, Y. (2024). Jump diffusion-informed neural networks with transfer learning for American option pricing. *Semantic Scholar*.

Sun, S., Wang, R., & An, B. (2023). Attention-based neural networks for financial market prediction. *arXiv preprint*.

Tan, W. L. (2024). Deep learning for options trading: An end-to-end approach. *arXiv preprint*.

Vittori, E., Trapletti, M., & Restelli, M. (2020). Option hedging with risk averse reinforcement learning. *AI in Finance*.

Wang, J., Liu, S., & Vuik, C. (2025). Controllable generation of implied volatility surfaces with variational autoencoders. *arXiv preprint*.

Wei, M., Wang, S., Pu, Y., & Wu, J. (2024). Multi-agent reinforcement learning for high-frequency trading strategy optimization. *Semantic Scholar*.

Wiedemann, R., Jacquier, A., & Gonon, L. (2024). Operator deep smoothing for implied volatility. *arXiv preprint*.

Yang, Y., Zheng, Y., & Hospedales, T. M. (2016). Gated neural networks for option pricing: Rationality by design. *Semantic Scholar*.

Zhang, W.-Y., Li, L., & Zhang, G. (2021). A two-step framework for arbitrage-free prediction of the implied volatility surface. *Quantitative Finance*.

Zheng, Y., Yang, Y., & Chen, B. (2019). Gated deep neural networks for implied volatility surfaces. *arXiv preprint*.

---

*Compiled from 65+ papers in the Master Thesis research database. Last updated: February 19, 2026.*
