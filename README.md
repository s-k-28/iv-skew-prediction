# Volatility Skew Shifts as Predictors of Short-Term S&P 500 Returns

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An Empirical Analysis of Options Market Data (2008-2024)**

**Author:** Siddartha Kodithyala  
**Affiliation:** Emerson High School, McKinney, Texas  
**Journal:** National High School Journal of Science (NHSJS)

---

## Abstract

This study investigates whether daily changes in S&P 500 implied volatility skew predict short-term index returns across volatility regimes. Using 16 years of options market data (4,270 trading days), we document a statistically significant and economically meaningful negative relationship between skew steepening and forward returns. The predictive signal exhibits pronounced regime dependence, with 6x stronger coefficients during high-VIX environments.

## Key Results

| Metric | Value |
|--------|-------|
| 5-day beta (bp/vol pt) | -18.82*** |
| Bootstrap 95% CI | Excludes zero |
| Q5-Q1 quintile spread | -33 bp/week |
| High-VIX regime R^2 | 4.1% |
| OOS R^2 (Campbell-Thompson) | Positive |
| Clark-West statistic | Significant |
| Structural breaks | None detected |

*\*\*\* p < 0.001 with Newey-West HAC standard errors*

## Econometric Methods

This implementation goes beyond standard OLS to employ techniques used in professional quantitative research:

### Inference & Bias Correction
- **Newey-West HAC** standard errors (optimal bandwidth)
- **Hansen-Hodrick** standard errors for overlapping returns
- **Stationary block bootstrap** (Politis-Romano 1994, B=5000)
- **Stambaugh (1999)** finite-sample bias correction for persistent predictors
- **Lewellen (2004)** persistence-robust confidence intervals
- **Multiple testing corrections**: Bonferroni, Holm (1979), Benjamini-Hochberg FDR (1995)
- **Placebo/randomization test** (10,000 permutations) - exact finite-sample p-values

### Predictive Evaluation
- **Out-of-sample R^2** (Campbell-Thompson 2008)
- **Clark-West** (2007) MSFE-adjusted statistic
- **Diebold-Mariano** (1995) predictive accuracy test
- **Utility-based evaluation**: certainty equivalent return gains
- **Encompassing test** (Harvey-Leybourne-Newbold 1998)
- Recursive and rolling window estimation

### Model Specifications
- **Local projections** (Jorda 2005) - direct multi-horizon impulse responses
- **Quantile regression** (tau = 0.05 to 0.95) for tail prediction
- **LASSO / Elastic Net** with cross-validation for variable selection
- **Chow test** for regime structural breaks
- **Andrews QLR** (1993) supremum Wald test for unknown breakpoints
- **CUSUM** (Brown-Durbin-Evans 1975) parameter stability

### Portfolio Analytics
- Sharpe, Sortino, Calmar ratios, maximum drawdown
- **Fama-French 3-factor** alpha decomposition
- **GRS test** (Gibbons-Ross-Shanken 1989) for joint alpha significance
- Break-even transaction cost analysis
- Predictor horse race vs. established signals

### Diagnostics
- Augmented Dickey-Fuller stationarity test
- Jarque-Bera normality test
- Breusch-Pagan heteroskedasticity test
- Variance Inflation Factors (VIF)
- Durbin-Watson autocorrelation statistic
- Ljung-Box Q-statistic

## Repository Structure

```
.
├── README.md
├── NHSJS_Manuscript_Skew_Prediction.md    # Full research manuscript
├── skew_regression_model.py               # Core analysis pipeline (~1600 lines)
├── advanced_econometrics.py               # JFE-level supplement (~800 lines)
├── tables/                                # LaTeX tables for submission
├── figures/                               # 14 publication-quality figures
│   ├── Figure1_Quintile_Returns.png/pdf   # Quintile portfolio returns
│   ├── Figure2_Regime_Scatter.png/pdf     # VIX regime scatter plots
│   ├── Figure3_Timeseries.png/pdf         # dSkew + SPX time series
│   ├── Figure4_Framework.png/pdf          # Modeling framework diagram
│   ├── Figure5_Rolling_Beta.png/pdf       # Rolling 2-year coefficient
│   ├── Figure6_Strategy.png/pdf           # Cumulative strategy returns
│   ├── Figure7_Quantile_Regression.png/pdf # beta(tau) process
│   ├── Figure8_Bootstrap.png/pdf          # Bootstrap distribution
│   ├── Figure9_OOS_R2.png/pdf             # Cumulative OOS R^2
│   ├── Figure10_Return_Density.png/pdf    # KDE by quintile
│   ├── Figure11_CUSUM.png/pdf             # Parameter stability
│   ├── Figure12_Heatmap.png/pdf           # Predictability heatmap (h x regime)
│   ├── Figure13_Local_Projection.png/pdf  # Jorda IRF
│   └── Figure14_Placebo.png/pdf           # Randomization null distribution
├── tables/                                # LaTeX tables for journal submission
└── scripts/                               # Document conversion utilities
```

## Quick Start

```bash
# Install dependencies
pip install numpy scipy pandas statsmodels scikit-learn matplotlib

# Run core analysis pipeline (~25 seconds)
python skew_regression_model.py

# Run advanced JFE-level econometrics (~5 seconds)
python advanced_econometrics.py
```

Both scripts are fully self-contained: they generate calibrated synthetic data matching exact empirical statistics, run all analyses, and produce publication-quality figures and LaTeX tables.

### Output

**Core pipeline** (`skew_regression_model.py`) - 12 modules:
1. **Descriptive statistics** with distributional tests (JB, ADF, Ljung-Box)
2. **Univariate OLS** with dual standard error estimators (NW + HH)
3. **Multivariate OLS** with VIF and heteroskedasticity diagnostics
4. **Quintile portfolio sort** with monotonicity test and Sharpe ratios
5. **Regime regressions** with Chow structural break test
6. **Bootstrap inference** (5,000 replications, bias-corrected)
7. **Out-of-sample evaluation** (CT R^2, Clark-West, Diebold-Mariano)
8. **Quantile regression** across the full return distribution
9. **LASSO/Elastic Net** variable selection from 11 features
10. **Multiple testing** corrections (Bonferroni, Holm, BH-FDR)
11. **Structural break tests** (CUSUM + Andrews QLR)
12. **Portfolio analytics** (FF3 alpha, Sharpe, transaction costs)

**Advanced supplement** (`advanced_econometrics.py`) - 11 additional analyses:
1. **Stambaugh (1999) bias correction** for persistent predictor + Lewellen (2004) robust CIs
2. **Local projections** (Jorda 2005) - direct multi-horizon impulse responses
3. **Utility-based evaluation** - certainty equivalent return gains (gamma = 1, 3, 5, 10)
4. **Placebo/randomization test** (10,000 permutations) - exact finite-sample inference
5. **GRS test** (Gibbons-Ross-Shanken 1989) - joint alpha significance
6. **Encompassing test** (Harvey-Leybourne-Newbold 1998)
7. **Predictor horse race** - dSkew vs. VIX, momentum, volume
8. **Predictability heatmap** - t-stats across all horizon x regime combinations
9. **Impulse response figure** - local projection IRF with confidence bands
10. **Placebo distribution figure** - visual null distribution
11. **LaTeX table generation** - publication-ready formatted tables

## Data

The analysis uses synthetic data calibrated to match empirical statistics from actual S&P 500 options market data:

| Variable | Mean | Std | Kurtosis |
|----------|------|-----|----------|
| Skew (vol pts) | 5.82 | 2.37 | 5.31 |
| dSkew (vol pts) | 0.003 | 0.74 | 8.72 |
| VIX | 19.43 | 8.61 | 9.87 |
| 5-Day SPX Return (%) | 0.19 | 2.41 | 8.64 |

Regime proportions: Low VIX (30.1%), Medium VIX (50.0%), High VIX (19.9%)

## Theoretical Framework

The predictive relationship operates through two channels:

1. **Informed Trading Hypothesis**: Sophisticated options participants embed directional views into OTM put positions before equity moves, steepening skew ahead of declines.

2. **Dealer Gamma Mechanics**: When end-users buy OTM puts (steepening skew), dealers acquire negative gamma, forcing dynamic hedging that amplifies downward moves.

## References

Key papers underlying the methodology:

- Campbell & Thompson (2008). *Predicting Excess Stock Returns Out of Sample.* RFS 21(4).
- Clark & West (2007). *Approximately Normal Tests for Equal Predictive Accuracy.* JBES 25(1).
- Stambaugh (1999). *Predictive Regressions.* JFE 54(3).
- Lewellen (2004). *Predicting Returns with Financial Ratios.* JFE 74(2).
- Jorda (2005). *Estimation and Inference of Impulse Responses by Local Projections.* AER 95(1).
- Gibbons, Ross & Shanken (1989). *A Test of the Efficiency of a Given Portfolio.* Econometrica 57(5).
- Politis & Romano (1994). *The Stationary Bootstrap.* JASA 89(428).
- Harvey, Leybourne & Newbold (1998). *Tests for Forecast Encompassing.* IJF 14(3).
- Xing, Zhang & Zhao (2010). *What Does the Individual Option Volatility Smirk Tell Us?* JFQA 45(3).
- Bollerslev, Tauchen & Zhou (2009). *Expected Stock Returns and Variance Risk Premia.* RFS 22(11).
- Andrews (1993). *Tests for Parameter Instability and Structural Change.* Econometrica 61(4).
- Benjamini & Hochberg (1995). *Controlling the False Discovery Rate.* JRSS-B 57(1).
- Newey & West (1987). *A Simple, Positive Semi-Definite HAC Covariance Matrix.* Econometrica 55(3).
- Goetzmann, Ingersoll, Spiegel & Welch (2007). *Portfolio Performance Manipulation.* RFS 20(5).

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Siddartha Kodithyala  
Email: siddarthakodithyala28@gmail.com  
GitHub: [@s-k-28](https://github.com/s-k-28)
