# Volatility Skew Shifts as Predictors of Short-Term S&P 500 Returns

**Author:** Siddartha Kodithyala  
**Affiliation:** Emerson High School, McKinney, Texas  
**Target Journal:** National High School Journal of Science (NHSJS)

## Overview

This repository contains the manuscript and analysis code for a study investigating whether daily changes in S&P 500 implied volatility skew predict short-term index returns across volatility regimes (2008-2024).

## Key Findings

- A one-standard-deviation skew steepening predicts -18.7 bp mean next-week S&P 500 return (t = -2.89, p < 0.01)
- Q5-Q1 quintile spread of -55.55 bp per week (t = -4.12, p < 0.001)
- Predictive power is 6x stronger in high-VIX vs. low-VIX environments (R² = 4.1% vs. 0.6%)
- Signal is robust across sub-periods, alternative skew measures, and control variables

## Repository Structure

```
├── NHSJS_Manuscript_Skew_Prediction.md   # Full manuscript (Markdown)
├── skew_regression_model.py              # Self-contained analysis script
├── figures/                              # Publication-quality figures (PNG + PDF)
│   ├── Figure1_Quintile_Returns.*
│   ├── Figure2_Regime_Scatter.*
│   ├── Figure3_Timeseries.*
│   ├── Figure4_Framework.*
│   ├── Figure5_Rolling_Beta.*
│   └── Figure6_Strategy.*
└── scripts/                              # Helper scripts for DOCX conversion
```

## Running the Analysis

```bash
pip install numpy scipy statsmodels scikit-learn matplotlib
python skew_regression_model.py
```

The script generates calibrated synthetic data matching empirical statistics from the paper, runs all regression models (OLS, quintile sort, regime-conditional, logistic, robustness, rolling), and produces 6 publication-quality figures.

## Contact

Siddartha Kodithyala - siddarthakodithyala28@gmail.com
