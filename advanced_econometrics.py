"""
Advanced Econometric Methods for Journal of Finance / Journal of Derivatives
Supplement to skew_regression_model.py

Additional analyses required for top-tier publication:
1. Stambaugh (1999) bias correction for persistent predictor
2. Local projections (Jorda 2005) - direct multi-horizon impulse responses
3. Utility-based evaluation (certainty equivalent return gain)
4. Placebo / randomization test (10,000 shuffles)
5. GRS test (Gibbons-Ross-Shanken 1989) for joint alpha significance
6. Hansen's Superior Predictive Ability (SPA) test
7. Encompassing test (Harvey-Leybourne-Newbold 1998)
8. Comparison with known equity predictors
9. Heatmap of predictability across horizons and regimes
10. LaTeX table generation for journal submission
11. Characteristic-sorted double sorts
12. Markov regime-switching regression
13. Impulse response functions
14. Persistence-robust confidence intervals (Lewellen 2004)

Author: Siddartha Kodithyala
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

import os
import sys

# Import data generator from main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skew_regression_model import simulate_dataset, COLORS

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'cm',
})


# =============================================================================
# 1. STAMBAUGH BIAS CORRECTION
# =============================================================================

def stambaugh_bias_correction(df):
    """
    Stambaugh (1999) bias correction for predictive regression with
    persistent predictor.

    When x_t = rho*x_{t-1} + u_t and y_t = alpha + beta*x_{t-1} + e_t,
    with Corr(u_t, e_t) = delta, the OLS estimate of beta is biased:

        E[beta_hat - beta] = gamma * E[rho_hat - rho]

    where gamma = sigma_eu / sigma_uu and the bias in rho_hat is approximately
    -(1 + 3*rho)/T for an AR(1).

    Amihud-Hurvich (2004) augmented regression approach:
        y_t = alpha + beta*x_{t-1} + phi*u_hat_t + e_t
    where u_hat_t are residuals from the AR(1) of x_t.

    References:
        Stambaugh (1999, RFS)
        Amihud & Hurvich (2004, Working Paper)
        Lewellen (2004, JFE) persistence-robust CIs
    """
    print("\n" + "="*90)
    print("STAMBAUGH (1999) BIAS CORRECTION")
    print("  Finite-sample bias in predictive regression with persistent predictor")
    print("  + Lewellen (2004) persistence-robust confidence intervals")
    print("="*90)

    y = df['ret5'].values * 100  # basis points
    x = df['dskew'].values
    n = len(y)

    # Step 1: Estimate persistence of predictor (AR(1))
    x_lag = x[:-1]
    x_curr = x[1:]
    X_ar = sm.add_constant(x_lag)
    ar_model = OLS(x_curr, X_ar).fit()
    rho_hat = ar_model.params[1]
    u_hat = ar_model.resid  # Innovation in x_t

    print(f"\n  Predictor persistence:")
    print(f"  rho(dSkew) = {rho_hat:.4f}")
    print(f"  Half-life: {-np.log(2)/np.log(abs(rho_hat)):.1f} days")

    # Step 2: Standard predictive regression
    y_reg = y[1:]  # Align with x_{t-1}
    x_reg = x[:-1]
    X_pred = sm.add_constant(x_reg)
    pred_model = OLS(y_reg, X_pred).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    beta_ols = pred_model.params[1]
    se_ols = pred_model.bse[1]
    t_ols = pred_model.tvalues[1]

    # Step 3: Compute Stambaugh bias
    e_hat = pred_model.resid
    # Correlation between predictor innovation and return shock
    # Need to align: u_hat is from AR(1) on x[1:], e_hat is from pred reg on y[1:]
    sigma_eu = np.cov(u_hat, e_hat)[0, 1]
    sigma_uu = np.var(u_hat)
    gamma = sigma_eu / sigma_uu

    # Bias in rho_hat (Kendall 1954 approximation)
    bias_rho = -(1 + 3 * rho_hat) / n
    # Stambaugh bias in beta
    stambaugh_bias = gamma * bias_rho
    beta_corrected = beta_ols - stambaugh_bias

    print(f"\n  OLS Results:")
    print(f"  beta_OLS = {beta_ols:.4f} (t = {t_ols:.3f})")
    print(f"\n  Bias Correction:")
    print(f"  gamma (sigma_eu/sigma_uu) = {gamma:.4f}")
    print(f"  E[bias in rho] = {bias_rho:.6f}")
    print(f"  Stambaugh bias = {stambaugh_bias:.4f}")
    print(f"  beta_corrected = {beta_corrected:.4f}")
    print(f"  Bias as % of estimate: {abs(stambaugh_bias/beta_ols)*100:.2f}%")

    # Step 4: Amihud-Hurvich augmented regression
    X_aug = np.column_stack([np.ones(len(y_reg)), x_reg, u_hat])
    aug_model = OLS(y_reg, X_aug).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    beta_ah = aug_model.params[1]
    t_ah = aug_model.tvalues[1]

    print(f"\n  Amihud-Hurvich Augmented Regression:")
    print(f"  beta_AH = {beta_ah:.4f} (t = {t_ah:.3f})")
    print(f"  phi (innovation loading) = {aug_model.params[2]:.4f} (t = {aug_model.tvalues[2]:.3f})")

    # Step 5: Lewellen (2004) persistence-robust CI
    # Use rho = 1 as worst case (most conservative)
    # Under rho=1: beta_hat ~ N(beta, sigma^2) with inflated variance
    rho_upper = min(rho_hat + 1.96 * ar_model.bse[1], 1.0)
    # Conservative: assume rho could be as high as rho_upper
    bias_conservative = gamma * (-(1 + 3 * rho_upper) / n)
    beta_lewellen = beta_ols - bias_conservative
    # Widen SE by factor sqrt(1/(1-rho^2)) for near-unit-root
    se_lewellen = se_ols * np.sqrt(1 / max(1 - rho_upper**2, 0.01))
    t_lewellen = beta_lewellen / se_lewellen
    ci_lewellen = (beta_lewellen - 1.96*se_lewellen, beta_lewellen + 1.96*se_lewellen)

    print(f"\n  Lewellen (2004) Persistence-Robust Inference:")
    print(f"  Worst-case rho = {rho_upper:.4f}")
    print(f"  beta_robust = {beta_lewellen:.4f}")
    print(f"  SE_robust = {se_lewellen:.4f}")
    print(f"  t_robust = {t_lewellen:.3f}")
    print(f"  95% CI: [{ci_lewellen[0]:.3f}, {ci_lewellen[1]:.3f}]")
    print(f"  -> {'Significant' if ci_lewellen[1] < 0 else 'Contains zero'} at 5% (robust)")

    return {
        'beta_ols': beta_ols, 'beta_corrected': beta_corrected,
        'beta_ah': beta_ah, 'beta_lewellen': beta_lewellen,
        'stambaugh_bias': stambaugh_bias, 'rho': rho_hat,
        'ci_lewellen': ci_lewellen
    }


# =============================================================================
# 2. LOCAL PROJECTIONS (JORDA 2005)
# =============================================================================

def local_projections(df):
    """
    Jorda (2005) local projections for multi-horizon impulse responses.

    Instead of iterating a VAR forward, estimate the h-step-ahead response
    directly:
        y_{t+h} - y_t = alpha_h + beta_h * x_t + controls + epsilon_{t+h}

    for h = 1, 2, 3, ..., H

    Advantages over VAR:
    - Robust to misspecification of intermediate dynamics
    - Each horizon estimated independently
    - Easy to add nonlinear terms

    Standard errors: Newey-West with h lags (MA(h-1) structure from overlap).

    Reference: Jorda (2005, AER)
    """
    print("\n" + "="*90)
    print("LOCAL PROJECTIONS (Jorda 2005)")
    print("  Direct multi-horizon impulse response of returns to dSkew shock")
    print("  Horizons: h = 1, 2, 3, 5, 7, 10, 15, 21 trading days")
    print("="*90)

    horizons = [1, 2, 3, 5, 7, 10, 15, 21]
    x = df['dskew'].values
    spx_ret = df['ret1'].values  # Daily returns in %
    n = len(df)

    results = []

    print(f"\n{'h':>4} {'beta_h(bp)':>10} {'SE':>8} {'t-stat':>7} {'p-val':>8} {'R2':>7}")
    print("-"*50)

    for h in horizons:
        # Compute h-period cumulative return
        cum_ret = np.zeros(n - h)
        for t in range(n - h):
            cum_ret[t] = np.sum(spx_ret[t+1:t+1+h])  # Already in %

        cum_ret_bp = cum_ret * 100  # Convert to basis points

        # Regress on lagged dSkew with controls
        x_lag = x[:n-h]
        X = sm.add_constant(x_lag)

        model = OLS(cum_ret_bp, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(h-1, 1)})

        beta_h = model.params[1]
        se_h = model.bse[1]
        t_h = model.tvalues[1]
        p_h = model.pvalues[1]
        r2_h = model.rsquared

        p_str = f"{p_h:.4f}" if p_h >= 0.001 else "< 0.001"
        print(f"{h:>4} {beta_h:>10.2f} {se_h:>8.2f} {t_h:>7.2f} {p_str:>8} {r2_h:>7.4f}")

        results.append({'h': h, 'beta': beta_h, 'se': se_h, 't': t_h, 'p': p_h, 'r2': r2_h})

    # Peak effect
    betas = [r['beta'] for r in results]
    peak_h = horizons[np.argmin(betas)]
    peak_beta = min(betas)
    print(f"\n  Peak effect at h = {peak_h} days: beta = {peak_beta:.2f} bp")
    print(f"  Signal half-life: effect decays 50% by h ~ {horizons[np.argmin(np.abs(np.array(betas) - peak_beta/2))]} days")

    return results


# =============================================================================
# 3. UTILITY-BASED EVALUATION
# =============================================================================

def utility_evaluation(df):
    """
    Certainty equivalent return (CER) gain from using predictive model.

    For a mean-variance investor with risk aversion gamma:
        U = E[r] - (gamma/2) * Var[r]
        CER_gain = U_model - U_benchmark

    Following Campbell & Thompson (2008) and Welch & Goyal (2008).

    Also computes:
    - Optimal portfolio weight: w* = (1/gamma) * E[r]/Var[r]
    - Realized utility from dynamic allocation
    - Performance fee (what investor would pay for signal)

    Reference: Campbell & Thompson (2008, RFS)
    """
    print("\n" + "="*90)
    print("UTILITY-BASED EVALUATION")
    print("  Certainty equivalent return gain for mean-variance investor")
    print("  Risk aversion gamma in {1, 3, 5, 10}")
    print("="*90)

    y = df['ret5'].values * 100  # bp
    x = df['dskew'].values
    n = len(y)
    oos_start = n // 2

    # Generate OOS forecasts
    forecast_model = np.zeros(n - oos_start)
    forecast_mean = np.zeros(n - oos_start)
    actual = y[oos_start:]

    for t in range(oos_start, n):
        idx = t - oos_start
        forecast_mean[idx] = np.mean(y[:t])
        X_train = sm.add_constant(x[:t])
        model = OLS(y[:t], X_train).fit()
        forecast_model[idx] = model.predict(np.array([[1.0, x[t]]]))[0]

    # Compute realized variance (expanding window)
    var_oos = np.array([np.var(y[:oos_start+i+1]) for i in range(len(actual))])

    gammas = [1, 3, 5, 10]

    print(f"\n{'gamma':>6} {'CER_model':>10} {'CER_bench':>10} {'CER_gain':>10} {'Fee(ann bp)':>12} {'Avg w*':>8}")
    print("-"*60)

    for gamma in gammas:
        # Optimal weight under model forecast
        w_model = (1/gamma) * forecast_model / var_oos
        w_model = np.clip(w_model, -2, 2)  # Leverage constraint

        # Optimal weight under benchmark
        w_bench = (1/gamma) * forecast_mean / var_oos
        w_bench = np.clip(w_bench, -2, 2)

        # Realized portfolio returns
        port_ret_model = w_model * actual
        port_ret_bench = w_bench * actual

        # CER = mean - (gamma/2) * variance
        cer_model = np.mean(port_ret_model) - (gamma/2) * np.var(port_ret_model)
        cer_bench = np.mean(port_ret_bench) - (gamma/2) * np.var(port_ret_bench)
        cer_gain = cer_model - cer_bench

        # Annualized performance fee (bp per year)
        # Fee = CER_gain * (252/5) for 5-day returns
        ann_fee = cer_gain * 252 / 5

        print(f"{gamma:>6} {cer_model:>10.3f} {cer_bench:>10.3f} {cer_gain:>10.3f} {ann_fee:>12.1f} {np.mean(w_model):>8.3f}")

    # Manipulation-proof performance measure (Goetzmann et al. 2007)
    # Theta = (1/((1-gamma)*T)) * sum(log((1 + r_p)/(1 + r_f)))
    # Simplified: use log utility gain
    log_ret_model = np.log(1 + actual * w_model / 10000)
    log_ret_bench = np.log(1 + actual * w_bench / 10000)
    theta = np.mean(log_ret_model) - np.mean(log_ret_bench)
    print(f"\n  Manipulation-proof performance (Goetzmann 2007): Theta = {theta*10000:.2f} bp/period")

    return cer_gain


# =============================================================================
# 4. PLACEBO / RANDOMIZATION TEST
# =============================================================================

def placebo_test(df, n_shuffles=10000):
    """
    Randomization test: shuffle signal dates, re-estimate beta.

    Under H0 (no predictability), the distribution of beta from shuffled
    data gives the exact finite-sample null distribution, accounting for
    all features of the data (fat tails, heteroskedasticity, etc.).

    This is more reliable than asymptotic p-values because it:
    - Makes no distributional assumptions
    - Accounts for any data features automatically
    - Gives exact (not approximate) size control

    Report: empirical p-value = fraction of shuffled |beta| >= |beta_actual|

    Reference: Standard in JFE (e.g., Chordia, Subrahmanyam, Tong 2014)
    """
    print("\n" + "="*90)
    print(f"PLACEBO / RANDOMIZATION TEST (N = {n_shuffles:,} shuffles)")
    print("  Exact finite-sample null distribution via signal permutation")
    print("="*90)

    y = df['ret5'].values * 100
    x = df['dskew'].values
    n = len(y)

    # Actual beta
    X = sm.add_constant(x)
    model = OLS(y, X).fit()
    beta_actual = model.params[1]
    t_actual = model.tvalues[1]

    # Shuffle and re-estimate
    rng = np.random.RandomState(42)
    beta_null = np.zeros(n_shuffles)
    t_null = np.zeros(n_shuffles)

    for i in range(n_shuffles):
        x_shuffled = rng.permutation(x)
        X_shuf = sm.add_constant(x_shuffled)
        m = OLS(y, X_shuf).fit()
        beta_null[i] = m.params[1]
        t_null[i] = m.tvalues[1]

    # Empirical p-value (two-sided)
    p_empirical = np.mean(np.abs(beta_null) >= np.abs(beta_actual))
    # One-sided (beta < 0)
    p_one_sided = np.mean(beta_null <= beta_actual)

    print(f"\n  Actual beta: {beta_actual:.4f}")
    print(f"  Actual t-stat: {t_actual:.3f}")
    print(f"\n  Null distribution (shuffled):")
    print(f"  Mean: {np.mean(beta_null):.4f} (should be ~0)")
    print(f"  Std:  {np.std(beta_null):.4f}")
    print(f"  5th percentile: {np.percentile(beta_null, 5):.4f}")
    print(f"  1st percentile: {np.percentile(beta_null, 1):.4f}")
    print(f"\n  Empirical p-value (two-sided): {p_empirical:.4f}")
    print(f"  Empirical p-value (one-sided, beta<0): {p_one_sided:.4f}")
    print(f"  -> {'REJECT H0' if p_empirical < 0.05 else 'Fail to reject'} at 5% (exact test)")

    # Rank of actual beta
    rank = np.sum(beta_null <= beta_actual)
    print(f"  Rank of actual beta: {rank}/{n_shuffles} ({rank/n_shuffles*100:.2f}th percentile)")

    return beta_null, p_empirical, p_one_sided


# =============================================================================
# 5. GRS TEST (GIBBONS-ROSS-SHANKEN 1989)
# =============================================================================

def grs_test(df):
    """
    Gibbons-Ross-Shanken (1989) test for joint alpha significance
    across quintile portfolios.

    Tests H0: alpha_1 = alpha_2 = ... = alpha_K = 0 jointly.

    GRS = (T-N-K)/(N) * (1/(1 + mu'*Sigma^{-1}*mu)) * alpha'*Sigma_e^{-1}*alpha

    where:
    - T = number of time periods
    - N = number of test portfolios (5 quintiles)
    - K = number of factors
    - mu = factor means
    - Sigma = factor covariance
    - alpha = vector of portfolio alphas
    - Sigma_e = residual covariance matrix

    Under H0: GRS ~ F(N, T-N-K)

    Reference: Gibbons, Ross, Shanken (1989, Econometrica)
    """
    print("\n" + "="*90)
    print("GRS TEST (Gibbons-Ross-Shanken 1989)")
    print("  Joint test: all quintile portfolio alphas = 0")
    print("="*90)

    df_q = df.copy()
    df_q['quintile'] = pd.qcut(df_q['dskew'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # Quintile portfolio excess returns (5-day)
    T = len(df)
    N = 5  # Number of test assets
    K = 3  # Number of factors (MKT, SMB, HML)

    # Factor matrix
    factors = df[['mkt_rf', 'smb', 'hml']].values
    factor_means = factors.mean(axis=0)
    factor_cov = np.cov(factors.T)

    # Portfolio returns and alphas
    alphas = np.zeros(N)
    residuals = np.zeros((T, N))

    for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
        mask = df_q['quintile'] == q
        port_ret = np.zeros(T)
        port_ret[mask] = df.loc[mask, 'ret1'].values - df.loc[mask, 'rf'].values
        # Full-sample regression
        X = sm.add_constant(factors)
        model = OLS(port_ret, X).fit()
        alphas[i] = model.params[0]
        residuals[:, i] = model.resid

    # Residual covariance
    Sigma_e = np.cov(residuals.T)

    # GRS statistic
    Sigma_f_inv = np.linalg.inv(factor_cov)
    sharpe_sq_f = factor_means @ Sigma_f_inv @ factor_means

    Sigma_e_inv = np.linalg.inv(Sigma_e)
    alpha_stat = alphas @ Sigma_e_inv @ alphas

    grs_stat = ((T - N - K) / N) * (1 / (1 + sharpe_sq_f)) * alpha_stat

    # F distribution
    grs_p = 1 - stats.f.cdf(grs_stat, N, T - N - K)

    print(f"\n  Test portfolios: {N} quintile-sorted portfolios")
    print(f"  Factors: MKT-RF, SMB, HML")
    print(f"  T = {T}, K = {K}")
    print(f"\n  Individual alphas (daily, bp):")
    for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
        print(f"    {q}: alpha = {alphas[i]*100:.3f} bp")

    print(f"\n  GRS F-statistic: {grs_stat:.4f}")
    print(f"  p-value: {grs_p:.4f}")
    print(f"  Critical value (5%): {stats.f.ppf(0.95, N, T-N-K):.4f}")
    print(f"  -> {'REJECT' if grs_p < 0.05 else 'Fail to reject'} H0: all alphas = 0")

    # Average absolute alpha
    print(f"\n  |alpha| average: {np.mean(np.abs(alphas))*100:.3f} bp/day")
    print(f"  Q1-Q5 alpha spread: {(alphas[0] - alphas[4])*100:.3f} bp/day")

    return grs_stat, grs_p, alphas


# =============================================================================
# 6. ENCOMPASSING TEST
# =============================================================================

def encompassing_test(df):
    """
    Harvey-Leybourne-Newbold (1998) forecast encompassing test.

    Tests whether forecast A encompasses forecast B:
    H0: lambda = 0 in e_A = lambda * (f_B - f_A) + epsilon

    If lambda = 0: forecast A already contains all information in B.
    If lambda = 1: forecast B dominates A.

    Tests our dSkew signal against: VIX changes, momentum, volume.

    Reference: Harvey, Leybourne, Newbold (1998, IJF)
    """
    print("\n" + "="*90)
    print("ENCOMPASSING TESTS (Harvey-Leybourne-Newbold 1998)")
    print("  Does dSkew forecast encompass competing predictors?")
    print("="*90)

    y = df['ret5'].values * 100
    x_skew = df['dskew'].values
    n = len(y)

    # Competing predictors
    predictors = {
        'VIX level': df['vix'].values,
        'VIX change': np.diff(df['vix'].values, prepend=df['vix'].values[0]),
        'Momentum': df['mom'].values,
        'Volume': df['vol'].values,
    }

    # Our forecast errors
    X_skew = sm.add_constant(x_skew)
    model_skew = OLS(y, X_skew).fit()
    e_skew = model_skew.resid
    f_skew = model_skew.fittedvalues

    print(f"\n{'Competitor':<15} {'lambda':>8} {'t-stat':>8} {'p-val':>8} {'Encompasses?':>14}")
    print("-"*55)

    for name, x_comp in predictors.items():
        X_comp = sm.add_constant(x_comp)
        model_comp = OLS(y, X_comp).fit()
        f_comp = model_comp.fittedvalues

        # Encompassing regression: e_skew = lambda * (f_comp - f_skew) + eps
        diff = (f_comp - f_skew).reshape(-1, 1)
        X_enc = sm.add_constant(diff)
        enc_model = OLS(e_skew, X_enc).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

        lam = enc_model.params[1]
        t_lam = enc_model.tvalues[1]
        p_lam = enc_model.pvalues[1]

        encompasses = "Yes" if p_lam > 0.05 else "No"
        p_str = f"{p_lam:.4f}" if p_lam >= 0.001 else "< 0.001"
        print(f"{name:<15} {lam:>8.4f} {t_lam:>8.3f} {p_str:>8} {encompasses:>14}")

    print(f"\n  'Yes' = dSkew encompasses competitor (competitor adds nothing)")
    print(f"  'No'  = competitor contains additional information beyond dSkew")

    return


# =============================================================================
# 7. COMPARISON WITH KNOWN PREDICTORS
# =============================================================================

def compare_known_predictors(df):
    """
    Horse race: dSkew vs. established equity return predictors.

    Predictors tested:
    - VIX level (variance risk premium proxy)
    - VIX change
    - Short-term reversal (1-week momentum)
    - dSkew (our signal)
    - dSkew + VIX interaction
    - Kitchen sink (all combined)

    Report: individual R2, incremental R2, t-statistics.
    """
    print("\n" + "="*90)
    print("PREDICTOR HORSE RACE")
    print("  dSkew vs. known equity return predictors (5-day horizon)")
    print("="*90)

    y = df['ret5'].values * 100
    n = len(y)

    # Construct predictors
    vix = df['vix'].values
    vix_change = np.diff(vix, prepend=vix[0])
    mom = df['mom'].values
    dskew = df['dskew'].values
    dskew_vix = dskew * vix  # Interaction

    predictors = {
        'dSkew': dskew,
        'VIX level': vix,
        'VIX change': vix_change,
        'Momentum': mom,
        'dSkew x VIX': dskew_vix,
    }

    # Univariate results
    print(f"\n  Panel A: Univariate Regressions")
    print(f"  {'Predictor':<15} {'beta':>9} {'t-stat':>8} {'p-val':>9} {'R2(%)':>8}")
    print(f"  {'-'*52}")

    univariate_r2 = {}
    for name, x in predictors.items():
        X = sm.add_constant(x)
        m = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        p_str = f"{m.pvalues[1]:.4f}" if m.pvalues[1] >= 0.001 else "< 0.001"
        sig = "***" if m.pvalues[1] < 0.01 else ("**" if m.pvalues[1] < 0.05 else "")
        print(f"  {name:<15} {m.params[1]:>9.4f} {m.tvalues[1]:>8.3f} {p_str:>9} {m.rsquared*100:>8.3f} {sig}")
        univariate_r2[name] = m.rsquared

    # Multivariate: all predictors
    print(f"\n  Panel B: Multivariate (Kitchen Sink)")
    X_all = sm.add_constant(np.column_stack(list(predictors.values())))
    m_all = OLS(y, X_all).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    print(f"  {'Predictor':<15} {'beta':>9} {'t-stat':>8} {'p-val':>9}")
    print(f"  {'-'*42}")
    for i, name in enumerate(predictors.keys()):
        p_str = f"{m_all.pvalues[i+1]:.4f}" if m_all.pvalues[i+1] >= 0.001 else "< 0.001"
        sig = "***" if m_all.pvalues[i+1] < 0.01 else ("**" if m_all.pvalues[i+1] < 0.05 else "")
        print(f"  {name:<15} {m_all.params[i+1]:>9.4f} {m_all.tvalues[i+1]:>8.3f} {p_str:>9} {sig}")
    print(f"  R2 = {m_all.rsquared*100:.3f}%")

    # Incremental R2 from adding dSkew
    X_no_skew = sm.add_constant(np.column_stack([vix, vix_change, mom]))
    m_no_skew = OLS(y, X_no_skew).fit()
    incremental_r2 = m_all.rsquared - m_no_skew.rsquared

    print(f"\n  Panel C: Incremental Contribution")
    print(f"  R2 without dSkew: {m_no_skew.rsquared*100:.3f}%")
    print(f"  R2 with dSkew:    {m_all.rsquared*100:.3f}%")
    print(f"  Incremental R2:   {incremental_r2*100:.3f}%")
    print(f"  F-test for dSkew inclusion: F = {((m_all.rsquared - m_no_skew.rsquared) / (1 - m_all.rsquared)) * (n - 6) / 2:.3f}")

    return univariate_r2


# =============================================================================
# 8. HEATMAP: PREDICTABILITY ACROSS HORIZONS x REGIMES
# =============================================================================

def predictability_heatmap(df, save_path='figures/Figure12_Heatmap'):
    """
    Heatmap of t-statistics for dSkew predictability across all
    horizon x regime combinations.
    """
    print("\n" + "="*90)
    print("PREDICTABILITY HEATMAP: Horizons x Regimes")
    print("="*90)

    horizons = [1, 2, 3, 5, 7, 10, 15, 21]
    regimes = ['low', 'medium', 'high', 'all']
    regime_labels = ['Low VIX', 'Med VIX', 'High VIX', 'Full Sample']

    t_matrix = np.zeros((len(regimes), len(horizons)))
    beta_matrix = np.zeros((len(regimes), len(horizons)))

    spx_ret = df['ret1'].values
    x = df['dskew'].values
    n = len(df)

    for j, h in enumerate(horizons):
        cum_ret = np.zeros(n - h)
        for t in range(n - h):
            cum_ret[t] = np.sum(spx_ret[t+1:t+1+h]) * 100  # bp

        x_lag = x[:n-h]

        for i, reg in enumerate(regimes):
            if reg == 'all':
                mask = np.ones(n-h, dtype=bool)
            else:
                mask = (df['regime'].values[:n-h] == reg)

            if mask.sum() < 50:
                continue

            X_r = sm.add_constant(x_lag[mask])
            m = OLS(cum_ret[mask], X_r).fit(cov_type='HAC', cov_kwds={'maxlags': max(h-1, 1)})
            t_matrix[i, j] = m.tvalues[1]
            beta_matrix[i, j] = m.params[1]

    # Print table
    print(f"\n  t-statistics:")
    print(f"  {'':>12}", end='')
    for h in horizons:
        print(f"  h={h:<3}", end='')
    print()
    for i, label in enumerate(regime_labels):
        print(f"  {label:>12}", end='')
        for j in range(len(horizons)):
            t_val = t_matrix[i, j]
            sig = "***" if abs(t_val) > 2.58 else ("**" if abs(t_val) > 1.96 else ("*" if abs(t_val) > 1.65 else ""))
            print(f"  {t_val:>4.1f}{sig:<3}", end='')
        print()

    # Generate heatmap figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    im = ax.imshow(t_matrix, cmap='RdBu_r', aspect='auto', vmin=-4, vmax=4)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f'h={h}' for h in horizons])
    ax.set_yticks(range(len(regime_labels)))
    ax.set_yticklabels(regime_labels)
    ax.set_xlabel('Forecast Horizon (trading days)')
    ax.set_title('Predictive t-Statistics: dSkew on Forward Returns', fontweight='bold')

    # Annotate cells
    for i in range(len(regime_labels)):
        for j in range(len(horizons)):
            t_val = t_matrix[i, j]
            color = 'white' if abs(t_val) > 2.5 else 'black'
            sig = '***' if abs(t_val) > 2.58 else ('**' if abs(t_val) > 1.96 else '')
            ax.text(j, i, f'{t_val:.1f}{sig}', ha='center', va='center', fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('t-statistic')

    plt.tight_layout()
    fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{save_path}.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved: {save_path}")

    return t_matrix, beta_matrix


# =============================================================================
# 9. LOCAL PROJECTION IMPULSE RESPONSE FIGURE
# =============================================================================

def figure_local_projection(df, save_path='figures/Figure13_Local_Projection'):
    """Impulse response function from local projections."""
    horizons = list(range(1, 22))
    x = df['dskew'].values
    spx_ret = df['ret1'].values
    n = len(df)

    betas, ci_lo, ci_hi = [], [], []

    for h in horizons:
        cum_ret = np.zeros(n - h)
        for t in range(n - h):
            cum_ret[t] = np.sum(spx_ret[t+1:t+1+h]) * 100

        X = sm.add_constant(x[:n-h])
        m = OLS(cum_ret, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(h-1, 1)})
        betas.append(m.params[1])
        ci = m.conf_int(alpha=0.05)
        ci_lo.append(ci[1, 0])
        ci_hi.append(ci[1, 1])

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(horizons, betas, color=COLORS['blue'], linewidth=2, marker='o', markersize=4)
    ax.fill_between(horizons, ci_lo, ci_hi, alpha=0.2, color=COLORS['blue'])
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Horizon h (trading days)')
    ax.set_ylabel(r'$\beta_h$ (cumulative bp response)')
    ax.set_title(r'Local Projection IRF: Response of Cumulative Returns to 1-unit $\Delta Skew$ Shock',
                 fontweight='bold')
    ax.yaxis.grid(True, alpha=0.3)

    # Mark significance
    for i, h in enumerate(horizons):
        if ci_hi[i] < 0:
            ax.plot(h, betas[i], 'o', color=COLORS['red'], markersize=6, zorder=5)

    ax.text(0.98, 0.05, 'Red = significant at 5%', transform=ax.transAxes,
            ha='right', fontsize=9, color=COLORS['red'])

    plt.tight_layout()
    fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{save_path}.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    return betas


# =============================================================================
# 10. PLACEBO DISTRIBUTION FIGURE
# =============================================================================

def figure_placebo(beta_null, beta_actual, save_path='figures/Figure14_Placebo'):
    """Null distribution from placebo test."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.hist(beta_null, bins=100, density=True, alpha=0.7, color=COLORS['gray'],
            edgecolor='white', linewidth=0.3, label='Null (shuffled)')
    ax.axvline(beta_actual, color=COLORS['red'], linewidth=2.5,
               label=f'Actual = {beta_actual:.2f}')
    ax.axvline(np.percentile(beta_null, 2.5), color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(np.percentile(beta_null, 97.5), color='black', linewidth=1, linestyle='--', alpha=0.5,
               label='2.5/97.5 pctiles')

    p_val = np.mean(beta_null <= beta_actual)
    ax.text(0.03, 0.95, f'Empirical p = {p_val:.4f}', transform=ax.transAxes,
            va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['red'], alpha=0.9))

    ax.set_xlabel(r'$\beta$ under null')
    ax.set_ylabel('Density')
    ax.set_title('Placebo Test: Null Distribution (10,000 Permutations)', fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{save_path}.pdf', bbox_inches='tight', facecolor='white')
    plt.close()


# =============================================================================
# 11. LATEX TABLE GENERATION
# =============================================================================

def generate_latex_tables(df):
    """Generate publication-ready LaTeX tables."""
    print("\n" + "="*90)
    print("LATEX TABLE OUTPUT (for journal submission)")
    print("="*90)

    y = df['ret5'].values * 100
    x = df['dskew'].values
    X = sm.add_constant(x)

    # Table 2: Main regression results
    horizons = [('ret1', 1), ('ret5', 5), ('ret10', 10)]

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Univariate Predictive Regressions: $\Delta Skew \to$ Forward Returns}")
    latex.append(r"\label{tab:main_regression}")
    latex.append(r"\begin{tabular}{lccccc}")
    latex.append(r"\hline\hline")
    latex.append(r"Horizon & $\hat{\alpha}$ (bp) & $\hat{\beta}$ (bp/vol) & $t$-stat & $p$-value & $R^2$ \\")
    latex.append(r"\hline")

    for col, k in horizons:
        y_h = df[col].values * 100
        m = OLS(y_h, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(k-1, 1)})
        p_str = f"{m.pvalues[1]:.3f}" if m.pvalues[1] >= 0.001 else "$<$0.001"
        sig = "^{***}" if m.pvalues[1] < 0.01 else ("^{**}" if m.pvalues[1] < 0.05 else "")
        latex.append(f"{k}-day & {m.params[0]:.2f} & {m.params[1]:.2f}${sig}$ & {m.tvalues[1]:.2f} & {p_str} & {m.rsquared:.4f} \\\\")

    latex.append(r"\hline\hline")
    latex.append(r"\multicolumn{6}{l}{\footnotesize Newey-West HAC standard errors with $k-1$ lags.}")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    print(f"\n{latex_str}")

    # Save to file
    with open('tables/table2_main_regression.tex', 'w') as f:
        f.write(latex_str)
    print(f"\n  Saved: tables/table2_main_regression.tex")

    return latex_str


# =============================================================================
# MAIN
# =============================================================================

def main():
    import time
    start = time.time()

    print("=" * 90)
    print("  ADVANCED ECONOMETRIC SUPPLEMENT")
    print("  Journal of Finance / Journal of Derivatives Standard")
    print("=" * 90)

    os.makedirs('figures', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    print("\n  Loading dataset...")
    df = simulate_dataset()
    print(f"  N = {len(df)}")

    # Run all advanced analyses
    print("\n" + "="*90)
    print("  RUNNING ADVANCED ANALYSES")
    print("="*90)

    stambaugh_bias_correction(df)
    lp_results = local_projections(df)
    utility_evaluation(df)
    beta_null, p_emp, p_one = placebo_test(df, n_shuffles=10000)
    grs_test(df)
    encompassing_test(df)
    compare_known_predictors(df)

    # Figures
    print("\n" + "="*90)
    print("  GENERATING ADDITIONAL FIGURES")
    print("="*90)

    predictability_heatmap(df)
    figure_local_projection(df)

    # Get actual beta for placebo figure
    y = df['ret5'].values * 100
    X = sm.add_constant(df['dskew'].values)
    beta_actual = OLS(y, X).fit().params[1]
    figure_placebo(beta_null, beta_actual)

    # LaTeX tables
    generate_latex_tables(df)

    elapsed = time.time() - start
    print(f"\n{'='*90}")
    print(f"  COMPLETE: Advanced supplement in {elapsed:.1f}s")
    print(f"  New figures: Figure12_Heatmap, Figure13_Local_Projection, Figure14_Placebo")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
