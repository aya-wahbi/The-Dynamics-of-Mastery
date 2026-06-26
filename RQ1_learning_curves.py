#!/usr/bin/env python3
"""


Learning-curve modelling pipeline for the "Dynamics of Mastery" project.
Fits power-law, exponential, and hybrid models per solver, runs robustness
checks (CV, outlier sensitivity, smoothed refit, profile likelihood for w),
and optionally does Bayesian comparison via PyMC.

Author: Aya Wahbi (01427598)
"""

import os
import sys
import json
import logging
import warnings
import pickle
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, norm

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- paths ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output") / "rq1"
RESID_DIR = OUTPUT_DIR / "residuals"
PYMC_DIR = OUTPUT_DIR / "pymc"
for d in [OUTPUT_DIR, RESID_DIR, PYMC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "rq1.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# --- cohort thresholds ---
MIN_COMPETITIONS = 15
MIN_SOLVES = 200
MIN_CAREER_YEARS = 2

# --- cli flags ---
USE_FULL_COHORT = "--full" in sys.argv
ENABLE_PYMC = "--pymc" in sys.argv
SKIP_ROBUSTNESS = "--no-robustness" in sys.argv
SAMPLE_SIZE = 50
N_TIERS = 5
PYMC_SUBSET = 10
if "--pymc-n" in sys.argv:
    idx = sys.argv.index("--pymc-n")
    PYMC_SUBSET = int(sys.argv[idx + 1])

MAX_CURVE_FIT_ITER = 20_000

# Kass-Raftery thresholds for 2*ln(BF)
KR_POSITIVE = 2
KR_STRONG = 6
KR_VERY_STRONG = 10

ASYMPTOTE_LOWER = 2.0   # physical floor — nobody solves a 3x3 in under 2s
ASYMPTOTE_FLOOR = 2.0
SE_IDENTIFIABILITY_THRESHOLD = 100.0


# ──────────────────────────────────────────────
#  Model definitions
# ──────────────────────────────────────────────

def power_law(n, a, b, c):
    return a * np.power(n, -b) + c


def exponential_decay(n, a, b, c):
    return a * np.exp(-b * n) + c


def hybrid(n, w, a, b, d, g, c):
    # g is the exponential rate ('e' in the proposal, renamed to dodge Euler)
    return w * a * np.power(n, -b) + (1 - w) * d * np.exp(-g * n) + c


MODEL_SPECS = {
    "power_law": {
        "func": power_law,
        "k": 3,
        "bounds": ([0, 0, ASYMPTOTE_LOWER], [np.inf, 5, np.inf]),
        "param_names": ["a", "b", "c"],
    },
    "exponential": {
        "func": exponential_decay,
        "k": 3,
        "bounds": ([0, 1e-8, ASYMPTOTE_LOWER], [np.inf, 5, np.inf]),
        "param_names": ["a", "b", "c"],
    },
    "hybrid": {
        "func": hybrid,
        "k": 6,
        "bounds": (
            [0, 0, 0, 0, 1e-8, ASYMPTOTE_LOWER],
            [1, np.inf, 5, np.inf, 5, np.inf],
        ),
        "param_names": ["w", "a", "b", "d", "g", "c"],
    },
}


def _estimate_exp_rate(n, t):
    """Quick-and-dirty initial guess for the exponential decay rate."""
    try:
        cut_early = np.percentile(np.sort(n), 10)
        cut_late = np.percentile(np.sort(n), 90)
        t_early = np.median(t[n <= cut_early])
        t_late = np.median(t[n >= cut_late])
        n_mid = np.median(n)
        if t_early <= t_late or t_early <= 0 or n_mid <= 0:
            return 0.005
        ratio = max((t_early - t_late) / t_early, 0.01)
        return float(np.clip(-np.log(1 - ratio) / n_mid, 1e-5, 0.1))
    except Exception:
        return 0.005


def _initial_guesses(n, t):
    """
    Build a list of starting-parameter vectors for each model,
    based on the data range and a rough exponential rate estimate.
    """
    t_range = float(np.ptp(t))
    t_min = float(np.min(t))
    b_exp = _estimate_exp_rate(n, t)

    def safe_c(val):
        return max(val, ASYMPTOTE_LOWER + 0.5)

    guesses = {
        "power_law": [
            [t_range, 0.3, safe_c(t_min)],
            [t_range * 1.5, 0.5, safe_c(t_min * 0.9)],
            [t_range * 0.8, 0.15, safe_c(t_min * 1.1)],
            [t_range * 2, 0.7, safe_c(t_min * 0.8)],
        ],
        "exponential": [
            [t_range, b_exp, safe_c(t_min)],
            [t_range * 1.5, b_exp * 2, safe_c(t_min * 0.9)],
            [t_range * 0.8, b_exp * 0.5, safe_c(t_min * 1.1)],
            [t_range * 2, b_exp * 4, safe_c(t_min * 0.8)],
        ],
        "hybrid": [
            [0.5, t_range, 0.3, t_range, b_exp, safe_c(t_min)],
            [0.7, t_range * 1.5, 0.5, t_range * 0.5, b_exp * 2, safe_c(t_min * 0.9)],
            [0.3, t_range * 0.8, 0.15, t_range * 1.2, b_exp * 0.5, safe_c(t_min * 1.1)],
            [0.9, t_range, 0.7, t_range, b_exp * 1.5, safe_c(t_min)],
            [0.1, t_range, 0.2, t_range * 1.5, b_exp * 3, safe_c(t_min * 0.9)],
        ],
    }
    return guesses


def _max_relative_se(popt, param_se):
    """Largest relative standard error across all fitted parameters."""
    rel = []
    for i in range(len(popt)):
        if np.isfinite(param_se[i]) and abs(popt[i]) > 1e-12:
            rel.append(abs(param_se[i] / popt[i]))
        else:
            rel.append(np.inf)
    return max(rel) if rel else np.inf


# ──────────────────────────────────────────────
#  SciPy curve fitting
# ──────────────────────────────────────────────

def _safe_fit(func, n, t, p0, bounds, maxfev=MAX_CURVE_FIT_ITER):
    """Wrapper around curve_fit that eats exceptions instead of crashing."""
    try:
        popt, pcov = curve_fit(func, n, t, p0=p0, bounds=bounds,
                               maxfev=maxfev, method="trf")
        residuals = t - func(n, *popt)
        rss = np.sum(residuals ** 2)
        return popt, pcov, rss
    except Exception:
        return None, None, np.inf


def fit_model(model_name, n, t):
    """
    Fit one model (by name) to the data, trying multiple initial guesses.
    Returns a dict with all the stats we care about, or None on failure.
    """
    spec = MODEL_SPECS[model_name]
    func = spec["func"]
    k = spec["k"]
    bounds = spec["bounds"]
    guesses = _initial_guesses(n, t)[model_name]

    best_popt = None
    best_pcov = None
    best_rss = np.inf

    for p0 in guesses:
        popt, pcov, rss = _safe_fit(func, n, t, p0, bounds)
        if rss < best_rss:
            best_popt, best_pcov, best_rss = popt, pcov, rss

    if best_popt is None:
        return None

    N = len(t)
    residuals = t - func(n, *best_popt)

    # information criteria (log-likelihood proportional term)
    ll = N * np.log(best_rss / N)
    bic = ll + k * np.log(N)
    aic = ll + 2 * k
    aicc = aic + (2 * k * (k + 1)) / max(N - k - 1, 1)

    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1.0 - best_rss / ss_tot
    adj_r2 = 1.0 - (1.0 - r2) * (N - 1) / (N - k - 1)

    mae = float(np.mean(np.abs(residuals)))
    med_ae = float(np.median(np.abs(residuals)))

    # parameter standard errors from covariance matrix
    if best_pcov is not None and np.isfinite(best_pcov).all():
        param_se = np.sqrt(np.diag(best_pcov))
    else:
        param_se = np.full(k, np.nan)

    max_rel = _max_relative_se(best_popt, param_se)
    params_identifiable = max_rel < SE_IDENTIFIABILITY_THRESHOLD

    # check if any param is sitting right on a bound
    hit_lower = []
    hit_upper = []
    for i, pname in enumerate(spec["param_names"]):
        if np.isclose(best_popt[i], bounds[0][i], rtol=1e-4):
            hit_lower.append(pname)
        if np.isclose(best_popt[i], bounds[1][i], rtol=1e-4):
            hit_upper.append(pname)

    c_implausible = best_popt[-1] < ASYMPTOTE_FLOOR

    return {
        "model": model_name,
        "popt": best_popt,
        "pcov": best_pcov,
        "param_se": param_se,
        "rss": best_rss,
        "bic": bic,
        "aic": aic,
        "aicc": aicc,
        "r_squared": r2,
        "adj_r2": adj_r2,
        "mae": mae,
        "med_ae": med_ae,
        "k": k,
        "residuals": residuals,
        "max_rel_se": max_rel,
        "params_identifiable": params_identifiable,
        "hit_lower": hit_lower,
        "hit_upper": hit_upper,
        "c_implausible": c_implausible,
    }


# ──────────────────────────────────────────────
#  Robustness & validation helpers
# ──────────────────────────────────────────────

def profile_likelihood_w(n, t, n_grid=30):
    """
    Profile-likelihood scan over the hybrid mixing weight w.
    Tells us whether w is actually identifiable or the RSS surface is flat.
    """
    w_grid = np.linspace(0.01, 0.99, n_grid)
    rss_profile = np.full(n_grid, np.inf)

    t_range = float(np.ptp(t))
    t_min = float(np.min(t))
    b_exp = _estimate_exp_rate(n, t)
    c_init = max(t_min, ASYMPTOTE_LOWER + 0.5)

    guess_bank = [
        [t_range, 0.3, t_range, b_exp, c_init],
        [t_range * 1.5, 0.5, t_range * 0.5, b_exp * 2, c_init * 0.95],
    ]
    fixed_bounds = ([0, 0, 0, 1e-8, ASYMPTOTE_LOWER],
                    [np.inf, 5, np.inf, 5, np.inf])

    for i, w_fixed in enumerate(w_grid):
        # build a reduced hybrid with w frozen
        def hybrid_fixed_w(na, a, b, d, g, c, *, _w=w_fixed):
            return _w * a * np.power(na, -b) + (1 - _w) * d * np.exp(-g * na) + c

        best_rss = np.inf
        for p0 in guess_bank:
            _, _, rss = _safe_fit(hybrid_fixed_w, n, t, p0, fixed_bounds)
            if rss < best_rss:
                best_rss = rss
        rss_profile[i] = best_rss

    best_idx = int(np.argmin(rss_profile))
    w_best = float(w_grid[best_idx])

    finite_vals = rss_profile[np.isfinite(rss_profile)]
    if len(finite_vals) > 5:
        rss_range = float(np.ptp(finite_vals))
        rss_min = float(np.min(finite_vals))
        range_pct = (rss_range / rss_min * 100) if rss_min > 0 else 0.0
        identifiable = range_pct > 1.0
    else:
        range_pct = 0.0
        identifiable = False

    return {
        "w_grid": w_grid,
        "rss_profile": rss_profile,
        "w_best_profile": w_best,
        "is_identifiable": identifiable,
        "rss_range_pct": range_pct,
    }


def temporal_cross_validate(n, t, train_frac=0.80):
    """
    Simple temporal train/test split. Train on the first 80% of solves,
    evaluate on the remaining 20%.
    """
    split = int(len(n) * train_frac)
    if split < 50 or (len(n) - split) < 20:
        return None

    n_train, t_train = n[:split], t[:split]
    n_test, t_test = n[split:], t[split:]

    results = {}
    for mn in MODEL_SPECS:
        fit = fit_model(mn, n_train, t_train)
        if fit is None:
            continue
        predicted = MODEL_SPECS[mn]["func"](n_test, *fit["popt"])
        rss_test = float(np.sum((t_test - predicted) ** 2))
        results[mn] = {
            "rss_train": fit["rss"],
            "rss_test": rss_test,
            "mse_test": rss_test / len(t_test),
        }

    if not results:
        return None

    best = min(results, key=lambda m: results[m]["mse_test"])
    return {
        "models": results,
        "cv_preferred": best,
        "n_train": len(n_train),
        "n_test": len(n_test),
    }


def flag_outliers(n, t, window=50, deviation_thresh=0.50, z_thresh=4.0):
    """Flag solves that are way off the rolling median or have extreme z-scores."""
    df = pd.DataFrame({"n": n, "t": t}).sort_values("n").reset_index(drop=True)
    rolling_med = df["t"].rolling(window, center=True, min_periods=10).median()
    deviation = (df["t"] - rolling_med) / rolling_med
    outlier_dev = deviation.abs() > deviation_thresh

    z = np.abs((df["t"] - df["t"].mean()) / df["t"].std())
    outlier_z = z > z_thresh

    is_outlier = (outlier_dev | outlier_z).values
    return is_outlier, int(is_outlier.sum())


def sensitivity_refit(n, t, is_outlier):
    """Re-fit all models after dropping flagged outliers, see if preference changes."""
    n_clean = n[~is_outlier]
    t_clean = t[~is_outlier]
    if len(n_clean) < 100:
        return None

    fits_clean = {mn: fit_model(mn, n_clean, t_clean) for mn in MODEL_SPECS}
    comp = compare_models(fits_clean)
    clean_pref = comp.iloc[0]["model"] if len(comp) else "none"

    return {
        "n_clean": len(n_clean),
        "n_removed": int(is_outlier.sum()),
        "pct_removed": float(100 * is_outlier.sum() / len(n)),
        "clean_preferred": clean_pref,
    }


def smoothed_robustness(n, t, window=25, subsample=5):
    """Fit on smoothed (rolling-median) data as a sanity check."""
    df = pd.DataFrame({"n": n, "t": t}).sort_values("n").reset_index(drop=True)
    df["smooth"] = df["t"].rolling(window, center=True, min_periods=10).median()
    df = df.dropna(subset=["smooth"]).iloc[::subsample]

    if len(df) < 50:
        return None

    ns = df["n"].values.astype(float)
    ts = df["smooth"].values.astype(float)

    fits_smooth = {mn: fit_model(mn, ns, ts) for mn in MODEL_SPECS}
    comp = compare_models(fits_smooth)
    smooth_pref = comp.iloc[0]["model"] if len(comp) else "none"

    r2_dict = {}
    for m, f in fits_smooth.items():
        if f is not None:
            r2_dict[m] = f["r_squared"]

    return {
        "n_points": len(df),
        "smooth_preferred": smooth_pref,
        "r2_smooth": r2_dict,
    }


def compute_effect_size(fits, preferred):
    """How much RSS improvement does the preferred model give over the runner-up?"""
    if preferred == "none" or fits.get(preferred) is None:
        return np.nan, "none"

    simple_rss = {}
    for m in ("power_law", "exponential"):
        if fits.get(m) is not None:
            simple_rss[m] = fits[m]["rss"]

    if not simple_rss:
        return np.nan, "none"

    baseline_model = min(simple_rss, key=simple_rss.get)
    baseline_rss = simple_rss[baseline_model]
    pref_rss = fits[preferred]["rss"]

    if baseline_rss <= 0:
        return 0.0, baseline_model
    return float((baseline_rss - pref_rss) / baseline_rss * 100), baseline_model


def consensus_preference(bic_pref, cv_pref, smooth_pref):
    """2-of-3 vote among BIC, CV, and smoothed fits."""
    votes = [bic_pref]
    if cv_pref:
        votes.append(cv_pref)
    if smooth_pref:
        votes.append(smooth_pref)

    winner, n_votes = Counter(votes).most_common(1)[0]
    if n_votes >= 2:
        return winner, "consensus"
    else:
        return bic_pref, "bic_only"


def heteroscedasticity_check(n, residuals):
    """Check if residual spread correlates with solve number (Spearman)."""
    abs_res = np.abs(residuals)
    if len(abs_res) < 20:
        return {"rho": np.nan, "p": np.nan, "decreasing_var": False}
    rho, p = spearmanr(n, abs_res)
    return {
        "rho": float(rho),
        "p": float(p),
        "decreasing_var": bool(rho < 0 and p < 0.05),
    }


# ──────────────────────────────────────────────
#  Model comparison (BIC + AICc tables)
# ──────────────────────────────────────────────

def compare_models(fits: dict):
    """Build a comparison table sorted by BIC, with delta-BIC evidence labels."""
    rows = []
    bic_vals = {m: f["bic"] for m, f in fits.items() if f is not None}
    if not bic_vals:
        return pd.DataFrame()

    best_bic = min(bic_vals.values())
    aicc_vals = {m: f["aicc"] for m, f in fits.items() if f is not None}
    best_aicc = min(aicc_vals.values()) if aicc_vals else np.nan

    for model_name, fit in fits.items():
        if fit is None:
            rows.append({
                "model": model_name, "bic": np.nan, "aicc": np.nan,
                "delta_bic": np.nan, "delta_aicc": np.nan,
                "approx_2lnBF": np.nan, "evidence": "no fit",
                "rss": np.nan, "r_squared": np.nan, "adj_r2": np.nan,
                "k": np.nan, "params_identifiable": None,
            })
            continue

        d_bic = fit["bic"] - best_bic
        d_aicc = fit["aicc"] - best_aicc

        if d_bic == 0:
            evidence = "best"
        elif d_bic < KR_POSITIVE:
            evidence = "negligible"
        elif d_bic < KR_STRONG:
            evidence = "positive"
        elif d_bic < KR_VERY_STRONG:
            evidence = "strong"
        else:
            evidence = "very strong"

        rows.append({
            "model": model_name,
            "bic": fit["bic"],
            "aicc": fit["aicc"],
            "delta_bic": d_bic,
            "delta_aicc": d_aicc,
            "approx_2lnBF": d_bic,
            "evidence": evidence,
            "rss": fit["rss"],
            "r_squared": fit["r_squared"],
            "adj_r2": fit["adj_r2"],
            "k": fit["k"],
            "params_identifiable": fit["params_identifiable"],
        })

    return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)


def wald_wolfowitz_runs_test(residuals):
    """Runs test on residual signs — checks for non-random structure."""
    signs = np.sign(residuals)
    signs = signs[signs != 0]
    if len(signs) < 10:
        return np.nan, np.nan, np.nan

    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)
    n_total = n_pos + n_neg

    runs = 1 + np.sum(np.diff(signs) != 0)
    expected_runs = 1 + 2 * n_pos * n_neg / n_total
    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_total)) / (n_total ** 2 * (n_total - 1))

    if var_runs <= 0:
        return runs, np.nan, np.nan

    z = (runs - expected_runs) / np.sqrt(var_runs)
    p = 2 * norm.sf(np.abs(z))
    return runs, z, p


# ──────────────────────────────────────────────
#  Bayesian model comparison (PyMC, optional)
# ──────────────────────────────────────────────

def _build_pymc_power_law(nd, td, t_range, t_min, t_std):
    with pm.Model() as model:
        a = pm.HalfNormal("a", sigma=t_range * 2)
        b = pm.HalfNormal("b", sigma=1.0)
        c = pm.TruncatedNormal("c", mu=t_min, sigma=max(t_min * 2, 1.0),
                               lower=ASYMPTOTE_LOWER)
        sigma = pm.HalfNormal("sigma", sigma=t_std)
        mu = a * (nd ** (-b)) + c
        pm.Normal("obs", mu=mu, sigma=sigma, observed=td)
    return model


def _build_pymc_exponential(nd, td, t_range, t_min, t_std):
    with pm.Model() as model:
        a = pm.HalfNormal("a", sigma=t_range * 2)
        b = pm.HalfNormal("b", sigma=0.01)
        c = pm.TruncatedNormal("c", mu=t_min, sigma=max(t_min * 2, 1.0),
                               lower=ASYMPTOTE_LOWER)
        sigma = pm.HalfNormal("sigma", sigma=t_std)
        mu = a * pm.math.exp(-b * nd) + c
        pm.Normal("obs", mu=mu, sigma=sigma, observed=td)
    return model


def _build_pymc_hybrid(nd, td, t_range, t_min, t_std):
    with pm.Model() as model:
        w = pm.Beta("w", alpha=2, beta=2)
        a = pm.HalfNormal("a", sigma=t_range * 2)
        b = pm.HalfNormal("b", sigma=1.0)
        d = pm.HalfNormal("d", sigma=t_range * 2)
        g = pm.HalfNormal("g", sigma=0.01)
        c = pm.TruncatedNormal("c", mu=t_min, sigma=max(t_min * 2, 1.0),
                               lower=ASYMPTOTE_LOWER)
        sigma = pm.HalfNormal("sigma", sigma=t_std)
        mu = w * a * (nd ** (-b)) + (1 - w) * d * pm.math.exp(-g * nd) + c
        pm.Normal("obs", mu=mu, sigma=sigma, observed=td)
    return model


PYMC_BUILDERS = {
    "power_law": _build_pymc_power_law,
    "exponential": _build_pymc_exponential,
    "hybrid": _build_pymc_hybrid,
}


def _safe_az_compare(traces):
    """
    Try az.compare with different API signatures depending on ArviZ version.
    Newer versions dropped the 'ic' kwarg; older ones require it.
    """
    try:
        return az.compare(traces)
    except TypeError:
        pass
    try:
        return az.compare(traces, ic="loo")
    except TypeError:
        pass
    try:
        return az.compare(traces, ic="waic")
    except TypeError:
        pass
    raise RuntimeError("az.compare failed with all known API signatures")


def bayesian_compare_solver(solver_id, n, t, draws=1000, tune=1000, chains=2):
    if not HAS_PYMC:
        return None

    nd = n.astype("float64")
    td = t.astype("float64")
    t_range = float(np.ptp(td))
    t_min = float(np.min(td))
    t_std = float(np.std(td))

    traces = {}
    for mn, builder in PYMC_BUILDERS.items():
        log.info(f"    PyMC  {mn} ...")
        try:
            model = builder(nd, td, t_range, t_min, t_std)
            with model:
                idata = pm.sample(
                    draws=draws, tune=tune, chains=chains, cores=1,
                    random_seed=42, return_inferencedata=True,
                    idata_kwargs={"log_likelihood": True},
                    progressbar=False,
                )
            traces[mn] = idata
        except Exception as exc:
            log.warning(f"    PyMC  {mn} FAILED for {solver_id}: {exc}")

    if len(traces) < 2:
        return None

    try:
        comp = _safe_az_compare(traces)
    except Exception as exc:
        log.warning(f"    az.compare failed for {solver_id}: {exc}")
        return None

    return {
        "traces": traces,
        "comp": comp,
        "preferred": comp.index[0] if len(comp) else None,
    }


# ──────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────

def find_tsv(pattern_parts):
    """Find a TSV file in DATA_DIR whose name contains all the given substrings."""
    candidates = list(DATA_DIR.glob("*.tsv"))
    for c in candidates:
        if all(p.lower() in c.name.lower() for p in pattern_parts):
            return c
    raise FileNotFoundError(
        f"No TSV matching {pattern_parts} in {DATA_DIR}. "
        f"Available: {[c.name for c in candidates]}"
    )


def load_data():
    results_path = find_tsv(["result", "export"])
    attempts_path = find_tsv(["attempt"])
    comps_path = find_tsv(["competition"])
    round_types_path = find_tsv(["round_type"])

    all_tsvs = sorted([f.name for f in DATA_DIR.glob("*.tsv")])
    log.info(f"TSV files found in {DATA_DIR}: {all_tsvs}")
    log.info(f"Loading results from {results_path.name} ...")

    results = pd.read_csv(results_path, sep="\t", low_memory=False)
    log.info(f"  results columns: {list(results.columns)}")
    log.info(f"  results shape:   {results.shape}")
    results.columns = [c.strip().lower() for c in results.columns]

    # figure out column names — different exports use different conventions
    person_col = next(
        (c for c in ["personid", "person_id", "person_wca_id", "wca_id"]
         if c in results.columns), None
    )
    if person_col is None:
        raise KeyError(f"Cannot find person column. Columns: {list(results.columns)}")

    event_col = next(
        (c for c in ["eventid", "event_id"] if c in results.columns), None
    )
    if event_col is None:
        raise KeyError("Cannot find event column.")

    comp_col = next(
        (c for c in ["competitionid", "competition_id"] if c in results.columns), None
    )
    result_id_col = next(
        (c for c in ["id", "result_id"] if c in results.columns), None
    )
    round_col = next(
        (c for c in ["roundtypeid", "round_type_id"] if c in results.columns), None
    )

    # keep only 3x3x3
    results_333 = results[results[event_col] == "333"].copy()
    log.info(f"  3x3x3 results: {len(results_333):,}")
    del results

    # competitions (for dates)
    comps = pd.read_csv(comps_path, sep="\t", low_memory=False)
    comps.columns = [c.strip().lower() for c in comps.columns]
    comp_id_col_c = next(
        (c for c in ["id", "competition_id"] if c in comps.columns), None
    )
    if "start_date" in comps.columns:
        comps["comp_date"] = pd.to_datetime(comps["start_date"], errors="coerce")
    elif {"year", "month", "day"}.issubset(comps.columns):
        comps["comp_date"] = pd.to_datetime(comps[["year", "month", "day"]], errors="coerce")
    else:
        raise KeyError("Cannot determine competition date.")
    comps_slim = comps[[comp_id_col_c, "comp_date"]].rename(
        columns={comp_id_col_c: "_comp_id"}
    )

    # round types (for ordering rounds within a comp)
    rt = pd.read_csv(round_types_path, sep="\t", low_memory=False)
    rt.columns = [c.strip().lower() for c in rt.columns]
    rt_id_col = next(
        (c for c in ["id", "round_type_id"] if c in rt.columns), None
    )
    rt_rank_col = next(
        (c for c in ["rank", "sort_order", "final"] if c in rt.columns), None
    )
    if rt_rank_col:
        rt["_round_rank"] = rt[rt_rank_col]
    else:
        rt["_round_rank"] = rt[rt_id_col].astype(str)
    rt_slim = rt[[rt_id_col, "_round_rank"]].rename(columns={rt_id_col: "_rt_id"})

    # individual attempts
    attempts = pd.read_csv(attempts_path, sep="\t", low_memory=False)
    attempts.columns = [c.strip().lower() for c in attempts.columns]
    att_result_id_col = next(
        (c for c in ["result_id", "id"] if c in attempts.columns), None
    )
    att_number_col = next(
        (c for c in ["attempt_number", "attempt_num", "num"] if c in attempts.columns),
        None,
    )
    att_value_col = next(
        (c for c in ["attempt_result", "value", "result"] if c in attempts.columns),
        None,
    )

    # merge everything together
    merged = results_333.merge(
        attempts, left_on=result_id_col, right_on=att_result_id_col,
        how="inner", suffixes=("", "_att"),
    )
    del results_333, attempts

    merged = merged.merge(comps_slim, left_on=comp_col, right_on="_comp_id", how="left")
    merged = merged.merge(rt_slim, left_on=round_col, right_on="_rt_id", how="left")

    # drop DNFs / DNS (coded as <= 0) and convert centiseconds -> seconds
    merged = merged[merged[att_value_col] > 0].copy()
    merged["solve_time_s"] = merged[att_value_col] / 100.0

    # chronological ordering within each solver
    sort_cols = [person_col, "comp_date", "_round_rank"]
    if att_number_col is not None:
        sort_cols.append(att_number_col)
    merged.sort_values(sort_cols, inplace=True)
    merged["solve_number"] = merged.groupby(person_col).cumcount() + 1

    keep = [person_col, comp_col, "comp_date", "solve_number", "solve_time_s"]
    if att_number_col is not None:
        keep.append(att_number_col)
    solves = merged[keep].copy().rename(
        columns={person_col: "solver_id", comp_col: "competition_id"}
    )
    log.info(f"  final solves DataFrame: {solves.shape}")
    return solves


# ──────────────────────────────────────────────
#  Cohort selection
# ──────────────────────────────────────────────

def select_cohort(solves):
    stats = solves.groupby("solver_id").agg(
        n_competitions=("competition_id", "nunique"),
        n_solves=("solve_number", "count"),
        first_comp=("comp_date", "min"),
        last_comp=("comp_date", "max"),
    ).reset_index()
    stats["career_days"] = (stats["last_comp"] - stats["first_comp"]).dt.days
    stats["career_years"] = stats["career_days"] / 365.25

    eligible = stats[
        (stats["n_competitions"] > MIN_COMPETITIONS)
        & (stats["n_solves"] > MIN_SOLVES)
        & (stats["career_years"] > MIN_CAREER_YEARS)
    ].copy()

    log.info(
        f"Cohort selection: {len(eligible)} solvers pass all thresholds "
        f"(from {len(stats)} total)"
    )

    # compute a "last observed average" from each solver's final 12 solves
    last_avg = (
        solves[solves["solver_id"].isin(eligible["solver_id"])]
        .sort_values(["solver_id", "solve_number"])
        .groupby("solver_id")
        .tail(12)
        .groupby("solver_id")["solve_time_s"]
        .mean()
        .rename("last_avg_s")
    )
    eligible = eligible.merge(last_avg, left_on="solver_id", right_index=True, how="left")
    cohort_solves = solves[solves["solver_id"].isin(eligible["solver_id"])].copy()
    return cohort_solves, eligible


def stratified_sample(cohort_info, n_total=SAMPLE_SIZE, n_tiers=N_TIERS):
    cohort_info = cohort_info.copy()

    tier_labels = ["T1_fast"]
    for i in range(1, n_tiers - 1):
        tier_labels.append(f"T{i+1}_mid")
    tier_labels.append(f"T{n_tiers}_slow")

    cohort_info["skill_tier"] = pd.qcut(
        cohort_info["last_avg_s"], q=n_tiers, labels=tier_labels
    )

    per_tier = max(1, n_total // n_tiers)
    chunks = []
    for tier in tier_labels:
        tier_df = cohort_info[cohort_info["skill_tier"] == tier]
        take = min(per_tier, len(tier_df))
        chunks.append(tier_df.nlargest(take, "n_solves"))
    sampled = pd.concat(chunks, ignore_index=True)

    log.info(
        f"\nStratified sampling: {len(sampled)} solvers selected "
        f"from {len(cohort_info)} eligible (target {n_total})"
    )
    log.info("  NOTE - sampling selects the most data-rich solvers per skill tier.")
    for tier in tier_labels:
        grp = sampled[sampled["skill_tier"] == tier]
        if len(grp):
            log.info(
                f"    {str(tier):20s}  n={len(grp):3d}  "
                f"last_avg=[{grp['last_avg_s'].min():.1f},{grp['last_avg_s'].max():.1f}]s  "
                f"solves=[{grp['n_solves'].min()},{grp['n_solves'].max()}]"
            )

    return sampled


# ──────────────────────────────────────────────
#  Per-solver analysis pipeline
# ──────────────────────────────────────────────

COLORS = {
    "power_law": "#e74c3c",
    "exponential": "#2980b9",
    "hybrid": "#27ae60",
}


def analyse_solver(solver_id, solver_df):
    n = solver_df["solve_number"].values.astype(float)
    t = solver_df["solve_time_s"].values.astype(float)

    # fit all three models
    fits = {mn: fit_model(mn, n, t) for mn in MODEL_SPECS}
    if all(v is None for v in fits.values()):
        log.warning(f"  {solver_id}: all fits failed.")
        return None

    comp_df = compare_models(fits)
    preferred = comp_df.iloc[0]["model"] if len(comp_df) else "none"
    clear_discrimination = (
        comp_df.iloc[1]["delta_bic"] > KR_STRONG if len(comp_df) >= 2 else False
    )

    # AICc winner (may differ from BIC)
    aicc_vals = {m: f["aicc"] for m, f in fits.items() if f is not None}
    aicc_preferred = min(aicc_vals, key=aicc_vals.get) if aicc_vals else "none"

    effect_pct, effect_vs = compute_effect_size(fits, preferred)

    # residual diagnostics on the BIC-preferred model
    best_fit = fits.get(preferred)
    runs_z, runs_p = np.nan, np.nan
    if best_fit:
        _, runs_z, runs_p = wald_wolfowitz_runs_test(best_fit["residuals"])

    hetero_result = (
        heteroscedasticity_check(n, best_fit["residuals"])
        if best_fit
        else {"rho": np.nan, "p": np.nan, "decreasing_var": False}
    )

    # robustness checks (skip if --no-robustness)
    profile_w_result = None
    cv_result = None
    outlier_result = None
    smooth_result = None
    n_outliers = 0
    pct_outliers = 0.0

    if not SKIP_ROBUSTNESS:
        if fits.get("hybrid"):
            profile_w_result = profile_likelihood_w(n, t)

        cv_result = temporal_cross_validate(n, t)

        is_outlier, n_outliers = flag_outliers(n, t)
        pct_outliers = 100 * n_outliers / len(n)
        if n_outliers > 0:
            outlier_result = sensitivity_refit(n, t, is_outlier)

        smooth_result = smoothed_robustness(n, t)

    # consensus vote
    cv_pref = cv_result["cv_preferred"] if cv_result else None
    smooth_pref = smooth_result["smooth_preferred"] if smooth_result else None
    consensus_model, consensus_type = consensus_preference(preferred, cv_pref, smooth_pref)

    # collect warnings about boundary hits, implausible asymptotes, SE issues
    boundary_warnings = []
    asymptote_warnings = []
    se_warnings = []

    for mn in MODEL_SPECS:
        f = fits[mn]
        if f is None:
            continue
        if f["hit_upper"]:
            boundary_warnings.append(f"{mn}: param(s) {f['hit_upper']} hit UPPER bound")
        if f["hit_lower"]:
            boundary_warnings.append(f"{mn}: param(s) {f['hit_lower']} hit LOWER bound")
        if f["c_implausible"]:
            asymptote_warnings.append(
                f"{mn}: c={f['popt'][-1]:.4g}s < {ASYMPTOTE_FLOOR}s (implausible)"
            )
        if not f["params_identifiable"]:
            se_warnings.append(
                f"{mn}: max_rel_SE={f['max_rel_se']:.0f}x "
                f"(>{SE_IDENTIFIABILITY_THRESHOLD}x)"
            )

    for w in boundary_warnings:
        log.warning(f"  {solver_id} BOUNDARY: {w}")
    for w in asymptote_warnings:
        log.warning(f"  {solver_id} ASYMPTOTE: {w}")
    for w in se_warnings:
        log.warning(f"  {solver_id} SE-IDENT: {w}")

    # assemble the summary dict
    summary = {
        "solver_id": solver_id,
        "n_solves": len(t),
        "preferred_model": preferred,
        "aicc_preferred_model": aicc_preferred,
        "consensus_model": consensus_model,
        "consensus_type": consensus_type,
        "clear_discrimination": clear_discrimination,
        "effect_pct": effect_pct,
        "effect_vs": effect_vs,
    }

    for mn in MODEL_SPECS:
        f = fits[mn]
        for stat in ["bic", "aicc", "rss", "r_squared", "adj_r2",
                      "mae", "med_ae", "max_rel_se"]:
            summary[f"{stat}_{mn}"] = f[stat] if f else np.nan
        summary[f"r2_{mn}"] = summary.get(f"r_squared_{mn}", np.nan)
        summary[f"params_ident_{mn}"] = f["params_identifiable"] if f else None

    summary.update({
        "runs_z": runs_z,
        "runs_p": runs_p,
        "hetero_rho": hetero_result["rho"],
        "hetero_p": hetero_result["p"],
        "hetero_decreasing": hetero_result["decreasing_var"],
        "comparison_table": comp_df,
        "fits": fits,
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
        "profile_w": profile_w_result,
        "cv_result": cv_result,
        "outlier_result": outlier_result,
        "smooth_result": smooth_result,
        "boundary_warnings": boundary_warnings,
        "asymptote_warnings": asymptote_warnings,
        "se_warnings": se_warnings,
    })

    for mn in MODEL_SPECS:
        f = fits[mn]
        pnames = MODEL_SPECS[mn]["param_names"]
        if f is not None:
            summary[f"params_{mn}"] = dict(zip(pnames, f["popt"].tolist()))
            summary[f"param_se_{mn}"] = dict(zip(pnames, f["param_se"].tolist()))
        else:
            summary[f"params_{mn}"] = None
            summary[f"param_se_{mn}"] = None

    summary["hybrid_w"] = fits["hybrid"]["popt"][0] if fits["hybrid"] else np.nan

    # save per-solver residuals
    residual_rows = []
    for mn in MODEL_SPECS:
        f = fits[mn]
        if f is None:
            continue
        predicted = MODEL_SPECS[mn]["func"](n, *f["popt"])
        residual_rows.append(pd.DataFrame({
            "solver_id": solver_id,
            "model": mn,
            "solve_number": n,
            "solve_time_s": t,
            "predicted": predicted,
            "residual": f["residuals"],
        }))
    if residual_rows:
        pd.concat(residual_rows, ignore_index=True).to_csv(
            RESID_DIR / f"{solver_id}.csv", index=False
        )

    return summary


# ──────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────

def plot_solver(solver_id, solver_df, summary, out_dir):
    n = solver_df["solve_number"].values.astype(float)
    t = solver_df["solve_time_s"].values.astype(float)
    fits = summary["fits"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # top-left: raw data + fitted curves
    ax = axes[0, 0]
    ax.scatter(n, t, s=4, alpha=0.2, color="grey", label="solves", rasterized=True)

    win = max(15, len(t) // 40)
    tmp = pd.DataFrame({"n": n, "t": t}).sort_values("n")
    tmp["smooth"] = tmp["t"].rolling(win, center=True, min_periods=5).median()
    ax.plot(tmp["n"], tmp["smooth"], color="black", lw=1.2, alpha=0.5,
            label="rolling median")

    n_dense = np.linspace(n.min(), n.max(), 500)
    for mn, f in fits.items():
        if f is None:
            continue
        curve = MODEL_SPECS[mn]["func"](n_dense, *f["popt"])
        is_best = (mn == summary["preferred_model"])
        ident_tag = "" if f["params_identifiable"] else " [!SE]"
        ax.plot(
            n_dense, curve,
            color=COLORS[mn],
            lw=2.5 if is_best else 1.2,
            ls="-" if is_best else "--",
            label=f"{mn} BIC={f['bic']:.0f} R\u00b2={f['r_squared']:.4f}{ident_tag}",
        )
    ax.set_xlabel("Solve number")
    ax.set_ylabel("Solve time (s)")
    title = f"Solver {solver_id}"
    if summary["consensus_type"] == "consensus":
        title += f"  [consensus: {summary['consensus_model']}]"
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")

    # top-right: residual scatter
    ax2 = axes[0, 1]
    bf = fits.get(summary["preferred_model"])
    if bf:
        ax2.scatter(n, bf["residuals"], s=4, alpha=0.25,
                    color=COLORS.get(summary["preferred_model"], "grey"),
                    rasterized=True)
        ax2.axhline(0, color="black", lw=0.8, ls="--")
        ax2.set_xlabel("Solve number")
        ax2.set_ylabel("Residual (s)")
        resid_title = f"Residuals - {summary['preferred_model']}"
        if not np.isnan(summary["runs_p"]):
            resid_title += f"  (runs p={summary['runs_p']:.3f})"
        ax2.set_title(resid_title)

    # bottom-left: log-log diagnostic
    ax3 = axes[1, 0]
    if bf:
        c_hat = bf["popt"][-1]
        shifted = t - c_hat
        valid = shifted > 0
        if valid.sum() > 50:
            ax3.scatter(np.log10(n[valid]), np.log10(shifted[valid]),
                        s=4, alpha=0.2, color="grey", rasterized=True)
            for mn2, f2 in fits.items():
                if f2 is None:
                    continue
                pred_shifted = MODEL_SPECS[mn2]["func"](n_dense, *f2["popt"]) - f2["popt"][-1]
                v2 = pred_shifted > 0
                if v2.sum() > 10:
                    ax3.plot(
                        np.log10(n_dense[v2]), np.log10(pred_shifted[v2]),
                        color=COLORS[mn2],
                        lw=2.2 if mn2 == summary["preferred_model"] else 1.0,
                        ls="-" if mn2 == summary["preferred_model"] else "--",
                        label=mn2,
                    )
            ax3.set_xlabel("log10(solve number)")
            ax3.set_ylabel("log10(solve time - \u0109)")
            ax3.set_title("Log-log diagnostic")
            ax3.legend(fontsize=7)
        else:
            ax3.set_visible(False)
    else:
        ax3.set_visible(False)

    # bottom-right: residual histogram
    ax4 = axes[1, 1]
    if bf:
        r = bf["residuals"]
        ax4.hist(r, bins=60,
                 color=COLORS.get(summary["preferred_model"], "grey"),
                 edgecolor="black", alpha=0.7, density=True)
        mu_r, std_r = np.mean(r), np.std(r)
        x_range = np.linspace(mu_r - 4 * std_r, mu_r + 4 * std_r, 200)
        ax4.plot(x_range, norm.pdf(x_range, mu_r, std_r), "k--", lw=1.2,
                 label=f"N({mu_r:.2f},{std_r:.2f}\u00b2)")
        ax4.set_xlabel("Residual (s)")
        ax4.set_ylabel("Density")
        ax4.set_title("Residual distribution")
        ax4.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / f"solver_{solver_id}.png", dpi=150)
    plt.close(fig)


def plot_profile_w_grid(all_summaries, out_dir, max_per_page=9):
    with_profile = [s for s in all_summaries if s.get("profile_w")]
    if not with_profile:
        return

    n_pages = int(np.ceil(len(with_profile) / max_per_page))
    for page in range(n_pages):
        batch = with_profile[page * max_per_page : (page + 1) * max_per_page]
        nrows = int(np.ceil(len(batch) / 3))
        fig, axes = plt.subplots(nrows, 3, figsize=(14, 3.5 * nrows))
        axes = np.atleast_2d(axes)

        for idx, s in enumerate(batch):
            row, col = divmod(idx, 3)
            ax = axes[row, col]
            pw = s["profile_w"]
            ax.plot(pw["w_grid"], pw["rss_profile"], "o-", ms=3, color="#27ae60", lw=1.2)
            ax.axvline(pw["w_best_profile"], color="red", ls="--", lw=1,
                       label=f"w*={pw['w_best_profile']:.2f}")
            if not np.isnan(s.get("hybrid_w", np.nan)):
                ax.axvline(s["hybrid_w"], color="blue", ls=":", lw=1,
                           label=f"fit w={s['hybrid_w']:.2f}")
            ident_str = "ident" if pw["is_identifiable"] else "FLAT"
            ax.set_title(f"{s['solver_id']}  ({ident_str})", fontsize=8)
            ax.set_xlabel("w", fontsize=7)
            ax.set_ylabel("RSS", fontsize=7)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)

        # hide unused subplots
        for idx in range(len(batch), nrows * 3):
            row, col = divmod(idx, 3)
            axes[row, col].set_visible(False)

        fig.suptitle("Profile likelihood for hybrid w", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        suffix = f"_p{page + 1}" if n_pages > 1 else ""
        fig.savefig(out_dir / f"profile_w{suffix}.png", dpi=150)
        plt.close(fig)

    log.info(f"Profile-w figures saved ({len(with_profile)} solvers, {n_pages} page(s)).")


def plot_summary(results_df, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # bar chart of preferred model counts
    ax = axes[0, 0]
    pref_counts = results_df["preferred_model"].value_counts()
    pref_counts.plot.bar(
        ax=ax,
        color=[COLORS.get(m, "#95a5a6") for m in pref_counts.index],
        edgecolor="black",
    )
    ax.set_title("Preferred model (lowest BIC)")
    ax.set_ylabel("Number of solvers")
    ax.tick_params(axis="x", rotation=20)
    for i, (label, count) in enumerate(pref_counts.items()):
        ax.text(i, count + 0.3, str(count), ha="center", fontsize=10, fontweight="bold")

    # pie chart: clear discrimination or not
    ax = axes[0, 1]
    disc_counts = results_df["clear_discrimination"].value_counts()
    ax.pie(
        disc_counts.values,
        labels=["Clear (dBIC>6)" if v else "Ambiguous" for v in disc_counts.index],
        autopct="%1.0f%%",
        startangle=90,
        colors=["#2ecc71" if v else "#e67e22" for v in disc_counts.index],
    )
    ax.set_title(f"Clear discrimination (dBIC > {KR_STRONG})")

    # histogram: delta-BIC between power law and exponential
    ax = axes[1, 0]
    delta_pl_exp = results_df["bic_power_law"] - results_df["bic_exponential"]
    ax.hist(delta_pl_exp.dropna(), bins=25, color="#8e44ad", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", lw=1.2, ls="--")
    ax.set_xlabel("dBIC (PL - Exp)")
    ax.set_ylabel("Count")
    ax.set_title("Power law vs Exponential")
    ax.text(
        0.02, 0.95,
        f"PL better: {(delta_pl_exp < 0).sum()}\nExp better: {(delta_pl_exp > 0).sum()}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # scatter: model preference by volume and skill
    ax = axes[1, 1]
    for mn, grp in results_df.groupby("preferred_model"):
        ax.scatter(
            grp["n_solves"], grp["last_avg_s"],
            label=mn, alpha=0.7, s=40,
            edgecolors="black", linewidths=0.4,
            color=COLORS.get(mn, "#95a5a6"),
        )
    ax.set_xlabel("Total solves")
    ax.set_ylabel("Last-observed avg (s)")
    ax.set_title("Model preference by volume & skill")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "rq1_summary.png", dpi=150)
    plt.close(fig)


def plot_hybrid_w(results_df, out_dir):
    w_vals = results_df["hybrid_w"].dropna()
    if len(w_vals) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(w_vals, bins=20, color="#27ae60", edgecolor="black", alpha=0.8)
    axes[0].axvline(0, color="red", ls="--")
    axes[0].axvline(1, color="blue", ls="--")
    axes[0].set_xlabel("w (hybrid weight)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of hybrid w")

    hybrid_only = results_df[results_df["preferred_model"] == "hybrid"].copy()
    if "skill_tier" in hybrid_only.columns and len(hybrid_only) > 0:
        tiers_sorted = sorted(hybrid_only["skill_tier"].dropna().unique())
        sns.stripplot(
            data=hybrid_only, x="skill_tier", y="hybrid_w",
            order=tiers_sorted, ax=axes[1], jitter=0.25,
            alpha=0.6, color="#27ae60", edgecolor="black",
            linewidth=0.4, size=7,
        )
    axes[1].set_ylabel("w")
    axes[1].set_title("Hybrid w by skill tier")
    axes[1].axhline(0, color="red", ls="--")
    axes[1].axhline(1, color="blue", ls="--")

    plt.tight_layout()
    fig.savefig(out_dir / "hybrid_w_analysis.png", dpi=150)
    plt.close(fig)


def plot_runs_investigation(results_df, out_dir):
    rd = results_df.copy()
    rd["runs_sig"] = rd["runs_p"] < 0.05
    sig = rd[rd["runs_sig"]]
    nonsig = rd[~rd["runs_sig"]]

    if len(sig) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in zip(
        axes,
        ["n_solves", "career_years", "last_avg_s"],
        ["Total solves", "Career span (yr)", "Last avg (s)"],
    ):
        parts = ax.violinplot(
            [nonsig[col].dropna(), sig[col].dropna()],
            positions=[0, 1], showmeans=True, showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-sig", "Sig"])
        ax.set_title(title)

    fig.suptitle("Runs-test: sig (p<0.05) vs non-sig", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "runs_test_investigation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aicc_vs_bic(results_df, out_dir):
    n_agree = (results_df["preferred_model"] == results_df["aicc_preferred_model"]).sum()
    n_total = len(results_df)

    fig, ax = plt.subplots(figsize=(6, 4))
    ct = pd.crosstab(results_df["preferred_model"], results_df["aicc_preferred_model"])
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax, linewidths=0.5)
    ax.set_xlabel("AICc preferred")
    ax.set_ylabel("BIC preferred")
    ax.set_title(f"BIC vs AICc: {n_agree}/{n_total} ({100 * n_agree / n_total:.0f}%)")

    plt.tight_layout()
    fig.savefig(out_dir / "aicc_vs_bic_agreement.png", dpi=150)
    plt.close(fig)


def plot_robustness_summary(results_df, out_dir):
    labels = []
    pcts = []

    cv_total = results_df["cv_preferred"].notna().sum()
    if cv_total > 0:
        labels.append(f"CV\n(n={cv_total})")
        agree = (results_df["preferred_model"] == results_df["cv_preferred"]).sum()
        pcts.append(100 * agree / cv_total)

    ol_total = results_df["outlier_preferred"].notna().sum()
    if ol_total > 0:
        labels.append(f"Outlier\n(n={ol_total})")
        agree = (results_df["preferred_model"] == results_df["outlier_preferred"]).sum()
        pcts.append(100 * agree / ol_total)

    sm_total = results_df["smooth_preferred"].notna().sum()
    if sm_total > 0:
        labels.append(f"Smooth\n(n={sm_total})")
        agree = (results_df["preferred_model"] == results_df["smooth_preferred"]).sum()
        pcts.append(100 * agree / sm_total)

    labels.append(f"AICc\n(n={len(results_df)})")
    aicc_agree = (results_df["preferred_model"] == results_df["aicc_preferred_model"]).sum()
    pcts.append(100 * aicc_agree / len(results_df))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    palette = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    bars = ax.bar(x, pcts, color=palette[:len(labels)], edgecolor="black", width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Agreement with BIC (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Robustness: agreement with BIC preference")
    ax.axhline(100, color="grey", ls=":", lw=0.8)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{pct:.0f}%", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "robustness_summary.png", dpi=150)
    plt.close(fig)


def plot_effect_size(results_df, out_dir):
    eff = results_df["effect_pct"].dropna()
    if len(eff) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(eff, bins=25, color="#f39c12", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", ls="--")
    ax.set_xlabel("% RSS improvement")
    ax.set_ylabel("Count")
    ax.set_title("Effect size")
    ax.text(
        0.98, 0.95,
        f"median={eff.median():.1f}%\nmean={eff.mean():.1f}%\n>5%: {(eff > 5).sum()}/{len(eff)}",
        transform=ax.transAxes, va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
    )

    plt.tight_layout()
    fig.savefig(out_dir / "effect_size.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("RQ1 - Learning-Curve Modeling  |  The Dynamics of Mastery")
    log.info(
        f"  v4.1  |  ASYMPTOTE_LOWER={ASYMPTOTE_LOWER}s  "
        f"|  SE_THRESHOLD={SE_IDENTIFIABILITY_THRESHOLD}x"
    )
    log.info("=" * 60)

    solves = load_data()
    solves_cohort, cohort_info = select_cohort(solves)
    del solves  # free memory

    cohort_info.to_csv(OUTPUT_DIR / "cohort_info_full.csv", index=False)
    log.info(f"\nFull eligible cohort ({len(cohort_info)} solvers):")
    log.info(f"  Solves - median {cohort_info['n_solves'].median():.0f}, "
             f"range [{cohort_info['n_solves'].min()},{cohort_info['n_solves'].max()}]")
    log.info(f"  Comps  - median {cohort_info['n_competitions'].median():.0f}")
    log.info(f"  Career - median {cohort_info['career_years'].median():.1f} yr")
    log.info(f"  Last avg - median {cohort_info['last_avg_s'].median():.2f} s")

    if USE_FULL_COHORT:
        analysis_cohort = cohort_info
        log.info(f"\n** FULL COHORT: {len(analysis_cohort)} solvers **")
    else:
        analysis_cohort = stratified_sample(cohort_info)
        log.info(
            f"\n** SAMPLED: {len(analysis_cohort)} solvers "
            f"(--full for all {len(cohort_info)}) **"
        )

    analysis_cohort.to_csv(OUTPUT_DIR / "cohort_info.csv", index=False)
    solver_ids = analysis_cohort["solver_id"].tolist()
    all_summaries = []

    grouped = solves_cohort.sort_values(
        ["solver_id", "solve_number"]
    ).groupby("solver_id")

    for i, sid in enumerate(solver_ids, 1):
        log.info(f"[{i}/{len(solver_ids)}]  Fitting solver {sid} ...")
        sdf = grouped.get_group(sid)
        summary = analyse_solver(sid, sdf)
        if summary is None:
            continue

        # attach cohort metadata
        ci = analysis_cohort[analysis_cohort["solver_id"] == sid].iloc[0]
        summary["n_competitions"] = ci["n_competitions"]
        summary["career_years"] = ci["career_years"]
        summary["last_avg_s"] = ci["last_avg_s"]
        if "skill_tier" in ci.index:
            summary["skill_tier"] = str(ci["skill_tier"])

        all_summaries.append(summary)
        plot_solver(sid, sdf, summary, OUTPUT_DIR)

        # per-solver log output
        pref = summary["preferred_model"]
        r2_val = summary.get(f"r2_{pref}", np.nan)
        mae_val = summary.get(f"mae_{pref}", np.nan)
        log.info(
            f"  Preferred(BIC): {pref}  R\u00b2={r2_val:.4f}  MAE={mae_val:.2f}s  "
            f"Discrim: {summary['clear_discrimination']}"
        )
        log.info(
            f"  Consensus: {summary['consensus_model']} ({summary['consensus_type']})  "
            f"AICc: {summary['aicc_preferred_model']}"
        )

        if not np.isnan(summary["runs_z"]):
            log.info(f"  Runs z={summary['runs_z']:.2f} p={summary['runs_p']:.4f}")
        if not np.isnan(summary["hetero_rho"]):
            log.info(
                f"  Heterosced rho={summary['hetero_rho']:.3f} "
                f"p={summary['hetero_p']:.4f} "
                f"decr={'YES' if summary['hetero_decreasing'] else 'no'}"
            )

        for mn in MODEL_SPECS:
            p = summary[f"params_{mn}"]
            se = summary[f"param_se_{mn}"]
            ident = summary.get(f"params_ident_{mn}")
            if p:
                param_str = "  ".join(
                    f"{k}={v:.4g}\u00b1{se[k]:.2g}" for k, v in p.items()
                )
                ident_tag = "  [NOT IDENT]" if not ident else ""
                log.info(f"    {mn:15s}  {param_str}{ident_tag}")

        if summary.get("cv_result"):
            cv = summary["cv_result"]
            log.info(f"  CV: {cv['cv_preferred']} (train={cv['n_train']},test={cv['n_test']})")
        if summary.get("outlier_result"):
            ol = summary["outlier_result"]
            log.info(
                f"  Outliers: {ol['n_removed']} ({ol['pct_removed']:.1f}%) "
                f"refit: {ol['clean_preferred']}"
            )
        if summary.get("smooth_result"):
            log.info(
                f"  Smoothed: {summary['smooth_result']['smooth_preferred']} "
                f"(n={summary['smooth_result']['n_points']})"
            )
        if summary.get("profile_w"):
            pw = summary["profile_w"]
            ident_str = "ident" if pw["is_identifiable"] else "NOT ident"
            log.info(
                f"  Profile w: {pw['w_best_profile']:.3f} "
                f"RSS%={pw['rss_range_pct']:.2f} {ident_str}"
            )
        if not np.isnan(summary.get("effect_pct", np.nan)):
            log.info(f"  Effect: {summary['effect_pct']:.2f}% over {summary['effect_vs']}")

    if not all_summaries:
        log.error("No solvers analysed.")
        sys.exit(1)

    # ---- optional PyMC Bayesian comparison ----
    if ENABLE_PYMC and HAS_PYMC:
        pymc_ids = []
        if "skill_tier" in analysis_cohort.columns:
            for tier in sorted(analysis_cohort["skill_tier"].unique()):
                tier_grp = analysis_cohort[analysis_cohort["skill_tier"] == tier]
                n_pick = min(max(1, PYMC_SUBSET // N_TIERS), len(tier_grp))
                pymc_ids.extend(
                    tier_grp.nsmallest(n_pick, "n_solves")["solver_id"].tolist()
                )
        else:
            pymc_ids = solver_ids[:PYMC_SUBSET]

        log.info(f"\nPyMC on {len(pymc_ids)} solvers ...")
        for j, sid in enumerate(pymc_ids, 1):
            sdf = grouped.get_group(sid)
            n_arr = sdf["solve_number"].values.astype(float)
            t_arr = sdf["solve_time_s"].values.astype(float)
            bc = bayesian_compare_solver(sid, n_arr, t_arr)
            if bc:
                log.info(f"  {sid}: preferred={bc['preferred']}")
    elif ENABLE_PYMC and not HAS_PYMC:
        log.warning("--pymc set but PyMC not installed.")

    # ---- assemble final results table ----
    records = []
    for s in all_summaries:
        rec = {
            k: s[k]
            for k in [
                "solver_id", "n_solves", "n_competitions", "career_years",
                "last_avg_s", "preferred_model", "aicc_preferred_model",
                "consensus_model", "consensus_type", "clear_discrimination",
                "hybrid_w", "effect_pct", "effect_vs", "runs_z", "runs_p",
                "hetero_rho", "hetero_p", "hetero_decreasing",
                "n_outliers", "pct_outliers",
            ]
        }

        for mn in MODEL_SPECS:
            for stat in ["bic", "aicc", "rss", "r2", "adj_r2", "mae",
                         "med_ae", "max_rel_se"]:
                col = f"{stat}_{mn}"
                rec[col] = s.get(col, np.nan)
            rec[f"params_ident_{mn}"] = s.get(f"params_ident_{mn}")

        cv = s.get("cv_result")
        rec["cv_preferred"] = cv["cv_preferred"] if cv else None

        ol = s.get("outlier_result")
        rec["outlier_preferred"] = ol["clean_preferred"] if ol else None

        sm = s.get("smooth_result")
        rec["smooth_preferred"] = sm["smooth_preferred"] if sm else None

        pw = s.get("profile_w")
        rec["w_profile_best"] = pw["w_best_profile"] if pw else np.nan
        rec["w_profile_identifiable"] = pw["is_identifiable"] if pw else None
        rec["w_profile_rss_range_pct"] = pw["rss_range_pct"] if pw else np.nan

        rec["boundary_warnings"] = "; ".join(s["boundary_warnings"]) if s["boundary_warnings"] else ""
        rec["asymptote_warnings"] = "; ".join(s["asymptote_warnings"]) if s["asymptote_warnings"] else ""
        rec["se_warnings"] = "; ".join(s["se_warnings"]) if s["se_warnings"] else ""

        for mn in MODEL_SPECS:
            rec[f"params_{mn}"] = json.dumps(s[f"params_{mn}"])
            rec[f"param_se_{mn}"] = json.dumps(s[f"param_se_{mn}"])

        if "skill_tier" in s:
            rec["skill_tier"] = s["skill_tier"]

        records.append(rec)

    results_df = pd.DataFrame(records)
    results_df.to_csv(OUTPUT_DIR / "rq1_results.csv", index=False)

    # ---- summary statistics ----
    n_total = len(results_df)
    n_clear = results_df["clear_discrimination"].sum()
    pct_clear = 100 * n_clear / n_total

    log.info("\n" + "=" * 60)
    log.info("RQ1 - RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(
        f"Solvers: {n_total}  Clear discrim: {n_clear}/{n_total} "
        f"({pct_clear:.1f}%)  [target>=60%]"
    )

    log.info("\nBIC preferred:")
    for m, c in results_df["preferred_model"].value_counts().items():
        log.info(f"  {m:20s} {c:3d} ({100 * c / n_total:.1f}%)")

    aicc_bic_agree = (
        results_df["preferred_model"] == results_df["aicc_preferred_model"]
    ).sum()
    log.info(f"\nAICc-BIC agreement: {aicc_bic_agree}/{n_total} ({100 * aicc_bic_agree / n_total:.1f}%)")
    log.info("AICc preferred:")
    for m, c in results_df["aicc_preferred_model"].value_counts().items():
        log.info(f"  {m:20s} {c:3d} ({100 * c / n_total:.1f}%)")

    log.info("\nConsensus (2-of-3: BIC,CV,smooth):")
    for m, c in results_df["consensus_model"].value_counts().items():
        log.info(f"  {m:20s} {c:3d} ({100 * c / n_total:.1f}%)")
    n_consensus = (results_df["consensus_type"] == "consensus").sum()
    log.info(f"  Achieved consensus: {n_consensus}/{n_total}  BIC-only: {n_total - n_consensus}/{n_total}")
    consensus_bic_agree = (results_df["preferred_model"] == results_df["consensus_model"]).sum()
    log.info(f"  Consensus==BIC: {consensus_bic_agree}/{n_total} ({100 * consensus_bic_agree / n_total:.1f}%)")

    # consensus bias caveat
    if not SKIP_ROBUSTNESS:
        sm_hybrid = (results_df["smooth_preferred"] == "hybrid").sum()
        sm_total = results_df["smooth_preferred"].notna().sum()
        if sm_total > 0:
            log.info(
                f"\n  NOTE: Smoothed-data fits selected hybrid for "
                f"{sm_hybrid}/{sm_total} ({100 * sm_hybrid / sm_total:.0f}%) solvers."
            )
            if sm_hybrid / sm_total > 0.8:
                log.info(
                    "  CAVEAT: Consensus may be biased toward hybrid because "
                    "smoothing systematically favours the most flexible model."
                )

    log.info(f"\nKPI: {'PASS' if pct_clear >= 60 else 'BELOW TARGET'} ({pct_clear:.1f}% vs 60%)")

    sig_runs = (results_df["runs_p"] < 0.05).sum()
    log.info(f"\nRuns-test sig (p<0.05): {sig_runs}/{n_total} ({100 * sig_runs / n_total:.1f}%)")
    n_hetero = results_df["hetero_decreasing"].sum()
    log.info(f"Heteroscedasticity (decr var): {n_hetero}/{n_total} ({100 * n_hetero / n_total:.1f}%)")

    log.info("\n-- Goodness of fit (preferred model) --")
    r2_arr = np.array([
        s.get(f"r2_{s['preferred_model']}", np.nan) for s in all_summaries
    ])
    r2_arr = r2_arr[np.isfinite(r2_arr)]
    mae_arr = np.array([
        s.get(f"mae_{s['preferred_model']}", np.nan) for s in all_summaries
    ])
    mae_arr = mae_arr[np.isfinite(mae_arr)]
    medae_arr = np.array([
        s.get(f"med_ae_{s['preferred_model']}", np.nan) for s in all_summaries
    ])
    medae_arr = medae_arr[np.isfinite(medae_arr)]

    if len(r2_arr):
        log.info(
            f"  R\u00b2: median={np.median(r2_arr):.4f} mean={np.mean(r2_arr):.4f} "
            f"range=[{np.min(r2_arr):.4f},{np.max(r2_arr):.4f}]"
        )
    if len(mae_arr):
        log.info(f"  MAE: median={np.median(mae_arr):.2f}s mean={np.mean(mae_arr):.2f}s")
    if len(medae_arr):
        log.info(
            f"  MedAE: median={np.median(medae_arr):.2f}s "
            f"(\u00b1{np.median(medae_arr):.1f}s for median solver)"
        )

    eff = results_df["effect_pct"].dropna()
    if len(eff):
        log.info(
            f"\n-- Effect size --\n"
            f"  median={eff.median():.2f}% mean={eff.mean():.2f}% "
            f">5%:{(eff > 5).sum()}/{len(eff)} <1%:{(eff < 1).sum()}/{len(eff)}"
        )

    log.info(f"\n-- SE identifiability (threshold: {SE_IDENTIFIABILITY_THRESHOLD}x) --")
    for mn in MODEL_SPECS:
        col = f"params_ident_{mn}"
        if col in results_df.columns:
            valid = results_df[col].dropna()
            if len(valid) > 0:
                n_ident = valid.sum()
                log.info(f"  {mn:15s} identifiable: {int(n_ident)}/{len(valid)} ({100 * n_ident / len(valid):.0f}%)")
            else:
                log.info(f"  {mn:15s} identifiable: N/A (column empty)")

    w_vals = results_df["hybrid_w"].dropna()
    if len(w_vals):
        log.info(
            f"\n-- Hybrid w --\n"
            f"  n={len(w_vals)} median={w_vals.median():.3f} "
            f"w<0.1:{(w_vals < 0.1).sum()} w>0.9:{(w_vals > 0.9).sum()} "
            f"mix:{((w_vals >= 0.1) & (w_vals <= 0.9)).sum()}"
        )

    if not SKIP_ROBUSTNESS:
        log.info("\n-- Robustness --")
        for label, col in [("CV", "cv_preferred"), ("Outlier", "outlier_preferred"),
                           ("Smooth", "smooth_preferred")]:
            valid = results_df[col].dropna()
            if len(valid):
                match = (results_df.loc[valid.index, "preferred_model"] == valid).sum()
                log.info(f"  {label}: agrees with BIC in {match}/{len(valid)} ({100 * match / len(valid):.0f}%)")

    if "skill_tier" in results_df.columns:
        log.info(
            "\n-- BIC pref by tier --\n"
            + pd.crosstab(results_df["skill_tier"], results_df["preferred_model"]).to_string()
        )
        log.info(
            "\n-- Consensus by tier --\n"
            + pd.crosstab(results_df["skill_tier"], results_df["consensus_model"]).to_string()
        )

    # ---- generate all the plots ----
    plot_summary(results_df, OUTPUT_DIR)
    plot_hybrid_w(results_df, OUTPUT_DIR)
    plot_runs_investigation(results_df, OUTPUT_DIR)
    plot_aicc_vs_bic(results_df, OUTPUT_DIR)
    plot_effect_size(results_df, OUTPUT_DIR)
    plot_profile_w_grid(all_summaries, OUTPUT_DIR)
    if not SKIP_ROBUSTNESS:
        plot_robustness_summary(results_df, OUTPUT_DIR)

    # ---- per-solver comparison tables (text file) ----
    with open(OUTPUT_DIR / "comparison_tables.txt", "w", encoding="utf-8") as f:
        for s in all_summaries:
            pref = s["preferred_model"]
            r2_val = s.get(f"r2_{pref}", np.nan)
            mae_val = s.get(f"mae_{pref}", np.nan)
            f.write(
                f"\n{'=' * 60}\n"
                f"Solver: {s['solver_id']} | BIC: {pref} | "
                f"Consensus: {s['consensus_model']} | "
                f"R\u00b2={r2_val:.4f} MAE={mae_val:.2f}s\n"
                f"{'=' * 60}\n"
            )
            f.write(s["comparison_table"].to_string(index=False) + "\n\nParams (\u00b1SE):\n")
            for mn in MODEL_SPECS:
                p = s[f"params_{mn}"]
                se = s[f"param_se_{mn}"]
                ident = s.get(f"params_ident_{mn}")
                if p:
                    param_str = "  ".join(
                        f"{k}={v:.4g}\u00b1{se[k]:.2g}" for k, v in p.items()
                    )
                    ident_tag = "  [NOT IDENT]" if not ident else ""
                    f.write(f"  {mn:15s} {param_str}{ident_tag}\n")
                else:
                    f.write(f"  {mn:15s} (failed)\n")
            f.write("\n")

    resid_files = sorted(RESID_DIR.glob("*.csv"))
    log.info(f"\nResiduals saved for {len(resid_files)} solvers in {RESID_DIR}/")
    log.info(f"All outputs written to {OUTPUT_DIR.resolve()}")
    log.info("Done.")


if __name__ == "__main__":
    main()
