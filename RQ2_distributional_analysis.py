#!/usr/bin/env python3
"""
RQ2_distributional_analysis.py  (v2.0)

Distributional evolution analysis for the "Dynamics of Mastery" project.
Splits each solver's career into phases, fits Gaussian / lognormal /
ex-Gaussian per phase, tracks how shape and scale features shift across
the career, and runs normalised two-sample KS to isolate genuine shape
changes from simple location/scale drift.

Author: Aya Wahbi (01427598)
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats as sp_stats
from scipy.stats import norm, lognorm, exponnorm, kstest, ks_2samp
from scipy.stats import skew as compute_skew, kurtosis as compute_kurtosis

np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- paths ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output") / "rq2"
PLOTS_DIR = OUTPUT_DIR / "solver_plots"
QQ_DIR = OUTPUT_DIR / "qq_plots"
RQ1_DIR = Path("output") / "rq1"
for d in [OUTPUT_DIR, PLOTS_DIR, QQ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "rq2.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# --- cohort thresholds (same as RQ1) ---
MIN_COMPETITIONS = 15
MIN_SOLVES = 200
MIN_CAREER_YEARS = 2
SAMPLE_SIZE = 50
N_TIERS = 5

# --- RQ2-specific ---
MIN_PHASE_SOLVES = 30
N_BOOTSTRAP = 500
ALPHA_SIG = 0.05
SPLIT_PRIMARY = "quartile"
SPLIT_ROBUSTNESS = ["tercile", "quintile"]
SHAPE_SHIFT_THRESHOLD = 20     # % relative change to count as a shape shift
SCALE_SHIFT_THRESHOLD = 20     # same idea but for scale features

USE_FULL_COHORT = "--full" in sys.argv
SKIP_ROBUSTNESS = "--no-robustness" in sys.argv

if "--bootstrap-n" in sys.argv:
    _idx = sys.argv.index("--bootstrap-n")
    N_BOOTSTRAP = int(sys.argv[_idx + 1])

DIST_COLORS = {
    "gaussian": "#e74c3c",
    "lognormal": "#2980b9",
    "exgaussian": "#27ae60",
}

# gaussian: (mu, sigma) = 2 free params
# lognormal: (s, scale) with loc pinned to 0 = 2 free params
# exgaussian: (K, loc, scale) = 3 free params
DIST_K = {"gaussian": 2, "lognormal": 2, "exgaussian": 3}


# ──────────────────────────────────────────────
#  Distribution fitting
# ──────────────────────────────────────────────

def _safe_fit_gaussian(data):
    try:
        mu, sigma = norm.fit(data)
        if sigma &lt;= 0:
            return None, -np.inf
        ll = float(np.sum(norm.logpdf(data, loc=mu, scale=sigma)))
        return (mu, sigma), ll
    except Exception:
        return None, -np.inf


def _safe_fit_lognormal(data):
    try:
        d = data[data &gt; 0]
        if len(d) &lt; 10:
            return None, -np.inf
        s, loc, scale = lognorm.fit(d, floc=0)
        if s &lt;= 0 or scale &lt;= 0:
            return None, -np.inf
        ll = float(np.sum(lognorm.logpdf(d, s, loc=0, scale=scale)))
        if not np.isfinite(ll):
            return None, -np.inf
        return (s, 0.0, scale), ll
    except Exception:
        return None, -np.inf


def _exgaussian_mom_init(data):
    """
    Method-of-moments starting values for ex-Gaussian (K, loc, scale).

    The ex-Gaussian has mean = mu + tau, var = sigma^2 + tau^2,
    skew_3 = 2*tau^3 / (sigma^2 + tau^2)^{3/2}.  We solve approximately
    and convert to scipy's parameterisation where K = tau/sigma.
    """
    m = float(np.mean(data))
    v = float(np.var(data, ddof=1))
    s = float(compute_skew(data, bias=False))

    # ex-Gaussian needs positive skew
    s = max(s, 0.1)

    # rough tau estimate from the skewness equation
    tau_est = max((s * v ** 1.5 / 2.0) ** (1.0 / 3.0), 0.01)
    sigma_est = max(np.sqrt(max(v - tau_est ** 2, 0.01)), 0.01)
    mu_est = m - tau_est

    K_est = tau_est / sigma_est
    return K_est, mu_est, sigma_est


def _safe_fit_exgaussian(data):
    """Fit ex-Gaussian trying several initialisation strategies."""
    d = data[data &gt; 0].astype(float)
    if len(d) &lt; 20:
        return None, -np.inf

    best_params = None
    best_ll = -np.inf

    # strategy 1: method-of-moments init
    try:
        K0, loc0, scale0 = _exgaussian_mom_init(d)
        K, loc, scale = exponnorm.fit(d, K0, loc=loc0, scale=scale0)
        if K &gt; 0 and scale &gt; 0:
            ll = float(np.sum(exponnorm.logpdf(d, K, loc=loc, scale=scale)))
            if np.isfinite(ll) and ll &gt; best_ll:
                best_params = (K, loc, scale)
                best_ll = ll
    except Exception:
        pass

    # strategy 2: let scipy figure it out on its own
    try:
        K, loc, scale = exponnorm.fit(d)
        if K &gt; 0 and scale &gt; 0:
            ll = float(np.sum(exponnorm.logpdf(d, K, loc=loc, scale=scale)))
            if np.isfinite(ll) and ll &gt; best_ll:
                best_params = (K, loc, scale)
                best_ll = ll
    except Exception:
        pass

    # strategy 3: sweep over a few K starting values
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1))
    for K_try in [0.5, 1.0, 2.0, 3.0, 5.0]:
        try:
            K, loc, scale = exponnorm.fit(d, K_try, loc=m - s * 0.5, scale=s * 0.5)
            if K &gt; 0 and scale &gt; 0:
                ll = float(np.sum(exponnorm.logpdf(d, K, loc=loc, scale=scale)))
                if np.isfinite(ll) and ll &gt; best_ll:
                    best_params = (K, loc, scale)
                    best_ll = ll
        except Exception:
            pass

    # strategy 4: fix K, optimise loc/scale, then unfreeze
    for K_fix in [0.5, 1.0, 2.0]:
        try:
            K, loc, scale = exponnorm.fit(d, f0=K_fix)
            K2, loc2, scale2 = exponnorm.fit(d, K, loc=loc, scale=scale)
            if K2 &gt; 0 and scale2 &gt; 0:
                ll = float(np.sum(exponnorm.logpdf(d, K2, loc=loc2, scale=scale2)))
                if np.isfinite(ll) and ll &gt; best_ll:
                    best_params = (K2, loc2, scale2)
                    best_ll = ll
        except Exception:
            pass

    if best_params is None:
        return None, -np.inf
    return best_params, best_ll


FITTERS = {
    "gaussian": _safe_fit_gaussian,
    "lognormal": _safe_fit_lognormal,
    "exgaussian": _safe_fit_exgaussian,
}

SCIPY_DISTS = {
    "gaussian": norm,
    "lognormal": lognorm,
    "exgaussian": exponnorm,
}


def _params_to_args(dist_name, params):
    """Convert our stored param tuples into scipy's (args, kwargs) form."""
    if dist_name == "gaussian":
        return (), {"loc": params[0], "scale": params[1]}
    elif dist_name == "lognormal":
        return (params[0],), {"loc": params[1], "scale": params[2]}
    elif dist_name == "exgaussian":
        return (params[0],), {"loc": params[1], "scale": params[2]}
    return (), {}


def _exgaussian_components(K, loc, scale):
    """Unpack scipy's (K, loc, scale) into the mu/sigma/tau parameterisation."""
    mu = loc
    sigma = scale
    tau = K * scale
    return mu, sigma, tau


def _lognormal_derived(s, scale):
    """Compute distribution-level stats from the lognormal shape and scale."""
    mu_underlying = float(np.log(scale))
    sigma_underlying = float(s)

    dist_mean = float(np.exp(mu_underlying + sigma_underlying ** 2 / 2))
    dist_var = float(
        (np.exp(sigma_underlying ** 2) - 1)
        * np.exp(2 * mu_underlying + sigma_underlying ** 2)
    )
    dist_skew = float(
        (np.exp(sigma_underlying ** 2) + 2)
        * np.sqrt(np.exp(sigma_underlying ** 2) - 1)
    )
    return {
        "s": sigma_underlying,
        "mu_log": mu_underlying,
        "dist_mean": dist_mean,
        "dist_var": dist_var,
        "dist_skew": dist_skew,
    }


# ──────────────────────────────────────────────
#  Model comparison (AIC / AICc)
# ──────────────────────────────────────────────

def compute_aic(ll, k):
    return 2.0 * k - 2.0 * ll


def compute_aicc(ll, k, n):
    aic = compute_aic(ll, k)
    if n - k - 1 &gt; 0:
        return aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    return aic


def fit_all_distributions(data):
    """Fit all three candidate distributions and pick the AIC winner."""
    n = len(data)
    results = {}

    for dname, fitter in FITTERS.items():
        params, ll = fitter(data)
        k = DIST_K[dname]
        if params is None:
            results[dname] = {
                "params": None, "ll": -np.inf, "aic": np.inf,
                "aicc": np.inf, "k": k, "fit_ok": False,
            }
        else:
            results[dname] = {
                "params": params, "ll": ll,
                "aic": compute_aic(ll, k),
                "aicc": compute_aicc(ll, k, n),
                "k": k, "fit_ok": True,
            }

    # AIC-preferred
    valid = {d: r for d, r in results.items() if r["fit_ok"]}
    if valid:
        pref = min(valid, key=lambda d: valid[d]["aic"])
        delta_min = valid[pref]["aic"]
        for d in results:
            results[d]["delta_aic"] = results[d]["aic"] - delta_min
    else:
        pref = "none"
        for d in results:
            results[d]["delta_aic"] = np.nan

    # also track which one has the best raw log-likelihood (no penalty)
    ll_valid = {d: r for d, r in results.items() if r["fit_ok"]}
    if ll_valid:
        ll_pref = max(ll_valid, key=lambda d: ll_valid[d]["ll"])
    else:
        ll_pref = "none"

    results["_preferred"] = pref
    results["_ll_preferred"] = ll_pref
    results["_n"] = n
    return results


# ──────────────────────────────────────────────
#  Bootstrap goodness-of-fit test
# ──────────────────────────────────────────────

def bootstrap_ks_gof(data, dist_name, params, n_boot=N_BOOTSTRAP):
    """
    Parametric bootstrap KS test: simulate from the fitted distribution,
    refit each time, and see how often the simulated KS stat exceeds the
    observed one.
    """
    dist = SCIPY_DISTS[dist_name]
    args, kwargs = _params_to_args(dist_name, params)

    def cdf_func(x):
        return dist.cdf(x, *args, **kwargs)

    ks_obs_val, _ = kstest(data, cdf_func)

    n = len(data)
    count = 0
    valid_boots = 0

    for _ in range(n_boot):
        sim = dist.rvs(*args, **kwargs, size=n)
        try:
            if dist_name == "gaussian":
                sp = norm.fit(sim)
                sim_cdf = lambda x, _sp=sp: norm.cdf(x, loc=_sp[0], scale=_sp[1])

            elif dist_name == "lognormal":
                sim_pos = sim[sim &gt; 0]
                if len(sim_pos) &lt; 10:
                    continue
                sp = lognorm.fit(sim_pos, floc=0)
                sim_cdf = lambda x, _sp=sp: lognorm.cdf(x, _sp[0], loc=0, scale=_sp[2])
                sim = sim_pos

            elif dist_name == "exgaussian":
                sim_pos = sim[sim &gt; 0]
                if len(sim_pos) &lt; 20:
                    continue
                sp = exponnorm.fit(
                    sim_pos, params[0], loc=params[1], scale=params[2]
                )
                if sp[0] &lt;= 0 or sp[2] &lt;= 0:
                    continue
                sim_cdf = lambda x, _sp=sp: exponnorm.cdf(
                    x, _sp[0], loc=_sp[1], scale=_sp[2]
                )
                sim = sim_pos
            else:
                continue

            ks_sim, _ = kstest(sim, sim_cdf)
            valid_boots += 1
            if ks_sim &gt;= ks_obs_val:
                count += 1

        except Exception:
            continue

    p_val = count / max(valid_boots, 1)
    return float(ks_obs_val), float(p_val)


# ──────────────────────────────────────────────
#  Normalised two-sample KS test
# ──────────────────────────────────────────────

def normalised_ks_test(data_early, data_late):
    """
    Z-score both samples independently, then run a two-sample KS.
    This tests whether the *shape* changed, stripping out any
    location/scale drift that we'd expect from learning anyway.
    """
    def zscore(x):
        m = np.mean(x)
        s = np.std(x, ddof=1)
        if s &lt; 1e-12:
            return x - m
        return (x - m) / s

    z_early = zscore(data_early)
    z_late = zscore(data_late)
    stat, p = ks_2samp(z_early, z_late)
    return float(stat), float(p)


# ──────────────────────────────────────────────
#  Descriptive features for a single phase
# ──────────────────────────────────────────────

def compute_features(data):
    n = len(data)
    if n &lt; 5:
        return {k: np.nan for k in [
            "n", "mean", "median", "std", "var", "cv",
            "skewness", "excess_kurtosis", "iqr",
            "p5", "p10", "p25", "p75", "p90", "p95",
            "p90_p10_ratio", "range", "min", "max",
            "tail_weight_upper", "tail_weight_lower",
        ]}

    q = np.percentile(data, [5, 10, 25, 50, 75, 90, 95])
    iqr = q[4] - q[2]
    mean_val = float(np.mean(data))
    std_val = float(np.std(data, ddof=1))

    return {
        "n": n,
        "mean": mean_val,
        "median": float(q[3]),
        "std": std_val,
        "var": float(std_val ** 2),
        "cv": float(std_val / mean_val) if mean_val &gt; 0 else np.nan,
        "skewness": float(compute_skew(data, bias=False)),
        "excess_kurtosis": float(compute_kurtosis(data, bias=False)),
        "iqr": float(iqr),
        "p5": float(q[0]),
        "p10": float(q[1]),
        "p25": float(q[2]),
        "p75": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
        "p90_p10_ratio": float(q[5] / q[1]) if q[1] &gt; 0 else np.nan,
        "range": float(np.ptp(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "tail_weight_upper": float((q[6] - q[3]) / iqr) if iqr &gt; 0 else np.nan,
        "tail_weight_lower": float((q[3] - q[0]) / iqr) if iqr &gt; 0 else np.nan,
    }


# ──────────────────────────────────────────────
#  Career-phase splitting
# ──────────────────────────────────────────────

def split_career(solver_df, method="quartile"):
    df = solver_df.sort_values("solve_number").reset_index(drop=True)
    n = len(df)

    if method == "quartile":
        q = 4
        labels = ["Q1_early", "Q2_mid_early", "Q3_mid_late", "Q4_late"]
    elif method == "tercile":
        q = 3
        labels = ["T1_early", "T2_mid", "T3_late"]
    elif method == "quintile":
        q = 5
        labels = ["P1", "P2", "P3", "P4", "P5"]
    else:
        raise ValueError(f"Unknown split method: {method}")

    cuts = np.linspace(0, n, q + 1, dtype=int)
    phases = []
    for i in range(q):
        chunk = df.iloc[cuts[i]:cuts[i + 1]]
        if len(chunk) &gt;= MIN_PHASE_SOLVES:
            phases.append((labels[i], chunk))

    return phases


# ──────────────────────────────────────────────
#  Data loading (mirrors RQ1)
# ──────────────────────────────────────────────

def find_tsv(pattern_parts):
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

    results_333 = results[results[event_col] == "333"].copy()
    log.info(f"  3x3x3 results: {len(results_333):,}")
    del results

    # competitions
    comps = pd.read_csv(comps_path, sep="\t", low_memory=False)
    comps.columns = [c.strip().lower() for c in comps.columns]
    comp_id_col_c = next(
        (c for c in ["id", "competition_id"] if c in comps.columns), None
    )
    if "start_date" in comps.columns:
        comps["comp_date"] = pd.to_datetime(comps["start_date"], errors="coerce")
    elif {"year", "month", "day"}.issubset(comps.columns):
        comps["comp_date"] = pd.to_datetime(
            comps[["year", "month", "day"]], errors="coerce"
        )
    else:
        raise KeyError("Cannot determine competition date.")

    comps_slim = comps[[comp_id_col_c, "comp_date"]].rename(
        columns={comp_id_col_c: "_comp_id"}
    )

    # round types
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

    # attempts
    attempts = pd.read_csv(attempts_path, sep="\t", low_memory=False)
    attempts.columns = [c.strip().lower() for c in attempts.columns]
    att_result_id_col = next(
        (c for c in ["result_id", "id"] if c in attempts.columns), None
    )
    att_number_col = next(
        (c for c in ["attempt_number", "attempt_num", "num"]
         if c in attempts.columns), None
    )
    att_value_col = next(
        (c for c in ["attempt_result", "value", "result"]
         if c in attempts.columns), None
    )

    # merge everything
    merged = results_333.merge(
        attempts, left_on=result_id_col, right_on=att_result_id_col,
        how="inner", suffixes=("", "_att"),
    )
    del results_333, attempts

    merged = merged.merge(comps_slim, left_on=comp_col, right_on="_comp_id", how="left")
    merged = merged.merge(rt_slim, left_on=round_col, right_on="_rt_id", how="left")

    # drop invalid (DNF/DNS) and convert centiseconds to seconds
    merged = merged[merged[att_value_col] &gt; 0].copy()
    merged["solve_time_s"] = merged[att_value_col] / 100.0

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
        (stats["n_competitions"] &gt; MIN_COMPETITIONS)
        &amp; (stats["n_solves"] &gt; MIN_SOLVES)
        &amp; (stats["career_years"] &gt; MIN_CAREER_YEARS)
    ].copy()
    log.info(
        f"Cohort selection: {len(eligible)} solvers pass all thresholds "
        f"(from {len(stats)} total)"
    )

    last_avg = (
        solves[solves["solver_id"].isin(eligible["solver_id"])]
        .sort_values(["solver_id", "solve_number"])
        .groupby("solver_id").tail(12)
        .groupby("solver_id")["solve_time_s"].mean()
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
    for tier in tier_labels:
        grp = sampled[sampled["skill_tier"] == tier]
        if len(grp):
            log.info(
                f"    {str(tier):20s}  n={len(grp):3d}  "
                f"last_avg=[{grp['last_avg_s'].min():.1f},{grp['last_avg_s'].max():.1f}]s  "
                f"solves=[{grp['n_solves'].min()},{grp['n_solves'].max()}]"
            )
    return sampled


def load_or_build_cohort(solves):
    """Try to reuse the RQ1 cohort; fall back to building one from scratch."""
    rq1_cohort_path = RQ1_DIR / "cohort_info.csv"
    if rq1_cohort_path.exists():
        log.info(f"Loading RQ1 cohort from {rq1_cohort_path} ...")
        cohort_info = pd.read_csv(rq1_cohort_path)
        ids = cohort_info["solver_id"].tolist()
        solves_cohort = solves[solves["solver_id"].isin(ids)].copy()
        log.info(f"  Loaded {len(cohort_info)} solvers from RQ1 cohort.")
        return solves_cohort, cohort_info
    else:
        log.info("RQ1 cohort not found; building cohort independently ...")
        solves_cohort, cohort_info = select_cohort(solves)
        if USE_FULL_COHORT:
            return solves_cohort, cohort_info
        sampled = stratified_sample(cohort_info)
        filtered = solves_cohort[
            solves_cohort["solver_id"].isin(sampled["solver_id"])
        ].copy()
        return filtered, sampled


# ──────────────────────────────────────────────
#  Per-solver analysis pipeline
# ──────────────────────────────────────────────

def analyse_solver_phase(solver_id, phase_label, data):
    """Fit distributions and compute features for one career phase."""
    t = data["solve_time_s"].values.astype(float)
    dist_results = fit_all_distributions(t)
    preferred = dist_results["_preferred"]
    ll_preferred = dist_results["_ll_preferred"]
    features = compute_features(t)

    # bootstrap goodness-of-fit for each distribution
    gof = {}
    for dname in ["gaussian", "lognormal", "exgaussian"]:
        dr = dist_results[dname]
        if dr["fit_ok"]:
            ks_stat, ks_p = bootstrap_ks_gof(t, dname, dr["params"], n_boot=N_BOOTSTRAP)
            gof[dname] = {"ks_stat": ks_stat, "ks_p": ks_p, "reject": ks_p &lt; ALPHA_SIG}
        else:
            gof[dname] = {"ks_stat": np.nan, "ks_p": np.nan, "reject": None}

    # ex-Gaussian decomposition (track even when lognormal wins on AIC)
    exg_components = None
    if dist_results["exgaussian"]["fit_ok"]:
        K, loc, scale = dist_results["exgaussian"]["params"]
        mu_eg, sigma_eg, tau_eg = _exgaussian_components(K, loc, scale)
        tau_total = sigma_eg + tau_eg
        exg_components = {
            "mu": mu_eg,
            "sigma": sigma_eg,
            "tau": tau_eg,
            "tau_frac": tau_eg / tau_total if tau_total &gt; 0 else np.nan,
        }

    # lognormal derived stats (same idea — track regardless of winner)
    ln_derived = None
    if dist_results["lognormal"]["fit_ok"]:
        s_ln, _, scale_ln = dist_results["lognormal"]["params"]
        ln_derived = _lognormal_derived(s_ln, scale_ln)

    return {
        "solver_id": solver_id,
        "phase": phase_label,
        "n_solves": len(t),
        "preferred_dist": preferred,
        "ll_preferred_dist": ll_preferred,
        "dist_results": dist_results,
        "features": features,
        "gof": gof,
        "exg_components": exg_components,
        "ln_derived": ln_derived,
    }


def _compute_deltas(first_feats, last_feats, feat_list):
    """Absolute and relative change for each feature between first and last phase."""
    deltas = {}
    for feat in feat_list:
        v1 = first_feats.get(feat, np.nan)
        v2 = last_feats.get(feat, np.nan)
        if np.isfinite(v1) and np.isfinite(v2):
            deltas[f"{feat}_first"] = v1
            deltas[f"{feat}_last"] = v2
            deltas[f"{feat}_delta"] = v2 - v1
            if abs(v1) &gt; 1e-9:
                deltas[f"{feat}_rel_change"] = (v2 - v1) / abs(v1) * 100
            else:
                deltas[f"{feat}_rel_change"] = np.nan
        else:
            deltas[f"{feat}_first"] = np.nan
            deltas[f"{feat}_last"] = np.nan
            deltas[f"{feat}_delta"] = np.nan
            deltas[f"{feat}_rel_change"] = np.nan
    return deltas


def analyse_solver(solver_id, solver_df, split_method="quartile"):
    """Full distributional analysis for one solver across career phases."""
    phases = split_career(solver_df, method=split_method)
    if len(phases) &lt; 2:
        log.warning(f"  {solver_id}: fewer than 2 phases ({split_method}), skipping.")
        return None

    phase_results = []
    for label, phase_df in phases:
        pr = analyse_solver_phase(solver_id, label, phase_df)
        phase_results.append(pr)

    first = phase_results[0]
    last = phase_results[-1]

    family_changed = first["preferred_dist"] != last["preferred_dist"]
    ll_family_changed = first["ll_preferred_dist"] != last["ll_preferred_dist"]

    # feature deltas between first and last phase
    all_feats = [
        "mean", "std", "var", "cv", "skewness", "excess_kurtosis",
        "iqr", "p90_p10_ratio", "tail_weight_upper",
    ]
    feature_deltas = _compute_deltas(first["features"], last["features"], all_feats)

    # separate shape shifts from scale shifts
    shape_feats = ["skewness", "excess_kurtosis", "tail_weight_upper", "cv"]
    scale_feats = ["var", "std", "mean"]

    shape_shift = any(
        abs(feature_deltas.get(f"{f}_rel_change", 0)) &gt; SHAPE_SHIFT_THRESHOLD
        for f in shape_feats
        if np.isfinite(feature_deltas.get(f"{f}_rel_change", np.nan))
    )
    scale_shift = any(
        abs(feature_deltas.get(f"{f}_rel_change", 0)) &gt; SCALE_SHIFT_THRESHOLD
        for f in scale_feats
        if np.isfinite(feature_deltas.get(f"{f}_rel_change", np.nan))
    )

    # normalised KS — tests shape change after removing location/scale
    t_first = phases[0][1]["solve_time_s"].values.astype(float)
    t_last = phases[-1][1]["solve_time_s"].values.astype(float)
    norm_ks_stat, norm_ks_p = normalised_ks_test(t_first, t_last)
    norm_ks_reject = norm_ks_p &lt; ALPHA_SIG

    # track lognormal shape parameter across phases
    ln_s_values = []
    for pr in phase_results:
        if pr["ln_derived"] is not None:
            ln_s_values.append(pr["ln_derived"]["s"])
        else:
            ln_s_values.append(np.nan)

    # track ex-Gaussian tau fraction across phases
    tau_fracs = []
    for pr in phase_results:
        if pr["exg_components"] is not None:
            tau_fracs.append(pr["exg_components"]["tau_frac"])
        else:
            tau_fracs.append(np.nan)

    # composite: did *any* kind of distributional shift happen?
    shift_detected = family_changed or shape_shift or norm_ks_reject

    # how many phases had a successful ex-Gaussian fit?
    exg_fit_ok = sum(
        1 for pr in phase_results if pr["dist_results"]["exgaussian"]["fit_ok"]
    )

    return {
        "solver_id": solver_id,
        "split_method": split_method,
        "n_phases": len(phase_results),
        "phase_results": phase_results,
        "family_changed": family_changed,
        "ll_family_changed": ll_family_changed,
        "pref_first": first["preferred_dist"],
        "pref_last": last["preferred_dist"],
        "ll_pref_first": first["ll_preferred_dist"],
        "ll_pref_last": last["ll_preferred_dist"],
        "shape_shift": shape_shift,
        "scale_shift": scale_shift,
        "norm_ks_stat": norm_ks_stat,
        "norm_ks_p": norm_ks_p,
        "norm_ks_reject": norm_ks_reject,
        "shift_detected": shift_detected,
        "feature_deltas": feature_deltas,
        "ln_s_values": ln_s_values,
        "tau_fracs": tau_fracs,
        "exg_fit_ok_count": exg_fit_ok,
    }


# ──────────────────────────────────────────────
#  Plotting — per solver
# ──────────────────────────────────────────────

def plot_solver_full(solver_id, solver_df, summary, out_dir):
    phases_data = split_career(solver_df, method=summary["split_method"])
    phase_results = summary["phase_results"]
    n_ph = len(phase_results)

    fig, axes = plt.subplots(2, n_ph, figsize=(5 * n_ph, 9))
    if n_ph == 1:
        axes = axes.reshape(2, 1)

    for j, ((label, pdf), pr) in enumerate(zip(phases_data, phase_results)):
        t = pdf["solve_time_s"].values.astype(float)
        ax_hist = axes[0, j]
        ax_qq = axes[1, j]

        # histogram with fitted PDFs overlaid
        ax_hist.hist(
            t, bins=50, density=True, color="grey", alpha=0.4,
            edgecolor="black", linewidth=0.3, label="data",
        )
        x_grid = np.linspace(max(t.min() - 0.5, 0.01), t.max() + 0.5, 300)

        for dname in ["gaussian", "lognormal", "exgaussian"]:
            dr = pr["dist_results"][dname]
            if not dr["fit_ok"]:
                continue
            dist = SCIPY_DISTS[dname]
            args, kwargs = _params_to_args(dname, dr["params"])
            y_pdf = dist.pdf(x_grid, *args, **kwargs)
            is_best = (dname == pr["preferred_dist"])
            ax_hist.plot(
                x_grid, y_pdf,
                color=DIST_COLORS[dname],
                lw=2.5 if is_best else 1.0,
                ls="-" if is_best else "--",
                label=f"{dname} AIC={dr['aic']:.0f}{' *' if is_best else ''}",
            )

        ax_hist.set_title(
            f"{pr['phase']}  (n={pr['n_solves']})\n"
            f"skew={pr['features']['skewness']:.2f}  "
            f"kurt={pr['features']['excess_kurtosis']:.2f}",
            fontsize=9,
        )
        ax_hist.set_xlabel("Solve time (s)", fontsize=8)
        if j == 0:
            ax_hist.set_ylabel("Density", fontsize=8)
        ax_hist.legend(fontsize=5.5, loc="upper right")
        ax_hist.tick_params(labelsize=7)

        # Q-Q plot for the AIC-preferred distribution
        pref_name = pr["preferred_dist"]
        if pref_name != "none" and pr["dist_results"][pref_name]["fit_ok"]:
            dr_pref = pr["dist_results"][pref_name]
            dist_pref = SCIPY_DISTS[pref_name]
            args_p, kwargs_p = _params_to_args(pref_name, dr_pref["params"])
            t_sorted = np.sort(t)
            n_pts = len(t_sorted)
            theo = dist_pref.ppf(
                (np.arange(1, n_pts + 1) - 0.5) / n_pts, *args_p, **kwargs_p
            )
            ax_qq.scatter(
                theo, t_sorted, s=4, alpha=0.3,
                color=DIST_COLORS[pref_name], rasterized=True,
            )
            lims = [min(theo.min(), t_sorted.min()), max(theo.max(), t_sorted.max())]
            ax_qq.plot(lims, lims, "k--", lw=0.8)
            ax_qq.set_title(f"Q-Q: {pref_name}", fontsize=9)
        else:
            ax_qq.set_title("Q-Q: N/A", fontsize=9)

        ax_qq.set_xlabel("Theoretical quantiles", fontsize=8)
        if j == 0:
            ax_qq.set_ylabel("Sample quantiles", fontsize=8)
        ax_qq.tick_params(labelsize=7)

    # build suptitle showing which kinds of shifts were detected
    shift_label = []
    if summary["shape_shift"]:
        shift_label.append("shape")
    if summary["scale_shift"]:
        shift_label.append("scale")
    if summary["norm_ks_reject"]:
        shift_label.append("norm-KS")
    if summary["family_changed"]:
        shift_label.append("family")
    sl = ", ".join(shift_label) if shift_label else "none"

    fig.suptitle(
        f"Solver {solver_id} \u2014 Distributional Evolution  [shifts: {sl}]",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"dist_{solver_id}.png", dpi=150)
    plt.close(fig)


def plot_feature_evolution(solver_id, summary, out_dir):
    phases = summary["phase_results"]
    labels = [p["phase"] for p in phases]

    feats_to_plot = [
        "mean", "std", "cv", "skewness",
        "excess_kurtosis", "iqr", "p90_p10_ratio", "tail_weight_upper",
    ]
    feat_display = {
        "mean": "Mean (s)", "std": "Std Dev (s)", "cv": "CV",
        "skewness": "Skewness", "excess_kurtosis": "Excess Kurtosis",
        "iqr": "IQR (s)", "p90_p10_ratio": "P90/P10",
        "tail_weight_upper": "Upper tail wt",
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    x = np.arange(len(labels))

    for idx, feat in enumerate(feats_to_plot):
        ax = axes[idx]
        vals = [p["features"].get(feat, np.nan) for p in phases]
        ax.plot(x, vals, "o-", color="#34495e", lw=1.8, ms=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=25)
        ax.set_title(feat_display.get(feat, feat), fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Feature Evolution \u2014 Solver {solver_id}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"feat_{solver_id}.png", dpi=140)
    plt.close(fig)


def plot_param_evolution(solver_id, summary, out_dir):
    """Lognormal s, ex-Gaussian tau fraction, and delta-AIC across phases."""
    phases = summary["phase_results"]
    labels = [p["phase"] for p in phases]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # lognormal shape parameter
    ax = axes[0]
    ax.plot(x, summary["ln_s_values"], "o-", color="#2980b9", lw=2, ms=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=25)
    ax.set_title("Lognormal shape (s)", fontsize=10)
    ax.set_ylabel("s")
    ax.grid(True, alpha=0.3)

    # ex-Gaussian tau fraction
    ax = axes[1]
    tf = summary["tau_fracs"]
    valid_mask = [np.isfinite(v) for v in tf]
    if any(valid_mask):
        ax.plot(x, tf, "o-", color="#27ae60", lw=2, ms=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=25)
    ax.set_title("Ex-Gaussian \u03c4/(\u03c3+\u03c4)", fontsize=10)
    ax.set_ylabel("\u03c4 fraction")
    ax.grid(True, alpha=0.3)

    # delta AIC: exgaussian minus lognormal
    ax = axes[2]
    daic = []
    for pr in phases:
        dr = pr["dist_results"]
        if dr["exgaussian"]["fit_ok"] and dr["lognormal"]["fit_ok"]:
            daic.append(dr["exgaussian"]["aic"] - dr["lognormal"]["aic"])
        else:
            daic.append(np.nan)
    bar_colors = ["#27ae60" if v &lt; 0 else "#e74c3c" for v in daic]
    ax.bar(x, daic, color=bar_colors, edgecolor="black", width=0.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=25)
    ax.set_title("\u0394AIC (ExGauss \u2212 LogN)\n&lt;0 = ExGauss better", fontsize=10)
    ax.set_ylabel("\u0394AIC")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Parameter Evolution \u2014 Solver {solver_id}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"params_{solver_id}.png", dpi=140)
    plt.close(fig)


# ──────────────────────────────────────────────
#  Plotting — summary level
# ──────────────────────────────────────────────

def _phase_sort_key(label):
    """Extract the number from a phase label for sorting."""
    return int("".join(filter(str.isdigit, label)) or "0")


def plot_preference_by_phase(all_summaries, out_dir):
    rows = []
    for s in all_summaries:
        for pr in s["phase_results"]:
            rows.append({
                "phase": pr["phase"],
                "preferred_aic": pr["preferred_dist"],
                "preferred_ll": pr["ll_preferred_dist"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in zip(
        axes,
        ["preferred_aic", "preferred_ll"],
        ["AIC-Preferred", "Best Log-Likelihood (no penalty)"],
    ):
        ct = pd.crosstab(df["phase"], df[col])
        ct = ct.reindex(columns=["gaussian", "lognormal", "exgaussian"], fill_value=0)
        phase_order = sorted(ct.index, key=_phase_sort_key)
        ct = ct.reindex(phase_order)
        ct.plot.bar(
            stacked=True, ax=ax,
            color=[DIST_COLORS.get(c, "grey") for c in ct.columns],
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xlabel("Career Phase")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(title="Distribution", fontsize=7)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(out_dir / "summary_preference_by_phase.png", dpi=150)
    plt.close(fig)


def plot_feature_evolution_summary(all_summaries, out_dir):
    feats = [
        "mean", "std", "cv", "skewness",
        "excess_kurtosis", "iqr", "p90_p10_ratio", "tail_weight_upper",
    ]
    feat_display = {
        "mean": "Mean (s)", "std": "Std Dev (s)", "cv": "CV",
        "skewness": "Skewness", "excess_kurtosis": "Excess Kurtosis",
        "iqr": "IQR (s)", "p90_p10_ratio": "P90/P10",
        "tail_weight_upper": "Upper tail wt",
    }

    # collect all phase labels across solvers
    phase_labels = []
    for s in all_summaries:
        for pr in s["phase_results"]:
            if pr["phase"] not in phase_labels:
                phase_labels.append(pr["phase"])
    phase_labels = sorted(phase_labels, key=_phase_sort_key)

    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    axes = axes.flatten()

    for idx, feat in enumerate(feats):
        ax = axes[idx]
        phase_vals = {pl: [] for pl in phase_labels}
        for s in all_summaries:
            for pr in s["phase_results"]:
                v = pr["features"].get(feat, np.nan)
                if np.isfinite(v) and pr["phase"] in phase_vals:
                    phase_vals[pr["phase"]].append(v)

        medians = [
            np.median(phase_vals[pl]) if phase_vals[pl] else np.nan
            for pl in phase_labels
        ]
        q25 = [
            np.percentile(phase_vals[pl], 25) if len(phase_vals[pl]) &gt; 2 else np.nan
            for pl in phase_labels
        ]
        q75 = [
            np.percentile(phase_vals[pl], 75) if len(phase_vals[pl]) &gt; 2 else np.nan
            for pl in phase_labels
        ]
        x = np.arange(len(phase_labels))
        ax.plot(x, medians, "o-", color="#2c3e50", lw=2, ms=6)
        ax.fill_between(x, q25, q75, alpha=0.2, color="#3498db")
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=7, rotation=25)
        ax.set_title(feat_display.get(feat, feat), fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Solver Feature Evolution (median \u00b1 IQR)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "summary_feature_evolution.png", dpi=150)
    plt.close(fig)


def plot_shift_detection_summary(all_summaries, out_dir):
    nt = len(all_summaries)
    n_shift = sum(1 for s in all_summaries if s["shift_detected"])
    n_family = sum(1 for s in all_summaries if s["family_changed"])
    n_shape = sum(1 for s in all_summaries if s["shape_shift"])
    n_scale = sum(1 for s in all_summaries if s["scale_shift"])
    n_normks = sum(1 for s in all_summaries if s["norm_ks_reject"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # pie: shift detected or not
    ax = axes[0]
    ax.pie(
        [n_shift, nt - n_shift],
        labels=[f"Shift\n({n_shift})", f"No shift\n({nt - n_shift})"],
        autopct="%1.0f%%", startangle=90,
        colors=["#2ecc71", "#e74c3c"],
    )
    ax.set_title(
        "Distributional Shift Detection\n"
        "(shape OR norm-KS OR family; target \u2265 50%)"
    )

    # bar: breakdown by type
    ax = axes[1]
    categories = ["Family\nchanged", "Shape\nshift", "Scale\nshift", "Normalised\nKS reject"]
    values = [n_family, n_shape, n_scale, n_normks]
    colors = ["#9b59b6", "#f39c12", "#3498db", "#e74c3c"]
    bars = ax.bar(categories, values, color=colors, edgecolor="black", width=0.55)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val}/{nt}", ha="center", fontsize=11, fontweight="bold",
        )
    ax.set_ylabel("Number of Solvers")
    ax.set_title("Types of Distributional Shift")
    ax.set_ylim(0, max(values) * 1.3 + 1)

    plt.tight_layout()
    fig.savefig(out_dir / "summary_shift_detection.png", dpi=150)
    plt.close(fig)


def plot_gof_summary(all_summaries, out_dir):
    rows = []
    for s in all_summaries:
        for pr in s["phase_results"]:
            for dname in ["gaussian", "lognormal", "exgaussian"]:
                g = pr["gof"][dname]
                if g["reject"] is not None:
                    rows.append({
                        "dist": dname,
                        "phase": pr["phase"],
                        "not_rejected": not g["reject"],
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # overall acceptance rate per distribution
    ax = axes[0]
    rates = df.groupby("dist")["not_rejected"].mean() * 100
    rates = rates.reindex(["gaussian", "lognormal", "exgaussian"])
    bars = ax.bar(
        rates.index, rates.values,
        color=[DIST_COLORS[d] for d in rates.index],
        edgecolor="black", width=0.5,
    )
    for i, (bar, val) in enumerate(zip(bars, rates.values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold",
        )
        # show raw counts below the axis
        total = len(df[df["dist"] == rates.index[i]])
        accepted = int(total * val / 100)
        ax.text(i, -5, f"{accepted}/{total}", ha="center", fontsize=8, color="grey")

    ax.set_ylabel("% phases NOT rejected (bootstrap KS)")
    ax.set_title("Goodness-of-Fit Acceptance Rate")
    ax.set_ylim(-8, 110)

    # acceptance rate by phase
    ax = axes[1]
    phase_order = sorted(df["phase"].unique(), key=_phase_sort_key)
    for dname in ["gaussian", "lognormal", "exgaussian"]:
        sub = df[df["dist"] == dname]
        rates_p = sub.groupby("phase")["not_rejected"].mean() * 100
        rates_p = rates_p.reindex(phase_order)
        ax.plot(
            range(len(phase_order)), rates_p.values, "o-",
            color=DIST_COLORS[dname], lw=1.8, ms=6, label=dname,
        )
    ax.set_xticks(range(len(phase_order)))
    ax.set_xticklabels(phase_order, fontsize=8, rotation=25)
    ax.set_ylabel("% NOT rejected")
    ax.set_title("GoF Acceptance by Phase")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "summary_gof.png", dpi=150)
    plt.close(fig)


def plot_tau_evolution(all_summaries, out_dir):
    phase_labels = []
    for s in all_summaries:
        for pr in s["phase_results"]:
            if pr["phase"] not in phase_labels:
                phase_labels.append(pr["phase"])
    phase_labels = sorted(phase_labels, key=_phase_sort_key)

    phase_tau = {pl: [] for pl in phase_labels}
    for s in all_summaries:
        for pr in s["phase_results"]:
            if (pr["exg_components"] is not None
                    and np.isfinite(pr["exg_components"]["tau_frac"])):
                phase_tau[pr["phase"]].append(pr["exg_components"]["tau_frac"])

    medians = [
        np.median(phase_tau[pl]) if phase_tau[pl] else np.nan
        for pl in phase_labels
    ]
    q25 = [
        np.percentile(phase_tau[pl], 25) if len(phase_tau[pl]) &gt; 2 else np.nan
        for pl in phase_labels
    ]
    q75 = [
        np.percentile(phase_tau[pl], 75) if len(phase_tau[pl]) &gt; 2 else np.nan
        for pl in phase_labels
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(phase_labels))
    ax.plot(x, medians, "o-", color="#27ae60", lw=2, ms=7)
    ax.fill_between(x, q25, q75, alpha=0.2, color="#27ae60")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, rotation=25)
    ax.set_ylabel("\u03c4 / (\u03c3 + \u03c4)  (exponential tail fraction)")
    ax.set_title(
        "Ex-Gaussian Tail Fraction Across Career Phases\n"
        "(cross-solver median \u00b1 IQR)"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_tau_evolution.png", dpi=150)
    plt.close(fig)


def plot_lognormal_s_evolution(all_summaries, out_dir):
    phase_labels = []
    for s in all_summaries:
        for pr in s["phase_results"]:
            if pr["phase"] not in phase_labels:
                phase_labels.append(pr["phase"])
    phase_labels = sorted(phase_labels, key=_phase_sort_key)

    phase_s = {pl: [] for pl in phase_labels}
    for s in all_summaries:
        for pr in s["phase_results"]:
            if pr["ln_derived"] is not None:
                phase_s[pr["phase"]].append(pr["ln_derived"]["s"])

    medians = [
        np.median(phase_s[pl]) if phase_s[pl] else np.nan
        for pl in phase_labels
    ]
    q25 = [
        np.percentile(phase_s[pl], 25) if len(phase_s[pl]) &gt; 2 else np.nan
        for pl in phase_labels
    ]
    q75 = [
        np.percentile(phase_s[pl], 75) if len(phase_s[pl]) &gt; 2 else np.nan
        for pl in phase_labels
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(phase_labels))
    ax.plot(x, medians, "o-", color="#2980b9", lw=2, ms=7)
    ax.fill_between(x, q25, q75, alpha=0.2, color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels, rotation=25)
    ax.set_ylabel("Lognormal shape parameter (s)")
    ax.set_title(
        "Lognormal Shape (s) Across Career Phases\n"
        "(cross-solver median \u00b1 IQR)"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_lognormal_s_evolution.png", dpi=150)
    plt.close(fig)


def plot_norm_ks_distribution(all_summaries, out_dir):
    pvals = [
        s["norm_ks_p"] for s in all_summaries if np.isfinite(s["norm_ks_p"])
    ]
    stats_vals = [
        s["norm_ks_stat"] for s in all_summaries if np.isfinite(s["norm_ks_stat"])
    ]
    if not pvals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(pvals, bins=20, color="#e74c3c", edgecolor="black", alpha=0.7)
    ax.axvline(0.05, color="red", ls="--", lw=1.5, label="\u03b1=0.05")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title("Normalised KS Test p-values\n(Q1 vs Q4, z-scored)")
    ax.legend()
    n_sig = sum(1 for p in pvals if p &lt; 0.05)
    ax.text(
        0.95, 0.95, f"Sig: {n_sig}/{len(pvals)}",
        transform=ax.transAxes, va="top", ha="right", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    ax = axes[1]
    ax.hist(stats_vals, bins=20, color="#3498db", edgecolor="black", alpha=0.7)
    ax.set_xlabel("KS statistic")
    ax.set_ylabel("Count")
    ax.set_title("Normalised KS Test Statistics")

    plt.tight_layout()
    fig.savefig(out_dir / "summary_normalised_ks.png", dpi=150)
    plt.close(fig)


def plot_feature_shift_ranking(all_summaries, out_dir):
    feats = [
        "mean", "std", "var", "cv", "skewness",
        "excess_kurtosis", "iqr", "p90_p10_ratio", "tail_weight_upper",
    ]
    feat_display = {
        "mean": "Mean", "std": "Std Dev", "var": "Variance", "cv": "CV",
        "skewness": "Skewness", "excess_kurtosis": "Excess Kurt",
        "iqr": "IQR", "p90_p10_ratio": "P90/P10",
        "tail_weight_upper": "Upper tail wt",
    }
    shape_feats = {"skewness", "excess_kurtosis", "tail_weight_upper", "cv"}

    median_abs_rel = {}
    for feat in feats:
        vals = [
            abs(s["feature_deltas"].get(f"{feat}_rel_change", np.nan))
            for s in all_summaries
            if np.isfinite(s["feature_deltas"].get(f"{feat}_rel_change", np.nan))
        ]
        median_abs_rel[feat] = np.median(vals) if vals else 0

    sorted_feats = sorted(median_abs_rel, key=median_abs_rel.get, reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sorted_feats))
    vals_sorted = [median_abs_rel[f] for f in sorted_feats]
    labels = [feat_display.get(f, f) for f in sorted_feats]
    colors = ["#f39c12" if f in shape_feats else "#3498db" for f in sorted_feats]

    bars = ax.bar(x, vals_sorted, color=colors, edgecolor="black", width=0.6)
    for bar, val in zip(bars, vals_sorted):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Median |relative change| (%)")
    ax.set_title(
        "Feature Shift Ranking (first \u2192 last phase)\n"
        "orange = shape features, blue = scale features"
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_feature_shift_ranking.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
#  Results assembly
# ──────────────────────────────────────────────

def build_phase_records(all_summaries):
    rows = []
    for s in all_summaries:
        for pr in s["phase_results"]:
            row = {
                "solver_id": s["solver_id"],
                "split_method": s["split_method"],
                "phase": pr["phase"],
                "n_solves": pr["n_solves"],
                "preferred_dist": pr["preferred_dist"],
                "ll_preferred_dist": pr["ll_preferred_dist"],
            }

            for dname in ["gaussian", "lognormal", "exgaussian"]:
                dr = pr["dist_results"][dname]
                row[f"aic_{dname}"] = dr["aic"]
                row[f"aicc_{dname}"] = dr["aicc"]
                row[f"ll_{dname}"] = dr["ll"]
                row[f"fit_ok_{dname}"] = dr["fit_ok"]
                row[f"delta_aic_{dname}"] = dr["delta_aic"]
                gof = pr["gof"][dname]
                row[f"ks_stat_{dname}"] = gof["ks_stat"]
                row[f"ks_p_{dname}"] = gof["ks_p"]
                row[f"ks_reject_{dname}"] = gof["reject"]

            for dname in ["gaussian", "lognormal", "exgaussian"]:
                dr = pr["dist_results"][dname]
                if dr["params"]:
                    row[f"params_{dname}"] = json.dumps(list(dr["params"]))
                else:
                    row[f"params_{dname}"] = None

            if pr["exg_components"]:
                for k, v in pr["exg_components"].items():
                    row[f"exg_{k}"] = v
            if pr["ln_derived"]:
                for k, v in pr["ln_derived"].items():
                    row[f"ln_{k}"] = v

            for k, v in pr["features"].items():
                row[f"feat_{k}"] = v

            rows.append(row)

    return pd.DataFrame(rows)


def build_solver_records(all_summaries):
    rows = []
    for s in all_summaries:
        row = {
            "solver_id": s["solver_id"],
            "split_method": s["split_method"],
            "n_phases": s["n_phases"],
            "pref_first": s["pref_first"],
            "pref_last": s["pref_last"],
            "ll_pref_first": s.get("ll_pref_first"),
            "ll_pref_last": s.get("ll_pref_last"),
            "family_changed": s["family_changed"],
            "ll_family_changed": s.get("ll_family_changed", False),
            "shape_shift": s["shape_shift"],
            "scale_shift": s["scale_shift"],
            "norm_ks_stat": s["norm_ks_stat"],
            "norm_ks_p": s["norm_ks_p"],
            "norm_ks_reject": s["norm_ks_reject"],
            "shift_detected": s["shift_detected"],
            "exg_fit_ok_count": s.get("exg_fit_ok_count", 0),
        }
        row.update(s["feature_deltas"])
        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("RQ2 \u2014 Distributional Evolution  |  The Dynamics of Mastery")
    log.info(
        f"  v2.0  |  bootstrap={N_BOOTSTRAP}  "
        f"|  min_phase_solves={MIN_PHASE_SOLVES}"
    )
    log.info(
        f"  shape_threshold={SHAPE_SHIFT_THRESHOLD}%  "
        f"|  scale_threshold={SCALE_SHIFT_THRESHOLD}%"
    )
    log.info("=" * 60)

    solves = load_data()
    solves_cohort, cohort_info = load_or_build_cohort(solves)
    del solves  # free memory

    cohort_info.to_csv(OUTPUT_DIR / "cohort_info.csv", index=False)
    solver_ids = cohort_info["solver_id"].tolist()
    log.info(f"\nAnalysis cohort: {len(solver_ids)} solvers")
    log.info(
        f"  Solves \u2014 median {cohort_info['n_solves'].median():.0f}, "
        f"range [{cohort_info['n_solves'].min()},{cohort_info['n_solves'].max()}]"
    )

    grouped = solves_cohort.sort_values(
        ["solver_id", "solve_number"]
    ).groupby("solver_id")

    all_summaries = []

    for i, sid in enumerate(solver_ids, 1):
        log.info(f"[{i}/{len(solver_ids)}]  Analysing solver {sid} ({SPLIT_PRIMARY}) ...")
        sdf = grouped.get_group(sid)
        summary = analyse_solver(sid, sdf, split_method=SPLIT_PRIMARY)
        if summary is None:
            continue

        # attach cohort metadata
        ci = cohort_info[cohort_info["solver_id"] == sid].iloc[0]
        summary["n_competitions"] = ci.get("n_competitions", np.nan)
        summary["career_years"] = ci.get("career_years", np.nan)
        summary["last_avg_s"] = ci.get("last_avg_s", np.nan)
        if "skill_tier" in ci.index:
            summary["skill_tier"] = str(ci["skill_tier"])

        all_summaries.append(summary)

        # per-solver plots
        plot_solver_full(sid, sdf, summary, PLOTS_DIR)
        plot_feature_evolution(sid, summary, PLOTS_DIR)
        plot_param_evolution(sid, summary, PLOTS_DIR)

        # per-solver log
        fd = summary["feature_deltas"]
        log.info(
            f"  AIC: {summary['pref_first']} -&gt; {summary['pref_last']}  "
            f"LL: {summary['ll_pref_first']} -&gt; {summary['ll_pref_last']}  "
            f"ExG fits: {summary['exg_fit_ok_count']}/{summary['n_phases']}"
        )
        log.info(
            f"  Shape shift: {summary['shape_shift']}  "
            f"Scale shift: {summary['scale_shift']}  "
            f"Norm-KS: stat={summary['norm_ks_stat']:.4f} "
            f"p={summary['norm_ks_p']:.4f} rej={summary['norm_ks_reject']}"
        )
        log.info(
            f"  Skewness: {fd.get('skewness_first', np.nan):.3f} -&gt; "
            f"{fd.get('skewness_last', np.nan):.3f}  "
            f"(\u0394={fd.get('skewness_delta', np.nan):.3f}, "
            f"rel={fd.get('skewness_rel_change', np.nan):.1f}%)"
        )
        log.info(
            f"  Variance: {fd.get('var_first', np.nan):.2f} -&gt; "
            f"{fd.get('var_last', np.nan):.2f}  "
            f"(rel={fd.get('var_rel_change', np.nan):.1f}%)"
        )

    if not all_summaries:
        log.error("No solvers analysed.")
        sys.exit(1)

    # ---- robustness with alternative splits ----
    robustness_summaries = {}
    if not SKIP_ROBUSTNESS:
        for split in SPLIT_ROBUSTNESS:
            log.info(f"\n--- Robustness: {split} split ---")
            rob_list = []
            for sid in solver_ids:
                sdf = grouped.get_group(sid)
                sr = analyse_solver(sid, sdf, split_method=split)
                if sr is not None:
                    rob_list.append(sr)
            robustness_summaries[split] = rob_list
            ns = sum(1 for s in rob_list if s["shift_detected"])
            nsh = sum(1 for s in rob_list if s["shape_shift"])
            nnk = sum(1 for s in rob_list if s["norm_ks_reject"])
            log.info(
                f"  {split}: {len(rob_list)} solvers, "
                f"shift={ns} shape={nsh} norm-KS={nnk}"
            )

    # ---- save CSVs ----
    build_phase_records(all_summaries).to_csv(
        OUTPUT_DIR / "rq2_phase_results.csv", index=False
    )
    build_solver_records(all_summaries).to_csv(
        OUTPUT_DIR / "rq2_solver_results.csv", index=False
    )
    for split, rob_list in robustness_summaries.items():
        if rob_list:
            build_solver_records(rob_list).to_csv(
                OUTPUT_DIR / f"rq2_solver_results_{split}.csv", index=False
            )
            build_phase_records(rob_list).to_csv(
                OUTPUT_DIR / f"rq2_phase_results_{split}.csv", index=False
            )

    # ---- summary plots ----
    log.info("\nGenerating summary plots ...")
    plot_preference_by_phase(all_summaries, OUTPUT_DIR)
    plot_feature_evolution_summary(all_summaries, OUTPUT_DIR)
    plot_shift_detection_summary(all_summaries, OUTPUT_DIR)
    plot_gof_summary(all_summaries, OUTPUT_DIR)
    plot_tau_evolution(all_summaries, OUTPUT_DIR)
    plot_lognormal_s_evolution(all_summaries, OUTPUT_DIR)
    plot_norm_ks_distribution(all_summaries, OUTPUT_DIR)
    plot_feature_shift_ranking(all_summaries, OUTPUT_DIR)

    # ---- summary log ----
    nt = len(all_summaries)
    n_shift = sum(1 for s in all_summaries if s["shift_detected"])
    n_family = sum(1 for s in all_summaries if s["family_changed"])
    n_ll_family = sum(1 for s in all_summaries if s.get("ll_family_changed", False))
    n_shape = sum(1 for s in all_summaries if s["shape_shift"])
    n_scale = sum(1 for s in all_summaries if s["scale_shift"])
    n_normks = sum(1 for s in all_summaries if s["norm_ks_reject"])
    pct_shift = 100 * n_shift / nt

    log.info("\n" + "=" * 60)
    log.info("RQ2 \u2014 RESULTS SUMMARY (v2.0)")
    log.info("=" * 60)
    log.info(f"Solvers analysed: {nt}")
    log.info("\n--- Shift detection (shape OR norm-KS OR family) ---")
    log.info(f"  Shift detected:     {n_shift}/{nt} ({pct_shift:.1f}%)  [target \u2265 50%]")
    log.info(f"  Family changed:     {n_family}/{nt} ({100 * n_family / nt:.1f}%)")
    log.info(f"  LL-family changed:  {n_ll_family}/{nt} ({100 * n_ll_family / nt:.1f}%)")
    log.info(f"  Shape shift:        {n_shape}/{nt} ({100 * n_shape / nt:.1f}%)")
    log.info(f"  Scale shift:        {n_scale}/{nt} ({100 * n_scale / nt:.1f}%)")
    log.info(f"  Norm-KS reject:     {n_normks}/{nt} ({100 * n_normks / nt:.1f}%)")
    log.info(
        f"\nKPI: {'PASS' if pct_shift &gt;= 50 else 'BELOW TARGET'} "
        f"({pct_shift:.1f}% vs 50%)"
    )

    log.info("\n-- AIC-preferred distribution: first phase --")
    for d, c in Counter(s["pref_first"] for s in all_summaries).most_common():
        log.info(f"  {d:15s} {c:3d} ({100 * c / nt:.1f}%)")

    log.info("-- AIC-preferred distribution: last phase --")
    for d, c in Counter(s["pref_last"] for s in all_summaries).most_common():
        log.info(f"  {d:15s} {c:3d} ({100 * c / nt:.1f}%)")

    log.info("-- LL-preferred (no penalty): first phase --")
    for d, c in Counter(s.get("ll_pref_first", "?") for s in all_summaries).most_common():
        log.info(f"  {d:15s} {c:3d} ({100 * c / nt:.1f}%)")

    log.info("-- LL-preferred (no penalty): last phase --")
    for d, c in Counter(s.get("ll_pref_last", "?") for s in all_summaries).most_common():
        log.info(f"  {d:15s} {c:3d} ({100 * c / nt:.1f}%)")

    # ex-Gaussian fit success rate
    exg_total = sum(s["n_phases"] for s in all_summaries)
    exg_ok = sum(s.get("exg_fit_ok_count", 0) for s in all_summaries)
    log.info(
        f"\n-- Ex-Gaussian fit success: {exg_ok}/{exg_total} phases "
        f"({100 * exg_ok / max(exg_total, 1):.1f}%) --"
    )

    # feature shift ranking
    feats = [
        "mean", "std", "var", "cv", "skewness",
        "excess_kurtosis", "iqr", "p90_p10_ratio", "tail_weight_upper",
    ]
    shape_feats_set = {"skewness", "excess_kurtosis", "tail_weight_upper", "cv"}

    log.info("\n-- Feature shift ranking (median |rel change| %) --")
    shift_ranking = {}
    for feat in feats:
        vals = [
            abs(s["feature_deltas"].get(f"{feat}_rel_change", np.nan))
            for s in all_summaries
            if np.isfinite(s["feature_deltas"].get(f"{feat}_rel_change", np.nan))
        ]
        shift_ranking[feat] = np.median(vals) if vals else 0

    for feat in sorted(shift_ranking, key=shift_ranking.get, reverse=True):
        tag = " [SHAPE]" if feat in shape_feats_set else " [SCALE]"
        log.info(f"  {feat:25s} {shift_ranking[feat]:7.1f}%{tag}")

    # bootstrap KS GoF acceptance rates
    log.info("\n-- Bootstrap KS GoF (% phases NOT rejected at \u03b1=0.05) --")
    for dname in ["gaussian", "lognormal", "exgaussian"]:
        total = 0
        accepted = 0
        for s in all_summaries:
            for pr in s["phase_results"]:
                g = pr["gof"][dname]
                if g["reject"] is not None:
                    total += 1
                    if not g["reject"]:
                        accepted += 1
        pct = 100 * accepted / total if total &gt; 0 else 0
        log.info(f"  {dname:15s} {accepted}/{total} ({pct:.1f}%)")

    # normalised KS summary
    log.info("\n-- Normalised KS test (Q1 vs Q4, z-scored) --")
    log.info(f"  Significant (p&lt;0.05): {n_normks}/{nt} ({100 * n_normks / nt:.1f}%)")
    pvals = [s["norm_ks_p"] for s in all_summaries if np.isfinite(s["norm_ks_p"])]
    if pvals:
        log.info(f"  p-value: median={np.median(pvals):.4f} mean={np.mean(pvals):.4f}")

    # robustness agreement
    if robustness_summaries:
        log.info("\n-- Robustness: shift-detection agreement --")
        primary = {s["solver_id"]: s["shift_detected"] for s in all_summaries}
        for split, rob_list in robustness_summaries.items():
            agree = sum(
                1 for sr in rob_list
                if sr["solver_id"] in primary
                and sr["shift_detected"] == primary[sr["solver_id"]]
            )
            total = sum(1 for sr in rob_list if sr["solver_id"] in primary)
            ns = sum(1 for sr in rob_list if sr["shift_detected"])
            nsh = sum(1 for sr in rob_list if sr["shape_shift"])
            nnk = sum(1 for sr in rob_list if sr["norm_ks_reject"])
            log.info(
                f"  {split:10s}  shift={ns}/{len(rob_list)}  shape={nsh}  norm-KS={nnk}  "
                f"agrees={agree}/{total} ({100 * agree / max(total, 1):.0f}%)"
            )

    # ---- per-solver comparison tables ----
    with open(OUTPUT_DIR / "comparison_tables.txt", "w", encoding="utf-8") as f:
        for s in all_summaries:
            f.write(
                f"\n{'=' * 60}\n"
                f"Solver: {s['solver_id']}  |  Shift: {s['shift_detected']}  |  "
                f"Shape: {s['shape_shift']}  Scale: {s['scale_shift']}  "
                f"Norm-KS: {s['norm_ks_reject']}\n"
                f"AIC: {s['pref_first']} \u2192 {s['pref_last']}  |  "
                f"LL: {s.get('ll_pref_first', '?')} \u2192 {s.get('ll_pref_last', '?')}\n"
                f"{'=' * 60}\n"
            )
            for pr in s["phase_results"]:
                f.write(
                    f"\n  Phase: {pr['phase']}  (n={pr['n_solves']})  "
                    f"AIC-Pref: {pr['preferred_dist']}  "
                    f"LL-Pref: {pr['ll_preferred_dist']}\n"
                )
                f.write(
                    f"  {'Distribution':15s} {'AIC':&gt;10s} {'\u0394AIC':&gt;8s} "
                    f"{'LL':&gt;12s} {'KS-stat':&gt;8s} {'KS-p':&gt;8s} {'fit':&gt;5s}\n"
                )
                for dname in ["gaussian", "lognormal", "exgaussian"]:
                    dr = pr["dist_results"][dname]
                    gof = pr["gof"][dname]
                    if dr["fit_ok"]:
                        f.write(
                            f"  {dname:15s} {dr['aic']:10.1f} {dr['delta_aic']:8.1f} "
                            f"{dr['ll']:12.1f} {gof['ks_stat']:8.4f} {gof['ks_p']:8.4f} "
                            f"{'OK':&gt;5s}\n"
                        )
                    else:
                        f.write(
                            f"  {dname:15s} {'':&gt;10s} {'':&gt;8s} "
                            f"{'':&gt;12s} {'':&gt;8s} {'':&gt;8s} {'FAIL':&gt;5s}\n"
                        )

                feat = pr["features"]
                f.write(
                    f"  Features: mean={feat['mean']:.2f}s  std={feat['std']:.2f}  "
                    f"skew={feat['skewness']:.3f}  kurt={feat['excess_kurtosis']:.3f}  "
                    f"cv={feat['cv']:.3f}\n"
                )
                if pr["exg_components"]:
                    ec = pr["exg_components"]
                    f.write(
                        f"  Ex-Gaussian: \u03bc={ec['mu']:.2f}  \u03c3={ec['sigma']:.2f}  "
                        f"\u03c4={ec['tau']:.2f}  \u03c4_frac={ec['tau_frac']:.3f}\n"
                    )
                if pr["ln_derived"]:
                    ld = pr["ln_derived"]
                    f.write(
                        f"  Lognormal:   s={ld['s']:.4f}  \u03bc_log={ld['mu_log']:.4f}  "
                        f"dist_skew={ld['dist_skew']:.3f}\n"
                    )
            f.write("\n")

    log.info(f"\nAll outputs written to {OUTPUT_DIR.resolve()}")
    log.info("Done.")


if __name__ == "__main__":
    main()
