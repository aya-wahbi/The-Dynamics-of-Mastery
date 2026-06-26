#!/usr/bin/env python3
"""
RQ3_fluctuation_analysis.py  

Detrended Fluctuation Analysis (DFA).

Takes the per-solver residual series from RQ1, runs DFA to estimate the
scaling exponent alpha, splits careers into early/late phases, builds
bootstrap CIs, runs shuffle-surrogate significance tests, and computes
complementary Power Spectral Density over solve order.

Author: Aya Wahbi (01427598)
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import linregress, pearsonr, spearmanr, wilcoxon, mannwhitneyu
from scipy.signal import welch

np.random.seed(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- paths ---
RQ1_DIR = Path("output") / "rq1"
RESID_DIR = RQ1_DIR / "residuals"
OUTPUT_DIR = Path("output") / "rq3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "rq3.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# --- cli flags ---
USE_FULL = "--full" in sys.argv
MIN_SERIES_LEN = 100       # need at least this many residuals to bother
MIN_PHASE_LEN = 100        # minimum solves per half for the early/late split
if "--min-phase" in sys.argv:
    _idx = sys.argv.index("--min-phase")
    MIN_PHASE_LEN = int(sys.argv[_idx + 1])

# --- DFA knobs ---
DFA_ORDER = 1               # linear detrending (DFA-1)
DFA_N_MIN = 10              # smallest window
DFA_N_MAX_FRAC = 0.25       # largest window = 25% of series length
DFA_N_POINTS = 30           # how many log-spaced window sizes
BOOTSTRAP_REPS = 1000
SURROGATE_REPS = 200
ALPHA_WN = 0.5              # white-noise baseline

# --- PSD knobs ---
PSD_NPERSEG_FRAC = 0.25     # Welch segment as fraction of series


# ──────────────────────────────────────────────
#  DFA implementation
# ──────────────────────────────────────────────

def dfa(series, order=DFA_ORDER, n_min=DFA_N_MIN,
        n_max_frac=DFA_N_MAX_FRAC, n_points=DFA_N_POINTS):
    """
    Detrended Fluctuation Analysis (Peng et al., 1994).

    Returns (scales, flucts, alpha, intercept, r_value).
    alpha is the slope of the log F(n) vs log n regression.
    """
    x = np.asarray(series, dtype=np.float64)
    N = len(x)
    if N < 2 * n_min:
        return None, None, np.nan, np.nan, np.nan

    # cumulative sum of the mean-centred series (the "profile")
    profile = np.cumsum(x - np.mean(x))

    # log-spaced window sizes
    n_max = max(int(N * n_max_frac), n_min + 1)
    scales = np.unique(
        np.logspace(np.log10(n_min), np.log10(n_max), n_points).astype(int)
    )
    scales = scales[scales >= n_min]
    if len(scales) < 4:
        return None, None, np.nan, np.nan, np.nan

    flucts = np.zeros(len(scales))

    for si, n in enumerate(scales):
        n_seg = N // n
        if n_seg < 1:
            flucts[si] = np.nan
            continue

        var_list = []
        # walk through non-overlapping segments in both directions
        for direction in [0, 1]:
            for v in range(n_seg):
                if direction == 0:
                    segment = profile[v * n : (v + 1) * n]
                else:
                    segment = profile[N - (v + 1) * n : N - v * n]

                idx = np.arange(len(segment))
                coeffs = np.polyfit(idx, segment, order)
                trend = np.polyval(coeffs, idx)
                var_list.append(np.mean((segment - trend) ** 2))

        flucts[si] = np.sqrt(np.mean(var_list))

    # drop NaN windows
    valid = np.isfinite(flucts) & (flucts > 0)
    scales = scales[valid]
    flucts = flucts[valid]

    if len(scales) < 4:
        return None, None, np.nan, np.nan, np.nan

    # log-log linear fit
    log_n = np.log10(scales.astype(float))
    log_f = np.log10(flucts)
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_f)

    return scales, flucts, slope, intercept, r_value


def dfa_alpha(series, **kwargs):
    """Just the exponent, nothing else."""
    _, _, alpha, _, _ = dfa(series, **kwargs)
    return alpha


def bootstrap_alpha(series, n_boot=BOOTSTRAP_REPS, block_frac=0.1, **dfa_kwargs):
    """
    Circular block-bootstrap CI for the DFA exponent.
    Blocks preserve local correlation while resampling.
    """
    N = len(series)
    block_len = max(int(N * block_frac), DFA_N_MIN)
    alphas = np.full(n_boot, np.nan)

    for b in range(n_boot):
        indices = []
        while len(indices) < N:
            start = np.random.randint(0, N)
            end = start + block_len
            indices.extend(np.arange(start, end) % N)
        boot_series = series[np.array(indices[:N])]
        alphas[b] = dfa_alpha(boot_series, **dfa_kwargs)

    alphas = alphas[np.isfinite(alphas)]
    if len(alphas) < n_boot * 0.5:
        return np.nan, np.nan, np.nan

    return (
        float(np.median(alphas)),
        float(np.percentile(alphas, 2.5)),
        float(np.percentile(alphas, 97.5)),
    )


def surrogate_test(series, alpha_obs, n_surr=SURROGATE_REPS, **dfa_kwargs):
    """
    Shuffle the series to destroy temporal correlations, re-estimate alpha.
    Returns fraction of surrogates where alpha >= observed (one-sided p).
    """
    count_ge = 0
    surr_alphas = np.full(n_surr, np.nan)

    for s in range(n_surr):
        perm = np.random.permutation(series)
        a = dfa_alpha(perm, **dfa_kwargs)
        surr_alphas[s] = a
        if np.isfinite(a) and a >= alpha_obs:
            count_ge += 1

    valid = surr_alphas[np.isfinite(surr_alphas)]
    p = count_ge / max(len(valid), 1)
    mean_surr = float(np.mean(valid)) if len(valid) else np.nan
    return p, mean_surr


# ──────────────────────────────────────────────
#  Power Spectral Density
# ──────────────────────────────────────────────

def compute_psd(series, nperseg_frac=PSD_NPERSEG_FRAC):
    """
    Welch PSD over solve order (not calendar time).
    Returns (freqs, psd, beta) where PSD ~ 1/f^beta.
    """
    N = len(series)
    nperseg = max(int(N * nperseg_frac), 32)
    nperseg = min(nperseg, N)

    freqs, psd = welch(
        series, fs=1.0, nperseg=nperseg,
        detrend="linear", scaling="density",
    )

    # fit the log-log slope (skip DC)
    valid = (freqs > 0) & (psd > 0)
    if valid.sum() < 4:
        return freqs, psd, np.nan

    lf = np.log10(freqs[valid])
    lp = np.log10(psd[valid])
    slope, _, _, _, _ = linregress(lf, lp)
    beta = -slope   # PSD ~ 1/f^beta  =>  log(PSD) ~ -beta*log(f)

    return freqs, psd, beta


# ──────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────

def load_rq1_results():
    """Grab the RQ1 summary to know each solver's preferred model."""
    path = RQ1_DIR / "rq1_results.csv"
    if not path.exists():
        log.error(f"RQ1 results not found at {path}. Run RQ1 first.")
        sys.exit(1)
    df = pd.read_csv(path)
    log.info(f"Loaded RQ1 results: {len(df)} solvers")
    return df


def load_cohort_info():
    path = RQ1_DIR / "cohort_info.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_residuals(solver_id, preferred_model):
    """
    Load one solver's residual CSV (produced by RQ1) and filter
    to just the preferred model's residuals.
    """
    path = RESID_DIR / f"{solver_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["model"] == preferred_model].sort_values("solve_number").reset_index(drop=True)
    if len(df) == 0:
        return None
    return df


# ──────────────────────────────────────────────
#  Per-solver analysis pipeline
# ──────────────────────────────────────────────

def analyse_solver(solver_id, resid_df):
    """
    Full RQ3 pipeline for one solver: DFA on the whole series,
    bootstrap CI, surrogate test, PSD, and early/late split.
    """
    residuals = resid_df["residual"].values.astype(np.float64)
    solve_nums = resid_df["solve_number"].values
    N = len(residuals)

    result = {
        "solver_id": solver_id,
        "n_residuals": N,
    }

    # --- full-series DFA ---
    scales, flucts, alpha, intercept, r_val = dfa(residuals)
    result["alpha_full"] = alpha
    result["alpha_full_r"] = r_val
    result["dfa_intercept"] = intercept

    if np.isfinite(alpha):
        med, lo, hi = bootstrap_alpha(residuals)
        result["alpha_full_median_boot"] = med
        result["alpha_full_ci_lo"] = lo
        result["alpha_full_ci_hi"] = hi

        p_surr, mean_surr = surrogate_test(residuals, alpha)
        result["alpha_full_surr_p"] = p_surr
        result["alpha_full_surr_mean"] = mean_surr
        result["alpha_full_sig"] = p_surr < 0.05
    else:
        result.update({
            "alpha_full_median_boot": np.nan,
            "alpha_full_ci_lo": np.nan,
            "alpha_full_ci_hi": np.nan,
            "alpha_full_surr_p": np.nan,
            "alpha_full_surr_mean": np.nan,
        })
        result["alpha_full_sig"] = False

    # --- PSD (full series) ---
    freqs, psd_vals, beta = compute_psd(residuals)
    result["beta_full"] = beta

    # --- early / late split ---
    mid = N // 2
    can_split = (mid >= MIN_PHASE_LEN) and ((N - mid) >= MIN_PHASE_LEN)
    result["phase_split_possible"] = can_split

    phase_results = []

    if can_split:
        for phase_label, phase_slice in [("early", residuals[:mid]),
                                          ("late", residuals[mid:])]:
            sc_p, fl_p, a_p, int_p, r_p = dfa(phase_slice)
            if np.isfinite(a_p):
                med_p, lo_p, hi_p = bootstrap_alpha(phase_slice)
            else:
                med_p, lo_p, hi_p = np.nan, np.nan, np.nan
            _, _, beta_p = compute_psd(phase_slice)

            result[f"alpha_{phase_label}"] = a_p
            result[f"alpha_{phase_label}_r"] = r_p
            result[f"alpha_{phase_label}_ci_lo"] = lo_p
            result[f"alpha_{phase_label}_ci_hi"] = hi_p
            result[f"beta_{phase_label}"] = beta_p
            result[f"n_{phase_label}"] = len(phase_slice)

            phase_results.append({
                "solver_id": solver_id,
                "phase": phase_label,
                "n": len(phase_slice),
                "alpha": a_p,
                "alpha_r": r_p,
                "alpha_ci_lo": lo_p,
                "alpha_ci_hi": hi_p,
                "beta": beta_p,
            })

        # how much did alpha change?
        if np.isfinite(result["alpha_early"]) and np.isfinite(result["alpha_late"]):
            result["delta_alpha"] = result["alpha_late"] - result["alpha_early"]
        else:
            result["delta_alpha"] = np.nan
    else:
        # fill in NaN placeholders when we can't split
        for tag in ["early", "late"]:
            for suf in ["", "_r", "_ci_lo", "_ci_hi"]:
                result[f"alpha_{tag}{suf}"] = np.nan
            result[f"beta_{tag}"] = np.nan
            result[f"n_{tag}"] = np.nan
        result["delta_alpha"] = np.nan

    # always include a "full" row in the phase table
    phase_results.insert(0, {
        "solver_id": solver_id,
        "phase": "full",
        "n": N,
        "alpha": alpha,
        "alpha_r": r_val,
        "alpha_ci_lo": result.get("alpha_full_ci_lo", np.nan),
        "alpha_ci_hi": result.get("alpha_full_ci_hi", np.nan),
        "beta": beta,
    })

    # stash raw arrays for plotting (these won't go into the CSV)
    result["_scales"] = scales
    result["_flucts"] = flucts
    result["_residuals"] = residuals
    result["_freqs"] = freqs
    result["_psd"] = psd_vals

    return result, phase_results


# ──────────────────────────────────────────────
#  Plotting — per solver
# ──────────────────────────────────────────────

PHASE_COLORS = {"full": "#2c3e50", "early": "#e74c3c", "late": "#2980b9"}


def plot_solver(solver_id, result, resid_df, out_dir):
    """4-panel figure: residuals, DFA log-log, PSD, alpha bar+CI."""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28)

    residuals = result["_residuals"]
    solve_nums = resid_df["solve_number"].values

    # --- panel A: residual time series ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(
        solve_nums, residuals, s=3, alpha=0.20,
        color="grey", rasterized=True,
    )

    # overlay rolling std on a twin axis
    w = max(20, len(residuals) // 40)
    roll_std = pd.Series(residuals).rolling(w, center=True, min_periods=10).std()
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        solve_nums, roll_std.values,
        color="#e67e22", lw=1.0, alpha=0.7, label="rolling std",
    )
    ax1_twin.set_ylabel("rolling std (s)", fontsize=8, color="#e67e22")
    ax1_twin.tick_params(axis="y", labelcolor="#e67e22", labelsize=7)

    if result["phase_split_possible"]:
        mid = len(residuals) // 2
        ax1.axvline(
            solve_nums[mid], color="red", ls="--", lw=0.9,
            label="early|late split",
        )

    ax1.set_xlabel("Solve number")
    ax1.set_ylabel("Residual (s)")
    ax1.set_title(f"Solver {solver_id} - residuals (n={len(residuals)})")
    ax1.legend(fontsize=7, loc="upper right")

    # --- panel B: DFA log-log ---
    ax2 = fig.add_subplot(gs[0, 1])
    scales = result["_scales"]
    flucts = result["_flucts"]

    if scales is not None and flucts is not None:
        ax2.loglog(
            scales, flucts, "o-", ms=4, color=PHASE_COLORS["full"],
            lw=1.2, label=f"full alpha={result['alpha_full']:.3f}",
        )

        # regression line
        log_n = np.log10(scales.astype(float))
        fit_line = 10 ** (result["alpha_full"] * log_n + result["dfa_intercept"])
        ax2.loglog(scales, fit_line, "--", color=PHASE_COLORS["full"], lw=0.8, alpha=0.6)

        # early/late curves if we have them
        if result["phase_split_possible"]:
            mid = len(residuals) // 2
            for phase_label, phase_slice, color in [
                ("early", residuals[:mid], PHASE_COLORS["early"]),
                ("late", residuals[mid:], PHASE_COLORS["late"]),
            ]:
                sc_p, fl_p, a_p, _, _ = dfa(phase_slice)
                if sc_p is not None:
                    ax2.loglog(
                        sc_p, fl_p, "s-", ms=3, color=color, lw=1.0,
                        alpha=0.7, label=f"{phase_label} alpha={a_p:.3f}",
                    )

        # white-noise reference line (alpha=0.5)
        ref = scales.astype(float) ** 0.5 * flucts[0] / scales[0] ** 0.5
        ax2.loglog(scales, ref, ":", color="grey", lw=0.8, alpha=0.5, label="alpha=0.5 ref")

    ax2.set_xlabel("Window size n")
    ax2.set_ylabel("F(n)")
    ax2.set_title("DFA log-log plot")
    ax2.legend(fontsize=7)

    # --- panel C: PSD ---
    ax3 = fig.add_subplot(gs[1, 0])
    freqs = result["_freqs"]
    psd = result["_psd"]

    if freqs is not None and len(freqs) > 1:
        valid = (freqs > 0) & (psd > 0)
        ax3.loglog(freqs[valid], psd[valid], "-", color="#8e44ad", lw=1.0, alpha=0.8)

        beta = result["beta_full"]
        if np.isfinite(beta):
            lf = np.log10(freqs[valid])
            fit_psd = 10 ** (-beta * lf + np.log10(psd[valid]).mean() + beta * lf.mean())
            ax3.loglog(
                freqs[valid], fit_psd, "--", color="black", lw=0.8,
                alpha=0.6, label=f"beta={beta:.2f}",
            )

        ax3.set_xlabel("Frequency (1/solve)")
        ax3.set_ylabel("PSD")
        ax3.set_title("Power Spectral Density")
        ax3.legend(fontsize=8)

    # --- panel D: alpha bar chart with CI ---
    ax4 = fig.add_subplot(gs[1, 1])
    bar_labels = []
    bar_alphas = []
    bar_ci_lo = []
    bar_ci_hi = []
    bar_colors = []

    for phase, color in PHASE_COLORS.items():
        a = result.get(f"alpha_{phase}", np.nan)
        if not np.isfinite(a):
            continue
        lo = result.get(f"alpha_{phase}_ci_lo", np.nan)
        hi = result.get(f"alpha_{phase}_ci_hi", np.nan)
        bar_labels.append(phase)
        bar_alphas.append(a)
        bar_ci_lo.append(a - lo if np.isfinite(lo) else 0)
        bar_ci_hi.append(hi - a if np.isfinite(hi) else 0)
        bar_colors.append(color)

    if bar_labels:
        y_pos = np.arange(len(bar_labels))
        ax4.barh(
            y_pos, bar_alphas, xerr=[bar_ci_lo, bar_ci_hi],
            color=bar_colors, edgecolor="black", height=0.5,
            capsize=4, alpha=0.8,
        )
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(bar_labels)
        ax4.axvline(0.5, color="red", ls="--", lw=1, label="alpha=0.5 (white noise)")
        ax4.axvline(1.0, color="blue", ls=":", lw=0.8, label="alpha=1.0 (1/f noise)")
        ax4.set_xlabel("alpha")
        ax4.set_title("DFA exponent +/- 95% CI")
        ax4.legend(fontsize=7, loc="lower right")
        ax4.set_xlim(0, max(1.5, max(bar_alphas) + 0.3))

        if result.get("alpha_full_sig"):
            ax4.text(
                0.98, 0.02, "* sig. vs shuffled (p<0.05)",
                transform=ax4.transAxes, ha="right", va="bottom",
                fontsize=7, color="green", fontstyle="italic",
            )

    plt.suptitle(f"RQ3 - Fluctuation Analysis: {solver_id}", fontsize=12, y=1.01)
    fig.savefig(out_dir / f"solver_{solver_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
#  Plotting — cohort level
# ──────────────────────────────────────────────

def plot_alpha_distribution(results_df, out_dir):
    af = results_df["alpha_full"].dropna()
    if len(af) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(af, bins=20, color="#2c3e50", edgecolor="black", alpha=0.8, density=False)
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label="alpha=0.5 (white noise)")
    ax.axvline(1.0, color="blue", ls=":", lw=1.2, label="alpha=1.0 (1/f)")
    ax.axvline(af.median(), color="#f39c12", ls="-", lw=1.5,
               label=f"median={af.median():.3f}")
    ax.set_xlabel("DFA exponent alpha")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of alpha (full career)  n={len(af)}")
    ax.legend(fontsize=9)

    txt = (
        f"median = {af.median():.3f}\n"
        f"mean   = {af.mean():.3f}\n"
        f"std    = {af.std():.3f}\n"
        f"alpha>0.5: {(af > 0.5).sum()}/{len(af)}\n"
        f"alpha>0.75: {(af > 0.75).sum()}/{len(af)}"
    )
    ax.text(
        0.97, 0.97, txt, transform=ax.transAxes, va="top", ha="right",
        fontsize=8, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(out_dir / "rq3_alpha_distribution.png", dpi=150)
    plt.close(fig)


def plot_early_vs_late(results_df, out_dir):
    sub = results_df.dropna(subset=["alpha_early", "alpha_late"]).copy()
    if len(sub) < 3:
        log.info("Too few solvers with early/late split for paired plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # paired dot plot
    ax = axes[0]
    for _, row in sub.iterrows():
        ax.plot(
            [0, 1], [row["alpha_early"], row["alpha_late"]],
            "o-", color="grey", alpha=0.4, ms=5,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Early career", "Late career"])
    ax.set_ylabel("alpha")
    ax.axhline(0.5, color="red", ls="--", lw=0.8)
    ax.set_title(f"Early vs Late alpha  (n={len(sub)})")

    # delta-alpha histogram
    ax = axes[1]
    da = sub["delta_alpha"].dropna()
    ax.hist(da, bins=15, color="#3498db", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=1.2)
    ax.axvline(
        da.median(), color="#f39c12", ls="-", lw=1.2,
        label=f"median delta_alpha={da.median():.3f}",
    )
    ax.set_xlabel("delta_alpha = alpha_late - alpha_early")
    ax.set_ylabel("Count")
    ax.set_title("Change in alpha across career")
    ax.legend(fontsize=8)

    # Wilcoxon test annotation
    if len(da) >= 5:
        try:
            stat, p = wilcoxon(da)
            test_str = f"Wilcoxon p={p:.4f}"
        except Exception:
            test_str = "Wilcoxon: N/A"
    else:
        test_str = f"n={len(da)} (too few)"
    ax.text(
        0.97, 0.97, test_str, transform=ax.transAxes, va="top", ha="right",
        fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    # scatter: early vs late
    ax = axes[2]
    ax.scatter(
        sub["alpha_early"], sub["alpha_late"], s=40,
        edgecolors="black", linewidths=0.5, alpha=0.7, color="#27ae60",
    )
    lims = [
        min(sub["alpha_early"].min(), sub["alpha_late"].min()) - 0.05,
        max(sub["alpha_early"].max(), sub["alpha_late"].max()) + 0.05,
    ]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="y=x")
    ax.set_xlabel("alpha early")
    ax.set_ylabel("alpha late")
    ax.set_title("Early vs Late scatter")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    fig.savefig(out_dir / "rq3_early_vs_late.png", dpi=150)
    plt.close(fig)


def plot_alpha_by_characteristics(results_df, cohort_info, out_dir):
    if cohort_info is None:
        return

    merged = results_df.merge(
        cohort_info[["solver_id", "n_solves", "career_years", "last_avg_s"]],
        on="solver_id", how="left", suffixes=("", "_ci"),
    )
    af = merged.dropna(subset=["alpha_full"])
    if len(af) < 5:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, xcol, xlabel in zip(
        axes,
        ["n_solves", "career_years", "last_avg_s"],
        ["Total solves", "Career span (yr)", "Last avg solve time (s)"],
    ):
        if xcol not in af.columns or af[xcol].isna().all():
            ax.set_visible(False)
            continue

        ax.scatter(
            af[xcol], af["alpha_full"], s=40, alpha=0.7,
            edgecolors="black", linewidths=0.4, color="#8e44ad",
        )
        ax.axhline(0.5, color="red", ls="--", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("alpha (full)")

        valid_pair = af[[xcol, "alpha_full"]].dropna()
        if len(valid_pair) >= 5:
            rho, p = spearmanr(valid_pair[xcol], valid_pair["alpha_full"])
            ax.set_title(f"rho={rho:.3f}, p={p:.3f}")
        else:
            ax.set_title(xlabel)

    plt.suptitle("DFA exponent vs solver characteristics", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "rq3_alpha_by_skill.png", dpi=150)
    plt.close(fig)


def plot_psd_gallery(all_results, out_dir, max_panels=9):
    subset = [
        r for r in all_results
        if r["_freqs"] is not None and len(r["_freqs"]) > 1
    ][:max_panels]
    if not subset:
        return

    ncols = 3
    nrows = int(np.ceil(len(subset) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, r in enumerate(subset):
        ri, ci = divmod(idx, ncols)
        ax = axes[ri, ci]
        f = r["_freqs"]
        p = r["_psd"]
        valid = (f > 0) & (p > 0)
        ax.loglog(f[valid], p[valid], color="#8e44ad", lw=0.8, alpha=0.8)
        beta = r["beta_full"]
        if np.isfinite(beta):
            ax.set_title(f"{r['solver_id']}  beta={beta:.2f}", fontsize=8)
        else:
            ax.set_title(r["solver_id"], fontsize=8)
        ax.tick_params(labelsize=7)

    # hide leftover subplots
    for idx in range(len(subset), nrows * ncols):
        ri, ci = divmod(idx, ncols)
        axes[ri, ci].set_visible(False)

    fig.suptitle("PSD gallery (1/f^beta)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "rq3_psd_gallery.png", dpi=150)
    plt.close(fig)


def plot_summary_dashboard(results_df, out_dir):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)

    af = results_df["alpha_full"].dropna()

    # panel 1: alpha histogram
    ax = fig.add_subplot(gs[0, 0])
    if len(af) >= 2:
        ax.hist(af, bins=15, color="#2c3e50", edgecolor="black", alpha=0.8)
        ax.axvline(0.5, color="red", ls="--", lw=1.2)
        ax.axvline(af.median(), color="#f39c12", lw=1.5)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Count")
    ax.set_title(f"Full alpha distribution (n={len(af)})")

    # panel 2: beta histogram
    ax = fig.add_subplot(gs[0, 1])
    bf = results_df["beta_full"].dropna()
    if len(bf) >= 2:
        ax.hist(bf, bins=15, color="#8e44ad", edgecolor="black", alpha=0.8)
        ax.axvline(0, color="red", ls="--", lw=1.2)
    ax.set_xlabel("beta (PSD exponent)")
    ax.set_ylabel("Count")
    ax.set_title(f"PSD beta distribution (n={len(bf)})")

    # panel 3: alpha vs beta (consistency check)
    ax = fig.add_subplot(gs[0, 2])
    both = results_df.dropna(subset=["alpha_full", "beta_full"])
    if len(both) >= 3:
        ax.scatter(
            both["alpha_full"], both["beta_full"], s=40,
            edgecolors="black", linewidths=0.4, alpha=0.7, color="#27ae60",
        )
        a_range = np.linspace(
            both["alpha_full"].min() - 0.1,
            both["alpha_full"].max() + 0.1, 50,
        )
        ax.plot(
            a_range, 2 * a_range - 1, "r--", lw=0.8, alpha=0.6,
            label="beta = 2*alpha - 1",
        )
        ax.legend(fontsize=7)
    ax.set_xlabel("alpha (DFA)")
    ax.set_ylabel("beta (PSD)")
    ax.set_title("alpha vs beta consistency")

    # panel 4: significance pie
    ax = fig.add_subplot(gs[1, 0])
    sig_col = results_df["alpha_full_sig"].dropna()
    if len(sig_col) > 0:
        counts = sig_col.value_counts()
        pie_labels = ["Significant" if v else "Not sig." for v in counts.index]
        pie_colors = ["#2ecc71" if v else "#e67e22" for v in counts.index]
        ax.pie(
            counts.values, labels=pie_labels, autopct="%1.0f%%",
            colors=pie_colors, startangle=90,
        )
    ax.set_title("Surrogate test (p<0.05)")

    # panel 5: delta-alpha histogram
    ax = fig.add_subplot(gs[1, 1])
    da = results_df["delta_alpha"].dropna()
    if len(da) >= 3:
        ax.hist(da, bins=15, color="#3498db", edgecolor="black", alpha=0.8)
        ax.axvline(0, color="red", ls="--", lw=1.2)
        ax.axvline(
            da.median(), color="#f39c12", lw=1.5,
            label=f"median={da.median():.3f}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("delta_alpha (late - early)")
    ax.set_ylabel("Count")
    ax.set_title(f"Career-phase change (n={len(da)})")

    # panel 6: KPI summary text box
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")

    n_total = len(results_df)
    n_computed = results_df["alpha_full"].notna().sum()
    n_sig = results_df["alpha_full_sig"].sum() if "alpha_full_sig" in results_df else 0
    n_split = results_df["phase_split_possible"].sum() if "phase_split_possible" in results_df else 0
    n_deviate = (af.sub(0.5).abs() > 0.1).sum() if len(af) else 0

    kpi_text = (
        f"RQ3 - KPI Summary\n"
        f"{'---' * 10}\n"
        f"Solvers analysed:     {n_total}\n"
        f"alpha computed:       {n_computed}/{n_total}\n"
        f"Sig. vs shuffled:     {n_sig}/{n_computed}\n"
        f"|alpha - 0.5| > 0.1: {n_deviate}/{n_computed}\n"
        f"Phase split possible: {n_split}/{n_total}\n"
        f"{'---' * 10}\n"
        f"Median alpha (full):  {af.median():.3f}\n"
        f"Mean alpha (full):    {af.mean():.3f}\n"
    )
    if len(da) >= 3:
        kpi_text += f"Median delta_alpha:   {da.median():.3f}\n"

    ax.text(
        0.05, 0.95, kpi_text, transform=ax.transAxes, va="top",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.9),
    )

    plt.suptitle("RQ3 - Fluctuation Analysis Summary", fontsize=13, y=1.01)
    fig.savefig(out_dir / "rq3_summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("RQ3 - Fluctuation Analysis  |  The Dynamics of Mastery")
    log.info(f"  v1.0  |  DFA order={DFA_ORDER}  |  min_phase={MIN_PHASE_LEN}")
    log.info("=" * 60)

    rq1_results = load_rq1_results()
    cohort_info = load_cohort_info()

    # solver -> preferred model mapping
    pref_map = dict(zip(rq1_results["solver_id"], rq1_results["preferred_model"]))

    resid_files = sorted(RESID_DIR.glob("*.csv"))
    log.info(f"Found {len(resid_files)} residual files in {RESID_DIR}")

    if not resid_files:
        log.error("No residual files found.  Run RQ1 first.")
        sys.exit(1)

    solver_ids = [f.stem for f in resid_files if f.stem in pref_map]
    log.info(f"Solvers with both residuals and preferred-model info: {len(solver_ids)}")

    if not USE_FULL and len(solver_ids) > 50:
        solver_ids = solver_ids[:50]
        log.info(f"  Capped to {len(solver_ids)} solvers (use --full for all)")

    # ---- per-solver loop ----
    all_results = []
    all_phase_rows = []
    n_skipped = 0

    for i, sid in enumerate(solver_ids, 1):
        preferred = pref_map[sid]
        resid_df = load_residuals(sid, preferred)

        if resid_df is None or len(resid_df) < MIN_SERIES_LEN:
            n_resid = len(resid_df) if resid_df is not None else 0
            log.info(
                f"[{i}/{len(solver_ids)}]  {sid}: skipped "
                f"(n={n_resid} < {MIN_SERIES_LEN})"
            )
            n_skipped += 1
            continue

        log.info(
            f"[{i}/{len(solver_ids)}]  {sid}  "
            f"(n={len(resid_df)}, model={preferred}) ..."
        )
        result, phase_rows = analyse_solver(sid, resid_df)
        all_results.append(result)
        all_phase_rows.extend(phase_rows)

        # quick log of the key numbers
        a = result["alpha_full"]
        if np.isfinite(result.get("alpha_full_ci_lo", np.nan)):
            ci_str = f"[{result['alpha_full_ci_lo']:.3f}, {result['alpha_full_ci_hi']:.3f}]"
        else:
            ci_str = "[N/A]"
        sig_str = "SIG" if result.get("alpha_full_sig") else "n.s."
        log.info(
            f"  alpha_full={a:.3f}  CI={ci_str}  {sig_str}  "
            f"beta={result['beta_full']:.2f}"
        )

        if result["phase_split_possible"]:
            log.info(
                f"  alpha_early={result['alpha_early']:.3f}  "
                f"alpha_late={result['alpha_late']:.3f}  "
                f"delta_alpha={result['delta_alpha']:.3f}"
            )
        else:
            log.info("  Phase split not possible (series too short)")

        plot_solver(sid, result, resid_df, OUTPUT_DIR)

    if not all_results:
        log.error("No solvers analysed successfully.")
        sys.exit(1)

    # ---- save CSVs (strip out the internal plotting arrays) ----
    csv_cols = [k for k in all_results[0].keys() if not k.startswith("_")]
    results_df = pd.DataFrame([{k: r[k] for k in csv_cols} for r in all_results])
    results_df.to_csv(OUTPUT_DIR / "rq3_results.csv", index=False)
    log.info(f"\nResults saved: {OUTPUT_DIR / 'rq3_results.csv'}")

    phase_df = pd.DataFrame(all_phase_rows)
    phase_df.to_csv(OUTPUT_DIR / "rq3_phase_results.csv", index=False)
    log.info(f"Phase results saved: {OUTPUT_DIR / 'rq3_phase_results.csv'}")

    # ---- cohort summary ----
    nt = len(results_df)
    af = results_df["alpha_full"].dropna()
    da = results_df["delta_alpha"].dropna()
    n_sig = results_df["alpha_full_sig"].sum() if "alpha_full_sig" in results_df else 0
    n_split = results_df["phase_split_possible"].sum()

    log.info("\n" + "=" * 60)
    log.info("RQ3 - RESULTS SUMMARY")
    log.info("=" * 60)
    log.info(f"Solvers analysed: {nt}  (skipped: {n_skipped})")
    log.info(f"alpha computed:       {len(af)}/{nt}")
    log.info(f"Phase split:      {n_split}/{nt} solvers")

    log.info("\n-- Full-career alpha --")
    if len(af):
        log.info(f"  median = {af.median():.4f}")
        log.info(f"  mean   = {af.mean():.4f}")
        log.info(f"  std    = {af.std():.4f}")
        log.info(f"  range  = [{af.min():.4f}, {af.max():.4f}]")
        log.info(f"  alpha > 0.5:  {(af > 0.5).sum()}/{len(af)}")
        log.info(f"  alpha > 0.75: {(af > 0.75).sum()}/{len(af)}")
        log.info(f"  |alpha-0.5|>0.1: {(af.sub(0.5).abs() > 0.1).sum()}/{len(af)}")

    log.info("\n-- Surrogate significance --")
    if len(af):
        log.info(
            f"  Significant (p<0.05): {n_sig}/{len(af)} "
            f"({100 * n_sig / len(af):.1f}%)"
        )
    else:
        log.info("  N/A")

    log.info("\n-- PSD beta --")
    bf = results_df["beta_full"].dropna()
    if len(bf):
        log.info(f"  median = {bf.median():.4f}")
        log.info(f"  beta > 0 (1/f-like): {(bf > 0).sum()}/{len(bf)}")

    log.info("\n-- Early vs Late --")
    if len(da) >= 3:
        log.info(f"  n pairs = {len(da)}")
        log.info(f"  median delta_alpha = {da.median():.4f}")
        log.info(f"  mean delta_alpha   = {da.mean():.4f}")
        log.info(f"  delta_alpha > 0 (late > early): {(da > 0).sum()}/{len(da)}")
        try:
            stat, p = wilcoxon(da)
            log.info(f"  Wilcoxon signed-rank: W={stat:.1f}, p={p:.4f}")
        except Exception as e:
            log.info(f"  Wilcoxon test failed: {e}")
    else:
        log.info(f"  Too few pairs ({len(da)}) for comparison")

    log.info("\n-- KPI Evaluation --")
    log.info("  Baseline: alpha=0.5 (white noise)")
    log.info("  Criterion: scaling exponent reported with CI; deviation from 0.5 assessed")
    ci_available = results_df["alpha_full_ci_lo"].notna().sum()
    log.info(f"  CIs available: {ci_available}/{nt}")
    kpi_pass = ci_available >= 0.8 * nt
    log.info(
        f"  KPI status: {'PASS' if kpi_pass else 'PARTIAL'} "
        f"({ci_available}/{nt} = {100 * ci_available / nt:.0f}% have CIs, target >=80%)"
    )

    # ---- text summary file ----
    with open(OUTPUT_DIR / "rq3_summary.txt", "w", encoding="utf-8") as f:
        f.write("RQ3 - Fluctuation Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Solvers analysed: {nt}\n")
        f.write(f"Skipped (too short): {n_skipped}\n\n")

        f.write("Full-career alpha:\n")
        if len(af):
            f.write(f"  median = {af.median():.4f}\n")
            f.write(f"  mean   = {af.mean():.4f}\n")
            f.write(f"  std    = {af.std():.4f}\n")
            f.write(f"  range  = [{af.min():.4f}, {af.max():.4f}]\n\n")

        f.write(f"Surrogate sig (p<0.05): {n_sig}/{len(af)}\n\n")

        if len(da) >= 3:
            f.write("Early vs Late:\n")
            f.write(f"  n pairs    = {len(da)}\n")
            f.write(f"  median delta_alpha  = {da.median():.4f}\n")
            f.write(f"  delta_alpha > 0:    {(da > 0).sum()}/{len(da)}\n")
            try:
                stat, p = wilcoxon(da)
                f.write(f"  Wilcoxon:  W={stat:.1f}, p={p:.4f}\n")
            except Exception:
                pass

        f.write("\nPer-solver results:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Solver':<15} {'n':>5} {'a_full':>7} {'CI_lo':>7} {'CI_hi':>7} "
            f"{'sig':>4} {'a_early':>8} {'a_late':>7} {'da':>7} {'beta':>6}\n"
        )
        f.write("-" * 80 + "\n")
        for _, row in results_df.iterrows():
            f.write(
                f"{row['solver_id']:<15} "
                f"{row['n_residuals']:5.0f} "
                f"{row['alpha_full']:7.3f} "
                f"{row.get('alpha_full_ci_lo', np.nan):7.3f} "
                f"{row.get('alpha_full_ci_hi', np.nan):7.3f} "
                f"{'*' if row.get('alpha_full_sig') else ' ':>4} "
                f"{row.get('alpha_early', np.nan):8.3f} "
                f"{row.get('alpha_late', np.nan):7.3f} "
                f"{row.get('delta_alpha', np.nan):7.3f} "
                f"{row.get('beta_full', np.nan):6.2f}\n"
            )

    log.info(f"Text summary saved: {OUTPUT_DIR / 'rq3_summary.txt'}")

    # ---- cohort-level plots ----
    plot_alpha_distribution(results_df, OUTPUT_DIR)
    plot_early_vs_late(results_df, OUTPUT_DIR)
    plot_alpha_by_characteristics(results_df, cohort_info, OUTPUT_DIR)
    plot_psd_gallery(all_results, OUTPUT_DIR)
    plot_summary_dashboard(results_df, OUTPUT_DIR)

    log.info(f"\nAll outputs written to {OUTPUT_DIR.resolve()}")
    log.info("Done.")


if __name__ == "__main__":
    main()
