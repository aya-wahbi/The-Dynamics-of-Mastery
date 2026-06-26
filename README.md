# The Dynamics of Mastery

**Non-Linear Dynamics and Statistical Physics of Human Skill Acquisition**


## Project Structure

```
├── data/                        # WCA TSV exports (not tracked — see below)
├── output/
│   ├── rq1/                     # Learning curve fits, residuals, plots
│   ├── rq2/                     # Distributional analysis results
│   └── rq3/                     # DFA results, PSD plots
├── RQ1_learning_curves.py       # Learning-curve modelling pipeline
├── RQ2_distributional_analysis.py  # Distributional evolution analysis
├── RQ3_fluctuation_analysis.py  # DFA + PSD analysis
├── requirements.txt
└── README.md
```

## Data

The analysis uses the [WCA Results Database Export](https://www.worldcubeassociation.org/export/results). Download the TSV export and place the files in a `data/` folder at the project root. The key files needed are:

- `WCA_export_results.tsv`
- `WCA_export_result_attempts.tsv`
- `WCA_export_competitions.tsv`
- `WCA_export_round_types.tsv`

> **Note:** The data files are large and not included in this repo. You'll need to download them yourself from the WCA website.

## Cohort

We filter for solvers with **>15 competitions**, **>200 recorded solves**, and **>2 years** of career span, then stratify-sample 50 solvers across 5 skill tiers for detailed analysis.

## How to run

```bash
# install dependencies
pip install -r requirements.txt

# RQ1 — learning curve fitting (run this first, RQ2 and RQ3 depend on it)
python RQ1_learning_curves.py

# RQ2 — distributional analysis
python RQ2_distributional_analysis.py

# RQ3 — fluctuation analysis (needs RQ1 residuals)
python RQ3_fluctuation_analysis.py
```

### Optional flags

| Flag | Script | What it does |
|------|--------|-------------|
| `--full` | RQ1, RQ2 | Run on the full eligible cohort instead of the 50-solver sample |
| `--pymc` | RQ1 | Enable Bayesian model comparison via PyMC (slow) |
| `--no-robustness` | RQ1, RQ2 | Skip robustness checks (faster runs) |

