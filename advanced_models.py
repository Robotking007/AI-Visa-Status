"""
Advanced Model Training — XGBoost, LightGBM, Stacking Ensemble
================================================================
Builds on the foundation in model_training.py:
  1. Train XGBoost and LightGBM with tuned defaults
  2. Fine-tune both with RandomizedSearchCV
  3. Build a Stacking Ensemble (Extra Trees + XGBoost + LightGBM → Ridge meta)
  4. Compare all advanced models against the existing Extra Trees baseline
  5. Save the best model (overwrites models/best_model.joblib if improved)
  6. Generate diagnostic plots

Same leakage-exclusion strategy as model_training.py — only submission-time
and historical-aggregate features are used.
"""

import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import randint, uniform

from sklearn.linear_model    import Ridge
from sklearn.ensemble        import (ExtraTreesRegressor,
                                     StackingRegressor)
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     cross_validate,
                                     RandomizedSearchCV,
                                     learning_curve)
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (mean_absolute_error,
                                     mean_squared_error,
                                     r2_score)

import xgboost  as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/engineered_visa_dataset.csv"
MODELS_DIR   = "models"
PLOTS_DIR    = "plots/advanced_models"
TARGET       = "processing_time_days"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

LEAKAGE_COLS = {
    "caseNumberFull",
    "Issued", "AP", "Ready", "Refused", "InTransit",
    "Transfer", "NVC", "potentialAP", "Refused221g",
    "is_issued", "is_refused", "is_administrative_processing",
    "is_ready", "is_refused_221g",
    "status_Administrative Processing", "status_Issued", "status_Ready",
    "status_Refused", "status_Refused221g", "status_Returned to NVC",
    "status_Transfer", "status_Transfer in Progress",
    "pt_deviation_from_consulate",
    "fiscal_year_mean_pt",
    "complexity_score",
}


# =============================================================================
# 1.  DATA
# =============================================================================

def load_and_prepare(path: str = DATA_PATH):
    print("\n" + "="*70)
    print("LOADING & PREPARING DATA")
    print("="*70)

    df = pd.read_csv(path, low_memory=False)
    drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(subset=[TARGET])

    if df["consulate"].dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["consulate"] = le.fit_transform(df["consulate"].astype(str))

    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET]).select_dtypes(include=["number"])
    X = X.fillna(X.median(numeric_only=True))

    print(f"  Shape: {X.shape[0]:,} x {X.shape[1]}  |  Target mean={y.mean():.1f}d")
    return X, y


def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# =============================================================================
# 2.  METRICS
# =============================================================================

def metrics(y_true, y_pred, label="") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
    if label:
        print(f"  {label:<35s}  MAE={mae:6.2f}d  RMSE={rmse:6.2f}d  "
              f"R2={r2:.4f}  MAPE={mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "predictions": y_pred}


# =============================================================================
# 3.  MODEL DEFINITIONS
# =============================================================================

def make_models():
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators     = 400,
            learning_rate    = 0.05,
            max_depth        = 6,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 5,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            n_jobs           = 1,      # 1 thread per model; RSCV handles outer parallelism
            random_state     = RANDOM_STATE,
            verbosity        = 0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators      = 400,
            learning_rate     = 0.05,
            num_leaves        = 63,
            max_depth         = -1,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            min_child_samples = 20,
            reg_alpha         = 0.1,
            reg_lambda        = 1.0,
            n_jobs            = 1,        # avoid OpenMP conflict with XGBoost on Windows
            force_row_wise    = True,     # disable col-wise parallelism
            random_state      = RANDOM_STATE,
            verbose           = -1,
        ),
    }


# =============================================================================
# 4.  BASELINE TRAIN
# =============================================================================

def train_baselines(X_tr, X_te, y_tr, y_te):
    print("\n" + "="*70)
    print("SECTION 4 — XGBoost & LightGBM BASELINE TRAINING")
    print("="*70)

    models  = make_models()
    results = {}
    trained = {}

    import sys
    for name, model in models.items():
        print(f"  Training {name} ...", flush=True)
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        y_pred  = model.predict(X_te)
        m = metrics(y_te, y_pred, label=name)
        m["train_time_s"] = round(elapsed, 2)
        results[name] = m
        trained[name] = model
        print(f"    (trained in {elapsed:.1f}s)", flush=True)
        sys.stdout.flush()

    return results, trained


# =============================================================================
# 5.  CROSS-VALIDATION
# =============================================================================

def cross_validate_models(trained: dict, X, y):
    """
    Run 3-fold CV on a 30% stratified subsample to estimate generalisation;
    fully sequential (n_jobs=1) to avoid Windows OpenMP crashes with XGBoost/LightGBM.
    """
    print("\n" + "="*70)
    print("SECTION 5 — 3-FOLD CROSS-VALIDATION (30% subsample, sequential)")
    print("="*70)

    # Subsample to 30% for speed; keeps class distribution intact
    n_sample = int(len(X) * 0.30)
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X), n_sample, replace=False)
    X_cv = X.iloc[idx]
    y_cv = y.iloc[idx]
    print(f"  CV sample: {len(X_cv):,} rows ({100*n_sample/len(X):.0f}% of full set)")

    cv_results = {}
    scoring = {
        "mae" : "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2"  : "r2",
    }
    for name, model in trained.items():
        print(f"\n  {name} ...", flush=True)
        t0 = time.time()
        cv = cross_validate(model, X_cv, y_cv, cv=3,
                            scoring=scoring, n_jobs=1, verbose=0)
        elapsed = time.time() - t0
        mae_mean  = round(float(-cv["test_mae"].mean()),   2)
        mae_std   = round(float(cv["test_mae"].std()),     2)
        rmse_mean = round(float(-cv["test_rmse"].mean()),  2)
        rmse_std  = round(float(cv["test_rmse"].std()),    2)
        r2_mean   = round(float(cv["test_r2"].mean()),     4)
        r2_std    = round(float(cv["test_r2"].std()),      4)
        cv_results[name] = {
            "CV_MAE_mean" : mae_mean,  "CV_MAE_std" : mae_std,
            "CV_RMSE_mean": rmse_mean, "CV_RMSE_std": rmse_std,
            "CV_R2_mean"  : r2_mean,   "CV_R2_std"  : r2_std,
        }
        print(f"    MAE : {mae_mean:.2f} +/- {mae_std:.2f}")
        print(f"    RMSE: {rmse_mean:.2f} +/- {rmse_std:.2f}")
        print(f"    R2  : {r2_mean:.4f} +/- {r2_std:.4f}")
        print(f"    (completed in {elapsed:.1f}s)", flush=True)

    return cv_results


# =============================================================================
# 6.  FINE-TUNING
# =============================================================================

XGB_PARAM_DIST = {
    "n_estimators"     : randint(200, 800),
    "learning_rate"    : uniform(0.01, 0.15),
    "max_depth"        : randint(3, 10),
    "subsample"        : uniform(0.6, 0.4),
    "colsample_bytree" : uniform(0.5, 0.5),
    "min_child_weight" : randint(1, 20),
    "reg_alpha"        : uniform(0, 1),
    "reg_lambda"       : uniform(0.5, 2),
    "gamma"            : uniform(0, 0.5),
    "n_jobs"           : [1],           # fixed: RSCV handles outer parallelism
}

LGBM_PARAM_DIST = {
    "n_estimators"      : randint(200, 800),
    "learning_rate"     : uniform(0.01, 0.15),
    "num_leaves"        : randint(20, 200),
    "max_depth"         : [-1, 6, 8, 10, 12],
    "subsample"         : uniform(0.6, 0.4),
    "colsample_bytree"  : uniform(0.5, 0.5),
    "min_child_samples" : randint(5, 50),
    "reg_alpha"         : uniform(0, 1),
    "reg_lambda"        : uniform(0.5, 2),
    "n_jobs"            : [1],          # keep fixed to avoid OpenMP conflicts
    "force_row_wise"    : [True],
}


def fine_tune(name: str, base_model, X_tr, y_tr, X_te, y_te, n_iter=10):
    print(f"\n  RandomizedSearchCV for {name} ({n_iter} iters, 3-fold, subsample=40%, sequential) ...")
    dist = XGB_PARAM_DIST if "XGBoost" in name else LGBM_PARAM_DIST

    # Use 40% subsample for speed in RSCV fitting
    n_sub = int(len(X_tr) * 0.40)
    rng   = np.random.RandomState(RANDOM_STATE)
    idx   = rng.choice(len(X_tr), n_sub, replace=False)
    X_sub = X_tr.iloc[idx]
    y_sub = y_tr.iloc[idx]
    print(f"  RSCV subsample: {len(X_sub):,} rows for search; refit on full train set.")

    search = RandomizedSearchCV(
        estimator           = base_model,
        param_distributions = dist,
        n_iter              = n_iter,
        cv                  = 3,
        scoring             = "neg_mean_absolute_error",
        n_jobs              = 1,     # fully sequential — avoids ALL OMP crashes
        random_state        = RANDOM_STATE,
        refit               = True,
        verbose             = 1,
    )
    t0 = time.time()
    search.fit(X_sub, y_sub)   # search on subsample
    t1 = time.time()

    # Refit best params on full training set for final model
    best_params = {k: v for k, v in search.best_params_.items()}
    best = type(base_model)(**best_params)
    best.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    y_pred = best.predict(X_te)
    m = metrics(y_te, y_pred, label=f"{name} (Tuned)")
    m["train_time_s"] = round(elapsed, 2)

    print(f"  Best params:")
    for k, v in search.best_params_.items():
        print(f"    {k}: {v}")
    print(f"  RSCV: {t1-t0:.1f}s  |  Final refit: {elapsed-(t1-t0):.1f}s  |  Total: {elapsed:.1f}s")

    return best, m


def tune_all(baseline_trained: dict, X_tr, y_tr, X_te, y_te):
    print("\n" + "="*70)
    print("SECTION 6 — HYPERPARAMETER FINE-TUNING")
    print("="*70)

    tuned_trained  = {}
    tuned_results  = {}

    for name, model in baseline_trained.items():
        base_clone = type(model)(**model.get_params())
        tuned_model, m = fine_tune(name, base_clone, X_tr, y_tr, X_te, y_te)
        tuned_trained[f"{name} (Tuned)"] = tuned_model
        tuned_results[f"{name} (Tuned)"] = m

    return tuned_trained, tuned_results


# =============================================================================
# 7.  STACKING ENSEMBLE
# =============================================================================

def train_stacking(X_tr, X_te, y_tr, y_te,
                   xgb_model, lgbm_model,
                   feature_names: list):
    print("\n" + "="*70)
    print("SECTION 7 — STACKING ENSEMBLE")
    print("  Base learners : Extra Trees (loaded) + XGBoost + LightGBM")
    print("  Meta learner  : Ridge (with StandardScaler)")
    print("="*70)

    # Load saved Extra Trees from previous run
    et_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if os.path.exists(et_path):
        saved_model = joblib.load(et_path)
        print(f"  Loaded saved Extra Trees from {et_path}")
        et_base = ("extra_trees", saved_model)
    else:
        et_base = ("extra_trees", ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE
        ))
        print("  No saved model found — training fresh Extra Trees")

    from sklearn.pipeline import Pipeline

    stack = StackingRegressor(
        estimators=[
            et_base,
            ("xgboost",  xgb_model),
            ("lightgbm", lgbm_model),
        ],
        final_estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=10.0)),
        ]),
        cv             = 3,
        n_jobs         = 1,       # sequential: avoids OMP crash on Windows
        passthrough    = False,
    )

    t0 = time.time()
    stack.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    y_pred  = stack.predict(X_te)
    m = metrics(y_te, y_pred, label="Stacking Ensemble")
    m["train_time_s"] = round(elapsed, 2)
    print(f"  Stacking trained in {elapsed:.1f}s")

    return stack, m


# =============================================================================
# 8.  PLOTS
# =============================================================================

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


def plot_comparison(all_results: dict, reference: dict, ref_name: str = "Extra Trees (Baseline)"):
    """Bar chart for all models: MAE, RMSE, R² — reference line marks previous best."""
    print("\n[Plots] Model comparison ...")

    # Build table
    rows = []
    for name, m in all_results.items():
        rows.append({"Model": name, "MAE": m["MAE"],
                     "RMSE": m["RMSE"], "R2": m["R2"]})
    df = pd.DataFrame(rows).sort_values("MAE")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Advanced Model Comparison", fontsize=14, fontweight="bold")

    for ax, col, color, ref_val in zip(
        axes,
        ["MAE", "RMSE", "R2"],
        ["steelblue", "coral", "mediumseagreen"],
        [reference["MAE"], reference["RMSE"], reference["R2"]],
    ):
        bars = ax.barh(df["Model"], df[col], color=color, edgecolor="white", alpha=0.85)
        ax.axvline(ref_val, color="black", ls="--", lw=1.5,
                   label=f"{ref_name}: {ref_val:.2f}")
        ax.set_title(col)
        ax.set_xlabel(f"{col} value")
        ax.legend(fontsize=8)
        for bar, val in zip(bars, df[col]):
            ax.text(val + max(df[col].abs()) * 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)

    _savefig(os.path.join(PLOTS_DIR, "01_model_comparison.png"))


def plot_actual_vs_predicted(y_te, all_results: dict, top_n: int = 3):
    print("\n[Plots] Actual vs predicted ...")
    names = sorted(all_results, key=lambda n: all_results[n]["MAE"])[:top_n]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        y_pred = all_results[name]["predictions"]
        idx    = np.random.choice(len(y_te), min(3000, len(y_te)), replace=False)
        ax.scatter(np.array(y_te)[idx], y_pred[idx],
                   alpha=0.25, s=8, color="steelblue", rasterized=True)
        mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect")
        ax.set_title(f"{name}\nMAE={all_results[name]['MAE']:.1f}d  "
                     f"R²={all_results[name]['R2']:.3f}", fontsize=10)
        ax.set_xlabel("Actual (days)")
        ax.set_ylabel("Predicted (days)")
        ax.legend(fontsize=8)

    plt.suptitle("Actual vs Predicted — Top-3 Models", fontsize=13, fontweight="bold")
    _savefig(os.path.join(PLOTS_DIR, "02_actual_vs_predicted.png"))


def plot_residuals(y_te, all_results: dict, top_n: int = 3):
    print("\n[Plots] Residuals ...")
    names = sorted(all_results, key=lambda n: all_results[n]["MAE"])[:top_n]
    fig, axes = plt.subplots(2, len(names), figsize=(6 * len(names), 10))
    if len(names) == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(names):
        y_pred    = all_results[name]["predictions"]
        residuals = np.array(y_te) - y_pred

        axes[0, col].hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        axes[0, col].axvline(0, color="red", ls="--", lw=1.5)
        axes[0, col].set_title(f"{name} — Residual Distribution", fontsize=9)
        axes[0, col].set_xlabel("Residual (days)")

        axes[1, col].scatter(y_pred, residuals, alpha=0.2, s=6, color="coral", rasterized=True)
        axes[1, col].axhline(0, color="black", ls="--")
        axes[1, col].set_title(f"{name} — Residuals vs Predicted", fontsize=9)
        axes[1, col].set_xlabel("Predicted (days)")
        axes[1, col].set_ylabel("Residual (days)")

    plt.suptitle("Residual Analysis — Top-3 Models", fontsize=13, fontweight="bold")
    _savefig(os.path.join(PLOTS_DIR, "03_residuals.png"))


def plot_feature_importance(model, feature_names: list, title: str, top_n: int = 20):
    print(f"\n[Plots] Feature importance — {title} ...")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("  Model has no feature_importances_ — skipping.")
        return

    imp_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(top_n))

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
            color="steelblue", edgecolor="white")
    ax.set_title(f"{title} — Top-{top_n} Features", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")

    slug = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    _savefig(os.path.join(PLOTS_DIR, f"04_feature_importance_{slug}.png"))

    print(f"\n  Top-10:\n{imp_df.head(10).to_string(index=False)}")


def plot_error_distribution(y_te, y_pred, model_name: str):
    print(f"\n[Plots] Error distribution — {model_name} ...")
    errors = np.abs(np.array(y_te) - y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Error Analysis — {model_name}", fontsize=13, fontweight="bold")

    axes[0].hist(errors, bins=50, color="steelblue", edgecolor="white")
    for q, c, ls in [(50, "red", "--"), (75, "orange", ":"), (90, "green", "-.")]:
        qv = np.percentile(errors, q)
        axes[0].axvline(qv, color=c, ls=ls, label=f"P{q}={qv:.1f}d")
    axes[0].set_title("Absolute Error Distribution")
    axes[0].set_xlabel("Absolute Error (days)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    sorted_err = np.sort(errors)
    cum_pct    = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    axes[1].plot(sorted_err, cum_pct, color="steelblue", lw=2)
    for tol, c in [(7, "red"), (14, "orange"), (30, "green")]:
        pct = np.mean(errors <= tol) * 100
        axes[1].axvline(tol, color=c, ls="--", label=f"Within {tol}d: {pct:.1f}%")
    axes[1].set_title("Cumulative Error Distribution")
    axes[1].set_xlabel("Absolute Error (days)")
    axes[1].set_ylabel("Cumulative % of Predictions")
    axes[1].legend()

    _savefig(os.path.join(PLOTS_DIR, "05_error_distribution_best.png"))


def plot_learning_curve(model, X_tr, y_tr, model_name: str, max_rows: int = 20000):
    """Learning curve on a capped subsample for speed."""
    print(f"\n[Plots] Learning curve — {model_name} (max {max_rows:,} rows) ...")
    if len(X_tr) > max_rows:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), max_rows, replace=False)
        X_s = X_tr.iloc[idx]
        y_s = y_tr.iloc[idx]
    else:
        X_s, y_s = X_tr, y_tr
    try:
        tr_sizes, tr_scores, cv_scores = learning_curve(
            model, X_s, y_s,
            train_sizes=np.linspace(0.2, 1.0, 5),
            cv=3, scoring="neg_mean_absolute_error",
            n_jobs=1, verbose=0       # n_jobs=1 avoids OMP crash on Windows
        )
    except Exception as e:
        print(f"  Skipped ({e})")
        return

    tr_mae = -tr_scores.mean(axis=1)
    cv_mae = -cv_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tr_sizes, tr_mae, "o-", color="steelblue", label="Train MAE")
    ax.fill_between(tr_sizes, tr_mae - tr_scores.std(axis=1),
                    tr_mae + tr_scores.std(axis=1), alpha=0.15, color="steelblue")
    ax.plot(tr_sizes, cv_mae, "o-", color="coral", label="CV MAE")
    ax.fill_between(tr_sizes, cv_mae - cv_scores.std(axis=1),
                    cv_mae + cv_scores.std(axis=1), alpha=0.15, color="coral")
    ax.set_title(f"Learning Curves — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("MAE (days)")
    ax.legend()

    slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    _savefig(os.path.join(PLOTS_DIR, f"06_learning_curves_{slug}.png"))


def plot_scorecard(all_results: dict):
    """Table-style heatmap of all models x metrics."""
    print("\n[Plots] Scorecard heatmap ...")
    rows = []
    for name, m in all_results.items():
        rows.append({"Model": name, "MAE": m["MAE"], "RMSE": m["RMSE"],
                     "R2": m["R2"], "MAPE": m["MAPE"]})
    df = pd.DataFrame(rows).set_index("Model").sort_values("MAE")

    # For R2, higher is better; for others, lower is better
    normed = df.copy()
    for col in ["MAE", "RMSE", "MAPE"]:
        normed[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min() + 1e-9)
    normed["R2"] = (df["R2"] - df["R2"].min()) / (df["R2"].max() - df["R2"].min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))
    sns.heatmap(normed, annot=df.round(2), fmt="g",
                cmap="RdYlGn", ax=ax, linewidths=0.5,
                cbar_kws={"label": "Normalised score (higher=better)"})
    ax.set_title("Model Scorecard (raw values annotated)", fontsize=12, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _savefig(os.path.join(PLOTS_DIR, "07_scorecard.png"))


# =============================================================================
# 9.  SAVE BEST
# =============================================================================

def save_best(all_results: dict, all_trained: dict,
              feature_names: list, existing_mae: float):
    print("\n" + "="*70)
    print("SAVING BEST MODEL")
    print("="*70)

    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["MAE"])
    best_name, best_m = ranked[0]
    best_model = all_trained[best_name]

    if best_m["MAE"] < existing_mae:
        model_path = os.path.join(MODELS_DIR, "best_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"  New best model: {best_name}  MAE={best_m['MAE']:.2f}d "
              f"(improved from {existing_mae:.2f}d)")
        print(f"  Saved to: {model_path}")

        feat_path = os.path.join(MODELS_DIR, "feature_names.json")
        with open(feat_path, "w") as f:
            json.dump(feature_names, f, indent=2)

        report = {
            "model_name"   : best_name,
            "train_date"   : pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "n_features"   : len(feature_names),
            "test_metrics" : {k: round(float(v), 4) for k, v in best_m.items()
                              if k != "predictions"},
            "all_models"   : [
                {"rank": i+1, "model": n,
                 "MAE":  round(m["MAE"],  2),
                 "RMSE": round(m["RMSE"], 2),
                 "R2":   round(m["R2"],   4),
                 "MAPE": round(m["MAPE"], 2)}
                for i, (n, m) in enumerate(ranked)
            ],
        }
        rpt_path = os.path.join(MODELS_DIR, "training_report_advanced.json")
        with open(rpt_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report: {rpt_path}")
    else:
        print(f"  Existing model (MAE={existing_mae:.2f}d) is still better "
              f"than {best_name} (MAE={best_m['MAE']:.2f}d). Not overwriting.")

    return best_name, best_m, best_model


# =============================================================================
# 10.  MAIN
# =============================================================================

EXISTING_ET_MAE  = 55.99   # from previous model_training.py run
EXISTING_ET_RMSE = 74.74
EXISTING_ET_R2   = 0.4254


def main():
    print("\n" + "="*70)
    print("VISA PROCESSING TIME — ADVANCED MODEL TRAINING")
    print("="*70)
    print(f"Run timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Baseline (Extra Trees, Tuned): MAE={EXISTING_ET_MAE}d  "
          f"RMSE={EXISTING_ET_RMSE}d  R2={EXISTING_ET_R2}")

    # ── Data ──────────────────────────────────────────────────────────────────
    X, y = load_and_prepare()
    feature_names = list(X.columns)
    X_tr, X_te, y_tr, y_te = split(X, y)

    # ── Baseline XGB + LGBM ───────────────────────────────────────────────────
    base_results, base_trained = train_baselines(X_tr, X_te, y_tr, y_te)

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv_results = cross_validate_models(base_trained, X, y)

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SECTION 6 — HYPERPARAMETER FINE-TUNING")
    print("="*70)
    tuned_trained = {}
    tuned_results = {}
    for name, model in base_trained.items():
        base_clone = type(model)(**model.get_params())
        tm, m = fine_tune(name, base_clone, X_tr, y_tr, X_te, y_te, n_iter=20)
        tuned_trained[f"{name} (Tuned)"] = tm
        tuned_results[f"{name} (Tuned)"] = m

    # ── Stacking ──────────────────────────────────────────────────────────────
    # Use tuned models as stacking base learners
    xgb_for_stack  = list(tuned_trained.values())[0]   # Tuned XGBoost
    lgbm_for_stack = list(tuned_trained.values())[1]   # Tuned LightGBM
    stack, stack_m = train_stacking(
        X_tr, X_te, y_tr, y_te, xgb_for_stack, lgbm_for_stack, feature_names
    )

    # ── Collect all results ───────────────────────────────────────────────────
    all_results = {}
    all_results.update(base_results)
    all_results.update(tuned_results)
    all_results["Stacking Ensemble"] = stack_m

    all_trained = {}
    all_trained.update(base_trained)
    all_trained.update(tuned_trained)
    all_trained["Stacking Ensemble"] = stack

    # ── Ranked summary ────────────────────────────────────────────────────────
    ranked = sorted(all_results.items(), key=lambda kv: kv[1]["MAE"])
    print("\n" + "="*70)
    print("RANKED RESULTS (all advanced models)")
    print("="*70)
    print(f"  {'Rank':<4} {'Model':<35} {'MAE':>7} {'RMSE':>7} {'R2':>7} {'MAPE':>7}")
    print("  " + "-"*65)
    for rank, (name, m) in enumerate(ranked, 1):
        print(f"  {rank:<4} {name:<35} {m['MAE']:>7.2f} {m['RMSE']:>7.2f} "
              f"{m['R2']:>7.4f} {m['MAPE']:>6.2f}%")
    print(f"\n  Previous best (Extra Trees Tuned): "
          f"MAE={EXISTING_ET_MAE}d  RMSE={EXISTING_ET_RMSE}d  R2={EXISTING_ET_R2}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SECTION 8 — GENERATING PLOTS")
    print("="*70)

    ref = {"MAE": EXISTING_ET_MAE, "RMSE": EXISTING_ET_RMSE, "R2": EXISTING_ET_R2}
    plot_comparison(all_results, reference=ref)
    plot_actual_vs_predicted(y_te, all_results, top_n=3)
    plot_residuals(y_te, all_results, top_n=3)

    # Feature importance for tuned XGBoost and LightGBM
    for name, model in tuned_trained.items():
        plot_feature_importance(model, feature_names, title=name)

    # Learning curves for best model
    best_adv_name = ranked[0][0]
    plot_learning_curve(all_trained[best_adv_name], X_tr, y_tr, best_adv_name)

    # Error distribution for best model
    plot_error_distribution(y_te, all_results[best_adv_name]["predictions"],
                            best_adv_name)

    plot_scorecard(all_results)

    # ── Save best ─────────────────────────────────────────────────────────────
    final_name, final_m, final_model = save_best(
        all_results, all_trained,
        feature_names, existing_mae=EXISTING_ET_MAE
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ADVANCED TRAINING COMPLETE — FINAL SUMMARY")
    print("="*70)
    print(f"\n  Best advanced model : {final_name}")
    print(f"  Test MAE            : {final_m['MAE']:.2f} days")
    print(f"  Test RMSE           : {final_m['RMSE']:.2f} days")
    print(f"  Test R2             : {final_m['R2']:.4f}")
    print(f"  Test MAPE           : {final_m['MAPE']:.2f}%")
    improvement = EXISTING_ET_MAE - final_m["MAE"]
    print(f"\n  MAE improvement vs Extra Trees baseline: {improvement:+.2f} days")
    print(f"\n  Artefacts saved  : {MODELS_DIR}/")
    print(f"  Plots saved      : {PLOTS_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
