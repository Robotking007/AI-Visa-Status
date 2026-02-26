"""
Model Training Pipeline — Visa Processing Time Prediction
==========================================================
Tasks:
  1. Train baseline regression models (Linear Regression, Random Forest,
     Extra Trees, Ridge, Gradient Boosting)
  2. Evaluate with MAE, RMSE, R², MAPE via 5-fold cross-validation
  3. Select the best model and fine-tune with RandomizedSearchCV
  4. Save artefacts (model, scaler, feature list, metrics report)
  5. Generate diagnostic plots (residuals, actual vs predicted,
     feature importance, learning curves)

Feature strategy
----------------
Only submission-time features and historical-aggregate features are used.
Post-submission outcome columns (Issued, AP, Refused, status_*, is_issued …)
and target-derived columns (pt_deviation_from_consulate, fiscal_year_mean_pt,
complexity_score) are excluded to prevent data leakage.
"""

import os
import time
import warnings
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.linear_model    import LinearRegression, Ridge, Lasso
from sklearn.ensemble        import (RandomForestRegressor,
                                     ExtraTreesRegressor,
                                     GradientBoostingRegressor)
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     RandomizedSearchCV,
                                     learning_curve)
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (mean_absolute_error,
                                     mean_squared_error,
                                     r2_score)
from scipy.stats             import randint, uniform

warnings.filterwarnings("ignore")

# ── Paths and constants ───────────────────────────────────────────────────────
DATA_PATH    = "data/engineered_visa_dataset.csv"
MODELS_DIR   = "models"
PLOTS_DIR    = "plots/model_training"
TARGET       = "processing_time_days"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


# =============================================================================
# 1.  DATA LOADING & FEATURE SELECTION
# =============================================================================

# Columns that are post-submission outcomes (data leakage) or not useful
LEAKAGE_COLS = {
    # IDs / strings
    "caseNumberFull",
    # Outcome status flags (known only after processing)
    "Issued", "AP", "Ready", "Refused", "InTransit",
    "Transfer", "NVC", "potentialAP", "Refused221g",
    "is_issued", "is_refused", "is_administrative_processing",
    "is_ready", "is_refused_221g",
    # One-hot outcome status
    "status_Administrative Processing", "status_Issued", "status_Ready",
    "status_Refused", "status_Refused221g", "status_Returned to NVC",
    "status_Transfer", "status_Transfer in Progress",
    # Target-derived engineered features
    "pt_deviation_from_consulate",
    "fiscal_year_mean_pt",       # uses same-year target values
    "complexity_score",          # built from outcome flags
}


def load_and_prepare(path: str = DATA_PATH):
    print("\n" + "="*70)
    print("LOADING & PREPARING DATA")
    print("="*70)

    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Drop target-leaking and outcome columns
    drop_cols = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"  Dropped {len(drop_cols)} leakage/outcome columns")

    # Drop rows missing the target
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"  Dropped {before - len(df):,} rows missing target. Remaining: {len(df):,}")

    # Encode 'consulate' (label encode since it's already stored as object in some runs)
    if df["consulate"].dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df["consulate"] = le.fit_transform(df["consulate"].astype(str))
        print("  Label-encoded 'consulate'")

    # Separate features and target
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET])

    # Keep only numeric columns
    X = X.select_dtypes(include=["number"])

    # Fill any remaining NaNs with column median
    X = X.fillna(X.median(numeric_only=True))

    print(f"\n  Feature matrix : {X.shape[0]:,} rows x {X.shape[1]} cols")
    print(f"  Target         : mean={y.mean():.1f}d  "
          f"std={y.std():.1f}d  "
          f"range=[{y.min():.0f}, {y.max():.0f}]")
    print(f"\n  Features used:\n  {list(X.columns)}")
    return X, y


# =============================================================================
# 2.  TRAIN / TEST SPLIT
# =============================================================================

def split(X: pd.DataFrame, y: pd.Series):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n  Train: {len(X_tr):,}   Test: {len(X_te):,}   "
          f"(split {100*(1-TEST_SIZE):.0f}/{100*TEST_SIZE:.0f})")
    return X_tr, X_te, y_tr, y_te


# =============================================================================
# 3.  METRICS HELPER
# =============================================================================

def compute_metrics(y_true, y_pred, label: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100

    if label:
        print(f"  {label:<30s}  MAE={mae:7.2f}d  "
              f"RMSE={rmse:7.2f}d  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# =============================================================================
# 4.  BASELINE MODELS
# =============================================================================

BASELINE_MODELS = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ]),
    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(alpha=0.5, max_iter=5000)),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=None,
        min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200, max_depth=None,
        min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        random_state=RANDOM_STATE
    ),
}


def train_baselines(X_tr, X_te, y_tr, y_te) -> dict:
    print("\n" + "="*70)
    print("SECTION 4 — BASELINE MODEL TRAINING")
    print("="*70)
    print(f"  {'Model':<30s}  {'MAE':>8s}  {'RMSE':>8s}  {'R2':>7s}  {'MAPE':>7s}")
    print("  " + "-"*68)

    results = {}
    trained  = {}

    for name, model in BASELINE_MODELS.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        y_pred  = model.predict(X_te)
        m = compute_metrics(y_te, y_pred, label=name)
        m["train_time_s"] = round(elapsed, 2)
        m["predictions"]  = y_pred
        results[name] = m
        trained[name] = model
        print(f"    (trained in {elapsed:.1f}s)")

    return results, trained


# =============================================================================
# 5.  CROSS-VALIDATION
# =============================================================================

def cross_validate_top(trained: dict, X: pd.DataFrame,
                       y: pd.Series, top_n: int = 3) -> dict:
    print("\n" + "="*70)
    print(f"SECTION 5 — {CV_FOLDS}-FOLD CROSS-VALIDATION (top-{top_n} models)")
    print("="*70)

    cv_results = {}
    # Use negative MAE as scoring (scikit-learn convention)
    for name, model in list(trained.items())[:top_n]:
        print(f"\n  {name} ...")
        scores_mae  = -cross_val_score(model, X, y,
                                       cv=CV_FOLDS, scoring="neg_mean_absolute_error",
                                       n_jobs=-1)
        scores_rmse = np.sqrt(-cross_val_score(model, X, y,
                                               cv=CV_FOLDS,
                                               scoring="neg_mean_squared_error",
                                               n_jobs=-1))
        scores_r2   = cross_val_score(model, X, y,
                                      cv=CV_FOLDS, scoring="r2",
                                      n_jobs=-1)
        cv_results[name] = {
            "CV_MAE_mean" : scores_mae.mean(),
            "CV_MAE_std"  : scores_mae.std(),
            "CV_RMSE_mean": scores_rmse.mean(),
            "CV_RMSE_std" : scores_rmse.std(),
            "CV_R2_mean"  : scores_r2.mean(),
            "CV_R2_std"   : scores_r2.std(),
        }
        print(f"    MAE : {scores_mae.mean():.2f} +/- {scores_mae.std():.2f}")
        print(f"    RMSE: {scores_rmse.mean():.2f} +/- {scores_rmse.std():.2f}")
        print(f"    R2  : {scores_r2.mean():.4f} +/- {scores_r2.std():.4f}")

    return cv_results


# =============================================================================
# 6.  HYPERPARAMETER FINE-TUNING
# =============================================================================

RF_PARAM_DIST = {
    "n_estimators"      : randint(100, 600),
    "max_depth"         : [None, 10, 20, 30, 40],
    "min_samples_split" : randint(2, 20),
    "min_samples_leaf"  : randint(1, 10),
    "max_features"      : ["sqrt", "log2", 0.5, 0.7, 1.0],
    "bootstrap"         : [True, False],
}

ET_PARAM_DIST = {
    "n_estimators"      : randint(100, 600),
    "max_depth"         : [None, 10, 20, 30],
    "min_samples_split" : randint(2, 20),
    "min_samples_leaf"  : randint(1, 10),
    "max_features"      : ["sqrt", "log2", 0.5, 0.7],
}

GB_PARAM_DIST = {
    "n_estimators"  : randint(100, 500),
    "learning_rate" : uniform(0.01, 0.3),
    "max_depth"     : randint(3, 8),
    "subsample"     : uniform(0.6, 0.4),
    "min_samples_leaf": randint(1, 20),
}


def fine_tune(best_name: str, X_tr, y_tr, X_te, y_te) -> tuple:
    print("\n" + "="*70)
    print(f"SECTION 6 — HYPERPARAMETER FINE-TUNING: {best_name}")
    print("="*70)

    if "Random Forest" in best_name:
        base   = RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)
        dist   = RF_PARAM_DIST
        n_iter = 30
    elif "Extra Trees" in best_name:
        base   = ExtraTreesRegressor(n_jobs=-1, random_state=RANDOM_STATE)
        dist   = ET_PARAM_DIST
        n_iter = 30
    elif "Gradient" in best_name:
        base   = GradientBoostingRegressor(random_state=RANDOM_STATE)
        dist   = GB_PARAM_DIST
        n_iter = 20
    else:
        # For linear models there's no meaningful search; return as-is
        print("  Linear model selected — no hyperparameter search needed.")
        return None, None

    print(f"  Running RandomizedSearchCV ({n_iter} iterations, {CV_FOLDS}-fold CV) ...")
    t0     = time.time()
    search = RandomizedSearchCV(
        estimator     = base,
        param_distributions = dist,
        n_iter        = n_iter,
        cv            = CV_FOLDS,
        scoring       = "neg_mean_absolute_error",
        n_jobs        = -1,
        random_state  = RANDOM_STATE,
        verbose       = 0,
        refit         = True,
    )
    search.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    best_model = search.best_estimator_
    y_pred     = best_model.predict(X_te)
    m          = compute_metrics(y_te, y_pred,
                                 label=f"Tuned {best_name}")

    print(f"\n  Best hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"    {k}: {v}")
    print(f"\n  Search completed in {elapsed:.1f}s")

    return best_model, m


# =============================================================================
# 7.  PLOTTING
# =============================================================================

def plot_model_comparison(results: dict):
    """Bar chart comparing MAE and RMSE across baseline models."""
    print("\n[Plots] Model comparison ...")
    names  = list(results.keys())
    maes   = [results[n]["MAE"]  for n in names]
    rmses  = [results[n]["RMSE"] for n in names]
    r2s    = [results[n]["R2"]   for n in names]

    x = np.arange(len(names))
    w = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Baseline Model Comparison", fontsize=14, fontweight="bold")

    for ax, vals, label, color in zip(
        axes,
        [maes, rmses, r2s],
        ["MAE (days)", "RMSE (days)", "R2 Score"],
        ["steelblue", "coral", "mediumseagreen"]
    ):
        bars = ax.bar(x, vals, color=color, edgecolor="white", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(label)
        ax.set_ylabel(label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_model_comparison.png")
    plt.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


def plot_actual_vs_predicted(y_te, results: dict, top_models: list):
    """Actual vs Predicted scatter for the top models."""
    print("\n[Plots] Actual vs predicted ...")
    n = len(top_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, top_models):
        y_pred = results[name]["predictions"]
        sample_idx  = np.random.choice(len(y_te), min(3000, len(y_te)), replace=False)
        ax.scatter(np.array(y_te)[sample_idx],
                   y_pred[sample_idx],
                   alpha=0.25, s=8, color="steelblue", rasterized=True)
        mn = min(y_te.min(), y_pred.min())
        mx = max(y_te.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
        ax.set_title(f"{name}\nMAE={results[name]['MAE']:.1f}d  "
                     f"R²={results[name]['R2']:.3f}",
                     fontsize=10)
        ax.set_xlabel("Actual Processing Time (days)")
        ax.set_ylabel("Predicted Processing Time (days)")
        ax.legend(fontsize=8)

    plt.suptitle("Actual vs Predicted Processing Time", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_actual_vs_predicted.png")
    plt.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


def plot_residuals(y_te, results: dict, top_models: list):
    """Residual distribution and residuals vs predicted for top models."""
    print("\n[Plots] Residual analysis ...")
    n    = len(top_models)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, name in enumerate(top_models):
        y_pred    = results[name]["predictions"]
        residuals = np.array(y_te) - y_pred

        # Residual distribution
        axes[0, col].hist(residuals, bins=50, color="steelblue",
                          edgecolor="white", alpha=0.8)
        axes[0, col].axvline(0, color="red", ls="--", lw=1.5)
        axes[0, col].set_title(f"{name} — Residual Distribution", fontsize=10)
        axes[0, col].set_xlabel("Residual (days)")
        axes[0, col].set_ylabel("Count")

        # Residuals vs predicted
        axes[1, col].scatter(y_pred, residuals,
                             alpha=0.25, s=8, color="coral", rasterized=True)
        axes[1, col].axhline(0, color="black", ls="--", lw=1)
        axes[1, col].set_title(f"{name} — Residuals vs Predicted", fontsize=10)
        axes[1, col].set_xlabel("Predicted (days)")
        axes[1, col].set_ylabel("Residual (days)")

    plt.suptitle("Residual Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_residuals.png")
    plt.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_names: list,
                            title: str = "Feature Importance", top_n: int = 20):
    """Horizontal bar chart of the top-N most important features."""
    print(f"\n[Plots] Feature importance ({title}) ...")

    # Handle Pipeline (extract inner model)
    actual_model = model
    if hasattr(model, "named_steps"):
        actual_model = model.named_steps.get("model", model)

    if not hasattr(actual_model, "feature_importances_"):
        # Try linear model coefficients
        if hasattr(actual_model, "coef_"):
            importances = np.abs(actual_model.coef_)
        else:
            print("  Model has no feature importances — skipping.")
            return None
    else:
        importances = actual_model.feature_importances_

    imp_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(top_n))

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    bars = ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
                   color="steelblue", edgecolor="white")
    ax.set_title(f"{title} — Top-{top_n} Features",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")

    plt.tight_layout()
    slug  = title.lower().replace(" ", "_")
    path  = os.path.join(PLOTS_DIR, f"04_feature_importance_{slug}.png")
    plt.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")
    print(f"\n  Top-10 features:\n"
          f"{imp_df.head(10).to_string(index=False)}")
    return imp_df


def plot_learning_curves(model, X_tr, y_tr, model_name: str):
    """Learning curves showing train vs CV score as training size grows."""
    print(f"\n[Plots] Learning curves for {model_name} ...")

    train_sizes = np.linspace(0.1, 1.0, 8)
    try:
        tr_sizes, tr_scores, cv_scores = learning_curve(
            model, X_tr, y_tr,
            train_sizes=train_sizes,
            cv=3, scoring="neg_mean_absolute_error",
            n_jobs=-1, verbose=0
        )
    except Exception as exc:
        print(f"  Skipped ({exc})")
        return

    tr_mae = -tr_scores.mean(axis=1)
    cv_mae = -cv_scores.mean(axis=1)
    tr_std = tr_scores.std(axis=1)
    cv_std = cv_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tr_sizes, tr_mae, "o-", color="steelblue", label="Train MAE")
    ax.fill_between(tr_sizes,
                    tr_mae - tr_std, tr_mae + tr_std,
                    alpha=0.15, color="steelblue")
    ax.plot(tr_sizes, cv_mae, "o-", color="coral", label="CV MAE")
    ax.fill_between(tr_sizes,
                    cv_mae - cv_std, cv_mae + cv_std,
                    alpha=0.15, color="coral")
    ax.set_title(f"Learning Curves — {model_name}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("MAE (days)")
    ax.legend()

    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    path = os.path.join(PLOTS_DIR, f"05_learning_curves_{slug}.png")
    plt.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


def plot_cv_box(cv_results: dict):
    """Boxplot comparing CV MAE distributions across models."""
    print("\n[Plots] CV score comparison ...")
    if not cv_results:
        return

    names    = list(cv_results.keys())
    mae_vals = [cv_results[n]["CV_MAE_mean"] for n in names]
    mae_stds = [cv_results[n]["CV_MAE_std"]  for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, mae_vals, yerr=mae_stds, capsize=5,
                  color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title(f"{CV_FOLDS}-Fold CV — Mean MAE with Std Dev",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("MAE (days)")
    for bar, val, std in zip(bars, mae_vals, mae_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "06_cv_mae_comparison.png")
    plt.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


def plot_tuning_improvement(baseline_m: dict, tuned_m: dict, model_name: str):
    """Before/After bar chart showing fine-tuning improvement."""
    print("\n[Plots] Fine-tuning improvement ...")
    metrics  = ["MAE", "RMSE", "R2", "MAPE"]
    baseline = [baseline_m.get(k, 0) for k in metrics]
    tuned    = [tuned_m.get(k, 0) for k in metrics]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, baseline, w, label="Baseline", color="coral",
           edgecolor="white", alpha=0.85)
    ax.bar(x + w/2, tuned,    w, label="Fine-tuned", color="steelblue",
           edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title(f"Fine-Tuning Impact — {model_name}",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Value")
    ax.legend()

    for i, (bv, tv) in enumerate(zip(baseline, tuned)):
        ax.text(i - w/2, bv + max(baseline) * 0.01,
                f"{bv:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, tv + max(baseline) * 0.01,
                f"{tv:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "07_tuning_improvement.png")
    plt.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


def plot_error_distribution_best(y_te, y_pred, model_name: str):
    """Absolute error distribution with quantile markers for the best model."""
    print("\n[Plots] Error distribution for best model ...")
    errors = np.abs(np.array(y_te) - y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Prediction Error Analysis — {model_name}",
                 fontsize=13, fontweight="bold")

    # Histogram of |error|
    axes[0].hist(errors, bins=50, color="steelblue", edgecolor="white")
    for q, c, ls in [(50, "red", "--"), (75, "orange", ":"), (90, "green", "-.")]:
        qv = np.percentile(errors, q)
        axes[0].axvline(qv, color=c, ls=ls, label=f"P{q}={qv:.1f}d")
    axes[0].set_title("Absolute Error Distribution")
    axes[0].set_xlabel("Absolute Error (days)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Cumulative error distribution
    sorted_err = np.sort(errors)
    cum_pct    = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    axes[1].plot(sorted_err, cum_pct, color="steelblue", lw=2)
    for tol, c in [(7, "red"), (14, "orange"), (30, "green")]:
        pct_within = np.mean(errors <= tol) * 100
        axes[1].axvline(tol, color=c, ls="--",
                        label=f"Within {tol}d: {pct_within:.1f}%")
    axes[1].set_title("Cumulative Error Distribution")
    axes[1].set_xlabel("Absolute Error (days)")
    axes[1].set_ylabel("Cumulative % of Predictions")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "08_error_distribution_best.png")
    plt.savefig(path)
    plt.close("all")
    print(f"  Saved: {path}")


# =============================================================================
# 8.  SAVE ARTEFACTS
# =============================================================================

def save_artefacts(model, feature_names: list,
                   metrics: dict, model_name: str,
                   extra_info: dict = None):
    print("\n" + "="*70)
    print("SAVING MODEL ARTEFACTS")
    print("="*70)

    # Save model
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved      : {model_path}")

    # Save feature list
    feat_path = os.path.join(MODELS_DIR, "feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Feature list     : {feat_path}")

    # Save metrics + meta
    report = {
        "model_name"    : model_name,
        "train_date"    : pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "n_features"    : len(feature_names),
        "test_metrics"  : {k: round(float(v), 4) for k, v in metrics.items()
                           if k not in ("predictions",)},
        "extra_info"    : extra_info or {},
    }
    report_path = os.path.join(MODELS_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Training report  : {report_path}")

    return model_path


# =============================================================================
# 9.  MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("VISA PROCESSING TIME PREDICTION — MODEL TRAINING")
    print("="*70)
    print(f"Run timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_and_prepare(DATA_PATH)
    feature_names = list(X.columns)

    X_tr, X_te, y_tr, y_te = split(X, y)

    # ── Baseline training ─────────────────────────────────────────────────────
    baseline_results, trained_models = train_baselines(X_tr, X_te, y_tr, y_te)

    # Rank by MAE (lower is better)
    ranked = sorted(baseline_results.items(), key=lambda kv: kv[1]["MAE"])
    print("\n" + "="*70)
    print("SECTION 4 — RANKED BASELINE RESULTS (by MAE)")
    print("="*70)
    print(f"  {'Rank':<5} {'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>7} {'MAPE':>7}")
    print("  " + "-"*65)
    for rank, (name, m) in enumerate(ranked, 1):
        print(f"  {rank:<5} {name:<30} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
              f"{m['R2']:>7.4f} {m['MAPE']:>6.2f}%")

    # ── Cross-validation on top-3 ─────────────────────────────────────────────
    top3_names  = [name for name, _ in ranked[:3]]
    top3_models = {n: trained_models[n] for n in top3_names}
    cv_results  = cross_validate_top(top3_models, X, y, top_n=3)

    # ── Fine-tune the best ────────────────────────────────────────────────────
    best_name     = ranked[0][0]
    baseline_best = ranked[0][1]

    tuned_model, tuned_metrics = fine_tune(
        best_name, X_tr, y_tr, X_te, y_te
    )

    # Use baseline if fine-tuning was skipped (linear models)
    if tuned_model is None:
        tuned_model   = trained_models[best_name]
        tuned_metrics = baseline_best

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SECTION 7 — GENERATING PLOTS")
    print("="*70)

    plot_model_comparison(baseline_results)
    plot_actual_vs_predicted(y_te, baseline_results, top3_names)
    plot_residuals(y_te, baseline_results, top3_names)

    # Feature importance for the best model (baseline)
    plot_feature_importance(
        trained_models[best_name], feature_names,
        title=f"{best_name} (Baseline)"
    )
    # Feature importance for fine-tuned model
    if tuned_model is not trained_models[best_name]:
        plot_feature_importance(
            tuned_model, feature_names,
            title=f"{best_name} (Fine-Tuned)"
        )

    # Learning curves for the best baseline (uses a smaller sample for speed)
    plot_learning_curves(trained_models[best_name], X_tr, y_tr, best_name)

    plot_cv_box(cv_results)

    # Show fine-tuning improvement only if it happened
    if tuned_model is not trained_models[best_name]:
        plot_tuning_improvement(baseline_best, tuned_metrics, best_name)

    # Error distribution for the best (tuned) model
    y_pred_best = tuned_model.predict(X_te)
    plot_error_distribution_best(y_te, y_pred_best, best_name)

    # ── Save artefacts ────────────────────────────────────────────────────────
    extra = {
        "baseline_ranking": [
            {"rank": i + 1, "model": n,
             "MAE": round(m["MAE"], 2), "RMSE": round(m["RMSE"], 2),
             "R2": round(m["R2"], 4), "MAPE": round(m["MAPE"], 2)}
            for i, (n, m) in enumerate(ranked)
        ],
        "cv_results": {
            n: {k: round(float(v), 4) for k, v in d.items()}
            for n, d in cv_results.items()
        },
    }
    save_artefacts(tuned_model, feature_names,
                   tuned_metrics, best_name, extra_info=extra)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TRAINING COMPLETE — FINAL SUMMARY")
    print("="*70)
    print(f"\n  Best model       : {best_name} (Fine-Tuned)")
    print(f"  Test MAE         : {tuned_metrics['MAE']:.2f} days")
    print(f"  Test RMSE        : {tuned_metrics['RMSE']:.2f} days")
    print(f"  Test R2          : {tuned_metrics['R2']:.4f}")
    print(f"  Test MAPE        : {tuned_metrics['MAPE']:.2f}%")
    print(f"\n  Artefacts saved  : {MODELS_DIR}/")
    print(f"  Plots saved      : {PLOTS_DIR}/")
    print("\n  Next steps:")
    print("    1. Load model: joblib.load('models/best_model.joblib')")
    print("    2. Load features: json.load(open('models/feature_names.json'))")
    print("    3. Review training_report.json for full metrics")
    print("    4. Consider XGBoost / LightGBM for further improvement")
    print("=" * 70)


if __name__ == "__main__":
    main()
