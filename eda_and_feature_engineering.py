"""
EDA and Feature Engineering for Visa Status Prediction
=======================================================
Tasks:
  1. Conduct EDA with visualizations (Matplotlib / Seaborn)
  2. Identify correlations between features and processing times
  3. Engineer features:  seasonal index, country-specific averages, and more
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend avoids display issues
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from datetime import datetime

warnings.filterwarnings("ignore")

# -- Plot style ---------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET = "processing_time_days"


# =============================================================================
# 1.  DATA LOADING
# =============================================================================

def load_data(path: str = "data/processed_visa_dataset.csv") -> pd.DataFrame:
    """Load the pre-processed visa dataset and do minimal type coercion."""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at '{path}'. "
            "Run data_preprocessing.py first."
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns")

    # Coerce numeric cols stored as objects
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    # Drop rows missing the target
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"  Dropped {before - len(df):,} rows missing '{TARGET}'. "
          f"Remaining: {len(df):,}")
    print(f"\nColumn dtype overview:\n{df.dtypes.value_counts().to_string()}")
    return df


# =============================================================================
# 2.  EDA -- VISUALIZATIONS
# =============================================================================

class EDAVisualizer:
    """All EDA visualisation methods. Each method saves a PNG file."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -------------------------------------------------------------------------
    def plot_processing_time_distribution(self):
        """2.1  Processing-time distribution (histogram + box plot)."""
        print("\n[EDA] 2.1 Processing-time distribution ...")
        data = self.df[TARGET].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Processing Time Distribution", fontsize=14, fontweight="bold")

        sns.histplot(data, bins=60, kde=True, ax=axes[0], color="steelblue")
        axes[0].set_title("Histogram with KDE")
        axes[0].set_xlabel("Processing Time (days)")
        axes[0].axvline(data.mean(), color="red", ls="--",
                        label=f"Mean={data.mean():.1f}d")
        axes[0].axvline(data.median(), color="orange", ls="--",
                        label=f"Median={data.median():.1f}d")
        axes[0].legend()

        axes[1].boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="steelblue", alpha=0.6))
        axes[1].set_title("Box Plot")
        axes[1].set_ylabel("Processing Time (days)")
        axes[1].set_xticks([1])
        axes[1].set_xticklabels(["All Applications"])

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "01_processing_time_distribution.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

        desc = data.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        print(f"\n  Stats (days):\n{desc.to_string()}")
        sk = stats.skew(data)
        ku = stats.kurtosis(data)
        print(f"  Skewness: {sk:.3f}  |  Kurtosis: {ku:.3f}")

    # -------------------------------------------------------------------------
    def plot_status_distribution(self):
        """2.2  Status distribution (bar + pie)."""
        print("\n[EDA] 2.2 Status distribution ...")
        status_cols = [c for c in self.df.columns if c.startswith("status_")]
        if not status_cols:
            print("  No status columns found - skipping.")
            return

        status_series = (self.df[status_cols]
                         .idxmax(axis=1)
                         .str.replace("status_", "", regex=False))
        counts = status_series.value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Visa Application Status Distribution",
                     fontsize=14, fontweight="bold")

        sns.barplot(x=counts.values, y=counts.index, ax=axes[0], palette="muted")
        axes[0].set_title("Application Count by Status")
        axes[0].set_xlabel("Count")
        axes[0].set_ylabel("Status")

        axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                    startangle=140,
                    colors=sns.color_palette("muted", len(counts)))
        axes[1].set_title("Status Proportion")

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "02_status_distribution.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

    # -------------------------------------------------------------------------
    def plot_processing_time_by_status(self):
        """2.3  Box + violin plots of processing time grouped by outcome status."""
        print("\n[EDA] 2.3 Processing time by status ...")
        status_cols = [c for c in self.df.columns if c.startswith("status_")]
        if not status_cols:
            print("  No status columns - skipping.")
            return

        tmp = self.df.copy()
        tmp["_status"] = (tmp[status_cols]
                          .idxmax(axis=1)
                          .str.replace("status_", "", regex=False))
        order = (tmp.groupby("_status")[TARGET]
                    .median()
                    .sort_values(ascending=False)
                    .index.tolist())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Processing Time by Visa Status", fontsize=14, fontweight="bold")

        sns.boxplot(data=tmp, x="_status", y=TARGET, order=order,
                    palette="Set2", ax=axes[0])
        axes[0].set_title("Box Plot")
        axes[0].set_xlabel("Status")
        axes[0].set_ylabel("Processing Time (days)")
        axes[0].tick_params(axis="x", rotation=30)

        sns.violinplot(data=tmp, x="_status", y=TARGET, order=order,
                       palette="Set3", ax=axes[1], cut=0)
        axes[1].set_title("Violin Plot")
        axes[1].set_xlabel("Status")
        axes[1].set_ylabel("Processing Time (days)")
        axes[1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "03_processing_time_by_status.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

    # -------------------------------------------------------------------------
    def plot_seasonal_patterns(self):
        """2.4  Monthly / quarterly / day-of-week / fiscal-year patterns."""
        print("\n[EDA] 2.4 Seasonal patterns ...")
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Seasonal Patterns in Processing Time",
                     fontsize=14, fontweight="bold")

        if "submit_month" in self.df.columns:
            monthly = (self.df.groupby("submit_month")[TARGET]
                              .agg(["median", "mean", "count"])
                              .reindex(range(1, 13)))
            monthly.index = month_labels
            monthly["median"].plot(kind="bar", ax=axes[0, 0],
                                   color="steelblue", edgecolor="white")
            axes[0, 0].set_title("Median Processing Time by Submission Month")
            axes[0, 0].set_xlabel("Month")
            axes[0, 0].set_ylabel("Days")
            axes[0, 0].tick_params(axis="x", rotation=45)

        if "submit_quarter" in self.df.columns:
            quarterly = (self.df.groupby("submit_quarter")[TARGET]
                                .median().reset_index())
            sns.barplot(data=quarterly, x="submit_quarter", y=TARGET,
                        palette="Blues_d", ax=axes[0, 1])
            axes[0, 1].set_title("Median Processing Time by Quarter")
            axes[0, 1].set_xlabel("Quarter")
            axes[0, 1].set_ylabel("Days")

        if "submit_day_of_week" in self.df.columns:
            dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            dow = (self.df.groupby("submit_day_of_week")[TARGET]
                          .median().reindex(range(7)))
            dow.index = dow_labels
            dow.plot(kind="bar", ax=axes[1, 0], color="coral", edgecolor="white")
            axes[1, 0].set_title("Median Processing Time by Day of Week")
            axes[1, 0].set_xlabel("Day")
            axes[1, 0].set_ylabel("Days")
            axes[1, 0].tick_params(axis="x", rotation=0)

        if "fiscal_year" in self.df.columns:
            fy = (self.df.groupby("fiscal_year")[TARGET]
                         .agg(["median", "count"])
                         .reset_index()
                         .sort_values("fiscal_year"))
            ax2 = axes[1, 1].twinx()
            axes[1, 1].bar(fy["fiscal_year"], fy["median"],
                           color="steelblue", alpha=0.7, label="Median days")
            ax2.plot(fy["fiscal_year"], fy["count"], color="red",
                     marker="o", label="Count")
            axes[1, 1].set_title("Processing Time & Volume by Fiscal Year")
            axes[1, 1].set_xlabel("Fiscal Year")
            axes[1, 1].set_ylabel("Median Processing Time (days)")
            ax2.set_ylabel("Application Count")
            axes[1, 1].tick_params(axis="x", rotation=30)
            axes[1, 1].legend(loc="upper left")
            ax2.legend(loc="upper right")

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "04_seasonal_patterns.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

    # -------------------------------------------------------------------------
    def plot_consulate_analysis(self, top_n: int = 20):
        """2.5  Volume and median processing time for the top-N consulates."""
        print(f"\n[EDA] 2.5 Top-{top_n} consulate analysis ...")
        if "consulate" not in self.df.columns:
            print("  'consulate' column not found - skipping.")
            return

        top_consulates = (self.df["consulate"]
                          .value_counts().head(top_n).index.tolist())
        sub = self.df[self.df["consulate"].isin(top_consulates)]
        vol = sub["consulate"].value_counts().loc[top_consulates]
        median_by_consulate = (sub.groupby("consulate")[TARGET]
                                  .median().sort_values(ascending=False))

        fig, axes = plt.subplots(2, 1, figsize=(16, 14))
        fig.suptitle(f"Top-{top_n} Consulates - Processing Time Analysis",
                     fontsize=14, fontweight="bold")

        sns.barplot(x=vol.values, y=vol.index, ax=axes[0], palette="Blues_r")
        axes[0].set_title("Application Volume by Consulate")
        axes[0].set_xlabel("Application Count")

        sns.barplot(x=median_by_consulate.values,
                    y=median_by_consulate.index,
                    ax=axes[1], palette="Oranges_r")
        axes[1].set_title("Median Processing Time by Consulate (days)")
        axes[1].set_xlabel("Median Days")

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "05_consulate_analysis.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

    # -------------------------------------------------------------------------
    def plot_region_analysis(self):
        """2.6  Processing time broken down by geographic region."""
        print("\n[EDA] 2.6 Region analysis ...")
        region_cols = [c for c in self.df.columns if c.startswith("region_")]
        if not region_cols:
            print("  No region columns - skipping.")
            return

        tmp = self.df.copy()
        tmp["_region"] = (tmp[region_cols]
                          .idxmax(axis=1)
                          .str.replace("region_", "", regex=False))
        order = (tmp.groupby("_region")[TARGET]
                    .median().sort_values(ascending=False).index.tolist())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Processing Time by Geographic Region",
                     fontsize=14, fontweight="bold")

        sns.boxplot(data=tmp, x="_region", y=TARGET,
                    order=order, palette="Set1", ax=axes[0])
        axes[0].set_title("Box Plot")
        axes[0].set_xlabel("Region")
        axes[0].set_ylabel("Processing Time (days)")

        reg_stats = (tmp.groupby("_region")[TARGET]
                        .agg(["mean", "median", "count"]).reset_index())
        sns.barplot(data=reg_stats, x="_region", y="mean",
                    order=order, palette="Set1", ax=axes[1])
        axes[1].set_title("Mean Processing Time by Region")
        axes[1].set_xlabel("Region")
        axes[1].set_ylabel("Mean Days")

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "06_region_analysis.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

    # -------------------------------------------------------------------------
    def plot_ap_rate_by_consulate(self, top_n: int = 15):
        """2.7  Top consulates by administrative processing rate."""
        print("\n[EDA] 2.7 Administrative processing (AP) rate ...")
        if ("consulate" not in self.df.columns or
                "is_administrative_processing" not in self.df.columns):
            print("  Required columns missing - skipping.")
            return

        ap = (self.df.groupby("consulate")["is_administrative_processing"]
                     .agg(["mean", "count"])
                     .rename(columns={"mean": "ap_rate"})
                     .query("count >= 50")
                     .sort_values("ap_rate", ascending=False)
                     .head(top_n))

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=ap["ap_rate"], y=ap.index, palette="Reds_r", ax=ax)
        ax.set_title(f"Top-{top_n} Consulates by Administrative Processing Rate",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("AP Rate (proportion)")
        ax.set_ylabel("Consulate")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "07_ap_rate_by_consulate.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

    # -------------------------------------------------------------------------
    def plot_missing_values(self):
        """2.8  Bar chart of columns with missing data."""
        print("\n[EDA] 2.8 Missing-values chart ...")
        missing_pct = self.df.isnull().mean() * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

        if missing_pct.empty:
            print("  No missing values in the dataset - skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(10, max(4, len(missing_pct) * 0.35)))
        sns.barplot(x=missing_pct.values, y=missing_pct.index,
                    palette="Reds_r", ax=ax)
        ax.set_title("Missing Value Percentage per Column",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Missing (%)")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "08_missing_values.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

    # -------------------------------------------------------------------------
    def plot_numeric_distributions(self, n_cols: int = 4):
        """2.9  Grid of histograms for all continuous numeric features."""
        print("\n[EDA] 2.9 Numeric feature distributions ...")
        numeric_df = self.df.select_dtypes(include=["number"])

        # Keep columns with reasonable variance and enough distinct values;
        # exclude binary flags (0/1) and the target itself
        numeric_cols = [
            c for c in numeric_df.columns
            if c != TARGET
            and numeric_df[c].nunique() > 10
            and numeric_df[c].std() > 0
        ][:16]

        if not numeric_cols:
            print("  No suitable numeric columns found - skipping.")
            return

        n_rows = -(-len(numeric_cols) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            data_col = numeric_df[col].dropna()
            try:
                axes[i].hist(data_col, bins=40, color="teal", edgecolor="white")
            except Exception:
                pass
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel("")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Numeric Feature Distributions",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "09_numeric_distributions.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

    def run_all(self):
        self.plot_processing_time_distribution()
        self.plot_status_distribution()
        self.plot_processing_time_by_status()
        self.plot_seasonal_patterns()
        self.plot_consulate_analysis()
        self.plot_region_analysis()
        self.plot_ap_rate_by_consulate()
        self.plot_missing_values()
        self.plot_numeric_distributions()


# =============================================================================
# 3.  CORRELATION ANALYSIS
# =============================================================================

class CorrelationAnalyzer:
    """Analyse and visualise correlations with the processing-time target."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -------------------------------------------------------------------------
    def plot_correlation_heatmap(self, max_cols: int = 25):
        """3.1  Heatmap of the correlation matrix (top features by target corr.)."""
        print("\n[Correlation] 3.1 Full numeric correlation heatmap ...")
        numeric_df = self.df.select_dtypes(include=["number"])
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

        if TARGET in numeric_df.columns:
            top_cols = (numeric_df.corr()[TARGET]
                                  .abs()
                                  .sort_values(ascending=False)
                                  .head(max_cols)
                                  .index.tolist())
            numeric_df = numeric_df[top_cols]

        corr = numeric_df.corr()
        size = max(10, len(corr) * 0.55)
        fig, ax = plt.subplots(figsize=(size, size))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, linewidths=0.4,
                    annot_kws={"size": 7}, ax=ax)
        ax.set_title("Feature Correlation Heatmap",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "10_correlation_heatmap.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")
        return corr

    # -------------------------------------------------------------------------
    def plot_top_feature_correlations(self, top_n: int = 15):
        """3.2  Horizontal bar chart of top-N feature correlations with target."""
        print(f"\n[Correlation] 3.2 Top-{top_n} features correlated with "
              f"'{TARGET}' ...")
        numeric_df = self.df.select_dtypes(include=["number"])
        if TARGET not in numeric_df.columns:
            print("  Target column not numeric - skipping.")
            return None

        corr = (numeric_df.corr()[TARGET]
                           .drop(TARGET, errors="ignore")
                           .sort_values(key=abs, ascending=False)
                           .head(top_n))

        colors = ["#e74c3c" if v > 0 else "#3498db" for v in corr.values]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(corr.index[::-1], corr.values[::-1],
                color=colors[::-1], edgecolor="white")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"Top-{top_n} Feature Correlations with Processing Time",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Pearson Correlation Coefficient")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "11_top_feature_correlations.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")
        print(f"\n  Top correlations:\n{corr.to_string()}")
        return corr

    # -------------------------------------------------------------------------
    def plot_scatter_top_features(self, top_n: int = 6):
        """3.3  Scatter plots with regression lines for the top-N features."""
        print(f"\n[Correlation] 3.3 Scatter plots for top-{top_n} features ...")
        numeric_df = self.df.select_dtypes(include=["number"])
        if TARGET not in numeric_df.columns:
            return

        top_features = (numeric_df.corr()[TARGET]
                                  .drop(TARGET, errors="ignore")
                                  .abs()
                                  .sort_values(ascending=False)
                                  .head(top_n)
                                  .index.tolist())

        n_cols = 3
        n_rows = -(-len(top_features) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(top_features):
            sample = (self.df[[col, TARGET]]
                          .dropna()
                          .sample(min(3000, len(self.df)), random_state=42))
            axes[i].scatter(sample[col], sample[TARGET],
                            alpha=0.3, s=10, color="steelblue", rasterized=True)
            m, b = np.polyfit(sample[col], sample[TARGET], 1)
            x_range = np.linspace(sample[col].min(), sample[col].max(), 100)
            axes[i].plot(x_range, m * x_range + b, color="red", lw=1.5)
            axes[i].set_xlabel(col, fontsize=9)
            axes[i].set_ylabel(TARGET, fontsize=9)
            axes[i].set_title(f"{col} vs {TARGET}", fontsize=9)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Scatter Plots - Top Correlated Features",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "12_scatter_top_features.png")
        plt.savefig(path)
        plt.close("all")
        print(f"  Saved: {path}")

    # -------------------------------------------------------------------------
    def plot_pairplot(self, top_n: int = 4):
        """3.4  Scatter matrix of the target with the top-N correlated features."""
        print(f"\n[Correlation] 3.4 Scatter matrix for top-{top_n} features ...")
        numeric_df = self.df.select_dtypes(include=["number"])
        if TARGET not in numeric_df.columns:
            return

        top_features = (numeric_df.corr()[TARGET]
                                  .drop(TARGET, errors="ignore")
                                  .abs()
                                  .sort_values(ascending=False)
                                  .head(top_n)
                                  .index.tolist())
        cols   = [TARGET] + top_features
        sample = (self.df[cols].dropna()
                               .sample(min(500, len(self.df)), random_state=42))

        try:
            from pandas.plotting import scatter_matrix
            fig, ax_arr = plt.subplots(
                len(cols), len(cols),
                figsize=(len(cols) * 3, len(cols) * 3)
            )
            scatter_matrix(sample, alpha=0.3, figsize=(len(cols) * 3, len(cols) * 3),
                           diagonal="hist", ax=ax_arr)
            plt.suptitle("Scatter Matrix - Target & Top Correlated Features",
                         y=1.02, fontsize=12, fontweight="bold")
            path = os.path.join(PLOTS_DIR, "13_scatter_matrix.png")
            plt.savefig(path, dpi=80)
            plt.close("all")
            print(f"  Saved: {path}")
        except Exception as exc:
            plt.close("all")
            print(f"  Scatter matrix skipped ({exc})")

    def run_all(self):
        corr_matrix = self.plot_correlation_heatmap()
        top_corr    = self.plot_top_feature_correlations()
        self.plot_scatter_top_features()
        self.plot_pairplot()
        return corr_matrix, top_corr


# =============================================================================
# 4.  FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Creates derived features:
      - Seasonal index          (month-level ratio to global mean)
      - Country / consulate aggregates  (mean & median PT, approval rate, AP rate)
      - Fiscal-year trend index
      - Case-complexity score
      - Peak-season flag
      - Cyclical (sin/cos) encoding of temporal features
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._lookup_tables: dict = {}

    # -------------------------------------------------------------------------
    def add_seasonal_index(self):
        """
        seasonal_index_month   = monthly mean PT  / global mean PT
        seasonal_index_quarter = quarterly mean PT / global mean PT
        Value > 1 indicates that period is slower than average.
        """
        print("\n[Feature Engineering] 4.1 Seasonal index ...")
        global_mean = self.df[TARGET].mean()

        if "submit_month" in self.df.columns:
            monthly_mean = (self.df.groupby("submit_month")[TARGET]
                                   .mean().rename("_monthly_mean"))
            self.df = self.df.join(monthly_mean, on="submit_month")
            self.df["seasonal_index_month"] = self.df["_monthly_mean"] / global_mean
            self.df.drop(columns=["_monthly_mean"], inplace=True)
            self._lookup_tables["monthly_seasonal_index"] = monthly_mean / global_mean
            print(f"  Monthly seasonal index:\n"
                  f"{(monthly_mean / global_mean).round(3).to_string()}")

        if "submit_quarter" in self.df.columns:
            quarterly_mean = (self.df.groupby("submit_quarter")[TARGET]
                                     .mean().rename("_quarterly_mean"))
            self.df = self.df.join(quarterly_mean, on="submit_quarter")
            self.df["seasonal_index_quarter"] = (
                self.df["_quarterly_mean"] / global_mean)
            self.df.drop(columns=["_quarterly_mean"], inplace=True)
            self._lookup_tables["quarterly_seasonal_index"] = (
                quarterly_mean / global_mean)

        # Peak-season flag: months where processing is >10% above average
        if "seasonal_index_month" in self.df.columns:
            self.df["is_peak_season"] = (
                self.df["seasonal_index_month"] > 1.1).astype(int)
            print(f"  Peak-season records flagged: "
                  f"{self.df['is_peak_season'].sum():,}")

        return self

    # -------------------------------------------------------------------------
    def add_consulate_features(self, min_samples: int = 30):
        """
        For each consulate compute (on self.df, i.e. training data):
          consulate_mean_pt
          consulate_median_pt
          consulate_std_pt
          consulate_volume
          consulate_approval_rate
          consulate_ap_rate
          consulate_refusal_rate
          consulate_221g_rate
          pt_deviation_from_consulate
        """
        print("\n[Feature Engineering] 4.2 Consulate-level aggregate features ...")
        if "consulate" not in self.df.columns:
            print("  'consulate' column not found - skipping.")
            return self

        agg_dict: dict = {TARGET: ["mean", "median", "std", "count"]}
        flag_map = {
            "is_issued":                    "consulate_approval_rate",
            "is_administrative_processing": "consulate_ap_rate",
            "is_refused":                   "consulate_refusal_rate",
            "is_refused_221g":              "consulate_221g_rate",
        }
        for col in flag_map:
            if col in self.df.columns:
                agg_dict[col] = "mean"

        consulate_stats = self.df.groupby("consulate").agg(agg_dict)
        # Flatten multi-level columns
        consulate_stats.columns = [
            "_".join(c).strip("_") for c in consulate_stats.columns
        ]
        consulate_stats = consulate_stats.rename(columns={
            f"{TARGET}_mean"  : "consulate_mean_pt",
            f"{TARGET}_median": "consulate_median_pt",
            f"{TARGET}_std"   : "consulate_std_pt",
            f"{TARGET}_count" : "consulate_volume",
        })
        for orig, new in flag_map.items():
            key = f"{orig}_mean"
            if key in consulate_stats.columns:
                consulate_stats = consulate_stats.rename(columns={key: new})

        # Floor stats for low-volume consulates
        low_vol = consulate_stats["consulate_volume"] < min_samples
        for col in ["consulate_mean_pt", "consulate_median_pt",
                    "consulate_std_pt"]:
            if col in consulate_stats.columns:
                consulate_stats.loc[low_vol, col] = consulate_stats[col].median()

        self._lookup_tables["consulate_stats"] = consulate_stats
        self.df = self.df.join(consulate_stats, on="consulate")

        if "consulate_mean_pt" in self.df.columns:
            self.df["pt_deviation_from_consulate"] = (
                self.df[TARGET] - self.df["consulate_mean_pt"]
            )

        print(f"  Added {len(consulate_stats.columns)} consulate features "
              f"for {len(consulate_stats)} consulates.")
        print(f"\n  Top-10 consulates by mean PT:\n"
              f"{consulate_stats['consulate_mean_pt'].sort_values(ascending=False).head(10).round(1).to_string()}")
        return self

    # -------------------------------------------------------------------------
    def add_fiscal_year_trend(self):
        """
        fiscal_year_mean_pt  = mean PT in that fiscal year
        fiscal_year_index    = normalised rank (0 = oldest, 1 = newest)
        """
        print("\n[Feature Engineering] 4.3 Fiscal-year trend features ...")
        if "fiscal_year" not in self.df.columns:
            print("  'fiscal_year' column not found - skipping.")
            return self

        fy_mean = (self.df.groupby("fiscal_year")[TARGET]
                          .mean().rename("_fy_mean"))
        self.df = self.df.join(fy_mean, on="fiscal_year")
        self.df.rename(columns={"_fy_mean": "fiscal_year_mean_pt"}, inplace=True)

        fy_sorted = sorted(self.df["fiscal_year"].unique())
        fy_rank   = {fy: i / max(len(fy_sorted) - 1, 1)
                     for i, fy in enumerate(fy_sorted)}
        self.df["fiscal_year_index"] = self.df["fiscal_year"].map(fy_rank)

        self._lookup_tables["fy_mean_pt"] = fy_mean
        print(f"  Fiscal-year mean PT:\n{fy_mean.round(1).to_string()}")
        return self

    # -------------------------------------------------------------------------
    def add_complexity_score(self):
        """
        Additive weighted score of flags that signal processing complexity.
        Higher score => expected longer / harder case.
        """
        print("\n[Feature Engineering] 4.4 Case-complexity score ...")
        flag_weights = {
            "is_administrative_processing": 3,
            "is_refused_221g"             : 2,
            "is_refused"                  : 2,
            "potentialAP"                 : 1,
        }
        score = pd.Series(0, index=self.df.index, dtype=float)
        used  = []
        for col, w in flag_weights.items():
            if col in self.df.columns:
                score += self.df[col].fillna(0) * w
                used.append(col)
        self.df["complexity_score"] = score
        print(f"  Components used: {used}")
        print(f"  Score distribution:\n{score.describe().round(2).to_string()}")
        return self

    # -------------------------------------------------------------------------
    def add_cyclical_encoding(self):
        """
        Encode periodic temporal features as sin/cos pairs.
        This lets models understand e.g. that month 12 is adjacent to month 1.
        """
        print("\n[Feature Engineering] 4.5 Cyclical (sin/cos) encoding ...")
        cyclic = {
            "submit_month"      : 12,
            "submit_day_of_week": 7,
            "submit_quarter"    : 4,
            "submit_day_of_year": 365,
        }
        for col, period in cyclic.items():
            if col in self.df.columns:
                self.df[f"{col}_sin"] = np.sin(2 * np.pi * self.df[col] / period)
                self.df[f"{col}_cos"] = np.cos(2 * np.pi * self.df[col] / period)
                print(f"  + {col}_sin / {col}_cos  (period={period})")
        return self

    # -------------------------------------------------------------------------
    def save(self, output_path: str = "data/engineered_visa_dataset.csv"):
        """Persist the engineered dataset to disk."""
        self.df.to_csv(output_path, index=False)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n  Engineered dataset saved: {output_path}  ({size_mb:.2f} MB)")
        print(f"  Shape: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")

    # -------------------------------------------------------------------------
    def plot_engineered_features(self):
        """4.7  Visualise the newly created features."""
        print("\n[Feature Engineering] 4.7 Plotting engineered features ...")

        # A.  Seasonal index bar chart
        if "seasonal_index_month" in self.df.columns:
            month_idx = (self.df.groupby("submit_month")["seasonal_index_month"]
                                .first().reindex(range(1, 13)))
            month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"]
            colors = [
                "#e74c3c" if v > 1.05 else
                "#27ae60" if v < 0.95 else
                "#3498db"
                for v in month_idx.fillna(1).values
            ]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(month_labels, month_idx.values, color=colors, edgecolor="white")
            ax.axhline(1.0, color="black", ls="--", lw=1,
                       label="Global avg (index = 1)")
            ax.set_title("Seasonal Index by Submission Month\n"
                         "(red > 1.05 = slower; green < 0.95 = faster)",
                         fontsize=12, fontweight="bold")
            ax.set_ylabel("Seasonal Index")
            ax.legend()
            plt.tight_layout()
            path = os.path.join(PLOTS_DIR, "14_seasonal_index.png")
            plt.savefig(path)
            plt.close("all")
            print(f"  Saved: {path}")

        # B.  Consulate: approval rate vs mean processing time (bubble chart)
        if ("consulate_mean_pt" in self.df.columns and
                "consulate_approval_rate" in self.df.columns and
                "consulate" in self.df.columns):

            cdf = (self.df.groupby("consulate")
                          .agg(
                              mean_pt      =("consulate_mean_pt",      "first"),
                              approval_rate=("consulate_approval_rate", "first"),
                              volume       =("consulate_mean_pt",       "count"),
                          )
                          .query("volume >= 50"))

            fig, ax = plt.subplots(figsize=(12, 7))
            sc = ax.scatter(
                cdf["approval_rate"], cdf["mean_pt"],
                s=np.sqrt(cdf["volume"]) * 2, alpha=0.6,
                c=cdf["mean_pt"], cmap="YlOrRd",
                edgecolors="grey", linewidths=0.4,
            )
            plt.colorbar(sc, ax=ax, label="Mean Processing Time (days)")
            ax.set_title("Consulate: Approval Rate vs Mean Processing Time\n"
                         "(bubble size proportional to application volume)",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Approval Rate")
            ax.set_ylabel("Mean Processing Time (days)")
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            plt.tight_layout()
            path = os.path.join(PLOTS_DIR, "15_consulate_approval_vs_pt.png")
            plt.savefig(path)
            plt.close("all")
            print(f"  Saved: {path}")

        # C.  Case complexity score vs processing time
        if "complexity_score" in self.df.columns:
            sg = (self.df.groupby("complexity_score")[TARGET]
                         .agg(["mean", "median", "count"]).reset_index())
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(sg["complexity_score"].astype(str),
                   sg["mean"], color="coral", label="Mean PT")
            ax.plot(sg["complexity_score"].astype(str),
                    sg["median"], color="navy", marker="o", label="Median PT")
            ax.set_title("Case Complexity Score vs Processing Time",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Complexity Score")
            ax.set_ylabel("Processing Time (days)")
            ax.legend()
            plt.tight_layout()
            path = os.path.join(PLOTS_DIR, "16_complexity_score_vs_pt.png")
            plt.savefig(path)
            plt.close("all")
            print(f"  Saved: {path}")

        # D.  Engineered features correlation with target
        eng_features = [
            "seasonal_index_month", "seasonal_index_quarter",
            "consulate_mean_pt", "consulate_median_pt",
            "consulate_approval_rate", "consulate_ap_rate",
            "fiscal_year_mean_pt", "complexity_score",
            "pt_deviation_from_consulate", "is_peak_season",
        ]
        avail = [f for f in eng_features
                 if f in self.df.columns and
                 pd.api.types.is_numeric_dtype(self.df[f])]

        if avail:
            corr_eng = (self.df[avail + [TARGET]]
                            .corr()[TARGET]
                            .drop(TARGET, errors="ignore")
                            .sort_values(key=abs, ascending=False))
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in corr_eng.values]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(corr_eng.index[::-1], corr_eng.values[::-1],
                    color=colors[::-1], edgecolor="white")
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title("Engineered Features - Correlation with Processing Time",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Pearson Correlation")
            plt.tight_layout()
            path = os.path.join(PLOTS_DIR, "17_engineered_features_correlation.png")
            plt.savefig(path)
            plt.close("all")
            print(f"  Saved: {path}")
            print(f"\n  Engineered feature correlations:\n"
                  f"{corr_eng.round(4).to_string()}")

    def run_all(self):
        (self
         .add_seasonal_index()
         .add_consulate_features()
         .add_fiscal_year_trend()
         .add_complexity_score()
         .add_cyclical_encoding())
        self.plot_engineered_features()
        self.save()
        return self.df


# =============================================================================
# 5.  SUMMARY REPORT
# =============================================================================

def print_eda_summary(df_original: pd.DataFrame,
                      df_engineered: pd.DataFrame,
                      top_corr: pd.Series):
    print("\n" + "="*70)
    print("EDA & FEATURE ENGINEERING - SUMMARY")
    print("="*70)
    print(f"\nOriginal dataset  : {df_original.shape[0]:,} rows x {df_original.shape[1]} cols")
    print(f"Engineered dataset: "
          f"{df_engineered.shape[0]:,} rows x {df_engineered.shape[1]} cols")

    new_cols = sorted(set(df_engineered.columns) - set(df_original.columns))
    print(f"\nNew features added ({len(new_cols)}):")
    for c in new_cols:
        print(f"  + {c}")

    if top_corr is not None and len(top_corr) > 0:
        print(f"\nTop-5 features correlated with '{TARGET}':")
        for feat, val in top_corr.head(5).items():
            print(f"  {feat:40s} {val:+.4f}")

    print(f"\nPlots saved to   : {PLOTS_DIR}/  (17 plots)")
    print("Engineered data  : data/engineered_visa_dataset.csv")
    print("="*70)


# =============================================================================
# 6.  MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("VISA STATUS PREDICTION - EDA & FEATURE ENGINEERING")
    print("="*70)
    print(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data("data/processed_visa_dataset.csv")
    df_original = df.copy()

    # Section 2 - EDA visualizations
    print("\n" + "-"*70)
    print("SECTION 2 - EXPLORATORY DATA ANALYSIS")
    print("-"*70)
    eda = EDAVisualizer(df)
    eda.run_all()

    # Section 3 - Correlation analysis
    print("\n" + "-"*70)
    print("SECTION 3 - CORRELATION ANALYSIS")
    print("-"*70)
    corr_analyzer = CorrelationAnalyzer(df)
    _, top_corr = corr_analyzer.run_all()

    # Section 4 - Feature engineering
    print("\n" + "-"*70)
    print("SECTION 4 - FEATURE ENGINEERING")
    print("-"*70)
    fe = FeatureEngineer(df)
    df_engineered = fe.run_all()

    # Final summary
    print_eda_summary(df_original, df_engineered, top_corr)


if __name__ == "__main__":
    main()
