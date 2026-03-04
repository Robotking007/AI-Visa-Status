"""
predictor.py — Core Prediction Engine
======================================
Loads the saved model and all preprocessing artefacts, then exposes a single
`predict()` function that accepts raw user inputs and returns:
  - predicted processing time (days)
  - ±1σ confidence interval
  - consulate-level historical stats
  - processing category (Fast / Normal / Slow)

Usage
-----
>>> from predictor import VisaPredictor
>>> pred = VisaPredictor()
>>> result = pred.predict(consulate="KGL", submit_date="2024-06-15", case_number=1234)
>>> print(result)
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, "models", "best_model.joblib")
FEATURE_NAMES_PATH= os.path.join(BASE_DIR, "models", "feature_names.json")
TRAINING_REPORT_PATH = os.path.join(BASE_DIR, "models", "training_report.json")
ENGINEERED_DATA   = os.path.join(BASE_DIR, "data", "engineered_visa_dataset.csv")

# Raw CEAC files used to recreate the label encoder (same order as training)
RAW_CEAC_FILES = [
    os.path.join(BASE_DIR, "data", "FY2020-ceac-current.csv"),
    os.path.join(BASE_DIR, "data", "FY2021-ceac-current.csv"),
    os.path.join(BASE_DIR, "data", "FY2022-ceac-current.csv"),
    os.path.join(BASE_DIR, "data", "FY2023-ceac-2023-06-24.csv"),
    os.path.join(BASE_DIR, "data", "FY2024-ceac-2024-10-01.csv"),
    os.path.join(BASE_DIR, "data", "FY2025-ceac-2025-10-01.csv"),
]

# Seasonal peak months (Oct–Jan tend to have higher volumes)
PEAK_MONTHS = {10, 11, 12, 1}

# Full consulate name lookup (3-letter code → city/country name)
CONSULATE_LABELS = {
    "ABD": "Abu Dhabi, UAE",
    "ABJ": "Abidjan, Ivory Coast",
    "ACC": "Accra, Ghana",
    "ACK": "Auckland, New Zealand",
    "ADD": "Addis Ababa, Ethiopia",
    "AKD": "Ankara, Turkey",
    "ALG": "Algiers, Algeria",
    "AMM": "Amman, Jordan",
    "AMS": "Amsterdam, Netherlands",
    "ANK": "Ankara (Alt), Turkey",
    "ANT": "Antananarivo, Madagascar",
    "ASN": "Asuncion, Paraguay",
    "ATA": "Athens, Greece",
    "ATH": "Athens (Alt), Greece",
    "BCH": "Bucharest, Romania",
    "BDP": "Bridgetown, Barbados",
    "BEN": "Bern, Switzerland",
    "BGH": "Belgrade, Serbia",
    "BGN": "Bangui, CAR",
    "BGT": "Bogota, Colombia",
    "BKK": "Bangkok, Thailand",
    "BLG": "Belgrade (Alt), Serbia",
    "BLZ": "Belize City, Belize",
    "BMB": "Bamako, Mali",
    "BNK": "Bangkok (Alt), Thailand",
    "BNS": "Buenos Aires, Argentina",
    "BRS": "Brasilia, Brazil",
    "BRT": "Beirut, Lebanon",
    "BTS": "Bratislava, Slovakia",
    "CDJ": "Ciudad Juarez, Mexico",
    "CHN": "Chennai, India",
    "CRO": "Canberra, Australia",
    "CSB": "Casablanca, Morocco",
    "DCK": "Dakar, Senegal",
    "DHB": "Dhahran, Saudi Arabia",
    "DKR": "Dakar (Alt), Senegal",
    "DRS": "Dresden, Germany",
    "DUB": "Dublin, Ireland",
    "DXB": "Dubai, UAE",
    "FRN": "Frankfurt, Germany",
    "GRN": "Grenada, Grenada",
    "GTM": "Guatemala City, Guatemala",
    "GUA": "Guangzhou, China",
    "HCM": "Ho Chi Minh City, Vietnam",
    "HKG": "Hong Kong",
    "IST": "Istanbul, Turkey",
    "JHN": "Johannesburg, South Africa",
    "KBL": "Kabul, Afghanistan",
    "KGL": "Kigali, Rwanda",
    "KHI": "Karachi, Pakistan",
    "KNG": "Kingston, Jamaica",
    "LGS": "Lagos, Nigeria",
    "LIL": "Lilongwe, Malawi",
    "LIM": "Lima, Peru",
    "LND": "London, UK",
    "LUA": "Luanda, Angola",
    "MAD": "Madrid, Spain",
    "MDS": "Madras, India",
    "MEX": "Mexico City, Mexico",
    "MNL": "Manila, Philippines",
    "MON": "Montreal, Canada",
    "MOS": "Moscow, Russia",
    "MTL": "Monterrey, Mexico",
    "MUM": "Mumbai, India",
    "NAI": "Nairobi, Kenya",
    "NDJ": "N'Djamena, Chad",
    "NEW": "New Delhi, India",
    "NKS": "Nicosia, Cyprus",
    "NRB": "Nairobi (Alt), Kenya",
    "OSL": "Oslo, Norway",
    "OUG": "Ouagadougou, Burkina Faso",
    "PAR": "Paris, France",
    "POR": "Port-of-Spain, Trinidad",
    "PRG": "Prague, Czech Republic",
    "QTO": "Quito, Ecuador",
    "RJD": "Riyadh, Saudi Arabia",
    "ROM": "Rome, Italy",
    "SAL": "San Salvador, El Salvador",
    "SAO": "São Paulo, Brazil",
    "SAR": "Santiago, Chile",
    "SEO": "Seoul, South Korea",
    "SGP": "Singapore",
    "SHN": "Shanghai, China",
    "SNE": "Shenyang, China",
    "STO": "Stockholm, Sweden",
    "SYD": "Sydney, Australia",
    "TIP": "Tripoli, Libya",
    "TKO": "Tokyo, Japan",
    "TLV": "Tel Aviv, Israel",
    "TOR": "Toronto, Canada",
    "UIO": "Quito (Alt), Ecuador",
    "VCN": "Vancouver, Canada",
    "VNK": "Vladivostok, Russia",
    "VNN": "Vienna, Austria",
    "VNT": "Vientiane, Laos",
    "WAR": "Warsaw, Poland",
    "WUH": "Wuhan, China",
    "YDE": "Yaoundé, Cameroon",
}


class VisaPredictor:
    """
    Loads the best model and provides the predict() method.
    On first call the data artefacts are built once and cached.
    """

    def __init__(self):
        print("[VisaPredictor] Loading model …")
        self.model = joblib.load(MODEL_PATH)

        with open(FEATURE_NAMES_PATH) as f:
            self.feature_names = json.load(f)

        with open(TRAINING_REPORT_PATH) as f:
            report = json.load(f)
        self.model_name = report.get("model_name", "Unknown")
        self.test_metrics = report.get("test_metrics", {})

        # Build label encoder and consulate statistics lookup
        self._build_artefacts()
        print(f"[VisaPredictor] Ready. Model={self.model_name}  "
              f"Consulates={len(self.consulate_stats)}")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_artefacts(self):
        """
        Recreate the LabelEncoder (same fit as in preprocessing) and build
        a consulate-stats lookup table from the engineered dataset.
        """
        # 1. Recreate LabelEncoder by fitting on same raw data
        dfs = []
        for f in RAW_CEAC_FILES:
            if os.path.exists(f):
                try:
                    d = pd.read_csv(
                        f, low_memory=False,
                        usecols=["consulate", "submitDate"]
                    )
                    dfs.append(d)
                except Exception:
                    pass

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.dropna(subset=["consulate", "submitDate"])
            le = LabelEncoder()
            le.fit(combined["consulate"].astype(str))
            self.le = le
            self.encoded_to_code = {i: le.classes_[i] for i in range(len(le.classes_))}
            self.code_to_encoded = {v: k for k, v in self.encoded_to_code.items()}
        else:
            self.le = None
            self.encoded_to_code = {}
            self.code_to_encoded = {}

        # 2. Build consulate statistics from engineered dataset
        if os.path.exists(ENGINEERED_DATA):
            eng = pd.read_csv(ENGINEERED_DATA, low_memory=False)
            stat_cols = [
                "consulate", "consulate_mean_pt", "consulate_median_pt",
                "consulate_std_pt", "consulate_volume", "consulate_approval_rate",
                "consulate_ap_rate", "consulate_refusal_rate", "consulate_221g_rate",
            ]
            stats_df = (
                eng[stat_cols]
                .drop_duplicates(subset=["consulate"])
                .set_index("consulate")
            )
            # Convert encoded int → 3-letter code for the key
            stats_df = stats_df.reset_index()
            stats_df["consulate_code"] = (
                stats_df["consulate"].map(self.encoded_to_code)
            )
            stats_df = stats_df.dropna(subset=["consulate_code"])
            self.consulate_stats = (
                stats_df.set_index("consulate_code")
                .to_dict(orient="index")
            )
            # Global fallback stats
            self.global_stats = {
                "consulate_mean_pt":      eng["consulate_mean_pt"].mean(),
                "consulate_median_pt":    eng["consulate_median_pt"].mean(),
                "consulate_std_pt":       eng["consulate_std_pt"].mean(),
                "consulate_volume":       eng["consulate_volume"].mean(),
                "consulate_approval_rate":eng["consulate_approval_rate"].mean(),
                "consulate_ap_rate":      eng["consulate_ap_rate"].mean(),
                "consulate_refusal_rate": eng["consulate_refusal_rate"].mean(),
                "consulate_221g_rate":    eng["consulate_221g_rate"].mean(),
            }

            # Fiscal year index (min fiscal_year → 0)
            fy_min = int(eng["fiscal_year"].min())
            fy_max = int(eng["fiscal_year"].max())
            self.fy_min = fy_min
            self.fy_max = fy_max
        else:
            self.consulate_stats = {}
            self.global_stats = {}
            self.fy_min = 0
            self.fy_max = 11

    def _seasonal_index(self, month: int, quarter: int) -> tuple[float, float]:
        """Simple seasonal indices matching the feature-engineering logic."""
        # Monthly index: peak months get 1.2, off-peak 0.9
        month_idx = 1.2 if month in PEAK_MONTHS else 0.9
        # Quarter index: Q1/Q4 slightly higher
        quarter_idx = 1.1 if quarter in (1, 4) else 0.95
        return month_idx, quarter_idx

    def _fiscal_year_index(self, year: int) -> float:
        """Normalise submission year to [0, 1] range."""
        span = max(self.fy_max - self.fy_min, 1)
        return (year - self.fy_min) / span

    def _encode_consulate(self, code: str) -> int:
        return self.code_to_encoded.get(code.upper(), 0)

    # ── Public API ──────────────────────────────────────────────────────────

    def list_consulates(self) -> list[dict]:
        """Return list of {code, label, mean_pt} dicts sorted by label."""
        result = []
        for code, stats in self.consulate_stats.items():
            result.append({
                "code": code,
                "label": CONSULATE_LABELS.get(code, code),
                "mean_pt": round(stats.get("consulate_mean_pt", 0), 1),
                "approval_rate": round(stats.get("consulate_approval_rate", 0) * 100, 1),
            })
        result.sort(key=lambda x: x["label"])
        return result

    def predict(
        self,
        consulate: str,
        submit_date: str | date,
        case_number: int = 1000,
    ) -> dict:
        """
        Predict visa processing time.

        Parameters
        ----------
        consulate   : 3-letter consulate code (e.g. "KGL")
        submit_date : ISO date string "YYYY-MM-DD" or datetime.date
        case_number : integer case number within the fiscal year (1-based)

        Returns
        -------
        dict with keys:
          predicted_days, lower_bound, upper_bound, category,
          consulate_code, consulate_label, consulate_stats,
          submit_date, features_used
        """
        # Parse date
        if isinstance(submit_date, (date, datetime)):
            dt = submit_date if isinstance(submit_date, datetime) else datetime.combine(submit_date, datetime.min.time())
        else:
            dt = datetime.strptime(str(submit_date), "%Y-%m-%d")

        code = consulate.upper().strip()
        consulate_encoded = self._encode_consulate(code)

        # Date features
        month       = dt.month
        quarter     = (month - 1) // 3 + 1
        dow         = dt.weekday()        # 0=Mon…6=Sun
        doy         = dt.timetuple().tm_yday
        submit_year = dt.year

        # Determine fiscal year (US: Oct–Sep)
        fiscal_year = submit_year if month < 10 else submit_year + 1
        fiscal_year_index = self._fiscal_year_index(fiscal_year)

        # Seasonal
        is_peak = int(month in PEAK_MONTHS)
        si_month, si_quarter = self._seasonal_index(month, quarter)

        # Cyclic encodings
        m_sin  = np.sin(2 * np.pi * month / 12)
        m_cos  = np.cos(2 * np.pi * month / 12)
        d_sin  = np.sin(2 * np.pi * dow / 7)
        d_cos  = np.cos(2 * np.pi * dow / 7)
        q_sin  = np.sin(2 * np.pi * quarter / 4)
        q_cos  = np.cos(2 * np.pi * quarter / 4)
        dy_sin = np.sin(2 * np.pi * doy / 365)
        dy_cos = np.cos(2 * np.pi * doy / 365)

        # Consulate historical stats
        cstats = self.consulate_stats.get(code, self.global_stats)

        row = {
            "consulate":               consulate_encoded,
            "fiscal_year":             fiscal_year,
            "caseNumber":              case_number,
            "submit_year":             submit_year,
            "submit_month":            month,
            "submit_quarter":          quarter,
            "submit_day_of_week":      dow,
            "submit_day_of_year":      doy,
            "case_number":             case_number,
            "seasonal_index_month":    si_month,
            "seasonal_index_quarter":  si_quarter,
            "is_peak_season":          is_peak,
            "consulate_mean_pt":       cstats.get("consulate_mean_pt",      580.0),
            "consulate_median_pt":     cstats.get("consulate_median_pt",    580.0),
            "consulate_std_pt":        cstats.get("consulate_std_pt",        98.0),
            "consulate_volume":        cstats.get("consulate_volume",       1000.0),
            "consulate_approval_rate": cstats.get("consulate_approval_rate",  0.75),
            "consulate_ap_rate":       cstats.get("consulate_ap_rate",        0.10),
            "consulate_refusal_rate":  cstats.get("consulate_refusal_rate",   0.15),
            "consulate_221g_rate":     cstats.get("consulate_221g_rate",      0.05),
            "fiscal_year_index":       fiscal_year_index,
            "submit_month_sin":        m_sin,
            "submit_month_cos":        m_cos,
            "submit_day_of_week_sin":  d_sin,
            "submit_day_of_week_cos":  d_cos,
            "submit_quarter_sin":      q_sin,
            "submit_quarter_cos":      q_cos,
            "submit_day_of_year_sin":  dy_sin,
            "submit_day_of_year_cos":  dy_cos,
        }

        # Align to model's expected feature order
        X = pd.DataFrame([row])[self.feature_names]

        predicted = float(self.model.predict(X)[0])
        predicted = max(0.0, predicted)

        # Confidence interval: use model RMSE as proxy for 1σ
        rmse = self.test_metrics.get("RMSE", 75.0)
        lower = max(0.0, predicted - rmse)
        upper = predicted + rmse

        # Processing category
        mean_pt = cstats.get("consulate_mean_pt", 580.0)
        std_pt  = cstats.get("consulate_std_pt",   98.0)
        if predicted < mean_pt - 0.5 * std_pt:
            category = "Fast"
            category_color = "green"
        elif predicted > mean_pt + 0.5 * std_pt:
            category = "Slow"
            category_color = "red"
        else:
            category = "Normal"
            category_color = "orange"

        return {
            "predicted_days":   round(predicted, 1),
            "lower_bound":      round(lower, 1),
            "upper_bound":      round(upper, 1),
            "predicted_months": round(predicted / 30.44, 1),
            "category":         category,
            "category_color":   category_color,
            "consulate_code":   code,
            "consulate_label":  CONSULATE_LABELS.get(code, code),
            "consulate_stats": {
                "mean_processing_days":   round(cstats.get("consulate_mean_pt", 0), 1),
                "median_processing_days": round(cstats.get("consulate_median_pt", 0), 1),
                "std_processing_days":    round(cstats.get("consulate_std_pt", 0), 1),
                "total_cases":            int(cstats.get("consulate_volume", 0)),
                "approval_rate_pct":      round(cstats.get("consulate_approval_rate", 0) * 100, 1),
                "ap_rate_pct":            round(cstats.get("consulate_ap_rate", 0) * 100, 1),
                "refusal_rate_pct":       round(cstats.get("consulate_refusal_rate", 0) * 100, 1),
                "refusal_221g_pct":       round(cstats.get("consulate_221g_rate", 0) * 100, 1),
            },
            "submit_date":      dt.strftime("%Y-%m-%d"),
            "model_name":       self.model_name,
            "model_mae_days":   round(self.test_metrics.get("MAE", 0), 1),
            "model_r2":         round(self.test_metrics.get("R2", 0), 4),
            "is_peak_season":   bool(is_peak),
            "features_used":    self.feature_names,
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_predictor_instance: VisaPredictor | None = None

def get_predictor() -> VisaPredictor:
    """Return a cached VisaPredictor (lazy singleton)."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = VisaPredictor()
    return _predictor_instance


if __name__ == "__main__":
    p = get_predictor()
    result = p.predict("KGL", "2024-06-15", 1234)
    print(json.dumps(result, indent=2, default=str))
    print(f"\nConsulates available: {len(p.list_consulates())}")
