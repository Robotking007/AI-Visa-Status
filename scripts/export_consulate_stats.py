"""
scripts/export_consulate_stats.py
===================================
One-time helper: reads local CEAC CSVs + the engineered dataset and writes
models/consulate_stats.json that predictor.py will use on Streamlit Cloud
(where the raw GBs of CSV files are not available).

Run once from the project root:
    python scripts/export_consulate_stats.py
"""

import os
import sys
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_CEAC_FILES = [
    os.path.join(BASE_DIR, "data", "FY2020-ceac-current.csv"),
    os.path.join(BASE_DIR, "data", "FY2021-ceac-current.csv"),
    os.path.join(BASE_DIR, "data", "FY2022-ceac-current.csv"),
    os.path.join(BASE_DIR, "data", "FY2023-ceac-2023-06-24.csv"),
    os.path.join(BASE_DIR, "data", "FY2024-ceac-2024-10-01.csv"),
    os.path.join(BASE_DIR, "data", "FY2025-ceac-2025-10-01.csv"),
]
ENGINEERED_DATA = os.path.join(BASE_DIR, "data", "engineered_visa_dataset.csv")
OUT_PATH        = os.path.join(BASE_DIR, "models", "consulate_stats.json")


def build_label_encoder():
    print("[1/3] Fitting LabelEncoder from CEAC files …")
    dfs = []
    for f in RAW_CEAC_FILES:
        if os.path.exists(f):
            try:
                d = pd.read_csv(f, low_memory=False, usecols=["consulate", "submitDate"])
                dfs.append(d)
                print(f"      ✓  {os.path.basename(f)}  ({len(d):,} rows)")
            except Exception as e:
                print(f"      ✗  {os.path.basename(f)}: {e}")
        else:
            print(f"      –  {os.path.basename(f)} not found, skipping")

    if not dfs:
        print("ERROR: No CEAC files found. Run from the project root where data/ lives.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["consulate", "submitDate"])
    le = LabelEncoder()
    le.fit(combined["consulate"].astype(str))
    print(f"      → {len(le.classes_)} unique consulate codes")
    return le


def build_consulate_stats(le: LabelEncoder):
    print("[2/3] Building consulate stats from engineered dataset …")
    if not os.path.exists(ENGINEERED_DATA):
        print(f"ERROR: {ENGINEERED_DATA} not found.")
        sys.exit(1)

    eng = pd.read_csv(ENGINEERED_DATA, low_memory=False)
    print(f"      → {len(eng):,} rows loaded")

    stat_cols = [
        "consulate", "consulate_mean_pt", "consulate_median_pt",
        "consulate_std_pt", "consulate_volume", "consulate_approval_rate",
        "consulate_ap_rate", "consulate_refusal_rate", "consulate_221g_rate",
    ]
    stats_df = (
        eng[stat_cols]
        .drop_duplicates(subset=["consulate"])
        .copy()
    )

    encoded_to_code = {i: le.classes_[i] for i in range(len(le.classes_))}
    stats_df["consulate_code"] = stats_df["consulate"].map(encoded_to_code)
    stats_df = stats_df.dropna(subset=["consulate_code"])

    consulate_stats = {}
    for _, row in stats_df.iterrows():
        code = row["consulate_code"]
        consulate_stats[code] = {
            "consulate_mean_pt":       round(float(row["consulate_mean_pt"]),       2),
            "consulate_median_pt":     round(float(row["consulate_median_pt"]),     2),
            "consulate_std_pt":        round(float(row["consulate_std_pt"]),        2),
            "consulate_volume":        int(row["consulate_volume"]),
            "consulate_approval_rate": round(float(row["consulate_approval_rate"]), 6),
            "consulate_ap_rate":       round(float(row["consulate_ap_rate"]),       6),
            "consulate_refusal_rate":  round(float(row["consulate_refusal_rate"]),  6),
            "consulate_221g_rate":     round(float(row["consulate_221g_rate"]),     6),
        }

    global_stats = {
        "consulate_mean_pt":       round(float(eng["consulate_mean_pt"].mean()),       2),
        "consulate_median_pt":     round(float(eng["consulate_median_pt"].mean()),     2),
        "consulate_std_pt":        round(float(eng["consulate_std_pt"].mean()),        2),
        "consulate_volume":        round(float(eng["consulate_volume"].mean()),        2),
        "consulate_approval_rate": round(float(eng["consulate_approval_rate"].mean()), 6),
        "consulate_ap_rate":       round(float(eng["consulate_ap_rate"].mean()),       6),
        "consulate_refusal_rate":  round(float(eng["consulate_refusal_rate"].mean()),  6),
        "consulate_221g_rate":     round(float(eng["consulate_221g_rate"].mean()),     6),
    }

    le_classes = le.classes_.tolist()
    fy_min = int(eng["fiscal_year"].min())
    fy_max = int(eng["fiscal_year"].max())

    return {
        "le_classes":       le_classes,
        "consulate_stats":  consulate_stats,
        "global_stats":     global_stats,
        "fy_min":           fy_min,
        "fy_max":           fy_max,
    }


def main():
    print("=" * 60)
    print("  Export Consulate Stats → models/consulate_stats.json")
    print("=" * 60)

    le = build_label_encoder()
    payload = build_consulate_stats(le)

    print("[3/3] Writing JSON …")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"      ✓  {OUT_PATH}  ({size_kb:.1f} KB)")
    print(f"      → {len(payload['consulate_stats'])} consulates exported")
    print("Done! Commit models/consulate_stats.json to git.")


if __name__ == "__main__":
    main()
