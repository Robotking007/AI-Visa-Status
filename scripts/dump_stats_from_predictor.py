"""
scripts/dump_stats_from_predictor.py  — run from project root
Usage: .venv\Scripts\python.exe scripts\dump_stats_from_predictor.py
Loads the predictor (uses local model + CSVs) and dumps consulate_stats.json.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import get_predictor

print("Loading predictor …")
p = get_predictor()

payload = {
    "le_classes":      list(p.le.classes_) if p.le is not None else [],
    "consulate_stats": {
        k: {sk: (float(sv) if hasattr(sv, 'item') else sv)
            for sk, sv in v.items() if sk != "consulate"}
        for k, v in p.consulate_stats.items()
    },
    "global_stats": {
        k: float(v) if hasattr(v, 'item') else v
        for k, v in p.global_stats.items()
    },
    "fy_min": int(p.fy_min),
    "fy_max": int(p.fy_max),
}

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "models", "consulate_stats.json")
with open(out, "w") as f:
    json.dump(payload, f, indent=2)

size_kb = os.path.getsize(out) / 1024
print(f"Written: {out}  ({size_kb:.1f} KB)")
print(f"Consulates: {len(payload['consulate_stats'])}")
print(f"LabelEncoder classes: {len(payload['le_classes'])}")
print(f"FY range: {payload['fy_min']} – {payload['fy_max']}")
print("Done! Commit models/consulate_stats.json to git.")
