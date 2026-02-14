# Quick Reference Guide

## Files Created

### Python Scripts
1. **`data_preprocessing.py`** - Main CEAC visa data preprocessing pipeline
2. **`h1b_data_preprocessing.py`** - H1B visa data processing
3. **`data_exploration.py`** - Data exploration and validation utilities
4. **`model_training_template.py`** - Baseline model training template

### Configuration
5. **`requirements.txt`** - Python package dependencies

### Documentation
6. **`DATA_PREPROCESSING_SUMMARY.md`** - Comprehensive preprocessing report
7. **`README.md`** - Updated with usage instructions

### Generated Outputs (in data/)
8. **`processed_visa_dataset.csv`** - Clean dataset (135,858 records × 37 features, 21.52 MB)
9. **`data_summary.txt`** - Statistical summary and dataset info
10. **`exploration_report.txt`** - Detailed exploratory analysis

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run preprocessing (creates cleaned dataset)
python data_preprocessing.py

# 3. Explore the processed data
python data_exploration.py

# 4. Train baseline models
python model_training_template.py

# 5. (Optional) Process H1B data
python h1b_data_preprocessing.py
```

---

## Dataset Summary

**Processed CEAC Dataset:**
- ✓ 135,858 records
- ✓ 37 features (all numeric/binary)
- ✓ Target: processing_time_days (313-730 days)
- ✓ 0 missing values (100% complete)
- ✓ Ready for ML modeling

**Features Include:**
- Temporal: year, month, quarter, day_of_week, day_of_year
- Binary indicators: is_issued, is_refused, is_administrative_processing
- Encoded: region, consulate, fiscal_year, status
- Original: case numbers, status flags

---

## Common Tasks

### Load Processed Data
```python
import pandas as pd
df = pd.read_csv('data/processed_visa_dataset.csv')
```

### Split Features and Target
```python
X = df.drop('processing_time_days', axis=1)
y = df['processing_time_days']
```

### Train a Model
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Evaluate
```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f} days, R²: {r2:.4f}")
```

---

## Processing Statistics

**Data Loaded:** 550,295 records (12 CEAC files)  
**Valid Processing Times:** 135,858 records (75% reduction due to missing dates)  
**Missing Values Handled:** 687,231 → 0  
**Features Created:** 37 total  
**Processing Time:** ~2-3 minutes  
**Output Size:** 21.52 MB  

---

## Target Variable: processing_time_days

- **Mean:** 577.95 days (~1.6 years)
- **Median:** 591.00 days
- **Std Dev:** 98.24 days
- **Range:** 313-730 days
- **Distribution:** Well-bounded, minimal outliers

---

## Next Steps

1. **Exploratory Analysis:**
   - Run `data_exploration.py` for detailed insights
   - Review `exploration_report.txt` for data quality checks

2. **Model Development:**
   - Use `model_training_template.py` as starting point
   - Try advanced models (XGBoost, LightGBM, Neural Networks)
   - Perform cross-validation
   - Tune hyperparameters

3. **Feature Engineering:**
   - Create interaction features
   - Add country/region groupings
   - Analyze seasonal patterns
   - Include more temporal features

4. **Deployment:**
   - Serialize best model (pickle/joblib)
   - Create prediction API (Flask/FastAPI)
   - Build web interface
   - Set up monitoring

---

## File Locations

```
Project Root/
├── data/
│   ├── processed_visa_dataset.csv    ← Main dataset
│   ├── data_summary.txt              ← Statistics
│   └── exploration_report.txt        ← Analysis
├── data_preprocessing.py             ← Main script
├── h1b_data_preprocessing.py         ← H1B script
├── data_exploration.py               ← Exploration
├── model_training_template.py        ← Modeling
├── requirements.txt                  ← Dependencies
├── DATA_PREPROCESSING_SUMMARY.md     ← Full report
└── README.md                         ← Documentation
```

---

## Troubleshooting

**Out of Memory?**
- Reduce `sample_size` in preprocessing scripts
- Process data in chunks
- Use fewer files

**Need More Data?**
- Set `sample_size=None` to process all records
- Combine with H1B data for additional features

**Custom Processing?**
- Modify parameters in `data_preprocessing.py`
- Create custom feature engineering in `create_feature_engineering()`
- Adjust encoding strategies in `encode_categorical_features()`

---

## Key Achievements ✓

✓ Structured dataset created  
✓ Missing values handled (100% complete)  
✓ Categorical encoding applied  
✓ Target labels generated (processing_time_days)  
✓ Feature engineering implemented  
✓ Data quality verified  
✓ Documentation completed  
✓ Ready for modeling  

---

**Last Updated:** February 14, 2026  
**Status:** Data Preprocessing Complete
