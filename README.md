# AI Enabled Visa Status Prediction and Processing Time Estimator

**Date**: February 14, 2026  
**Status**: Data Collection and Preprocessing Complete ✓

---

## Project Statement

Visa applicants often face long waiting times and uncertainty regarding the progress of their applications. This project aims to develop a predictive analytics system that estimates visa processing times based on historical data. By analyzing past application records—including applicant country, application type, processing office, and seasonal variations—this system will provide applicants with data-driven estimates of how long their application may take, improving transparency and applicant experience.

## Outcomes

- **Data-Driven Estimates**: Predict approximate processing times for visa applications.
- **Trend Analysis**: Identify seasonal or regional patterns in visa approvals.
- **User-Friendly Tool**: Provide applicants with an easy-to-use web-based estimator.
- **Improved Transparency**: Reduce applicant uncertainty with data-backed predictions.

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

## Modules to be Implemented

### Module 1: Data Collection & Preprocessing
- Gather historical visa data (application dates, decision dates, applicant demographics, processing centers).
- Clean and preprocess data (handle missing values, normalize formats).
- Compute target variable: number of days between submission and decision.

### Module 2: Exploratory Data Analysis (EDA)
- Visualize processing time distributions across visa types and regions.
- Identify trends based on seasons, workload at processing centers, and applicant origin.
- Generate feature importance insights.

### Module 3: Predictive Modeling
- Train regression models (Linear Regression, Random Forest, Gradient Boosting).
- Compare models using evaluation metrics (MAE, RMSE, R² score).
- Tune hyperparameters for optimal performance.

### Module 4: Processing Time Estimator Engine
- Build a prediction engine where users input application details (visa type, country, date, office).
- Return estimated processing time range (e.g., 30–45 days).
- Provide confidence intervals for predictions.

### Module 5: Deployment & Web Application
- Develop a web app with input forms for applicants.
- Display predictions, charts, and past processing trends.
- Enable API endpoints for integration with government or agency platforms.

---

## Technologies Used

- Python
- Machine Learning (Scikit-learn)
- Data Analysis (Pandas, NumPy)
- Visualization (Matplotlib, Seaborn)
- Web Framework (Flask/Django)
- API Development

### Dependencies
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

### System Requirements
- Python 3.8+
- RAM: 8GB minimum (16GB recommended for full dataset)
- Storage: 5GB for data files

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Robotking007/AI-Visa-Status.git

# Navigate to project directory
cd AI-Visa-Status

# Install dependencies
pip install -r requirements.txt
```

---

## Data Sources

### Available Datasets

The project contains **34 CSV files** in the `data/` folder:

1. **CEAC Data Files (12 files)**:
   - FY2013 through FY2025
   - Contains visa application and processing information
   - Total records combined: 550,295+

2. **H1B Visa Data (4 files)**:
   - h1b_2012_2022.csv (59.8 MB)
   - h1b_data_fy2011_fy2018.csv (912.9 MB)
   - h1b_disclosure_data_2015_2019.csv (150.0 MB)
   - h1b_kaggle.csv (492.3 MB)

3. **LCA Disclosure Data (6 files)**:
   - FY2020 through FY2024 (combined: 2.8 GB)
   - Labor Condition Application data

4. **Auxiliary/Lookup Tables (8 files)**:
   - academic.csv, academic_detail.csv
   - field_of_study.csv
   - origin.csv, status.csv
   - source_of_fund.csv
   - global_visa_status.csv

5. **Other Visa Data (4 files)**:
   - visaData.csv
   - VISA_Details files
   - ndsap_visa_info

### Primary Datasets:
- **CEAC Data (FY2013-FY2025)**: Consular Electronic Application Center data containing visa application processing information
- **H1B Data (2012-2022)**: H1B visa petition data with employer and geographic information
- **LCA Disclosure Data (FY2020-FY2024)**: Labor Condition Application data

### Auxiliary Data:
- **Lookup Tables**: academic, field_of_study, origin, status
- **Global Visa Status**: Visa requirements between countries

---

## Processing Pipeline

### Scripts Created

1. **`data_preprocessing.py`** (Main Pipeline)
   - Processes CEAC visa data
   - Comprehensive data cleaning and feature engineering
   - Generates target variable (processing_time_days)
   - Handles missing values and categorical encoding
   - Removes outliers

2. **`h1b_data_preprocessing.py`**
   - Specialized processor for H1B visa data
   - Employer and geographic analysis
   - Approval/denial prediction features

3. **`data_exploration.py`**
   - Data quality checks
   - Statistical analysis
   - Missing value analysis
   - Correlation analysis
   - Generates comprehensive reports

4. **`model_training_template.py`**
   - Baseline model training
   - Model comparison framework
   - Feature importance analysis
   - Evaluation metrics

### Preprocessing Steps Implemented

#### 1. Data Loading
- ✓ Load and combine multiple CEAC files
- ✓ Handle large file sizes with sampling
- ✓ Extract fiscal year from filenames
- ✓ Total records loaded: 550,295

#### 2. Target Variable Generation
- ✓ Calculate processing_time_days = statusDate - submitDate
- ✓ Remove negative values (data errors)
- ✓ Valid records with processing time: 135,858
- ✓ Statistics:
  - Mean: 577.95 days
  - Median: 591.00 days
  - Range: 313-730 days

#### 3. Missing Value Handling
- ✓ Categorical: Filled with mode or 'Unknown'
- ✓ Numerical: Filled with median
- ✓ Reduction: 687,231 → 0 missing values
- ✓ 100% data completeness achieved

#### 4. Feature Engineering
- ✓ Temporal features (5): year, month, quarter, day_of_week, day_of_year
- ✓ Binary indicators (5): is_issued, is_refused, is_administrative_processing, is_ready, is_refused_221g
- ✓ Case number features
- ✓ Total features created: 37

#### 5. Categorical Encoding
- ✓ One-Hot Encoding: 2 features (region, status)
- ✓ Label Encoding: 3 features (caseNumberFull, consulate, fiscal_year)
- ✓ Strategy: Low cardinality (<10) → One-Hot; High cardinality → Label

#### 6. Outlier Removal
- ✓ IQR-based method
- ✓ Threshold: 3 standard deviations
- ✓ Outliers removed: 0 (data already well-bounded)

---

## Output Dataset

### Processed CEAC Dataset

**File**: `data/processed_visa_dataset.csv`

**Dimensions**:
- Records: 135,858
- Features: 37
- File Size: 21.52 MB
- Memory Usage: 25.91 MB

**Feature Breakdown**:
- Numeric features: 20
- Binary/Boolean features: 17
- Categorical features: 0 (all encoded)

**Data Quality**:
- Missing values: 0 (100% complete)
- Duplicates: None
- Outliers: Removed
- Data validity: All checks passed ✓

**Key Features**:
```
1. caseNumberFull (encoded)
2. consulate (encoded)
3. fiscal_year (encoded)
4. caseNumber
5-13. Status indicators (Issued, AP, Ready, Refused, etc.)
14. processing_time_days (TARGET)
15-19. Temporal features (submit_year, submit_month, etc.)
20-24. Engineered binary features
25-37. One-hot encoded features (region, status)
```

### Additional Output Files

**1. Data Summary Report**
- **File**: `data/data_summary.txt`
- Contains: Dataset information and statistics, column details and data types, descriptive statistics, memory usage analysis

**2. Exploration Report**
- **File**: `data/exploration_report.txt`
- Contains: Basic dataset information, missing values analysis, numerical features summary, categorical features summary, target variable analysis, correlation analysis, data quality checks

---

## Target Variable: processing_time_days

- **Mean:** 577.95 days (~1.6 years)
- **Median:** 591.00 days
- **Std Dev:** 98.24 days
- **Range:** 313-730 days
- **Distribution:** Well-bounded, minimal outliers

---

## Model Training Readiness

The processed dataset is now ready for:

### Regression Tasks (Primary)
**Target**: `processing_time_days`

**Recommended Models**:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost / LightGBM
- Neural Networks

**Evaluation Metrics**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

### Classification Tasks (Secondary)
**Targets**: 
- `is_issued` (binary)
- `is_refused` (binary)
- `status` (multi-class)

**Recommended Models**:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Neural Networks

---

## Model Training

Use the processed data to train prediction models:

```bash
python model_training_template.py
```

This template demonstrates:
- Loading processed data
- Train/test split (80/20)
- Training baseline models (Linear Regression, Random Forest, Gradient Boosting)
- Model evaluation (RMSE, MAE, R²)
- Feature importance analysis

### Customization

Modify preprocessing parameters in the scripts:

```python
# In data_preprocessing.py
processor = VisaDataProcessor(data_dir='data')
processed_df = processor.process_ceac_dataset(
    sample_size=50000,      # Number of records per file (None = all)
    remove_outliers=True    # Enable/disable outlier removal
)
```

---

## Usage Examples

### 1. Load Processed Data

```python
import pandas as pd
df = pd.read_csv('data/processed_visa_dataset.csv')
```

### 2. Split Features and Target

```python
X = df.drop('processing_time_days', axis=1)
y = df['processing_time_days']
```

### 3. Train a Model

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 4. Evaluate

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f} days, R²: {r2:.4f}")
```

### 5. Explore Raw Data

```python
from data_exploration import explore_raw_data
explore_raw_data(data_dir='data')
```

### 6. Process Custom Dataset

```python
from data_preprocessing import VisaDataProcessor

processor = VisaDataProcessor(data_dir='data')
df = processor.load_ceac_data(sample_size=100000)
df = processor.calculate_processing_time(df)
df = processor.handle_missing_values(df)
processor.save_processed_data('custom_output.csv')
```

---

## Processing Statistics

**Data Loaded:** 550,295 records (12 CEAC files)  
**Valid Processing Times:** 135,858 records (75% reduction due to missing dates)  
**Missing Values Handled:** 687,231 → 0  
**Features Created:** 37 total  
**Processing Time:** ~2-3 minutes  
**Output Size:** 21.52 MB  

### Performance
- Processing time: ~2-3 minutes (50k records per file)
- Memory usage: ~26 MB for processed dataset
- Scalable to millions of records

---

## File Structure

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
└── README.md                         ← Documentation
```

---

## Next Steps

### Immediate Tasks
1. ✓ Data collection complete
2. ✓ Data cleaning complete
3. ✓ Feature engineering complete
4. ✓ Dataset saved and documented

### Upcoming Tasks
1. **Model Development**:
   - Train baseline models
   - Hyperparameter tuning
   - Cross-validation
   - Model selection

2. **Advanced Feature Engineering**:
   - Interaction features
   - Polynomial features
   - Country/region groupings
   - Seasonal patterns

3. **Model Optimization**:
   - Feature selection
   - Dimensionality reduction
   - Ensemble methods
   - Deep learning approaches

4. **Deployment Preparation**:
   - Model serialization (pickle/joblib)
   - API development (Flask/FastAPI)
   - Web application integration
   - Documentation and monitoring

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

## Data Preprocessing Achievements

✓ **34 datasets** identified and cataloged  
✓ **550,295 records** loaded and combined  
✓ **135,858 clean records** ready for modeling  
✓ **37 features** engineered and encoded  
✓ **0 missing values** (100% complete)  
✓ **4 preprocessing scripts** created  
✓ **3 output files** generated (dataset + 2 reports)  
✓ **Target variable** calculated and validated  
✓ **Data quality** verified and documented  

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Robotking007**

## Acknowledgments

- Thanks to all contributors and supporters of this project.
- Data sources and references will be added as the project develops.

---

**Last Updated:** February 14, 2026  
**Author**: GitHub Copilot  
**Project**: AI Enabled Visa Status Prediction and Processing Time
