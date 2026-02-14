# AI Enabled Visa Status Prediction and Processing Time Estimator

## Project Statement

Visa applicants often face long waiting times and uncertainty regarding the progress of their applications. This project aims to develop a predictive analytics system that estimates visa processing times based on historical data. By analyzing past application records—including applicant country, application type, processing office, and seasonal variations—this system will provide applicants with data-driven estimates of how long their application may take, improving transparency and applicant experience.

## Outcomes

- **Data-Driven Estimates**: Predict approximate processing times for visa applications.
- **Trend Analysis**: Identify seasonal or regional patterns in visa approvals.
- **User-Friendly Tool**: Provide applicants with an easy-to-use web-based estimator.
- **Improved Transparency**: Reduce applicant uncertainty with data-backed predictions.

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

## Technologies Used

- Python
- Machine Learning (Scikit-learn)
- Data Analysis (Pandas, NumPy)
- Visualization (Matplotlib, Seaborn)
- Web Framework (Flask/Django)
- API Development

## Installation

```bash
# Clone the repository
git clone https://github.com/Robotking007/AI-Visa-Status.git

# Navigate to project directory
cd AI-Visa-Status

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

The project includes multiple visa-related datasets in the `data/` folder:

### Primary Datasets:
- **CEAC Data (FY2013-FY2025)**: Consular Electronic Application Center data containing visa application processing information
- **H1B Data (2012-2022)**: H1B visa petition data with employer and geographic information
- **LCA Disclosure Data (FY2020-FY2024)**: Labor Condition Application data

### Auxiliary Data:
- **Lookup Tables**: academic, field_of_study, origin, status
- **Global Visa Status**: Visa requirements between countries

## Data Preprocessing

### Quick Start

Run the complete preprocessing pipeline:

```bash
# Process CEAC visa data (creates cleaned dataset for modeling)
python data_preprocessing.py

# Process H1B data separately
python h1b_data_preprocessing.py

# Explore and analyze the processed data
python data_exploration.py
```

### Preprocessing Features

The `data_preprocessing.py` script performs:

1. **Data Loading**: 
   - Loads and combines multiple CEAC files (FY2013-FY2025)
   - Samples data for memory efficiency (configurable)
   
2. **Target Variable Generation**:
   - Calculates `processing_time_days` = statusDate - submitDate
   - Removes invalid/negative processing times

3. **Missing Value Handling**:
   - Categorical features: Filled with mode or 'Unknown'
   - Numerical features: Filled with median values
   - Achieved 100% data completeness

4. **Feature Engineering**:
   - Temporal features: year, month, quarter, day of week, day of year
   - Binary indicators: is_issued, is_refused, is_administrative_processing
   - Case number features

5. **Categorical Encoding**:
   - One-Hot Encoding for low cardinality features (<10 unique values)
   - Label Encoding for high cardinality features
   
6. **Outlier Removal**:
   - IQR-based outlier detection and removal
   - Configurable threshold

### Output Files

After running the preprocessing:

- **`data/processed_visa_dataset.csv`**: Cleaned, encoded dataset ready for modeling (135,858 records × 37 features)
- **`data/data_summary.txt`**: Comprehensive data summary and statistics
- **`data/exploration_report.txt`**: Detailed exploratory data analysis report

### Dataset Statistics

**Processed CEAC Dataset:**
- **Records**: 135,858 visa applications
- **Features**: 37 (20 numeric, 17 binary)
- **Target Variable**: processing_time_days
  - Mean: 577.95 days
  - Median: 591.00 days
  - Range: 313-730 days
- **Missing Values**: None (100% complete)
- **File Size**: 21.52 MB

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

## Usage Examples

### 1. Explore Raw Data

```python
from data_exploration import explore_raw_data
explore_raw_data(data_dir='data')
```

### 2. Process Custom Dataset

```python
from data_preprocessing import VisaDataProcessor

processor = VisaDataProcessor(data_dir='data')
df = processor.load_ceac_data(sample_size=100000)
df = processor.calculate_processing_time(df)
df = processor.handle_missing_values(df)
processor.save_processed_data('custom_output.csv')
```

### 3. Train Your Own Model

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load processed data
df = pd.read_csv('data/processed_visa_dataset.csv')
X = df.drop('processing_time_days', axis=1)
y = df['processing_time_days']

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Robotking007**

## Acknowledgments

- Thanks to all contributors and supporters of this project.
- Data sources and references will be added as the project develops.
