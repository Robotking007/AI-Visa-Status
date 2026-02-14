"""
Model Training Template
This script demonstrates how to use the processed data for building prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(filepath='data/processed_visa_dataset.csv'):
    """
    Load the processed dataset.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def prepare_features_and_target(df, target_col='processing_time_days'):
    """
    Separate features and target variable.
    """
    print(f"\nPreparing features and target variable...")
    
    # Separate target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    print(f"\nSplitting data (test size: {test_size*100:.0f}%)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Testing set: {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance.
    """
    print(f"\n{model_name} Performance:")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  RMSE: {rmse:.2f} days")
    print(f"  MAE: {mae:.2f} days")
    print(f"  R² Score: {r2:.4f}")
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': y_pred}


def train_baseline_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple baseline models.
    """
    print("\n" + "="*80)
    print("TRAINING BASELINE MODELS")
    print("="*80)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        results[model_name] = evaluate_model(model, X_test, y_test, model_name)
    
    return results, models


def feature_importance_analysis(model, feature_names, top_n=15):
    """
    Analyze and display feature importance.
    """
    if hasattr(model, 'feature_importances_'):
        print(f"\nTop {top_n} Most Important Features:")
        
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(importances.head(top_n).to_string(index=False))
        
        return importances
    else:
        print("Model does not support feature importance analysis.")
        return None


def main():
    """
    Main execution function for model training.
    """
    print("\n" + "="*80)
    print("VISA PROCESSING TIME PREDICTION - MODEL TRAINING")
    print("="*80 + "\n")
    
    # Load data
    df = load_processed_data('data/processed_visa_dataset.csv')
    
    # Prepare features and target
    X, y = prepare_features_and_target(df, target_col='processing_time_days')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Train baseline models
    results, models = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Feature importance analysis (using Random Forest)
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    rf_model = models['Random Forest']
    feature_importance_analysis(rf_model, X.columns, top_n=15)
    
    # Summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    summary_df = pd.DataFrame(results).T
    summary_df = summary_df[['rmse', 'mae', 'r2']]
    print(summary_df.to_string())
    
    # Best model
    best_model_name = summary_df['rmse'].idxmin()
    print(f"\nBest Model: {best_model_name}")
    print(f"  RMSE: {summary_df.loc[best_model_name, 'rmse']:.2f} days")
    print(f"  R² Score: {summary_df.loc[best_model_name, 'r2']:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80 + "\n")
    
    print("Next steps:")
    print("  1. Fine-tune the best performing model")
    print("  2. Perform cross-validation for robust evaluation")
    print("  3. Try advanced models (XGBoost, LightGBM, Neural Networks)")
    print("  4. Implement feature engineering techniques")
    print("  5. Deploy the model for production use")
    print("\n")


if __name__ == "__main__":
    main()
