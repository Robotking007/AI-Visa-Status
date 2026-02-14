"""
Data Exploration and Validation Utilities
This script provides tools to explore and validate the visa datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


class DataExplorer:
    """
    Utilities for exploring and validating visa datasets.
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """
        Load the dataset.
        """
        if os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path, low_memory=False)
            print(f"  Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
        else:
            print(f"File not found: {self.data_path}")
    
    def basic_info(self):
        """
        Display basic information about the dataset.
        """
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*80)
        print("BASIC DATASET INFORMATION")
        print("="*80 + "\n")
        
        print(f"Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"\nMemory Usage: {self.df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
        print("\nColumn Data Types:")
        print(self.df.dtypes.value_counts())
        
        print("\nFirst 5 Rows:")
        print(self.df.head())
        
        print("\nDataset Info:")
        self.df.info()
    
    def missing_values_analysis(self):
        """
        Analyze missing values in the dataset.
        """
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*80)
        print("MISSING VALUES ANALYSIS")
        print("="*80 + "\n")
        
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percent': missing_percent.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percent', ascending=False
        )
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
            print(f"\nTotal columns with missing values: {len(missing_df)}")
        else:
            print("No missing values found!")
    
    def numerical_summary(self):
        """
        Summary statistics for numerical columns.
        """
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*80)
        print("NUMERICAL FEATURES SUMMARY")
        print("="*80 + "\n")
        
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            print(self.df[numeric_cols].describe().T)
        else:
            print("No numerical columns found.")
    
    def categorical_summary(self):
        """
        Summary statistics for categorical columns.
        """
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*80)
        print("CATEGORICAL FEATURES SUMMARY")
        print("="*80 + "\n")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            for col in categorical_cols[:10]:  # Show first 10
                print(f"\n{col}:")
                print(f"  Unique values: {self.df[col].nunique()}")
                print(f"  Top 5 values:")
                print(self.df[col].value_counts().head())
        else:
            print("No categorical columns found.")
    
    def target_variable_analysis(self, target_col='processing_time_days'):
        """
        Analyze the target variable.
        """
        if self.df is None:
            print("No data loaded.")
            return
        
        if target_col not in self.df.columns:
            print(f"Target column '{target_col}' not found.")
            return
        
        print("\n" + "="*80)
        print(f"TARGET VARIABLE ANALYSIS: {target_col}")
        print("="*80 + "\n")
        
        print(f"Count: {self.df[target_col].count():,}")
        print(f"Mean: {self.df[target_col].mean():.2f}")
        print(f"Median: {self.df[target_col].median():.2f}")
        print(f"Std Dev: {self.df[target_col].std():.2f}")
        print(f"Min: {self.df[target_col].min():.2f}")
        print(f"Max: {self.df[target_col].max():.2f}")
        
        print("\nPercentiles:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = self.df[target_col].quantile(p/100)
            print(f"  {p}th: {val:.2f}")
    
    def correlation_analysis(self, target_col='processing_time_days', top_n=15):
        """
        Analyze correlations with target variable.
        """
        if self.df is None:
            print("No data loaded.")
            return
        
        if target_col not in self.df.columns:
            print(f"Target column '{target_col}' not found.")
            return
        
        print("\n" + "="*80)
        print(f"CORRELATION ANALYSIS WITH {target_col}")
        print("="*80 + "\n")
        
        # Calculate correlations
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        correlations = self.df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        
        print(f"Top {top_n} features correlated with {target_col}:")
        print(correlations.head(top_n))
    
    def data_quality_check(self):
        """
        Perform comprehensive data quality checks.
        """
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*80)
        print("DATA QUALITY CHECK")
        print("="*80 + "\n")
        
        issues = []
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates:,} duplicate rows ({duplicates/len(self.df)*100:.2f}%)")
        
        # Check for constant columns
        constant_cols = [col for col in self.df.columns if self.df[col].nunique() == 1]
        if constant_cols:
            issues.append(f"Found {len(constant_cols)} constant columns: {constant_cols}")
        
        # Check for high cardinality
        high_cardinality = []
        for col in self.df.select_dtypes(include=['object']).columns:
            if self.df[col].nunique() > len(self.df) * 0.9:
                high_cardinality.append(col)
        if high_cardinality:
            issues.append(f"High cardinality columns (>90% unique): {high_cardinality}")
        
        # Check for negative values in processing time
        if 'processing_time_days' in self.df.columns:
            negative = (self.df['processing_time_days'] < 0).sum()
            if negative > 0:
                issues.append(f"Found {negative:,} negative processing times")
        
        if issues:
            print("Data Quality Issues Found:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("✓ No data quality issues found!")
    
    def generate_full_report(self, output_file='data/exploration_report.txt'):
        """
        Generate a comprehensive exploration report.
        """
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80 + "\n")
        
        # Redirect print to file
        import sys
        original_stdout = sys.stdout
        
        with open(output_file, 'w') as f:
            sys.stdout = f
            
            print("="*80)
            print("DATA EXPLORATION REPORT")
            print("="*80)
            print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Dataset: {self.data_path}")
            
            self.basic_info()
            self.missing_values_analysis()
            self.numerical_summary()
            self.categorical_summary()
            
            if 'processing_time_days' in self.df.columns:
                self.target_variable_analysis()
                self.correlation_analysis()
            
            self.data_quality_check()
        
        sys.stdout = original_stdout
        
        print(f"Report saved to: {output_file}")


def explore_raw_data(data_dir='data'):
    """
    Quick exploration of raw data files.
    """
    print("\n" + "="*80)
    print("RAW DATA FILES EXPLORATION")
    print("="*80 + "\n")
    
    # Check CEAC files
    ceac_files = [
        'FY2024-ceac-2024-10-01.csv',
        'FY2023-ceac-2023-06-24.csv',
        'FY2022-ceac-current.csv'
    ]
    
    for filename in ceac_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"\n{filename}:")
            try:
                df = pd.read_csv(filepath, nrows=1000)
                print(f"  Columns: {list(df.columns)}")
                print(f"  Sample size: {len(df)} rows")
                print(f"  Status distribution:")
                if 'status' in df.columns:
                    print(df['status'].value_counts())
            except Exception as e:
                print(f"  Error: {e}")


def main():
    """
    Main execution function.
    """
    # Check if processed data exists
    processed_file = 'data/processed_visa_dataset.csv'
    
    if os.path.exists(processed_file):
        print(f"\nExploring processed dataset: {processed_file}")
        explorer = DataExplorer(processed_file)
        explorer.generate_full_report('data/exploration_report.txt')
    else:
        print(f"\nProcessed dataset not found: {processed_file}")
        print("Exploring raw data files...")
        explore_raw_data()


if __name__ == "__main__":
    main()
