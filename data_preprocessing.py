"""
Data Preprocessing Pipeline for Visa Status Prediction
This script processes raw visa data from multiple sources and creates a cleaned dataset for modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class VisaDataProcessor:
    """
    A comprehensive data processor for visa datasets.
    Handles CEAC data, H1B data, and auxiliary datasets.
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.label_encoders = {}
        self.processed_data = None
        
    def load_ceac_data(self, sample_size=None):
        """
        Load and combine CEAC (Consular Electronic Application Center) data files.
        These contain visa application and processing information.
        """
        print("Loading CEAC data...")
        ceac_files = glob.glob(os.path.join(self.data_dir, 'FY*-ceac-*.csv'))
        
        dfs = []
        for file in ceac_files:
            print(f"  Reading {os.path.basename(file)}...")
            try:
                df = pd.read_csv(file, low_memory=False)
                # Extract fiscal year from filename
                fiscal_year = os.path.basename(file).split('-')[0].replace('FY', '')
                df['fiscal_year'] = fiscal_year
                
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                    
                dfs.append(df)
            except Exception as e:
                print(f"  Error reading {file}: {e}")
                continue
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"  Loaded {len(combined_df):,} records from {len(dfs)} files")
            return combined_df
        return None
    
    def load_h1b_data(self, sample_size=None):
        """
        Load H1B visa data.
        """
        print("Loading H1B data...")
        h1b_file = os.path.join(self.data_dir, 'h1b_2012_2022.csv')
        
        if os.path.exists(h1b_file):
            try:
                df = pd.read_csv(h1b_file, low_memory=False)
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                print(f"  Loaded {len(df):,} H1B records")
                return df
            except Exception as e:
                print(f"  Error reading H1B data: {e}")
        return None
    
    def load_auxiliary_data(self):
        """
        Load auxiliary/lookup tables.
        """
        print("Loading auxiliary data...")
        aux_data = {}
        
        # Load lookup tables
        aux_files = {
            'status': 'status.csv',
            'origin': 'origin.csv',
            'academic': 'academic.csv',
            'field_of_study': 'field_of_study.csv',
            'global_visa_status': 'global_visa_status.csv'
        }
        
        for key, filename in aux_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    aux_data[key] = pd.read_csv(filepath)
                    print(f"  Loaded {key}: {len(aux_data[key])} records")
                except Exception as e:
                    print(f"  Error loading {key}: {e}")
        
        return aux_data
    
    def calculate_processing_time(self, df):
        """
        Calculate processing time in days (target variable).
        Processing time = statusDate - submitDate
        """
        print("Calculating processing time...")
        
        if 'submitDate' in df.columns and 'statusDate' in df.columns:
            # Convert to datetime
            df['submitDate'] = pd.to_datetime(df['submitDate'], errors='coerce')
            df['statusDate'] = pd.to_datetime(df['statusDate'], errors='coerce')
            
            # Calculate processing time in days
            df['processing_time_days'] = (df['statusDate'] - df['submitDate']).dt.days
            
            # Remove negative values (data errors)
            df.loc[df['processing_time_days'] < 0, 'processing_time_days'] = np.nan
            
            print(f"  Processing time statistics:")
            print(f"    Mean: {df['processing_time_days'].mean():.2f} days")
            print(f"    Median: {df['processing_time_days'].median():.2f} days")
            print(f"    Min: {df['processing_time_days'].min():.2f} days")
            print(f"    Max: {df['processing_time_days'].max():.2f} days")
            print(f"    Missing: {df['processing_time_days'].isna().sum():,} records")
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values using various strategies based on data type and distribution.
        """
        print("Handling missing values...")
        initial_missing = df.isnull().sum().sum()
        
        # Strategy 1: Fill categorical columns with 'Unknown' or most frequent value
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                # Use mode for columns with low cardinality, 'Unknown' for others
                if df[col].nunique() < 50:
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        # Strategy 2: Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        # Exclude target variable from filling
        numerical_cols = [col for col in numerical_cols if col != 'processing_time_days']
        
        for col in numerical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
        
        final_missing = df.isnull().sum().sum()
        print(f"  Reduced missing values from {initial_missing:,} to {final_missing:,}")
        
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features using Label Encoding and One-Hot Encoding.
        """
        print("Encoding categorical features...")
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Exclude date columns if any are still objects
        date_cols = ['submitDate', 'statusDate', '2nlDate']
        categorical_cols = [col for col in categorical_cols if col not in date_cols]
        
        # Strategy: Use Label Encoding for high cardinality, One-Hot for low cardinality
        low_cardinality_threshold = 10
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            if unique_count <= low_cardinality_threshold:
                # One-Hot Encoding for low cardinality
                print(f"  One-Hot encoding {col} ({unique_count} unique values)")
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
            else:
                # Label Encoding for high cardinality
                print(f"  Label encoding {col} ({unique_count} unique values)")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def create_feature_engineering(self, df):
        """
        Create additional features that might be useful for modeling.
        """
        print("Engineering additional features...")
        
        # Extract temporal features from submitDate
        if 'submitDate' in df.columns:
            df['submit_year'] = df['submitDate'].dt.year
            df['submit_month'] = df['submitDate'].dt.month
            df['submit_quarter'] = df['submitDate'].dt.quarter
            df['submit_day_of_week'] = df['submitDate'].dt.dayofweek
            df['submit_day_of_year'] = df['submitDate'].dt.dayofyear
        
        # Create binary features from existing columns (handle NaN values)
        if 'Issued' in df.columns:
            df['is_issued'] = df['Issued'].fillna(0).astype(int)
        if 'Refused' in df.columns:
            df['is_refused'] = df['Refused'].fillna(0).astype(int)
        if 'AP' in df.columns:
            df['is_administrative_processing'] = df['AP'].fillna(0).astype(int)
        if 'Ready' in df.columns:
            df['is_ready'] = df['Ready'].fillna(0).astype(int)
        if 'Refused221g' in df.columns:
            df['is_refused_221g'] = df['Refused221g'].fillna(0).astype(int)
        
        # Create case number features
        if 'caseNumber' in df.columns:
            df['case_number'] = pd.to_numeric(df['caseNumber'], errors='coerce')
        
        return df
    
    def remove_outliers(self, df, column='processing_time_days', method='iqr', threshold=3):
        """
        Remove outliers from the processing time column.
        """
        print(f"Removing outliers from {column}...")
        
        if column not in df.columns:
            return df
        
        initial_count = len(df)
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores < threshold]
        
        final_count = len(df)
        removed_count = initial_count - final_count
        print(f"  Removed {removed_count:,} outlier records ({removed_count/initial_count*100:.2f}%)")
        
        return df
    
    def process_ceac_dataset(self, sample_size=50000, remove_outliers=True):
        """
        Main processing pipeline for CEAC data.
        """
        print("\n" + "="*80)
        print("VISA DATA PREPROCESSING PIPELINE")
        print("="*80 + "\n")
        
        # Load data
        df = self.load_ceac_data(sample_size=sample_size)
        
        if df is None or len(df) == 0:
            print("No data loaded. Exiting...")
            return None
        
        print(f"\nInitial dataset shape: {df.shape}")
        print(f"Initial columns: {list(df.columns)}")
        
        # Calculate processing time (target variable)
        df = self.calculate_processing_time(df)
        
        # Remove records without target variable
        df = df.dropna(subset=['processing_time_days'])
        print(f"\nAfter removing records without processing time: {df.shape}")
        
        # Feature engineering
        df = self.create_feature_engineering(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Drop date columns (already extracted features)
        date_cols = ['submitDate', 'statusDate', '2nlDate']
        df.drop([col for col in date_cols if col in df.columns], axis=1, inplace=True)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Remove outliers
        if remove_outliers:
            df = self.remove_outliers(df, column='processing_time_days', threshold=3)
        
        # Store processed data
        self.processed_data = df
        
        print(f"\nFinal dataset shape: {df.shape}")
        print(f"Final columns ({len(df.columns)}): {list(df.columns)}")
        
        return df
    
    def save_processed_data(self, output_file='data/processed_visa_dataset.csv'):
        """
        Save the processed dataset to a CSV file.
        """
        if self.processed_data is not None:
            print(f"\nSaving processed data to {output_file}...")
            self.processed_data.to_csv(output_file, index=False)
            print(f"  Saved {len(self.processed_data):,} records")
            print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        else:
            print("No processed data to save.")
    
    def generate_data_summary(self):
        """
        Generate a summary report of the processed dataset.
        """
        if self.processed_data is None:
            print("No processed data available.")
            return
        
        print("\n" + "="*80)
        print("DATA SUMMARY REPORT")
        print("="*80 + "\n")
        
        df = self.processed_data
        
        print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"\nTarget Variable: processing_time_days")
        print(f"  Mean: {df['processing_time_days'].mean():.2f} days")
        print(f"  Median: {df['processing_time_days'].median():.2f} days")
        print(f"  Std Dev: {df['processing_time_days'].std():.2f} days")
        print(f"  Min: {df['processing_time_days'].min():.2f} days")
        print(f"  Max: {df['processing_time_days'].max():.2f} days")
        
        print(f"\nFeature Types:")
        print(f"  Numeric features: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
        print(f"  Categorical features: {len(df.select_dtypes(include=['object']).columns)}")
        
        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  No missing values!")
        
        print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
        # Save summary to file
        summary_file = 'data/data_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n")
            f.write(df.info().__str__())
            f.write("\n\nDescriptive Statistics:\n")
            f.write(df.describe().to_string())
            f.write("\n\nColumn Names:\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i:3d}. {col}\n")
        
        print(f"\nDetailed summary saved to: {summary_file}")


def main():
    """
    Main execution function.
    """
    # Initialize processor
    processor = VisaDataProcessor(data_dir='data')
    
    # Process CEAC dataset
    # Adjust sample_size based on your system's memory
    # Set to None to process all data
    processed_df = processor.process_ceac_dataset(
        sample_size=50000,  # Process 50,000 records per file for demo
        remove_outliers=True
    )
    
    if processed_df is not None:
        # Save processed data
        processor.save_processed_data('data/processed_visa_dataset.csv')
        
        # Generate summary report
        processor.generate_data_summary()
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review the processed dataset: data/processed_visa_dataset.csv")
        print("  2. Check the summary report: data/data_summary.txt")
        print("  3. Use this data for model training and evaluation")
        print("\n")


if __name__ == "__main__":
    main()
