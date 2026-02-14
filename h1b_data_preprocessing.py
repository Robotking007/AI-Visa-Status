"""
H1B Visa Data Preprocessing Pipeline
This script specifically processes H1B visa datasets with employer and petition information.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class H1BDataProcessor:
    """
    Specialized processor for H1B visa datasets.
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.label_encoders = {}
        self.processed_data = None
    
    def load_h1b_data(self, filename='h1b_2012_2022.csv', sample_size=None):
        """
        Load H1B visa data.
        """
        print(f"Loading H1B data from {filename}...")
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        
        try:
            # Read in chunks if file is large
            if sample_size:
                df = pd.read_csv(filepath, low_memory=False, nrows=sample_size)
            else:
                # Try to read full file
                df = pd.read_csv(filepath, low_memory=False)
            
            print(f"  Loaded {len(df):,} H1B records")
            print(f"  Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"  Error reading H1B data: {e}")
            return None
    
    def handle_missing_values(self, df):
        """
        Handle missing values in H1B dataset.
        """
        print("\nHandling missing values...")
        initial_missing = df.isnull().sum().sum()
        
        # Fill employer with 'Unknown'
        if 'Employer' in df.columns:
            df['Employer'].fillna('Unknown', inplace=True)
        
        # Fill NAICS with 'Unknown'
        if 'NAICS' in df.columns:
            df['NAICS'].fillna('Unknown', inplace=True)
        
        # Fill State with 'Unknown'
        if 'State' in df.columns:
            df['State'].fillna('Unknown', inplace=True)
        
        # Fill City with 'Unknown'
        if 'City' in df.columns:
            df['City'].fillna('Unknown', inplace=True)
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        final_missing = df.isnull().sum().sum()
        print(f"  Reduced missing values from {initial_missing:,} to {final_missing:,}")
        
        return df
    
    def create_features(self, df):
        """
        Create additional features for H1B data.
        """
        print("\nCreating additional features...")
        
        # Create petition type features
        if 'IA' in df.columns:
            df['is_initial_approval'] = df['IA'].astype(int)
        if 'ID' in df.columns:
            df['is_initial_denial'] = df['ID'].astype(int)
        if 'CA' in df.columns:
            df['is_continuing_approval'] = df['CA'].astype(int)
        if 'CD' in df.columns:
            df['is_continuing_denial'] = df['CD'].astype(int)
        
        # Create approval indicator
        if 'IA' in df.columns and 'CA' in df.columns:
            df['is_approved'] = ((df['IA'] == 1) | (df['CA'] == 1)).astype(int)
        
        # Create denial indicator
        if 'ID' in df.columns and 'CD' in df.columns:
            df['is_denied'] = ((df['ID'] == 1) | (df['CD'] == 1)).astype(int)
        
        # Employer features: Count applications per employer
        if 'Employer' in df.columns:
            employer_counts = df['Employer'].value_counts()
            df['employer_application_count'] = df['Employer'].map(employer_counts)
        
        # Geographic features
        if 'State' in df.columns:
            state_counts = df['State'].value_counts()
            df['state_application_count'] = df['State'].map(state_counts)
        
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features.
        """
        print("\nEncoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            if unique_count <= 10:
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
    
    def process_h1b_dataset(self, sample_size=100000):
        """
        Main processing pipeline for H1B data.
        """
        print("\n" + "="*80)
        print("H1B VISA DATA PREPROCESSING PIPELINE")
        print("="*80 + "\n")
        
        # Load data
        df = self.load_h1b_data(sample_size=sample_size)
        
        if df is None or len(df) == 0:
            print("No data loaded. Exiting...")
            return None
        
        print(f"\nInitial dataset shape: {df.shape}")
        
        # Create features
        df = self.create_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Store processed data
        self.processed_data = df
        
        print(f"\nFinal dataset shape: {df.shape}")
        print(f"Final columns ({len(df.columns)}): {list(df.columns)[:10]}...")
        
        return df
    
    def save_processed_data(self, output_file='data/processed_h1b_dataset.csv'):
        """
        Save the processed dataset.
        """
        if self.processed_data is not None:
            print(f"\nSaving processed data to {output_file}...")
            self.processed_data.to_csv(output_file, index=False)
            print(f"  Saved {len(self.processed_data):,} records")
            print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        else:
            print("No processed data to save.")
    
    def generate_summary(self):
        """
        Generate summary statistics.
        """
        if self.processed_data is None:
            print("No processed data available.")
            return
        
        print("\n" + "="*80)
        print("H1B DATA SUMMARY")
        print("="*80 + "\n")
        
        df = self.processed_data
        
        print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        if 'is_approved' in df.columns:
            approval_rate = df['is_approved'].mean() * 100
            print(f"\nApproval Rate: {approval_rate:.2f}%")
        
        if 'is_denied' in df.columns:
            denial_rate = df['is_denied'].mean() * 100
            print(f"Denial Rate: {denial_rate:.2f}%")
        
        if 'Fiscal Year' in df.columns:
            print(f"\nFiscal Years: {df['Fiscal Year'].min()} - {df['Fiscal Year'].max()}")
        
        print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")


def main():
    """
    Main execution function.
    """
    processor = H1BDataProcessor(data_dir='data')
    
    # Process H1B dataset
    processed_df = processor.process_h1b_dataset(sample_size=100000)
    
    if processed_df is not None:
        # Save processed data
        processor.save_processed_data('data/processed_h1b_dataset.csv')
        
        # Generate summary
        processor.generate_summary()
        
        print("\n" + "="*80)
        print("H1B PREPROCESSING COMPLETE!")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
