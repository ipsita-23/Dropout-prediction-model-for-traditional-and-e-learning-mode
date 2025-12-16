"""
Preprocessing Pipeline for Student Dropout Prediction

This module provides a reusable preprocessing pipeline including:
- Missing value handling
- Feature scaling (StandardScaler)
- SMOTE for class imbalance
- Train-test split
- Feature engineering utilities
- Pipeline serialization with joblib

Author: AI Project
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DropoutPreprocessor:
    """
    Preprocessing pipeline for student dropout prediction dataset.
    """
    
    def __init__(self, apply_smote: bool = True, test_size: float = 0.2, 
                 random_state: int = 42):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        apply_smote : bool
            Whether to apply SMOTE for class imbalance (default: True)
        test_size : float
            Proportion of dataset to include in test split (default: 0.2)
        random_state : int
            Random seed for reproducibility (default: 42)
        """
        self.apply_smote = apply_smote
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=random_state) if apply_smote else None
        self.feature_names = None
        self.is_fitted = False
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with missing values handled
        """
        logger.info("Handling missing values...")
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
            
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"  ✓ Filled {col} with median: {median_val:.2f}")
        else:
            logger.info("  ✓ No missing values found")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        pd.DataFrame
            Dataframe with engineered features
        """
        logger.info("Performing feature engineering...")
        
        df = df.copy()
        
        # Create interaction features
        df['attendance_academic_interaction'] = (
            df['attendance_percentage'] * df['academic_score']
        )
        
        df['risk_score'] = (
            (100 - df['attendance_percentage']) * 0.3 +
            (10 - df['academic_score']) * 10 * 0.3 +
            df['indiscipline_count'] * 10 * 0.2 +
            (100 - df['online_engagement_score']) * 0.2
        )
        
        # Create categorical bins for attendance
        df['attendance_category'] = pd.cut(
            df['attendance_percentage'],
            bins=[0, 60, 75, 90, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        df['attendance_category'] = df['attendance_category'].astype(str)
        
        # Create academic performance category
        df['academic_category'] = pd.cut(
            df['academic_score'],
            bins=[0, 5, 7, 8.5, 10],
            labels=['Poor', 'Average', 'Good', 'Excellent']
        )
        df['academic_category'] = df['academic_category'].astype(str)
        
        logger.info("  ✓ Created interaction features")
        logger.info("  ✓ Created risk score feature")
        logger.info("  ✓ Created categorical features")
        
        return df
    
    def prepare_features_target(self, df: pd.DataFrame, 
                               target_col: str = 'dropout') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column (default: 'dropout')
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Features and target
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor and transform training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        y : pd.Series
            Target series
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Transformed features and target
        """
        logger.info("Fitting preprocessor...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        logger.info("  ✓ Features scaled using StandardScaler")
        
        # Apply SMOTE if enabled
        if self.apply_smote:
            logger.info("Applying SMOTE for class imbalance...")
            X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)
            logger.info(f"  ✓ After SMOTE - Shape: {X_resampled.shape}")
            logger.info(f"  ✓ Class distribution:\n{pd.Series(y_resampled).value_counts()}")
            self.is_fitted = True
            return X_resampled, y_resampled
        else:
            self.is_fitted = True
            return X_scaled.values, y.values
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature dataframe
        
        Returns:
        --------
        np.ndarray
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        return X_scaled.values
    
    def train_test_split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Split data into train and test sets.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature array
        y : np.ndarray
            Target array
        
        Returns:
        --------
        Tuple
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data (test_size={self.test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y, shuffle=True
        )
        
        logger.info(f"  ✓ Train set: {X_train.shape[0]} samples")
        logger.info(f"  ✓ Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_pipeline(self, filepath: str = 'models/preprocessor.pkl'):
        """
        Save preprocessor pipeline to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the pipeline
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        pipeline_data = {
            'scaler': self.scaler,
            'smote': self.smote,
            'apply_smote': self.apply_smote,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'random_state': self.random_state
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"  ✓ Preprocessor saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str = 'models/preprocessor.pkl'):
        """
        Load preprocessor pipeline from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to load the pipeline from
        
        Returns:
        --------
        DropoutPreprocessor
            Loaded preprocessor instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        pipeline_data = joblib.load(filepath)
        
        preprocessor = cls(
            apply_smote=pipeline_data['apply_smote'],
            random_state=pipeline_data['random_state']
        )
        
        preprocessor.scaler = pipeline_data['scaler']
        preprocessor.smote = pipeline_data['smote']
        preprocessor.feature_names = pipeline_data['feature_names']
        preprocessor.is_fitted = pipeline_data['is_fitted']
        
        logger.info(f"  ✓ Preprocessor loaded from {filepath}")
        
        return preprocessor


def preprocess_dataset(dataset_path: str = 'data/dataset.csv',
                      apply_smote: bool = True,
                      test_size: float = 0.2,
                      random_state: int = 42,
                      save_pipeline: bool = True) -> Tuple:
    """
    Complete preprocessing pipeline for the dataset.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset CSV file
    apply_smote : bool
        Whether to apply SMOTE (default: True)
    test_size : float
        Test set proportion (default: 0.2)
    random_state : int
        Random seed (default: 42)
    save_pipeline : bool
        Whether to save the preprocessor (default: True)
    
    Returns:
    --------
    Tuple
        X_train, X_test, y_train, y_test, preprocessor
    """
    logger.info("="*60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset loaded: {df.shape}")
    
    # Initialize preprocessor
    preprocessor = DropoutPreprocessor(
        apply_smote=apply_smote,
        test_size=test_size,
        random_state=random_state
    )
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Feature engineering (optional - can be skipped if not needed)
    # df = preprocessor.feature_engineering(df)
    
    # Prepare features and target
    X, y = preprocessor.prepare_features_target(df)
    
    # Fit and transform
    X_transformed, y_transformed = preprocessor.fit_transform(X, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = preprocessor.train_test_split_data(
        X_transformed, y_transformed
    )
    
    # Save pipeline
    if save_pipeline:
        preprocessor.save_pipeline('models/preprocessor.pkl')
    
    logger.info("="*60)
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Run preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(
        dataset_path='data/dataset.csv',
        apply_smote=True,
        test_size=0.2,
        random_state=42
    )
    
    print("\n✅ Preprocessing completed!")
    print(f"📊 Training set: {X_train.shape}")
    print(f"📊 Test set: {X_test.shape}")
    print(f"💾 Preprocessor saved to: models/preprocessor.pkl")
