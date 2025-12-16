"""
Synthetic Dataset Generator for Student Dropout Prediction

This script generates a realistic synthetic dataset of 5,000 student records
with features that correlate with dropout probability.

Author: AI Project
Date: 2024
"""

import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_dataset(n_samples=5000, random_state=42):
    """
    Generate synthetic student dataset with realistic dropout correlations.
    
    Parameters:
    -----------
    n_samples : int
        Number of student records to generate (default: 5000)
    random_state : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with student features and dropout labels
    """
    np.random.seed(random_state)
    logger.info(f"Generating {n_samples} synthetic student records...")
    
    # Initialize lists to store data
    data = {
        'attendance_percentage': [],
        'academic_score': [],
        'indiscipline_count': [],
        'online_engagement_score': [],
        'dropout': []
    }
    
    for i in range(n_samples):
        # Generate base features with some randomness
        # Low attendance correlates with higher dropout
        base_attendance = np.random.normal(75, 20)  # Mean 75%, std 20
        attendance = np.clip(base_attendance, 0, 100)
        
        # Low academic score correlates with higher dropout
        base_academic = np.random.normal(6.5, 2.0)  # Mean 6.5/10, std 2.0
        academic_score = np.clip(base_academic, 0, 10)
        
        # High indiscipline correlates with higher dropout
        base_indiscipline = np.random.exponential(2.0)  # Exponential distribution
        indiscipline_count = np.clip(base_indiscipline, 0, 10)
        
        # Low engagement correlates with higher dropout
        base_engagement = np.random.normal(70, 25)  # Mean 70%, std 25
        online_engagement_score = np.clip(base_engagement, 0, 100)
        
        # Calculate dropout probability based on features
        # Lower attendance -> higher dropout probability
        attendance_factor = (100 - attendance) / 100
        
        # Lower academic score -> higher dropout probability
        academic_factor = (10 - academic_score) / 10
        
        # Higher indiscipline -> higher dropout probability
        indiscipline_factor = indiscipline_count / 10
        
        # Lower engagement -> higher dropout probability
        engagement_factor = (100 - online_engagement_score) / 100
        
        # Combined dropout probability (weighted)
        dropout_prob = (
            0.35 * attendance_factor +
            0.30 * academic_factor +
            0.20 * indiscipline_factor +
            0.15 * engagement_factor
        )
        
        # Add noise to avoid perfect linearity
        noise = np.random.normal(0, 0.15)
        dropout_prob = np.clip(dropout_prob + noise, 0, 1)
        
        # Convert probability to binary label
        dropout = 1 if dropout_prob > 0.5 else 0
        
        # Add some randomness to make it more realistic
        # Sometimes good students drop out (5% chance)
        if dropout == 0 and np.random.random() < 0.05:
            dropout = 1
        
        # Sometimes poor students don't drop out (10% chance)
        if dropout == 1 and np.random.random() < 0.10:
            dropout = 0
        
        # Store the data
        data['attendance_percentage'].append(round(attendance, 2))
        data['academic_score'].append(round(academic_score, 2))
        data['indiscipline_count'].append(round(indiscipline_count, 2))
        data['online_engagement_score'].append(round(online_engagement_score, 2))
        data['dropout'].append(dropout)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    logger.info(f"Dataset generated successfully!")
    logger.info(f"Dropout rate: {df['dropout'].mean() * 100:.2f}%")
    logger.info(f"Shape: {df.shape}")
    
    return df


def save_dataset(df, filepath='data/dataset.csv'):
    """
    Save dataset to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to save
    filepath : str
        Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    logger.info(f"Dataset saved to {filepath}")
    
    # Display summary statistics
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"\nTotal Records: {len(df)}")
    print(f"\nFeature Statistics:")
    print(df.describe())
    print(f"\nDropout Distribution:")
    print(df['dropout'].value_counts())
    print(f"\nDropout Rate: {df['dropout'].mean() * 100:.2f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    # Generate dataset
    dataset = generate_synthetic_dataset(n_samples=5000, random_state=42)
    
    # Save dataset
    save_dataset(dataset, filepath='data/dataset.csv')
    
    print("\n✅ Dataset generation completed successfully!")
    print(f"📁 Dataset saved to: data/dataset.csv")
