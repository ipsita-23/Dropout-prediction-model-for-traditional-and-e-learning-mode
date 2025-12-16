"""
Feature Importance Plotting Utility

This module provides utilities for visualizing feature importance
from trained models.

Author: AI Project
Date: 2024
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_feature_importance(model_path: str = 'models/dropout_model.pkl',
                            feature_names: Optional[list] = None,
                            output_path: str = 'models/feature_importance.png',
                            figsize: tuple = (10, 6)):
    """
    Plot feature importance from a trained model.
    
    Parameters:
    -----------
    model_path : str
        Path to trained model file
    feature_names : list
        List of feature names (if None, uses default)
    output_path : str
        Path to save the plot
    figsize : tuple
        Figure size
    """
    if feature_names is None:
        feature_names = [
            'attendance_percentage',
            'academic_score',
            'indiscipline_count',
            'online_engagement_score'
        ]
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Get feature importance
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance = model.feature_importances_
    
    # Create DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.barplot(data=df_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(df_importance['importance']):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature importance plot saved to {output_path}")
    
    plt.close()
    
    return df_importance


if __name__ == "__main__":
    try:
        df = plot_feature_importance()
        print("\n✅ Feature importance plot generated!")
        print("\n📊 Feature Importance Ranking:")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"❌ Error: {e}")


