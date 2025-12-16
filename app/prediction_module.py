"""
Dropout Prediction Module

This module loads the trained ML model and makes dropout predictions
for students based on their features.

Author: AI Project
Date: 2024
"""

import joblib
import numpy as np
import pandas as pd
import os
import sys
import logging
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DropoutPredictor:
    """
    Dropout prediction module using trained ML model.
    """
    
    def __init__(self, model_path: str = 'models/dropout_model.pkl',
                 scaler_path: str = 'models/scaler.pkl'):
        """
        Initialize dropout predictor.
        
        Parameters:
        -----------
        model_path : str
            Path to trained model file
        scaler_path : str
            Path to scaler file (optional, for preprocessing)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = ['attendance_percentage', 'academic_score', 
                             'indiscipline_count', 'online_engagement_score']
        self.feature_importance = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler."""
        # Load model
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            logger.error("Please train the model in Google Colab and download it to models/")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Load scaler if available
        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"✅ Scaler loaded from {self.scaler_path}")
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}")
        else:
            logger.warning(f"Scaler file not found: {self.scaler_path}")
            logger.warning("Using default StandardScaler - results may vary")
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            logger.info("✅ Feature importance loaded")
    
    def prepare_features(self, attendance_percentage: float,
                        academic_score: float,
                        indiscipline_count: float,
                        online_engagement_score: float) -> np.ndarray:
        """
        Prepare feature vector for prediction.
        
        Parameters:
        -----------
        attendance_percentage : float
            Attendance percentage (0-100)
        academic_score : float
            Academic score (0-10)
        indiscipline_count : float
            Indiscipline count (0-10)
        online_engagement_score : float
            Online engagement score (0-100)
        
        Returns:
        --------
        np.ndarray
            Prepared feature array
        """
        features = np.array([[
            attendance_percentage,
            academic_score,
            indiscipline_count,
            online_engagement_score
        ]])
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features
    
    def predict(self, attendance_percentage: float,
               academic_score: float,
               indiscipline_count: float,
               online_engagement_score: float) -> Dict:
        """
        Predict dropout probability for a student.
        
        Parameters:
        -----------
        attendance_percentage : float
            Attendance percentage (0-100)
        academic_score : float
            Academic score (0-10)
        indiscipline_count : float
            Indiscipline count (0-10)
        online_engagement_score : float
            Online engagement score (0-100)
        
        Returns:
        --------
        Dict
            Prediction results with probability, prediction, and explanations
        """
        # Prepare features
        features = self.prepare_features(
            attendance_percentage, academic_score,
            indiscipline_count, online_engagement_score
        )
        
        # Make prediction
        dropout_probability = self.model.predict_proba(features)[0][1]
        dropout_prediction = int(dropout_probability > 0.5)
        
        # Get contributing factors
        contributing_factors = self._get_contributing_factors(
            attendance_percentage, academic_score,
            indiscipline_count, online_engagement_score
        )
        
        # Get highest contributing factor
        highest_factor = max(contributing_factors.items(), key=lambda x: abs(x[1]))
        
        result = {
            'dropout_prediction': dropout_prediction,
            'dropout_probability': round(dropout_probability, 4),
            'risk_level': self._get_risk_level(dropout_probability),
            'contributing_factors': contributing_factors,
            'highest_contributing_factor': {
                'feature': highest_factor[0],
                'impact': highest_factor[1]
            },
            'explanation': self._generate_explanation(
                dropout_probability, dropout_prediction,
                attendance_percentage, academic_score,
                indiscipline_count, online_engagement_score,
                highest_factor
            )
        }
        
        return result
    
    def _get_contributing_factors(self, attendance: float, academic: float,
                                  indiscipline: float, engagement: float) -> Dict:
        """
        Calculate contributing factors based on feature importance and values.
        
        Parameters:
        -----------
        attendance : float
            Attendance percentage
        academic : float
            Academic score
        indiscipline : float
            Indiscipline count
        engagement : float
            Online engagement score
        
        Returns:
        --------
        Dict
            Contributing factors with impact scores
        """
        factors = {}
        
        if self.feature_importance:
            # Use actual feature importance
            factors['attendance_percentage'] = (
                self.feature_importance['attendance_percentage'] * 
                (100 - attendance) / 100  # Lower attendance = higher risk
            )
            factors['academic_score'] = (
                self.feature_importance['academic_score'] * 
                (10 - academic) / 10  # Lower score = higher risk
            )
            factors['indiscipline_count'] = (
                self.feature_importance['indiscipline_count'] * 
                indiscipline / 10  # Higher count = higher risk
            )
            factors['online_engagement_score'] = (
                self.feature_importance['online_engagement_score'] * 
                (100 - engagement) / 100  # Lower engagement = higher risk
            )
        else:
            # Default equal importance
            factors['attendance_percentage'] = (100 - attendance) / 100
            factors['academic_score'] = (10 - academic) / 10
            factors['indiscipline_count'] = indiscipline / 10
            factors['online_engagement_score'] = (100 - engagement) / 100
        
        return factors
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Get risk level based on probability.
        
        Parameters:
        -----------
        probability : float
            Dropout probability (0-1)
        
        Returns:
        --------
        str
            Risk level
        """
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Medium"
        elif probability < 0.8:
            return "High"
        else:
            return "Very High"
    
    def _generate_explanation(self, probability: float, prediction: int,
                             attendance: float, academic: float,
                             indiscipline: float, engagement: float,
                             highest_factor: tuple) -> str:
        """
        Generate human-readable explanation of the prediction.
        
        Parameters:
        -----------
        probability : float
            Dropout probability
        prediction : int
            Predicted dropout (0 or 1)
        attendance : float
            Attendance percentage
        academic : float
            Academic score
        indiscipline : float
            Indiscipline count
        engagement : float
            Online engagement score
        highest_factor : tuple
            (feature_name, impact) of highest contributing factor
        
        Returns:
        --------
        str
            Explanation text
        """
        feature_name, impact = highest_factor
        
        explanation = f"Based on the student's profile, the dropout probability is {probability*100:.1f}%.\n\n"
        
        if prediction == 1:
            explanation += "⚠️ **PREDICTION: HIGH RISK OF DROPOUT**\n\n"
        else:
            explanation += "✅ **PREDICTION: LOW RISK OF DROPOUT**\n\n"
        
        explanation += "**Key Factors:**\n"
        explanation += f"- Attendance: {attendance:.1f}% ({'Good' if attendance >= 75 else 'Needs Improvement'})\n"
        explanation += f"- Academic Score: {academic:.1f}/10 ({'Good' if academic >= 7 else 'Needs Improvement'})\n"
        explanation += f"- Indiscipline Count: {indiscipline:.1f} ({'Low' if indiscipline <= 2 else 'High'})\n"
        explanation += f"- Online Engagement: {engagement:.1f}% ({'Good' if engagement >= 70 else 'Needs Improvement'})\n\n"
        
        explanation += f"**Highest Contributing Factor:** {feature_name.replace('_', ' ').title()}\n"
        explanation += f"This factor has the most significant impact on the prediction.\n\n"
        
        if prediction == 1:
            explanation += "**Recommendations:**\n"
            if attendance < 75:
                explanation += "- Improve attendance through regular follow-ups\n"
            if academic < 7:
                explanation += "- Provide additional academic support\n"
            if indiscipline > 2:
                explanation += "- Address behavioral issues\n"
            if engagement < 70:
                explanation += "- Increase online engagement activities\n"
        
        return explanation
    
    def predict_from_student_data(self, student_data: Dict) -> Dict:
        """
        Predict dropout from student database record.
        
        Parameters:
        -----------
        student_data : Dict
            Student record from database
        
        Returns:
        --------
        Dict
            Prediction results
        """
        return self.predict(
            attendance_percentage=student_data.get('attendance_percentage', 0),
            academic_score=student_data.get('academic_score', 0),
            indiscipline_count=student_data.get('indiscipline_count', 0),
            online_engagement_score=student_data.get('online_engagement_score', 0)
        )


def predict_dropout(attendance_percentage: float, academic_score: float,
                   indiscipline_count: float, online_engagement_score: float) -> Dict:
    """
    Convenience function to predict dropout.
    
    Parameters:
    -----------
    attendance_percentage : float
        Attendance percentage (0-100)
    academic_score : float
        Academic score (0-10)
    indiscipline_count : float
        Indiscipline count (0-10)
    online_engagement_score : float
        Online engagement score (0-100)
    
    Returns:
    --------
    Dict
        Prediction results
    """
    predictor = DropoutPredictor()
    return predictor.predict(
        attendance_percentage, academic_score,
        indiscipline_count, online_engagement_score
    )


if __name__ == "__main__":
    # Test prediction module
    print("="*60)
    print("DROPOUT PREDICTION MODULE TEST")
    print("="*60)
    
    try:
        predictor = DropoutPredictor()
        
        # Test with sample data
        print("\n📊 Testing with sample student data:")
        print("  - Attendance: 65%")
        print("  - Academic Score: 5.5/10")
        print("  - Indiscipline Count: 3")
        print("  - Online Engagement: 55%")
        
        result = predictor.predict(
            attendance_percentage=65.0,
            academic_score=5.5,
            indiscipline_count=3.0,
            online_engagement_score=55.0
        )
        
        print(f"\n✅ Prediction Results:")
        print(f"  Dropout Probability: {result['dropout_probability']*100:.2f}%")
        print(f"  Prediction: {'HIGH RISK' if result['dropout_prediction'] == 1 else 'LOW RISK'}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Highest Contributing Factor: {result['highest_contributing_factor']['feature']}")
        print(f"\n📝 Explanation:")
        print(result['explanation'])
        
    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
        print("   Please train the model in Google Colab first!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

