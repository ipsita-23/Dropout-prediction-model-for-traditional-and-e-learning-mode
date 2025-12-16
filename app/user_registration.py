"""
User Registration Module

This module handles new student registration including:
- Collecting student information
- Face capture and encoding
- Database insertion
- Initial dropout prediction

Author: AI Project
Date: 2024
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.init_db import get_db_connection, add_student, update_student_prediction
from app.face_recognition_module import register_new_face, FaceRecognitionModule
from app.prediction_module import DropoutPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UserRegistration:
    """
    Handles new user registration process.
    """
    
    def __init__(self, db_path: str = 'database/students.db',
                faces_dir: str = 'faces'):
        """
        Initialize user registration.
        
        Parameters:
        -----------
        db_path : str
            Path to database file
        faces_dir : str
            Directory for face encodings
        """
        self.db_path = db_path
        self.faces_dir = faces_dir
        self.face_module = FaceRecognitionModule()
    
    def collect_student_info(self) -> dict:
        """
        Collect student information from user input.
        
        Returns:
        --------
        dict
            Student information dictionary
        """
        print("\n" + "="*60)
        print("NEW STUDENT REGISTRATION")
        print("="*60)
        
        name = input("\nEnter student name: ").strip()
        if not name:
            raise ValueError("Student name cannot be empty")
        
        print("\nEnter academic information:")
        try:
            academic_score = float(input("  Academic Score (0-10): "))
            if not (0 <= academic_score <= 10):
                raise ValueError("Academic score must be between 0 and 10")
        except ValueError as e:
            print(f"Invalid input: {e}")
            academic_score = None
        
        try:
            indiscipline_count = int(input("  Indiscipline Count (0-10): "))
            if not (0 <= indiscipline_count <= 10):
                raise ValueError("Indiscipline count must be between 0 and 10")
        except ValueError as e:
            print(f"Invalid input: {e}")
            indiscipline_count = 0
        
        try:
            online_engagement_score = float(input("  Online Engagement Score (0-100): "))
            if not (0 <= online_engagement_score <= 100):
                raise ValueError("Engagement score must be between 0 and 100")
        except ValueError as e:
            print(f"Invalid input: {e}")
            online_engagement_score = None
        
        return {
            'name': name,
            'academic_score': academic_score,
            'indiscipline_count': indiscipline_count,
            'online_engagement_score': online_engagement_score
        }
    
    def register_student(self, student_info: dict = None, 
                        num_frames: int = 25) -> dict:
        """
        Complete student registration process.
        
        Parameters:
        -----------
        student_info : dict
            Student information (if None, will prompt for input)
        num_frames : int
            Number of frames to capture for face encoding
        
        Returns:
        --------
        dict
            Registration result with student ID and prediction
        """
        try:
            # Collect student information
            if student_info is None:
                student_info = self.collect_student_info()
            
            name = student_info['name']
            
            # Capture face and generate encoding
            print("\n" + "="*60)
            print("FACE CAPTURE")
            print("="*60)
            print(f"\nPlease look at the camera for face registration...")
            print(f"Capturing {num_frames} frames...")
            print("Press Q to cancel during capture\n")
            
            user_id, encoding = register_new_face(
                user_name=name,
                faces_dir=self.faces_dir,
                num_frames=num_frames
            )
            
            # Get face encoding path
            face_encoding_path = os.path.join(self.faces_dir, user_id, 'encoding.npy')
            
            # Add to database
            conn = get_db_connection(self.db_path)
            try:
                student_id = add_student(
                    conn=conn,
                    name=name,
                    face_encoding_path=face_encoding_path,
                    academic_score=student_info.get('academic_score'),
                    indiscipline_count=student_info.get('indiscipline_count', 0),
                    online_engagement_score=student_info.get('online_engagement_score')
                )
                
                logger.info(f"Student registered: {name} (ID: {student_id})")
                
                # Run initial dropout prediction
                prediction_result = self._run_initial_prediction(
                    conn, student_id, student_info
                )
                
                return {
                    'success': True,
                    'student_id': student_id,
                    'user_id': user_id,
                    'name': name,
                    'face_encoding_path': face_encoding_path,
                    'prediction': prediction_result
                }
            
            finally:
                conn.close()
        
        except KeyboardInterrupt:
            logger.info("Registration cancelled by user")
            return {
                'success': False,
                'message': 'Registration cancelled'
            }
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            return {
                'success': False,
                'message': f'Registration failed: {str(e)}'
            }
    
    def _run_initial_prediction(self, conn, student_id: int, student_info: dict) -> dict:
        """
        Run initial dropout prediction for new student.
        
        Parameters:
        -----------
        conn
            Database connection
        student_id : int
            Student ID
        student_info : dict
            Student information
        
        Returns:
        --------
        dict
            Prediction result
        """
        try:
            predictor = DropoutPredictor()
            
            # Use provided values or defaults
            attendance = student_info.get('attendance_percentage', 0)
            academic = student_info.get('academic_score', 0)
            indiscipline = student_info.get('indiscipline_count', 0)
            engagement = student_info.get('online_engagement_score', 0)
            
            # Make prediction
            prediction_result = predictor.predict(
                attendance_percentage=attendance,
                academic_score=academic if academic is not None else 0,
                indiscipline_count=indiscipline,
                online_engagement_score=engagement if engagement is not None else 0
            )
            
            # Update database with prediction
            update_student_prediction(
                conn=conn,
                student_id=student_id,
                dropout_prediction=prediction_result['dropout_prediction'],
                dropout_probability=prediction_result['dropout_probability']
            )
            
            logger.info(f"Initial prediction completed for student ID {student_id}")
            return prediction_result
        
        except Exception as e:
            logger.warning(f"Could not run prediction: {e}")
            return None
    
    def register_student_with_gui(self, name: str, academic_score: float = None,
                                 indiscipline_count: int = 0,
                                 online_engagement_score: float = None,
                                 num_frames: int = 25) -> dict:
        """
        Register student with provided information (for GUI use).
        
        Parameters:
        -----------
        name : str
            Student name
        academic_score : float
            Academic score (0-10)
        indiscipline_count : int
            Indiscipline count (0-10)
        online_engagement_score : float
            Online engagement score (0-100)
        num_frames : int
            Number of frames to capture
        
        Returns:
        --------
        dict
            Registration result
        """
        student_info = {
            'name': name,
            'academic_score': academic_score,
            'indiscipline_count': indiscipline_count,
            'online_engagement_score': online_engagement_score
        }
        
        return self.register_student(student_info, num_frames)


def register_new_student() -> dict:
    """
    Convenience function to register a new student.
    
    Returns:
    --------
    dict
        Registration result
    """
    registration = UserRegistration()
    return registration.register_student()


if __name__ == "__main__":
    # Test registration
    print("="*60)
    print("USER REGISTRATION MODULE TEST")
    print("="*60)
    
    try:
        result = register_new_student()
        
        if result['success']:
            print("\n✅ Registration successful!")
            print(f"   Student ID: {result['student_id']}")
            print(f"   Name: {result['name']}")
            print(f"   User ID: {result['user_id']}")
            
            if result.get('prediction'):
                pred = result['prediction']
                print(f"\n📊 Initial Dropout Prediction:")
                print(f"   Probability: {pred['dropout_probability']*100:.2f}%")
                print(f"   Risk Level: {pred['risk_level']}")
        else:
            print(f"\n❌ Registration failed: {result.get('message', 'Unknown error')}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Registration cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")

