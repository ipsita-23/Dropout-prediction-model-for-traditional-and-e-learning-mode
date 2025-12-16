"""
Facial Recognition Module

This module handles face detection, encoding, and recognition using:
- face_recognition library
- OpenCV for video capture
- NumPy for encoding storage

Author: AI Project
Date: 2024
"""

import cv2
import face_recognition
import numpy as np
import os
import sys
import json
import logging
from typing import Optional, Tuple, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceRecognitionModule:
    """
    Face recognition module for student attendance system.
    """
    
    def __init__(self, tolerance: float = 0.6):
        """
        Initialize face recognition module.
        
        Parameters:
        -----------
        tolerance : float
            Face distance threshold for matching (default: 0.6)
            Lower values = more strict matching
        """
        self.tolerance = tolerance
        self.known_encodings = {}
        self.known_names = []
        self.face_cascade = None
        
    def load_known_faces(self, faces_dir: str = 'faces'):
        """
        Load all known face encodings from the faces directory.
        
        Parameters:
        -----------
        faces_dir : str
            Directory containing user face encodings
        """
        self.known_encodings = {}
        self.known_names = []
        
        if not os.path.exists(faces_dir):
            logger.warning(f"Faces directory not found: {faces_dir}")
            return
        
        logger.info(f"Loading known faces from {faces_dir}...")
        
        for user_folder in os.listdir(faces_dir):
            user_path = os.path.join(faces_dir, user_folder)
            if not os.path.isdir(user_path):
                continue
            
            encoding_path = os.path.join(user_path, 'encoding.npy')
            meta_path = os.path.join(user_path, 'meta.json')
            
            if os.path.exists(encoding_path) and os.path.exists(meta_path):
                try:
                    encoding = np.load(encoding_path)
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    user_name = meta.get('name', user_folder)
                    self.known_encodings[user_name] = encoding
                    self.known_names.append(user_name)
                    logger.info(f"  ✓ Loaded face for: {user_name}")
                except Exception as e:
                    logger.error(f"  ✗ Error loading {user_folder}: {e}")
        
        logger.info(f"Loaded {len(self.known_names)} known faces")
    
    def capture_face_frames(self, num_frames: int = 25, camera_index: int = 0) -> List[np.ndarray]:
        """
        Capture multiple frames from webcam for face encoding.
        
        Parameters:
        -----------
        num_frames : int
            Number of frames to capture (default: 25)
        camera_index : int
            Camera device index (default: 0)
        
        Returns:
        --------
        List[np.ndarray]
            List of captured frames with detected faces
        """
        logger.info(f"Capturing {num_frames} frames from camera...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        frames_with_faces = []
        frame_count = 0
        
        try:
            while len(frames_with_faces) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    break
                
                # Convert BGR to RGB (face_recognition uses RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces in the frame
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if len(face_locations) == 1:
                    # Only accept frames with exactly one face
                    frames_with_faces.append(rgb_frame)
                    frame_count += 1
                    
                    # Display progress
                    display_frame = frame.copy()
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Capturing: {len(frames_with_faces)}/{num_frames}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Capture - Press Q to cancel', display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Capture cancelled by user")
                        break
                elif len(face_locations) > 1:
                    # Multiple faces detected
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "Multiple faces detected! Show only one face.", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Face Capture - Press Q to cancel', display_frame)
                    cv2.waitKey(1)
                else:
                    # No face detected
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "No face detected. Please look at the camera.", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Face Capture - Press Q to cancel', display_frame)
                    cv2.waitKey(1)
                
                # Small delay to avoid capturing too fast
                cv2.waitKey(50)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Captured {len(frames_with_faces)} frames with faces")
        return frames_with_faces
    
    def generate_face_encoding(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Generate face encoding from multiple frames (average encoding).
        
        Parameters:
        -----------
        frames : List[np.ndarray]
            List of RGB frames with faces
        
        Returns:
        --------
        Optional[np.ndarray]
            Average face encoding, or None if no faces detected
        """
        if not frames:
            logger.error("No frames provided")
            return None
        
        logger.info(f"Generating face encoding from {len(frames)} frames...")
        
        encodings = []
        for frame in frames:
            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) == 1:
                encoding = face_recognition.face_encodings(frame, face_locations)[0]
                encodings.append(encoding)
        
        if not encodings:
            logger.error("No face encodings generated")
            return None
        
        # Average the encodings for better accuracy
        avg_encoding = np.mean(encodings, axis=0)
        logger.info(f"Generated encoding from {len(encodings)} frames")
        
        return avg_encoding
    
    def save_face_encoding(self, encoding: np.ndarray, user_id: str, 
                          user_name: str, faces_dir: str = 'faces'):
        """
        Save face encoding and metadata to disk.
        
        Parameters:
        -----------
        encoding : np.ndarray
            Face encoding to save
        user_id : str
            Unique user identifier (e.g., 'user_001')
        user_name : str
            User's full name
        faces_dir : str
            Directory to save encodings (default: 'faces')
        """
        os.makedirs(faces_dir, exist_ok=True)
        
        user_folder = os.path.join(faces_dir, user_id)
        os.makedirs(user_folder, exist_ok=True)
        
        encoding_path = os.path.join(user_folder, 'encoding.npy')
        meta_path = os.path.join(user_folder, 'meta.json')
        
        # Save encoding
        np.save(encoding_path, encoding)
        
        # Save metadata
        metadata = {
            'user_id': user_id,
            'name': user_name,
            'encoding_path': encoding_path
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Face encoding saved for {user_name} ({user_id})")
        logger.info(f"  Encoding: {encoding_path}")
        logger.info(f"  Metadata: {meta_path}")
    
    def recognize_face(self, frame: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Recognize a face in a frame.
        
        Parameters:
        -----------
        frame : np.ndarray
            RGB frame to search for faces
        
        Returns:
        --------
        Optional[Tuple[str, float]]
            (name, distance) if face recognized, None otherwise
        """
        if not self.known_encodings:
            logger.warning("No known faces loaded")
            return None
        
        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            return None
        
        # Get encoding for the first face
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        if not face_encodings:
            return None
        
        face_encoding = face_encodings[0]
        
        # Compare with known faces
        best_match = None
        best_distance = float('inf')
        
        for name, known_encoding in self.known_encodings.items():
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
            
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        # Check if match is within tolerance
        if best_distance <= self.tolerance:
            return (best_match, best_distance)
        
        return None
    
    def recognize_face_realtime(self, camera_index: int = 0) -> Optional[Tuple[str, float]]:
        """
        Real-time face recognition from webcam.
        
        Parameters:
        -----------
        camera_index : int
            Camera device index (default: 0)
        
        Returns:
        --------
        Optional[Tuple[str, float]]
            (name, distance) if face recognized, None otherwise
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Recognize face
                result = self.recognize_face(rgb_frame)
                
                # Draw on frame
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    
                    if result:
                        name, distance = result
                        color = (0, 255, 0)  # Green for recognized
                        label = f"{name} ({distance:.2f})"
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        label = "Unknown"
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, label, (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow('Face Recognition - Press Q to quit', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Return result if face recognized
                if result:
                    cap.release()
                    cv2.destroyAllWindows()
                    return result
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return None
    
    def get_next_user_id(self, faces_dir: str = 'faces') -> str:
        """
        Get the next available user ID.
        
        Parameters:
        -----------
        faces_dir : str
            Directory containing user folders
        
        Returns:
        --------
        str
            Next user ID (e.g., 'user_001')
        """
        if not os.path.exists(faces_dir):
            return 'user_001'
        
        existing_ids = []
        for folder in os.listdir(faces_dir):
            if folder.startswith('user_') and os.path.isdir(os.path.join(faces_dir, folder)):
                try:
                    num = int(folder.split('_')[1])
                    existing_ids.append(num)
                except:
                    pass
        
        if not existing_ids:
            return 'user_001'
        
        next_num = max(existing_ids) + 1
        return f'user_{next_num:03d}'


# Convenience functions
def register_new_face(user_name: str, faces_dir: str = 'faces', 
                     num_frames: int = 25) -> Tuple[str, np.ndarray]:
    """
    Register a new face by capturing frames and generating encoding.
    
    Parameters:
    -----------
    user_name : str
        Name of the user
    faces_dir : str
        Directory to save encodings
    num_frames : int
        Number of frames to capture
    
    Returns:
    --------
    Tuple[str, np.ndarray]
        (user_id, encoding)
    """
    module = FaceRecognitionModule()
    
    # Get next user ID
    user_id = module.get_next_user_id(faces_dir)
    
    # Capture frames
    frames = module.capture_face_frames(num_frames=num_frames)
    
    if not frames:
        raise RuntimeError("Failed to capture face frames")
    
    # Generate encoding
    encoding = module.generate_face_encoding(frames)
    
    if encoding is None:
        raise RuntimeError("Failed to generate face encoding")
    
    # Save encoding
    module.save_face_encoding(encoding, user_id, user_name, faces_dir)
    
    return user_id, encoding


def recognize_face_from_frame(frame: np.ndarray, faces_dir: str = 'faces') -> Optional[str]:
    """
    Recognize a face from a frame.
    
    Parameters:
    -----------
    frame : np.ndarray
        RGB frame
    faces_dir : str
        Directory containing known faces
    
    Returns:
    --------
    Optional[str]
        Recognized name, or None
    """
    module = FaceRecognitionModule()
    module.load_known_faces(faces_dir)
    
    result = module.recognize_face(frame)
    if result:
        return result[0]
    return None

