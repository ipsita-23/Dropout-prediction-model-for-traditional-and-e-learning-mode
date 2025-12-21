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
        self.tolerance = tolerance
        self.known_encodings = {}
        self.known_names = []
        self.face_cascade = None
        
    def load_known_faces(self, faces_dir: str = 'faces'):
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
        logger.info(f"Capturing {num_frames} frames from camera...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        
        frames_with_faces = []
        
        try:
            while len(frames_with_faces) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if rgb_frame.dtype != np.uint8:
                    rgb_frame = rgb_frame.astype(np.uint8)

                if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
                    logger.warning(f"Invalid frame shape: {rgb_frame.shape}")
                    continue

                rgb_frame = np.ascontiguousarray(rgb_frame)

                face_locations = face_recognition.face_locations(rgb_frame)
                
                if len(face_locations) == 1:
                    frames_with_faces.append(rgb_frame.copy())
                    
                    display_frame = frame.copy()
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        f"Capturing: {len(frames_with_faces)}/{num_frames}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow('Face Capture - Press Q to cancel', display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Capture cancelled by user")
                        break
                else:
                    cv2.imshow('Face Capture - Press Q to cancel', frame)
                    cv2.waitKey(1)
                
                cv2.waitKey(50)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Captured {len(frames_with_faces)} frames with faces")
        return frames_with_faces
    
    def generate_face_encoding(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        if not frames:
            logger.error("No frames provided")
            return None
        
        logger.info(f"Generating face encoding from {len(frames)} frames...")
        
        encodings = []
        for frame in frames:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            face_locations = face_recognition.face_locations(frame)

            if len(face_locations) == 1:
                encoding = face_recognition.face_encodings(frame, face_locations)[0]
                encodings.append(encoding)
        
        if not encodings:
            logger.error("No face encodings generated")
            return None
        
        avg_encoding = np.mean(encodings, axis=0)
        logger.info(f"Generated encoding from {len(encodings)} frames")
        return avg_encoding
    
    def save_face_encoding(self, encoding: np.ndarray, user_id: str, user_name: str, faces_dir: str = 'faces'):
        os.makedirs(faces_dir, exist_ok=True)
        user_folder = os.path.join(faces_dir, user_id)
        os.makedirs(user_folder, exist_ok=True)
        
        np.save(os.path.join(user_folder, 'encoding.npy'), encoding)
        
        with open(os.path.join(user_folder, 'meta.json'), 'w') as f:
            json.dump(
                {'user_id': user_id, 'name': user_name},
                f,
                indent=2
            )
        
        logger.info(f"Face encoding saved for {user_name} ({user_id})")
    
    def recognize_face(self, frame: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.known_encodings:
            logger.warning("No known faces loaded")
            return None

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning(f"Invalid frame shape in recognize_face: {frame.shape}")
            return None

        frame = np.ascontiguousarray(frame)

        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            return None
        
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        if not face_encodings:
            return None
        
        face_encoding = face_encodings[0]
        
        best_match = None
        best_distance = float('inf')
        
        for name, known_encoding in self.known_encodings.items():
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
            if distance < best_distance:
                best_distance = distance
                best_match = name
        
        if best_distance <= self.tolerance:
            return best_match, best_distance
        
        return None
