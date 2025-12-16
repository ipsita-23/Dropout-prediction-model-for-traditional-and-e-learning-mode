"""
Main Application - Student Dropout Prediction + Facial Recognition Attendance System

This is the main application that integrates:
- Real-time facial recognition
- Attendance marking
- Dropout prediction
- User registration

Author: AI Project
Date: 2024
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.face_recognition_module import FaceRecognitionModule
from app.attendance_manager import AttendanceManager
from app.prediction_module import DropoutPredictor
from app.user_registration import UserRegistration
from database.init_db import get_db_connection, get_student_by_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StudentDropoutApp:
    """
    Main application class for student dropout prediction system.
    """
    
    def __init__(self, root):
        """
        Initialize the application.
        
        Parameters:
        -----------
        root : tk.Tk
            Tkinter root window
        """
        self.root = root
        self.root.title("Student Dropout Prediction & Attendance System")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Initialize modules
        self.face_module = FaceRecognitionModule()
        self.face_module.load_known_faces()
        self.attendance_manager = AttendanceManager()
        self.predictor = None
        self.registration = UserRegistration()
        
        # Try to load predictor
        try:
            self.predictor = DropoutPredictor()
        except Exception as e:
            logger.warning(f"Could not load predictor: {e}")
            messagebox.showwarning(
                "Model Not Found",
                "Trained model not found. Please train the model in Google Colab first.\n"
                "Some features may not work."
            )
        
        # Camera state
        self.camera_running = False
        self.camera_thread = None
        self.current_student = None
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        """Create the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Student Dropout Prediction & Attendance System",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_frame, text="Actions", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Buttons
        ttk.Button(
            left_panel,
            text="🔍 Scan Face & Identify",
            command=self.scan_face,
            width=25
        ).grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        
        ttk.Button(
            left_panel,
            text="📝 Mark Attendance",
            command=self.mark_attendance,
            width=25
        ).grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        
        ttk.Button(
            left_panel,
            text="🔮 Check Dropout Prediction",
            command=self.check_prediction,
            width=25
        ).grid(row=2, column=0, pady=5, sticky=tk.W+tk.E)
        
        ttk.Button(
            left_panel,
            text="➕ Register New Student",
            command=self.register_new_student,
            width=25
        ).grid(row=3, column=0, pady=5, sticky=tk.W+tk.E)
        
        ttk.Button(
            left_panel,
            text="📊 View All Students",
            command=self.view_all_students,
            width=25
        ).grid(row=4, column=0, pady=5, sticky=tk.W+tk.E)
        
        ttk.Separator(left_panel, orient='horizontal').grid(row=5, column=0, sticky=tk.W+tk.E, pady=10)
        
        ttk.Button(
            left_panel,
            text="🔄 Refresh Face Database",
            command=self.refresh_faces,
            width=25
        ).grid(row=6, column=0, pady=5, sticky=tk.W+tk.E)
        
        ttk.Button(
            left_panel,
            text="❌ Exit",
            command=self.root.quit,
            width=25
        ).grid(row=7, column=0, pady=5, sticky=tk.W+tk.E)
        
        # Right panel - Output
        right_panel = ttk.LabelFrame(main_frame, text="Output", padding="10")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            right_panel,
            width=50,
            height=30,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Initial message
        self.log_output("="*60)
        self.log_output("Student Dropout Prediction System")
        self.log_output("="*60)
        self.log_output("\nWelcome! Please select an action from the left panel.")
        self.log_output("\nMake sure your webcam is connected and ready.")
    
    def log_output(self, message: str):
        """Add message to output text area."""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update()
    
    def update_status(self, message: str):
        """Update status bar."""
        self.status_label.config(text=message)
        self.root.update()
    
    def scan_face(self):
        """Scan face and identify student."""
        self.update_status("Scanning face...")
        self.log_output("\n" + "="*60)
        self.log_output("FACE SCANNING")
        self.log_output("="*60)
        self.log_output("\nPlease look at the camera...")
        self.log_output("Press 'Q' in the camera window to quit\n")
        
        try:
            result = self.face_module.recognize_face_realtime()
            
            if result:
                name, distance = result
                self.current_student = name
                self.log_output(f"\n✅ Face recognized!")
                self.log_output(f"   Name: {name}")
                self.log_output(f"   Distance: {distance:.4f}")
                self.log_output(f"\nYou can now:")
                self.log_output(f"  - Mark attendance")
                self.log_output(f"  - Check dropout prediction")
                self.update_status(f"Recognized: {name}")
            else:
                self.log_output("\n❌ Face not recognized")
                self.log_output("   This person is not registered in the system.")
                self.log_output("   Please register as a new student first.")
                self.current_student = None
                self.update_status("Face not recognized")
        
        except Exception as e:
            self.log_output(f"\n❌ Error: {e}")
            self.update_status("Error during face scanning")
            logger.error(f"Error scanning face: {e}")
    
    def mark_attendance(self):
        """Mark attendance for current student."""
        if not self.current_student:
            messagebox.showwarning(
                "No Student Selected",
                "Please scan and identify a face first."
            )
            return
        
        self.update_status("Marking attendance...")
        self.log_output("\n" + "="*60)
        self.log_output("MARKING ATTENDANCE")
        self.log_output("="*60)
        
        try:
            result = self.attendance_manager.mark_attendance_by_name(self.current_student)
            
            if result['success']:
                self.log_output(f"\n✅ Attendance marked successfully!")
                self.log_output(f"   Student: {result['student_name']}")
                self.log_output(f"   Attendance: {result['attendance_count']}/{result['total_classes']}")
                self.log_output(f"   Percentage: {result['attendance_percentage']:.2f}%")
                self.log_output(f"   Date: {result['date']}")
                self.update_status(f"Attendance marked for {result['student_name']}")
            else:
                if result.get('already_marked'):
                    self.log_output(f"\n⚠️ {result['message']}")
                    self.log_output(f"   Student: {self.current_student}")
                else:
                    self.log_output(f"\n❌ {result['message']}")
                self.update_status("Attendance marking failed")
        
        except Exception as e:
            self.log_output(f"\n❌ Error: {e}")
            self.update_status("Error marking attendance")
            logger.error(f"Error marking attendance: {e}")
    
    def check_prediction(self):
        """Check dropout prediction for current student."""
        if not self.current_student:
            messagebox.showwarning(
                "No Student Selected",
                "Please scan and identify a face first."
            )
            return
        
        if not self.predictor:
            messagebox.showerror(
                "Model Not Available",
                "Trained model not found. Please train the model first."
            )
            return
        
        self.update_status("Checking prediction...")
        self.log_output("\n" + "="*60)
        self.log_output("DROPOUT PREDICTION")
        self.log_output("="*60)
        
        try:
            # Get student data from database
            conn = get_db_connection()
            student_data = get_student_by_name(conn, self.current_student)
            conn.close()
            
            if not student_data:
                self.log_output(f"\n❌ Student data not found")
                return
            
            # Make prediction
            result = self.predictor.predict_from_student_data(student_data)
            
            # Display results
            self.log_output(f"\n📊 Prediction Results for {self.current_student}:")
            self.log_output(f"   Dropout Probability: {result['dropout_probability']*100:.2f}%")
            self.log_output(f"   Risk Level: {result['risk_level']}")
            self.log_output(f"   Prediction: {'HIGH RISK' if result['dropout_prediction'] == 1 else 'LOW RISK'}")
            
            self.log_output(f"\n📈 Contributing Factors:")
            for feature, impact in result['contributing_factors'].items():
                self.log_output(f"   - {feature.replace('_', ' ').title()}: {impact:.4f}")
            
            self.log_output(f"\n🎯 Highest Contributing Factor:")
            self.log_output(f"   {result['highest_contributing_factor']['feature'].replace('_', ' ').title()}")
            
            self.log_output(f"\n📝 Explanation:")
            self.log_output(result['explanation'])
            
            self.update_status(f"Prediction checked for {self.current_student}")
        
        except Exception as e:
            self.log_output(f"\n❌ Error: {e}")
            self.update_status("Error checking prediction")
            logger.error(f"Error checking prediction: {e}")
    
    def register_new_student(self):
        """Register a new student."""
        self.update_status("Starting registration...")
        self.log_output("\n" + "="*60)
        self.log_output("NEW STUDENT REGISTRATION")
        self.log_output("="*60)
        self.log_output("\nPlease follow the prompts in the console/terminal window.")
        self.log_output("The registration window will open separately.\n")
        
        # Run registration in a separate thread to avoid blocking
        def run_registration():
            try:
                result = self.registration.register_student()
                
                if result['success']:
                    self.log_output(f"\n✅ Registration successful!")
                    self.log_output(f"   Student ID: {result['student_id']}")
                    self.log_output(f"   Name: {result['name']}")
                    self.log_output(f"   User ID: {result['user_id']}")
                    
                    if result.get('prediction'):
                        pred = result['prediction']
                        self.log_output(f"\n📊 Initial Prediction:")
                        self.log_output(f"   Probability: {pred['dropout_probability']*100:.2f}%")
                        self.log_output(f"   Risk Level: {pred['risk_level']}")
                    
                    # Refresh face database
                    self.face_module.load_known_faces()
                    self.update_status("Registration completed")
                else:
                    self.log_output(f"\n❌ Registration failed: {result.get('message', 'Unknown error')}")
                    self.update_status("Registration failed")
            
            except Exception as e:
                self.log_output(f"\n❌ Error during registration: {e}")
                self.update_status("Registration error")
                logger.error(f"Error during registration: {e}")
        
        thread = threading.Thread(target=run_registration, daemon=True)
        thread.start()
    
    def view_all_students(self):
        """View all registered students."""
        self.update_status("Loading students...")
        self.log_output("\n" + "="*60)
        self.log_output("ALL REGISTERED STUDENTS")
        self.log_output("="*60)
        
        try:
            stats = self.attendance_manager.get_all_attendance_stats()
            
            if not stats:
                self.log_output("\n⚠️ No students registered yet.")
                return
            
            self.log_output(f"\n📊 Total Students: {len(stats)}\n")
            self.log_output(f"{'Name':<30} {'Attendance':<15} {'%':<10} {'Last Date':<15}")
            self.log_output("-" * 70)
            
            for stat in stats:
                name = stat['name'][:28]  # Truncate if too long
                attendance = f"{stat['attendance_count']}/{stat['total_classes']}"
                percentage = f"{stat['attendance_percentage']:.1f}%"
                last_date = stat['last_attendance_date'] or "Never"
                
                self.log_output(f"{name:<30} {attendance:<15} {percentage:<10} {last_date:<15}")
            
            self.update_status(f"Loaded {len(stats)} students")
        
        except Exception as e:
            self.log_output(f"\n❌ Error: {e}")
            self.update_status("Error loading students")
            logger.error(f"Error viewing students: {e}")
    
    def refresh_faces(self):
        """Refresh the face database."""
        self.update_status("Refreshing face database...")
        self.log_output("\n" + "="*60)
        self.log_output("REFRESHING FACE DATABASE")
        self.log_output("="*60)
        
        try:
            self.face_module.load_known_faces()
            num_faces = len(self.face_module.known_names)
            self.log_output(f"\n✅ Face database refreshed!")
            self.log_output(f"   Loaded {num_faces} known faces")
            self.update_status(f"Loaded {num_faces} faces")
        
        except Exception as e:
            self.log_output(f"\n❌ Error: {e}")
            self.update_status("Error refreshing faces")
            logger.error(f"Error refreshing faces: {e}")


def main():
    """Main function to run the application."""
    # Initialize database
    try:
        from database.init_db import init_database
        init_database()
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")
    
    # Create and run application
    root = tk.Tk()
    app = StudentDropoutApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application closed by user")


if __name__ == "__main__":
    main()


