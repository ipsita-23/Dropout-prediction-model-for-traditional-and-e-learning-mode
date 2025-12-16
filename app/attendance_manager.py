"""
Attendance Manager Module

This module handles student attendance tracking with:
- Duplicate attendance prevention (same day)
- Automatic attendance percentage calculation
- Database updates

Author: AI Project
Date: 2024
"""

import sqlite3
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.init_db import get_db_connection, check_attendance_today, update_student_attendance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttendanceManager:
    """
    Manages student attendance tracking.
    """
    
    def __init__(self, db_path: str = 'database/students.db'):
        """
        Initialize attendance manager.
        
        Parameters:
        -----------
        db_path : str
            Path to the database file
        """
        self.db_path = db_path
    
    def mark_attendance(self, student_id: int, attendance_date: str = None) -> dict:
        """
        Mark attendance for a student.
        
        Parameters:
        -----------
        student_id : int
            Student ID
        attendance_date : str
            Attendance date (ISO format), defaults to today
        
        Returns:
        --------
        dict
            Result dictionary with status and message
        """
        if attendance_date is None:
            attendance_date = datetime.now().date().isoformat()
        
        conn = get_db_connection(self.db_path)
        
        try:
            # Check if already marked today
            if check_attendance_today(conn, student_id):
                logger.warning(f"Attendance already marked for student ID {student_id} today")
                return {
                    'success': False,
                    'message': 'Attendance already marked for today',
                    'already_marked': True
                }
            
            # Get student info before update
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, attendance_count, total_classes, attendance_percentage 
                FROM students WHERE id = ?
            """, (student_id,))
            student_info = cursor.fetchone()
            
            if not student_info:
                return {
                    'success': False,
                    'message': f'Student ID {student_id} not found'
                }
            
            name, old_count, old_total, old_percentage = student_info
            
            # Update attendance
            update_student_attendance(conn, student_id, attendance_date)
            
            # Get updated values
            cursor.execute("""
                SELECT attendance_count, total_classes, attendance_percentage 
                FROM students WHERE id = ?
            """, (student_id,))
            new_count, new_total, new_percentage = cursor.fetchone()
            
            logger.info(f"Attendance marked for {name} (ID: {student_id})")
            logger.info(f"  Previous: {old_count}/{old_total} ({old_percentage:.2f}%)")
            logger.info(f"  Updated:  {new_count}/{new_total} ({new_percentage:.2f}%)")
            
            return {
                'success': True,
                'message': 'Attendance marked successfully',
                'student_name': name,
                'attendance_count': new_count,
                'total_classes': new_total,
                'attendance_percentage': round(new_percentage, 2),
                'date': attendance_date
            }
        
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }
        
        finally:
            conn.close()
    
    def mark_attendance_by_name(self, student_name: str, attendance_date: str = None) -> dict:
        """
        Mark attendance by student name.
        
        Parameters:
        -----------
        student_name : str
            Student name
        attendance_date : str
            Attendance date (ISO format), defaults to today
        
        Returns:
        --------
        dict
            Result dictionary with status and message
        """
        conn = get_db_connection(self.db_path)
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM students WHERE name = ?", (student_name,))
            row = cursor.fetchone()
            
            if not row:
                return {
                    'success': False,
                    'message': f'Student "{student_name}" not found'
                }
            
            student_id = row[0]
            return self.mark_attendance(student_id, attendance_date)
        
        finally:
            conn.close()
    
    def get_attendance_stats(self, student_id: int) -> dict:
        """
        Get attendance statistics for a student.
        
        Parameters:
        -----------
        student_id : int
            Student ID
        
        Returns:
        --------
        dict
            Attendance statistics
        """
        conn = get_db_connection(self.db_path)
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, attendance_count, total_classes, attendance_percentage,
                       last_attendance_date
                FROM students WHERE id = ?
            """, (student_id,))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            name, count, total, percentage, last_date = row
            
            return {
                'student_id': student_id,
                'name': name,
                'attendance_count': count,
                'total_classes': total,
                'attendance_percentage': round(percentage, 2),
                'last_attendance_date': last_date
            }
        
        finally:
            conn.close()
    
    def get_all_attendance_stats(self) -> list:
        """
        Get attendance statistics for all students.
        
        Returns:
        --------
        list
            List of attendance statistics dictionaries
        """
        conn = get_db_connection(self.db_path)
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, attendance_count, total_classes, 
                       attendance_percentage, last_attendance_date
                FROM students
                ORDER BY name
            """)
            
            rows = cursor.fetchall()
            
            return [
                {
                    'student_id': row[0],
                    'name': row[1],
                    'attendance_count': row[2],
                    'total_classes': row[3],
                    'attendance_percentage': round(row[4], 2),
                    'last_attendance_date': row[5]
                }
                for row in rows
            ]
        
        finally:
            conn.close()


def mark_attendance_for_student(student_id: int) -> dict:
    """
    Convenience function to mark attendance.
    
    Parameters:
    -----------
    student_id : int
        Student ID
    
    Returns:
    --------
    dict
        Result dictionary
    """
    manager = AttendanceManager()
    return manager.mark_attendance(student_id)


if __name__ == "__main__":
    # Test attendance manager
    manager = AttendanceManager()
    
    print("="*60)
    print("ATTENDANCE MANAGER TEST")
    print("="*60)
    
    # Get all students
    stats = manager.get_all_attendance_stats()
    
    if stats:
        print(f"\n📊 Found {len(stats)} students:")
        for stat in stats[:5]:  # Show first 5
            print(f"  - {stat['name']}: {stat['attendance_count']}/{stat['total_classes']} ({stat['attendance_percentage']:.2f}%)")
    else:
        print("\n⚠️ No students found in database")
        print("   Please register students first using user_registration.py")
    
    print("\n✅ Attendance manager is ready!")

