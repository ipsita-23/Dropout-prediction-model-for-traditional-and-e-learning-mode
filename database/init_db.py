"""
Database Initialization Script

This script initializes the SQLite database for the student dropout prediction system.

Author: AI Project
Date: 2024
"""

import sqlite3
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_database(db_path: str = 'database/students.db'):
    """
    Initialize the SQLite database with the students table.
    
    Parameters:
    -----------
    db_path : str
        Path to the database file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    logger.info(f"Initializing database at {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create students table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        face_encoding_path TEXT,
        attendance_count INTEGER DEFAULT 0,
        total_classes INTEGER DEFAULT 0,
        attendance_percentage REAL DEFAULT 0.0,
        academic_score REAL,
        indiscipline_count INTEGER DEFAULT 0,
        online_engagement_score REAL,
        last_attendance_date TEXT,
        dropout_prediction REAL,
        dropout_probability REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    cursor.execute(create_table_query)
    
    # Create index on name for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_name ON students(name)
    """)
    
    # Create index on face_encoding_path
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_face_path ON students(face_encoding_path)
    """)
    
    conn.commit()
    conn.close()
    
    logger.info("✅ Database initialized successfully!")
    logger.info(f"   Table: students")
    logger.info(f"   Location: {db_path}")


def get_db_connection(db_path: str = 'database/students.db'):
    """
    Get a database connection.
    
    Parameters:
    -----------
    db_path : str
        Path to the database file
    
    Returns:
    --------
    sqlite3.Connection
        Database connection
    """
    # Initialize database if it doesn't exist
    if not os.path.exists(db_path):
        init_database(db_path)
    
    return sqlite3.connect(db_path)


def add_student(conn: sqlite3.Connection, name: str, face_encoding_path: str = None,
               academic_score: float = None, indiscipline_count: int = 0,
               online_engagement_score: float = None) -> int:
    """
    Add a new student to the database.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    name : str
        Student name
    face_encoding_path : str
        Path to face encoding file
    academic_score : float
        Academic score (0-10)
    indiscipline_count : int
        Number of indiscipline incidents
    online_engagement_score : float
        Online engagement score (0-100)
    
    Returns:
    --------
    int
        Student ID
    """
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO students (
        name, face_encoding_path, academic_score, 
        indiscipline_count, online_engagement_score, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?)
    """
    
    cursor.execute(insert_query, (
        name, face_encoding_path, academic_score,
        indiscipline_count, online_engagement_score, datetime.now().isoformat()
    ))
    
    student_id = cursor.lastrowid
    conn.commit()
    
    logger.info(f"Student added: {name} (ID: {student_id})")
    return student_id


def get_student_by_name(conn: sqlite3.Connection, name: str) -> dict:
    """
    Get student by name.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    name : str
        Student name
    
    Returns:
    --------
    dict
        Student record, or None if not found
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE name = ?", (name,))
    row = cursor.fetchone()
    
    if row:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    return None


def get_student_by_face_path(conn: sqlite3.Connection, face_path: str) -> dict:
    """
    Get student by face encoding path.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    face_path : str
        Path to face encoding file
    
    Returns:
    --------
    dict
        Student record, or None if not found
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE face_encoding_path = ?", (face_path,))
    row = cursor.fetchone()
    
    if row:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    return None


def update_student_attendance(conn: sqlite3.Connection, student_id: int, 
                            attendance_date: str = None):
    """
    Update student attendance.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    student_id : int
        Student ID
    attendance_date : str
        Attendance date (ISO format), defaults to today
    """
    if attendance_date is None:
        attendance_date = datetime.now().date().isoformat()
    
    cursor = conn.cursor()
    
    # Get current values
    cursor.execute("SELECT attendance_count, total_classes FROM students WHERE id = ?", 
                  (student_id,))
    row = cursor.fetchone()
    
    if row:
        attendance_count, total_classes = row
        
        # Update counts
        new_attendance_count = attendance_count + 1
        new_total_classes = total_classes + 1
        new_percentage = (new_attendance_count / new_total_classes) * 100 if new_total_classes > 0 else 0
        
        # Update database
        update_query = """
        UPDATE students SET
            attendance_count = ?,
            total_classes = ?,
            attendance_percentage = ?,
            last_attendance_date = ?,
            updated_at = ?
        WHERE id = ?
        """
        
        cursor.execute(update_query, (
            new_attendance_count, new_total_classes, new_percentage,
            attendance_date, datetime.now().isoformat(), student_id
        ))
        
        conn.commit()
        logger.info(f"Attendance updated for student ID {student_id}")
    else:
        logger.warning(f"Student ID {student_id} not found")


def update_student_prediction(conn: sqlite3.Connection, student_id: int,
                             dropout_prediction: int, dropout_probability: float):
    """
    Update student dropout prediction.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    student_id : int
        Student ID
    dropout_prediction : int
        Predicted dropout (0 or 1)
    dropout_probability : float
        Dropout probability (0-1)
    """
    cursor = conn.cursor()
    
    update_query = """
    UPDATE students SET
        dropout_prediction = ?,
        dropout_probability = ?,
        updated_at = ?
    WHERE id = ?
    """
    
    cursor.execute(update_query, (
        dropout_prediction, dropout_probability,
        datetime.now().isoformat(), student_id
    ))
    
    conn.commit()
    logger.info(f"Prediction updated for student ID {student_id}")


def get_all_students(conn: sqlite3.Connection) -> list:
    """
    Get all students from the database.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    
    Returns:
    --------
    list
        List of student records
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY name")
    rows = cursor.fetchall()
    
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def check_attendance_today(conn: sqlite3.Connection, student_id: int) -> bool:
    """
    Check if student has already marked attendance today.
    
    Parameters:
    -----------
    conn : sqlite3.Connection
        Database connection
    student_id : int
        Student ID
    
    Returns:
    --------
    bool
        True if attendance already marked today, False otherwise
    """
    today = datetime.now().date().isoformat()
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT last_attendance_date FROM students 
        WHERE id = ? AND last_attendance_date = ?
    """, (student_id, today))
    
    return cursor.fetchone() is not None


if __name__ == "__main__":
    # Initialize database
    init_database('database/students.db')
    
    # Test connection
    conn = get_db_connection('database/students.db')
    print("\n✅ Database initialized and connection tested!")
    print(f"📁 Database location: database/students.db")
    
    # Show table structure
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(students)")
    columns = cursor.fetchall()
    
    print("\n📋 Table Structure:")
    print("-" * 60)
    for col in columns:
        print(f"  {col[1]:<25} {col[2]:<15} {'NOT NULL' if col[3] else ''}")
    print("-" * 60)
    
    conn.close()


