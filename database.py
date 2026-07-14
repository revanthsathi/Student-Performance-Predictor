import sqlite3
from datetime import datetime

DATABASE_NAME = "student_performance.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            registration_number TEXT,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_marks REAL NOT NULL,
            sleep_hours REAL NOT NULL,
            assignments_completed INTEGER NOT NULL,
            class_participation INTEGER NOT NULL,
            screen_time REAL NOT NULL,
            extracurricular_hours REAL NOT NULL,
            stress_level INTEGER NOT NULL,
            predicted_score REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def normalize_name(name):
    return " ".join(name.strip().lower().split())


def get_previous_prediction(student_name, registration_number):
    connection = get_connection()

    normalized_name = normalize_name(student_name)
    registration_number = registration_number.strip().lower()

    if registration_number:
        result = connection.execute(
            """
            SELECT *
            FROM predictions
            WHERE LOWER(registration_number) = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (registration_number,)
        ).fetchone()
    else:
        result = connection.execute(
            """
            SELECT *
            FROM predictions
            WHERE normalized_name = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (normalized_name,)
        ).fetchone()

    connection.close()

    return dict(result) if result else None


def save_prediction(data, predicted_score):
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO predictions (
            student_name,
            normalized_name,
            registration_number,
            study_hours,
            attendance,
            previous_marks,
            sleep_hours,
            assignments_completed,
            class_participation,
            screen_time,
            extracurricular_hours,
            stress_level,
            predicted_score,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["student_name"].strip(),
            normalize_name(data["student_name"]),
            data["registration_number"].strip(),
            data["study_hours"],
            data["attendance"],
            data["previous_marks"],
            data["sleep_hours"],
            data["assignments_completed"],
            data["class_participation"],
            data["screen_time"],
            data["extracurricular_hours"],
            data["stress_level"],
            float(predicted_score),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    connection.commit()
    connection.close()


def get_student_history(student_name="", registration_number=""):
    connection = get_connection()

    normalized_name = normalize_name(student_name)
    registration_number = registration_number.strip().lower()

    if registration_number:
        results = connection.execute(
            """
            SELECT *
            FROM predictions
            WHERE LOWER(registration_number) = ?
            ORDER BY id DESC
            """,
            (registration_number,)
        ).fetchall()

    elif normalized_name:
        results = connection.execute(
            """
            SELECT *
            FROM predictions
            WHERE normalized_name = ?
            ORDER BY id DESC
            """,
            (normalized_name,)
        ).fetchall()

    else:
        results = []

    connection.close()

    return [dict(row) for row in results]