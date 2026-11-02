import sqlite3
from pathlib import Path

# Database location
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "vegetable_history.db"


# Create database and table
def init_db():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vegetable TEXT NOT NULL,
            predicted_stage TEXT NOT NULL,
            confidence REAL NOT NULL,
            temperature REAL,
            humidity REAL,
            gas_value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# Save prediction
def save_prediction(data, result):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            vegetable,
            predicted_stage,
            confidence,
            temperature,
            humidity,
            gas_value
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data["Vegetable_Type"],
        result["predicted_stage"],
        result["confidence"],
        data["DHT22_Temp_C"],
        data["DHT22_Humidity_Percent"],
        data["MQ135_ADC_Raw"]
    ))

    conn.commit()
    conn.close()


# Retrieve prediction history
def get_predictions():

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
    """)

    records = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return records