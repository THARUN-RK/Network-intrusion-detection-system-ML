"""
Network Intrusion Detection System (NIDS) - SQLite Database Module
Maintains persistent audit logs of predictions, security alerts, and analytics.
Uses parameterized queries for injection safety.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_db_connection(db_path: Path = config.DATABASE_PATH) -> sqlite3.Connection:
    """Creates a connection to the SQLite database with Row factory enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = config.DATABASE_PATH) -> None:
    """Initializes the database schema with necessary tables and indexes."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Predictions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_identifier TEXT,
            predicted_category TEXT NOT NULL,
            confidence REAL NOT NULL,
            security_status TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            destination_port INTEGER,
            flow_duration REAL,
            metadata_json TEXT
        )
    """)
    
    # 2. Alerts Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            alert_level TEXT NOT NULL,
            attack_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)
    
    # 3. Create indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(security_status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_category ON predictions(predicted_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_level ON security_alerts(alert_level)")
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at: {db_path}")


def save_prediction(
    predicted_category: str,
    confidence: float,
    security_status: str,
    record_identifier: Optional[str] = None,
    destination_port: Optional[int] = None,
    flow_duration: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Path = config.DATABASE_PATH
) -> int:
    """Stores a single prediction record with parameterized query."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    meta_json = json.dumps(metadata) if metadata else "{}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO predictions (
            record_identifier, predicted_category, confidence, security_status,
            timestamp, destination_port, flow_duration, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record_identifier or f"flow-{int(datetime.now().timestamp()*1000)}",
        predicted_category,
        float(confidence),
        security_status,
        now_str,
        destination_port,
        flow_duration,
        meta_json
    ))
    
    pred_id = cursor.lastrowid
    
    # Generate automatic security alert if an attack is detected
    if security_status == "ATTACK":
        alert_level = "CRITICAL" if confidence >= 85.0 else "SUSPICIOUS"
        desc = f"Detected {predicted_category} threat with {confidence:.1f}% confidence."
        cursor.execute("""
            INSERT INTO security_alerts (prediction_id, alert_level, attack_type, confidence, timestamp, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (pred_id, alert_level, predicted_category, float(confidence), now_str, desc))
        
    conn.commit()
    conn.close()
    return pred_id


def save_predictions_batch(
    records: List[Dict[str, Any]],
    db_path: Path = config.DATABASE_PATH
) -> int:
    """Stores a batch of predictions efficiently in a single transaction."""
    if not records:
        return 0
        
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pred_tuples = []
    alert_tuples = []
    
    for r in records:
        meta_json = json.dumps(r.get("metadata", {}))
        sec_status = r.get("security_status", "BENIGN")
        conf = float(r.get("confidence", 0.0))
        cat = r.get("predicted_category", "BENIGN")
        
        pred_tuples.append((
            r.get("record_identifier") or f"flow-{int(datetime.now().timestamp()*1000)}",
            cat,
            conf,
            sec_status,
            now_str,
            r.get("destination_port"),
            r.get("flow_duration"),
            meta_json
        ))
        
    cursor.executemany("""
        INSERT INTO predictions (
            record_identifier, predicted_category, confidence, security_status,
            timestamp, destination_port, flow_duration, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, pred_tuples)
    
    # Insert alerts for attacks
    for r in records:
        if r.get("security_status") == "ATTACK":
            conf = float(r.get("confidence", 0.0))
            cat = r.get("predicted_category", "ATTACK")
            level = "CRITICAL" if conf >= 85.0 else "SUSPICIOUS"
            desc = f"Batch detection: {cat} ({conf:.1f}% confidence)"
            alert_tuples.append((None, level, cat, conf, now_str, desc))
            
    if alert_tuples:
        cursor.executemany("""
            INSERT INTO security_alerts (prediction_id, alert_level, attack_type, confidence, timestamp, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, alert_tuples)
        
    conn.commit()
    inserted_count = len(pred_tuples)
    conn.close()
    return inserted_count


def get_prediction_history(
    limit: int = 100,
    offset: int = 0,
    status_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    db_path: Path = config.DATABASE_PATH
) -> List[Dict[str, Any]]:
    """Fetches paginated prediction history with optional filtering."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if status_filter:
        query += " AND security_status = ?"
        params.append(status_filter)
        
    if category_filter:
        query += " AND predicted_category = ?"
        params.append(category_filter)
        
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recent_alerts(limit: int = 20, db_path: Path = config.DATABASE_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent security alerts."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM security_alerts
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_overview_statistics(db_path: Path = config.DATABASE_PATH) -> Dict[str, Any]:
    """Generates real-time SOC dashboard analytics from prediction history."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE security_status = 'BENIGN'")
    normal_connections = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE security_status = 'ATTACK'")
    detected_attacks = cursor.fetchone()[0]
    
    attack_percentage = (detected_attacks / total_records * 100.0) if total_records > 0 else 0.0
    
    # Most frequent attack
    cursor.execute("""
        SELECT predicted_category, COUNT(*) as cnt
        FROM predictions
        WHERE security_status = 'ATTACK'
        GROUP BY predicted_category
        ORDER BY cnt DESC LIMIT 1
    """)
    top_attack_row = cursor.fetchone()
    top_attack = top_attack_row[0] if top_attack_row else "None"
    
    # Attack breakdown
    cursor.execute("""
        SELECT predicted_category, COUNT(*) as cnt
        FROM predictions
        GROUP BY predicted_category
        ORDER BY cnt DESC
    """)
    category_counts = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        "total_records_analysed": total_records,
        "total_normal_connections": normal_connections,
        "total_detected_attacks": detected_attacks,
        "attack_detection_percentage": round(attack_percentage, 2),
        "most_frequent_attack": top_attack,
        "category_distribution": category_counts
    }


if __name__ == "__main__":
    init_db()
    # Test insertion
    p_id = save_prediction("DDoS", 98.4, "ATTACK", destination_port=80, flow_duration=12000.0)
    print(f"Sample prediction inserted with ID: {p_id}")
    stats = get_overview_statistics()
    print("Overview Stats:", stats)
