"""
Network Intrusion Detection System (NIDS) & Security Operations Center (SOC) Database
Multi-Database Engine: Supports both SQLite (local development) and PostgreSQL (production).
Maintains models for:
- Predictions (network flow classification logs with IP tracking)
- Security Alerts (threat alerts with severity & resolution status)
- Users (RBAC authentication with admin and analyst roles)
- Incidents (SOC incident triage and investigation tracking)
- IP Reputation & Monitoring (source/destination IP behavioral profiling)
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

def utc_now():
    """Returns timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Index, desc, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session, relationship

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base Model
Base = declarative_base()

# Multi-Database Configuration
# Reads DATABASE_URL from environment (e.g. PostgreSQL on Render/AWS) or defaults to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{config.DATABASE_PATH.as_posix()}")

# Handle Render's postgres:// prefix -> postgresql:// for SQLAlchemy compatibility
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


# ==============================================================================
# ORM ENTITY MODELS
# ==============================================================================

class PredictionModel(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    record_identifier = Column(String(100), index=True)
    predicted_category = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    security_status = Column(String(20), nullable=False, index=True)  # BENIGN, ATTACK, ANOMALY
    severity = Column(String(20), default="LOW", index=True)          # LOW, MEDIUM, HIGH, CRITICAL
    is_anomaly = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=utc_now, index=True)
    
    # IP & Flow Tracking
    source_ip = Column(String(45), default="192.168.1.100", index=True)
    destination_ip = Column(String(45), default="10.0.0.1", index=True)
    source_port = Column(Integer, default=0)
    destination_port = Column(Integer, default=80)
    protocol = Column(String(10), default="TCP")
    flow_duration = Column(Float, default=0.0)
    metadata_json = Column(Text, default="{}")

    alerts = relationship("SecurityAlertModel", back_populates="prediction", cascade="all, delete-orphan")


class SecurityAlertModel(Base):
    __tablename__ = "security_alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    alert_level = Column(String(20), nullable=False, index=True)       # LOW, MEDIUM, HIGH, CRITICAL
    attack_type = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    status = Column(String(30), default="NEW", index=True)            # NEW, INVESTIGATING, RESOLVED, FALSE_POSITIVE
    source_ip = Column(String(45), default="192.168.1.100")
    destination_ip = Column(String(45), default="10.0.0.1")
    timestamp = Column(DateTime, default=utc_now, index=True)
    description = Column(Text)

    prediction = relationship("PredictionModel", back_populates="alerts")


class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="analyst", nullable=False)      # admin, analyst
    full_name = Column(String(100), default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)


class IncidentModel(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    incident_code = Column(String(50), unique=True, nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    severity = Column(String(20), default="HIGH", index=True)         # LOW, MEDIUM, HIGH, CRITICAL
    status = Column(String(30), default="OPEN", index=True)           # OPEN, IN_PROGRESS, RESOLVED, CLOSED
    assigned_to = Column(String(50), default="Unassigned")
    source_ip = Column(String(45))
    target_ip = Column(String(45))
    created_at = Column(DateTime, default=utc_now, index=True)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    investigator_notes = Column(Text, default="")


class IPReputationModel(Base):
    __tablename__ = "ip_reputation"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ip_address = Column(String(45), unique=True, nullable=False, index=True)
    total_flows = Column(Integer, default=0)
    attack_count = Column(Integer, default=0)
    reputation_score = Column(Float, default=100.0)                   # 0.0 (Malicious) - 100.0 (Clean)
    threat_level = Column(String(20), default="SAFE", index=True)     # SAFE, SUSPICIOUS, MALICIOUS
    is_blocked = Column(Boolean, default=False, index=True)
    country = Column(String(50), default="Unknown")
    last_seen = Column(DateTime, default=utc_now)


# ==============================================================================
# DATABASE INITIALIZATION & COMPATIBILITY LAYER
# ==============================================================================

def get_db():
    """Dependency for FastAPI route handlers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_connection(db_path: Path = config.DATABASE_PATH):
    """Raw connection backward-compatibility for SQLite direct callers."""
    import sqlite3
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initializes tables, handles schema migrations, and seeds default admin and analyst users."""
    if db_path is None:
        config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database schema initialized using engine: {engine.url.drivername}")
    
    # Auto-migration for existing SQLite databases
    if engine.url.drivername.startswith("sqlite"):
        with engine.connect() as conn:
            # Check predictions columns
            pred_cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(predictions)").fetchall()]
            new_pred_columns = [
                ("severity", "TEXT DEFAULT 'LOW'"),
                ("is_anomaly", "BOOLEAN DEFAULT 0"),
                ("source_ip", "TEXT DEFAULT '192.168.1.100'"),
                ("destination_ip", "TEXT DEFAULT '10.0.0.1'"),
                ("source_port", "INTEGER DEFAULT 0"),
                ("protocol", "TEXT DEFAULT 'TCP'")
            ]
            for col_name, col_type in new_pred_columns:
                if col_name not in pred_cols:
                    conn.exec_driver_sql(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Migrated predictions table: added column {col_name}")

            # Check security_alerts columns
            alert_cols = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(security_alerts)").fetchall()]
            new_alert_columns = [
                ("status", "TEXT DEFAULT 'NEW'"),
                ("source_ip", "TEXT DEFAULT '192.168.1.100'"),
                ("destination_ip", "TEXT DEFAULT '10.0.0.1'")
            ]
            for col_name, col_type in new_alert_columns:
                if col_name not in alert_cols:
                    conn.exec_driver_sql(f"ALTER TABLE security_alerts ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Migrated security_alerts table: added column {col_name}")
            conn.commit()
    
    # Seed default SOC users if empty
    session = SessionLocal()
    try:
        user_count = session.query(UserModel).count()
        if user_count == 0:
            import bcrypt
            
            admin_pw = bcrypt.hashpw("Admin@12345".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            analyst_pw = bcrypt.hashpw("Analyst@12345".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            
            admin_user = UserModel(
                username="admin",
                email="admin@soc.defense.corp",
                hashed_password=admin_pw,
                role="admin",
                full_name="Chief Security Officer (CSO)"
            )
            analyst_user = UserModel(
                username="analyst",
                email="analyst@soc.defense.corp",
                hashed_password=analyst_pw,
                role="analyst",
                full_name="Tier-2 SOC Analyst"
            )
            session.add_all([admin_user, analyst_user])
            session.commit()
            logger.info("Seeded default SOC users: 'admin' and 'analyst'")

        # Seed baseline network traffic flows if empty
        pred_count = session.query(PredictionModel).count()
        if pred_count == 0:
            seed_flows = [
                ("BENIGN", 99.4, "BENIGN", "LOW", False, "192.168.1.105", "142.250.190.46", 49152, 443, "TCP", 1240.0),
                ("BENIGN", 98.2, "BENIGN", "LOW", False, "192.168.1.102", "192.168.1.1", 51220, 80, "TCP", 850.0),
                ("BENIGN", 97.8, "BENIGN", "LOW", False, "192.168.1.110", "172.217.16.206", 53100, 443, "TCP", 2100.0),
                ("BENIGN", 99.1, "BENIGN", "LOW", False, "192.168.1.108", "8.8.8.8", 61002, 53, "UDP", 45.0),
                ("DDoS", 100.0, "ATTACK", "CRITICAL", False, "185.220.101.5", "10.0.0.1", 44102, 80, "TCP", 15200.0),
                ("DoS", 99.8, "ATTACK", "HIGH", False, "194.26.29.112", "10.0.0.1", 38910, 80, "TCP", 9800.0),
                ("PortScan", 100.0, "ATTACK", "MEDIUM", False, "45.33.32.156", "10.0.0.1", 55210, 22, "TCP", 120.0),
                ("Bot", 92.5, "ATTACK", "HIGH", False, "103.251.167.20", "10.0.0.5", 58120, 8080, "TCP", 4500.0),
                ("Web Attack", 90.3, "ATTACK", "HIGH", False, "198.51.100.42", "10.0.0.1", 41200, 80, "TCP", 3100.0),
                ("FTP-Patator", 96.0, "ATTACK", "MEDIUM", False, "203.0.113.88", "10.0.0.2", 34510, 21, "TCP", 890.0),
                ("SSH-Patator", 98.2, "ATTACK", "HIGH", False, "198.51.100.15", "10.0.0.2", 52190, 22, "TCP", 1200.0),
                ("Infiltration", 99.1, "ATTACK", "CRITICAL", False, "172.16.0.4", "10.0.0.10", 49800, 445, "TCP", 7600.0),
            ]
            for cat, conf, sec_st, sev, is_ano, src_ip, dst_ip, s_port, d_port, proto, dur in seed_flows:
                rec_id = f"flow-{src_ip}-{dst_ip}-{s_port}-{d_port}"
                pred_obj = PredictionModel(
                    record_identifier=rec_id,
                    predicted_category=cat,
                    confidence=conf,
                    security_status=sec_st,
                    severity=sev,
                    is_anomaly=is_ano,
                    source_ip=src_ip,
                    destination_ip=dst_ip,
                    source_port=s_port,
                    destination_port=d_port,
                    protocol=proto,
                    flow_duration=dur
                )
                session.add(pred_obj)
                session.flush()

                # Generate alert for attack or anomaly
                if sec_st in ["ATTACK", "ANOMALY"]:
                    alert_obj = SecurityAlertModel(
                        prediction_id=pred_obj.id,
                        alert_level=sev,
                        attack_type=cat,
                        confidence=conf,
                        status="NEW",
                        source_ip=src_ip,
                        destination_ip=dst_ip,
                        description=f"Automated threat triage: {cat} detected with {conf:.1f}% confidence."
                    )
                    session.add(alert_obj)

                # Track IP reputation
                ip_entry = session.query(IPReputationModel).filter_by(ip_address=src_ip).first()
                if not ip_entry:
                    th_level = "MALICIOUS" if sec_st == "ATTACK" else "SAFE"
                    rep_sc = 15.0 if sec_st == "ATTACK" else 95.0
                    ip_entry = IPReputationModel(
                        ip_address=src_ip,
                        total_flows=1,
                        attack_count=1 if sec_st == "ATTACK" else 0,
                        reputation_score=rep_sc,
                        threat_level=th_level,
                        is_blocked=(sev == "CRITICAL")
                    )
                    session.add(ip_entry)

            session.commit()
            logger.info("Seeded initial baseline network flows and security alerts.")
    except Exception as e:
        session.rollback()
        logger.error(f"Error seeding default users: {e}")
    finally:
        session.close()


# ==============================================================================
# PREDICTIONS & ALERTS LOGIC
# ==============================================================================

def map_severity(category: str, confidence: float, is_anomaly: bool = False) -> str:
    """Maps classified threat and confidence to standard SOC severity."""
    if is_anomaly:
        return "HIGH"
    if category == "BENIGN":
        return "LOW"
    if category in ["DDoS", "Heartbleed", "Infiltration"]:
        return "CRITICAL"
    if category in ["DoS", "Web Attack", "Bot"]:
        return "HIGH" if confidence >= 80.0 else "MEDIUM"
    return "MEDIUM" if confidence >= 75.0 else "LOW"


def update_ip_profile(session, ip: str, is_attack: bool) -> None:
    """Updates behavioral profiling and reputation score of an IP address."""
    if not ip or ip in ["Unknown", "127.0.0.1", "localhost"]:
        return
        
    profile = session.query(IPReputationModel).filter_by(ip_address=ip).first()
    if not profile:
        profile = IPReputationModel(
            ip_address=ip,
            total_flows=1,
            attack_count=1 if is_attack else 0,
            reputation_score=70.0 if is_attack else 100.0,
            threat_level="SUSPICIOUS" if is_attack else "SAFE"
        )
        session.add(profile)
    else:
        profile.total_flows += 1
        if is_attack:
            profile.attack_count += 1
            # Penalize reputation score proportionally
            profile.reputation_score = max(0.0, profile.reputation_score - 15.0)
        else:
            profile.reputation_score = min(100.0, profile.reputation_score + 0.5)
            
        profile.last_seen = utc_now()
        if profile.reputation_score < 40.0:
            profile.threat_level = "MALICIOUS"
        elif profile.reputation_score < 75.0:
            profile.threat_level = "SUSPICIOUS"
        else:
            profile.threat_level = "SAFE"


def save_prediction(
    predicted_category: str,
    confidence: float,
    security_status: str,
    record_identifier: Optional[str] = None,
    destination_port: Optional[int] = None,
    flow_duration: Optional[float] = None,
    source_ip: Optional[str] = None,
    destination_ip: Optional[str] = None,
    protocol: Optional[str] = "TCP",
    is_anomaly: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None
) -> int:
    """Saves classification result, tracks IP reputation, and creates alerts."""
    session = SessionLocal()
    try:
        now = utc_now()
        src_ip = source_ip or "192.168.1.100"
        dst_ip = destination_ip or "10.0.0.1"
        severity = map_severity(predicted_category, confidence, is_anomaly)
        meta_json = json.dumps(metadata) if metadata else "{}"
        
        pred = PredictionModel(
            record_identifier=record_identifier or f"flow-{int(datetime.now().timestamp()*1000)}",
            predicted_category=predicted_category,
            confidence=float(confidence),
            security_status=security_status,
            severity=severity,
            is_anomaly=is_anomaly,
            timestamp=now,
            source_ip=src_ip,
            destination_ip=dst_ip,
            destination_port=destination_port,
            protocol=protocol or "TCP",
            flow_duration=flow_duration,
            metadata_json=meta_json
        )
        session.add(pred)
        session.flush()
        pred_id = pred.id
        
        # Automatic alert generation for attacks and zero-day anomalies
        if security_status == "ATTACK" or is_anomaly:
            alert_desc = f"Detected {predicted_category} threat from {src_ip} targeting {dst_ip}:{destination_port or 80} ({confidence:.1f}% confidence)"
            alert = SecurityAlertModel(
                prediction_id=pred_id,
                alert_level=severity,
                attack_type=predicted_category,
                confidence=float(confidence),
                status="NEW",
                source_ip=src_ip,
                destination_ip=dst_ip,
                timestamp=now,
                description=alert_desc
            )
            session.add(alert)
            update_ip_profile(session, src_ip, is_attack=True)
        else:
            update_ip_profile(session, src_ip, is_attack=False)
            
        session.commit()
        return pred_id
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save prediction: {e}")
        return 0
    finally:
        session.close()


def save_predictions_batch(
    records: List[Dict[str, Any]],
    db_path: Optional[Path] = None
) -> int:
    """Stores multiple records efficiently."""
    if not records:
        return 0
    session = SessionLocal()
    try:
        now = utc_now()
        for r in records:
            cat = r.get("predicted_category", "BENIGN")
            conf = float(r.get("confidence", 0.0))
            sec_status = r.get("security_status", "BENIGN")
            is_ano = bool(r.get("is_anomaly", False))
            src_ip = r.get("source_ip", "192.168.1.100")
            dst_ip = r.get("destination_ip", "10.0.0.1")
            sev = map_severity(cat, conf, is_ano)
            
            pred = PredictionModel(
                record_identifier=r.get("record_identifier") or f"batch-flow-{int(datetime.now().timestamp()*1000)}",
                predicted_category=cat,
                confidence=conf,
                security_status=sec_status,
                severity=sev,
                is_anomaly=is_ano,
                timestamp=now,
                source_ip=src_ip,
                destination_ip=dst_ip,
                destination_port=r.get("destination_port", 80),
                protocol=r.get("protocol", "TCP"),
                flow_duration=r.get("flow_duration", 0.0),
                metadata_json=json.dumps(r.get("metadata", {}))
            )
            session.add(pred)
            session.flush()
            
            if sec_status == "ATTACK" or is_ano:
                alert = SecurityAlertModel(
                    prediction_id=pred.id,
                    alert_level=sev,
                    attack_type=cat,
                    confidence=conf,
                    status="NEW",
                    source_ip=src_ip,
                    destination_ip=dst_ip,
                    timestamp=now,
                    description=f"Batch alert: {cat} from {src_ip}"
                )
                session.add(alert)
                update_ip_profile(session, src_ip, is_attack=True)
            else:
                update_ip_profile(session, src_ip, is_attack=False)
                
        session.commit()
        return len(records)
    except Exception as e:
        session.rollback()
        logger.error(f"Batch prediction save failed: {e}")
        return 0
    finally:
        session.close()


def get_prediction_history(
    limit: int = 100,
    offset: int = 0,
    status_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    severity_filter: Optional[str] = None,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Retrieves paginated prediction audit history."""
    session = SessionLocal()
    try:
        query = session.query(PredictionModel)
        if status_filter:
            query = query.filter(PredictionModel.security_status == status_filter)
        if category_filter:
            query = query.filter(PredictionModel.predicted_category == category_filter)
        if severity_filter:
            query = query.filter(PredictionModel.severity == severity_filter)
            
        rows = query.order_by(desc(PredictionModel.id)).offset(offset).limit(limit).all()
        return [
            {
                "id": r.id,
                "record_identifier": r.record_identifier,
                "predicted_category": r.predicted_category,
                "confidence": r.confidence,
                "security_status": r.security_status,
                "severity": r.severity,
                "is_anomaly": r.is_anomaly,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
                "source_ip": r.source_ip,
                "destination_ip": r.destination_ip,
                "destination_port": r.destination_port,
                "protocol": r.protocol,
                "flow_duration": r.flow_duration
            }
            for r in rows
        ]
    finally:
        session.close()


def get_recent_alerts(limit: int = 20, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetches real-time security alerts."""
    session = SessionLocal()
    try:
        rows = session.query(SecurityAlertModel).order_by(desc(SecurityAlertModel.id)).limit(limit).all()
        return [
            {
                "id": r.id,
                "prediction_id": r.prediction_id,
                "alert_level": r.alert_level,
                "attack_type": r.attack_type,
                "confidence": r.confidence,
                "status": r.status,
                "source_ip": r.source_ip,
                "destination_ip": r.destination_ip,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
                "description": r.description
            }
            for r in rows
        ]
    finally:
        session.close()


def update_alert_status(alert_id: int, new_status: str) -> bool:
    """Updates status of a security incident alert."""
    session = SessionLocal()
    try:
        alert = session.query(SecurityAlertModel).filter_by(id=alert_id).first()
        if alert:
            alert.status = new_status
            session.commit()
            return True
        return False
    finally:
        session.close()


def get_overview_statistics(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Computes SOC overview statistics."""
    session = SessionLocal()
    try:
        total_records = session.query(PredictionModel).count()
        normal_cnt = session.query(PredictionModel).filter(PredictionModel.security_status == "BENIGN").count()
        attack_cnt = session.query(PredictionModel).filter(PredictionModel.security_status == "ATTACK").count()
        anomaly_cnt = session.query(PredictionModel).filter(PredictionModel.is_anomaly == True).count()
        
        attack_pct = (attack_cnt / total_records * 100.0) if total_records > 0 else 0.0
        
        # Most frequent attack
        top_cat = session.query(
            PredictionModel.predicted_category, func.count(PredictionModel.id).label("cnt")
        ).filter(PredictionModel.security_status == "ATTACK").group_by(
            PredictionModel.predicted_category
        ).order_by(desc("cnt")).first()
        top_attack = top_cat[0] if top_cat else "None"
        
        # Category breakdown
        cat_counts_query = session.query(
            PredictionModel.predicted_category, func.count(PredictionModel.id)
        ).group_by(PredictionModel.predicted_category).all()
        category_distribution = {row[0]: row[1] for row in cat_counts_query}
        
        # Severity breakdown
        sev_counts_query = session.query(
            PredictionModel.severity, func.count(PredictionModel.id)
        ).group_by(PredictionModel.severity).all()
        severity_distribution = {row[0]: row[1] for row in sev_counts_query}
        
        return {
            "total_records_analysed": total_records,
            "total_normal_connections": normal_cnt,
            "total_detected_attacks": attack_cnt,
            "total_anomalies_detected": anomaly_cnt,
            "attack_detection_percentage": round(attack_pct, 2),
            "most_frequent_attack": top_attack,
            "category_distribution": category_distribution,
            "severity_distribution": severity_distribution
        }
    finally:
        session.close()


# ==============================================================================
# IP MONITORING & REPUTATION ENGINE
# ==============================================================================

def get_tracked_ips(limit: int = 50, threat_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns profiled IP addresses with reputation scores."""
    session = SessionLocal()
    try:
        query = session.query(IPReputationModel)
        if threat_filter and threat_filter != "All":
            query = query.filter(IPReputationModel.threat_level == threat_filter)
        rows = query.order_by(desc(IPReputationModel.attack_count), IPReputationModel.reputation_score).limit(limit).all()
        return [
            {
                "ip_address": r.ip_address,
                "total_flows": r.total_flows,
                "attack_count": r.attack_count,
                "reputation_score": round(r.reputation_score, 1),
                "threat_level": r.threat_level,
                "is_blocked": r.is_blocked,
                "country": r.country,
                "last_seen": r.last_seen.strftime("%Y-%m-%d %H:%M:%S") if r.last_seen else ""
            }
            for r in rows
        ]
    finally:
        session.close()


def toggle_ip_block(ip_address: str, block: bool) -> bool:
    """Blocks or unblocks a specific IP address."""
    session = SessionLocal()
    try:
        profile = session.query(IPReputationModel).filter_by(ip_address=ip_address).first()
        if not profile:
            profile = IPReputationModel(
                ip_address=ip_address,
                is_blocked=block,
                reputation_score=0.0 if block else 100.0,
                threat_level="MALICIOUS" if block else "SAFE"
            )
            session.add(profile)
        else:
            profile.is_blocked = block
            if block:
                profile.threat_level = "MALICIOUS"
                profile.reputation_score = min(profile.reputation_score, 10.0)
                
        session.commit()
        return True
    finally:
        session.close()


# ==============================================================================
# INCIDENT TRIAGE & INVESTIGATION WORKFLOW
# ==============================================================================

def create_incident(
    title: str,
    description: str,
    severity: str = "HIGH",
    source_ip: str = "Unknown",
    target_ip: str = "Unknown",
    assigned_to: str = "Unassigned"
) -> Dict[str, Any]:
    """Creates a new formal incident report for investigation."""
    session = SessionLocal()
    try:
        inc_code = f"INC-{int(utc_now().timestamp())}"
        inc = IncidentModel(
            incident_code=inc_code,
            title=title,
            description=description,
            severity=severity,
            status="OPEN",
            assigned_to=assigned_to,
            source_ip=source_ip,
            target_ip=target_ip,
            created_at=utc_now()
        )
        session.add(inc)
        session.commit()
        return {
            "id": inc.id,
            "incident_code": inc.incident_code,
            "title": inc.title,
            "status": inc.status,
            "severity": inc.severity
        }
    finally:
        session.close()


def get_incidents(status_filter: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves all tracked SOC incidents."""
    session = SessionLocal()
    try:
        query = session.query(IncidentModel)
        if status_filter and status_filter != "All":
            query = query.filter(IncidentModel.status == status_filter)
        rows = query.order_by(desc(IncidentModel.id)).limit(limit).all()
        return [
            {
                "id": r.id,
                "incident_code": r.incident_code,
                "title": r.title,
                "description": r.description,
                "severity": r.severity,
                "status": r.status,
                "assigned_to": r.assigned_to,
                "source_ip": r.source_ip,
                "target_ip": r.target_ip,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
                "updated_at": r.updated_at.strftime("%Y-%m-%d %H:%M:%S") if r.updated_at else "",
                "notes": r.investigator_notes
            }
            for r in rows
        ]
    finally:
        session.close()


def update_incident_status(incident_code: str, new_status: str, notes: Optional[str] = None) -> bool:
    """Updates status and investigator notes on an incident."""
    session = SessionLocal()
    try:
        inc = session.query(IncidentModel).filter_by(incident_code=incident_code).first()
        if inc:
            inc.status = new_status
            if notes:
                inc.investigator_notes = f"{inc.investigator_notes}\n[{utc_now().strftime('%Y-%m-%d %H:%M:%S')}] {notes}".strip()
            inc.updated_at = utc_now()
            session.commit()
            return True
        return False
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
    # Test insertion
    p_id = save_prediction(
        predicted_category="DDoS",
        confidence=98.4,
        security_status="ATTACK",
        source_ip="192.168.1.105",
        destination_ip="10.0.0.5",
        destination_port=80,
        flow_duration=12000.0
    )
    print(f"Sample prediction inserted with ID: {p_id}")
    stats = get_overview_statistics()
    print("Overview Stats:", stats)
    ips = get_tracked_ips(limit=5)
    print("Tracked IPs:", ips)
