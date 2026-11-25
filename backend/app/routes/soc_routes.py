"""
FastAPI Routes: Security Operations Center (SOC) Advanced Management
Includes Incident Investigation, IP Monitoring, Alert Triage, and PDF/CSV Report Generation.
"""

import io
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Response, Query
from fastapi.responses import StreamingResponse

import database
from backend.app.schemas import (
    IncidentCreateRequest,
    IncidentUpdateRequest,
    IncidentResponse,
    IPReputationResponse,
    IPBlockRequest
)

router = APIRouter(prefix="/soc", tags=["SOC Management & Incident Response"])


# ==============================================================================
# INCIDENT INVESTIGATION WORKFLOW
# ==============================================================================

@router.post("/incidents", response_model=IncidentResponse)
def create_new_incident(request: IncidentCreateRequest):
    """Creates a formal cybersecurity incident for investigation."""
    try:
        inc = database.create_incident(
            title=request.title,
            description=request.description,
            severity=request.severity,
            source_ip=request.source_ip,
            target_ip=request.target_ip,
            assigned_to=request.assigned_to
        )
        # Fetch full object
        incidents = database.get_incidents(limit=1)
        return incidents[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create incident: {str(e)}")


@router.get("/incidents", response_model=List[IncidentResponse])
def list_incidents(status: Optional[str] = "All", limit: int = 50):
    """Lists tracked SOC incidents filtered by status (OPEN, IN_PROGRESS, RESOLVED, CLOSED)."""
    try:
        status_val = status if isinstance(status, str) else "All"
        limit_val = limit if isinstance(limit, int) else 50
        return database.get_incidents(status_filter=status_val, limit=limit_val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/incidents/{incident_code}")
def update_incident(incident_code: str, request: IncidentUpdateRequest):
    """Updates status and adds investigator notes to an ongoing incident."""
    success = database.update_incident_status(
        incident_code=incident_code,
        new_status=request.status,
        notes=request.notes
    )
    if not success:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"message": "Incident updated successfully", "incident_code": incident_code, "status": request.status}


# ==============================================================================
# IP MONITORING & REPUTATION ENGINE
# ==============================================================================

@router.get("/ips", response_model=List[IPReputationResponse])
def get_ip_profiles(threat_level: Optional[str] = "All", limit: int = 50):
    """Retrieves monitored IP addresses, reputation scores, and threat levels."""
    try:
        threat_val = threat_level if isinstance(threat_level, str) else "All"
        limit_val = limit if isinstance(limit, int) else 50
        return database.get_tracked_ips(limit=limit_val, threat_filter=threat_val)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ips/block")
def toggle_ip_blocking(request: IPBlockRequest):
    """Enforces firewall block or unblocks a suspicious/malicious IP."""
    try:
        success = database.toggle_ip_block(ip_address=request.ip_address, block=request.block)
        action = "blocked" if request.block else "unblocked"
        return {"message": f"IP {request.ip_address} has been successfully {action}.", "is_blocked": request.block}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# ALERT RESOLUTION & FALSE POSITIVE TRIAGE
# ==============================================================================

@router.patch("/alerts/{alert_id}/status")
def set_alert_status(alert_id: int, status: str = Query(..., regex="^(NEW|INVESTIGATING|RESOLVED|FALSE_POSITIVE)$")):
    """Updates the triage status of an individual security alert."""
    success = database.update_alert_status(alert_id=alert_id, new_status=status)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": f"Alert {alert_id} status updated to {status}", "alert_id": alert_id, "status": status}


# ==============================================================================
# REPORT GENERATION (PDF & CSV)
# ==============================================================================

@router.get("/reports/pdf")
def generate_pdf_report():
    """Generates an executive-level PDF Incident & Security Audit Report."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f172a"),
            alignment=1
        )
        subtitle_style = ParagraphStyle(
            name="SubTitleStyle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            alignment=1
        )
        section_style = ParagraphStyle(
            name="SectionStyle",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=12,
            spaceAfter=6
        )
        
        # Header
        elements.append(Paragraph("🛡️ NETWORK SECURITY OPERATIONS CENTER (SOC)", title_style))
        elements.append(Paragraph(f"Executive Threat & Audit Report — Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", subtitle_style))
        elements.append(Spacer(1, 15))
        
        # Overview Stats
        stats = database.get_overview_statistics()
        elements.append(Paragraph("1. Executive Traffic & Threat Summary", section_style))
        summary_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Analyzed Flows", f"{stats['total_records_analysed']:,}", "Detected Attacks", f"{stats['total_detected_attacks']:,}"],
            ["Normal Connections", f"{stats['total_normal_connections']:,}", "Attack Ratio", f"{stats['attack_detection_percentage']:.2f}%"],
            ["Top Threat Vector", str(stats['most_frequent_attack']), "Zero-Day Anomalies", f"{stats.get('total_anomalies_detected', 0):,}"]
        ]
        t_summary = Table(summary_data, colWidths=[130, 130, 130, 130])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(t_summary)
        elements.append(Spacer(1, 15))
        
        # Recent Alerts Table
        elements.append(Paragraph("2. Critical Security Incidents & Alerts", section_style))
        alerts = database.get_recent_alerts(limit=8)
        if alerts:
            alert_rows = [["ID", "Timestamp", "Severity", "Attack Type", "Source IP", "Status"]]
            for a in alerts:
                alert_rows.append([
                    str(a["id"]),
                    a["timestamp"][:16],
                    a["alert_level"],
                    a["attack_type"],
                    a["source_ip"],
                    a.get("status", "NEW")
                ])
            t_alerts = Table(alert_rows, colWidths=[35, 110, 65, 120, 110, 80])
            t_alerts.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_alerts)
        else:
            elements.append(Paragraph("No critical alerts logged.", styles["Normal"]))
            
        elements.append(Spacer(1, 15))
        
        # Tracked Host Profiles
        elements.append(Paragraph("3. Top Suspicious & Malicious Hosts", section_style))
        ips = database.get_tracked_ips(limit=6)
        if ips:
            ip_rows = [["IP Address", "Total Flows", "Attacks", "Reputation", "Threat Level", "Blocked"]]
            for ip in ips:
                ip_rows.append([
                    ip["ip_address"],
                    str(ip["total_flows"]),
                    str(ip["attack_count"]),
                    f"{ip['reputation_score']}/100",
                    ip["threat_level"],
                    "YES" if ip["is_blocked"] else "NO"
                ])
            t_ips = Table(ip_rows, colWidths=[120, 70, 70, 80, 100, 80])
            t_ips.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_ips)
            
        doc.build(elements)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=nids_soc_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")
