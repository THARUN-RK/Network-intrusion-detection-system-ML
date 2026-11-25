"""
Unit and integration tests for Real-Time Traffic Simulator, /api/stats, /api/timeseries,
dynamic IP reputation decay, and table pruning.
"""

import pytest
import asyncio
from datetime import datetime, timezone
import database
import predict
from backend.app.simulator import NetworkTrafficSimulator
from backend.app.routes.api_live import get_live_soc_stats, get_live_soc_timeseries, toggle_simulator, get_model_evaluation_metrics, download_model_report


def test_simulator_pool_indexing():
    sim = NetworkTrafficSimulator()
    assert len(sim.benign_pool) > 0, "Simulator failed to index benign sample flows"
    assert len(sim.attack_pool) > 0, "Simulator failed to index attack sample flows"
    
    flow, src_ip, dst_ip = sim._get_next_flow()
    assert isinstance(flow, dict)
    assert len(flow) > 30, "Flow features dictionary should contain full feature set"
    assert src_ip is not None
    assert dst_ip is not None


def test_simulator_subtle_noise():
    sim = NetworkTrafficSimulator()
    original_flow = {"Flow Duration": 100000.0, "Total Fwd Packets": 50.0, "Destination Port": 80}
    noised = sim._apply_subtle_noise(original_flow)
    assert noised["Destination Port"] == 80  # Ports should not be altered
    assert noised["Flow Duration"] >= 0.0
    assert noised["Total Fwd Packets"] >= 0.0


def test_simulator_single_step():
    sim = NetworkTrafficSimulator()
    predictor = predict.get_predictor()
    
    flow, src, dst = sim._get_next_flow()
    assert flow is not None
    flow_with_noise = sim._apply_subtle_noise(flow)
    
    initial_count = database.get_live_stats(5)["total_flows"]
    result = predictor.predict_single_flow(
        flow_with_noise,
        source_ip=src,
        destination_ip=dst,
        log_to_db=True
    )
    after_count = database.get_live_stats(5)["total_flows"]
    
    assert result["prediction"] is not None
    assert "status" in result
    assert "confidence" in result
    assert after_count >= initial_count + 1, "Simulation step must log a prediction record to the database"


def test_live_stats_endpoint_schema():
    stats = get_live_soc_stats(window_minutes=5)
    required_keys = [
        "total_flows", "benign_count", "attack_count", "detection_percentage",
        "flows_per_minute", "window_flows", "trend_percentage",
        "attack_classes_last_5m", "attack_distribution", "simulator_enabled"
    ]
    for k in required_keys:
        assert k in stats, f"Missing key '{k}' in /api/stats response"

    assert stats["total_flows"] >= stats["benign_count"] + stats["attack_count"]
    assert 0.0 <= stats["detection_percentage"] <= 100.0


def test_timeseries_endpoint_schema():
    timeseries = get_live_soc_timeseries(window_minutes=30)
    assert isinstance(timeseries, list)
    assert len(timeseries) == 30, "Timeseries endpoint must return exactly 30 one-minute buckets"
    for bucket in timeseries:
        assert "minute" in bucket
        assert "total" in bucket
        assert "benign" in bucket
        assert "attack" in bucket
        assert bucket["total"] == bucket["benign"] + bucket["attack"]


def test_simulator_toggle():
    res1 = toggle_simulator(enabled=False)
    assert res1["simulator_enabled"] is False

    res2 = toggle_simulator(enabled=True)
    assert res2["simulator_enabled"] is True


def test_dynamic_ip_reputation_and_autoblock():
    session = database.SessionLocal()
    test_ip = "198.51.100.99"
    try:
        # Clear any prior test entries
        session.query(database.IPReputationModel).filter_by(ip_address=test_ip).delete()
        session.commit()

        # Step 1: Malicious critical flow penalty
        database.update_ip_profile(session, test_ip, is_attack=True, severity="CRITICAL")
        session.commit()
        profile = session.query(database.IPReputationModel).filter_by(ip_address=test_ip).first()
        assert profile is not None
        assert profile.reputation_score <= 75.0

        # Step 2: Multiple attacks push score below threshold -> auto-block
        database.update_ip_profile(session, test_ip, is_attack=True, severity="CRITICAL")
        database.update_ip_profile(session, test_ip, is_attack=True, severity="HIGH")
        session.commit()
        profile = session.query(database.IPReputationModel).filter_by(ip_address=test_ip).first()
        assert profile.reputation_score <= 35.0
        assert profile.is_blocked is True
        assert profile.threat_level == "MALICIOUS"

        # Step 3: Manual unblock
        database.toggle_ip_block(test_ip, block=False)
        profile = session.query(database.IPReputationModel).filter_by(ip_address=test_ip).first()
        assert profile.is_blocked is False
    finally:
        session.query(database.IPReputationModel).filter_by(ip_address=test_ip).delete()
        session.commit()
        session.close()


def test_table_pruning():
    session = database.SessionLocal()
    try:
        # Call pruning with a generous threshold to ensure no exceptions
        deleted = database.prune_old_predictions(session, max_records=50000)
        assert deleted >= 0
    finally:
        session.close()


def test_model_metrics_endpoint():
    res = get_model_evaluation_metrics()
    assert isinstance(res, dict)
    if res["available"]:
        assert "accuracy" in res
        assert res["accuracy"] >= 0.90
        assert "Support Vector Machine" in res["model_type"]
        assert res["kernel"] == "rbf"
        assert res["feature_count"] == 71
        assert "macro_f1_score" in res
    else:
        assert res["message"] == "Evaluation results not available"

    report_resp = download_model_report()
    assert report_resp is not None


def test_app_simulator_lifecycle_integration():
    import time
    from backend.app.main import health_check
    from backend.app.simulator import traffic_simulator
    from backend.app.routes.api_live import get_live_soc_stats

    # Check initial records
    s0 = get_live_soc_stats(window_minutes=5)
    initial_flows = s0["total_flows"]

    # Start simulator
    traffic_simulator.running = True
    traffic_simulator.catch_up_sync(max_catchup=4)
    time.sleep(6.0)
    # Simulator / catch-up generates flows
    traffic_simulator.catch_up_sync(max_catchup=4)

    # Assert health diagnostics
    h = health_check()
    assert h["simulator_running"] is True
    assert h["last_tick_at"] is not None
    assert h["flows_generated"] > 0

    # Assert total records increased
    s1 = get_live_soc_stats(window_minutes=5)
    new_flows = s1["total_flows"]
    assert new_flows > initial_flows, f"Expected total flows to increase from {initial_flows}, got {new_flows}"




