"""
Unit tests for data preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import config
import preprocessing


def test_clean_column_name():
    assert preprocessing.clean_column_name("  Flow   Duration  ") == "Flow Duration"
    assert preprocessing.clean_column_name("Destination Port\n") == "Destination Port"


def test_canonicalize_label():
    assert preprocessing.canonicalize_label("BENIGN") == "BENIGN"
    assert preprocessing.canonicalize_label("DoS Hulk") == "DoS"
    assert preprocessing.canonicalize_label("DoS GoldenEye") == "DoS"
    assert preprocessing.canonicalize_label("DDoS") == "DDoS"
    assert preprocessing.canonicalize_label("PortScan") == "PortScan"
    assert preprocessing.canonicalize_label("FTP-Patator") == "FTP-Patator"
    assert preprocessing.canonicalize_label("SSH-Patator") == "SSH-Patator"
    assert preprocessing.canonicalize_label("Web Attack – Brute Force") == "Web Attack"
    assert preprocessing.canonicalize_label("Heartbleed") == "Heartbleed"
    assert preprocessing.canonicalize_label(np.nan) == "BENIGN"


def test_network_flow_sanitizer():
    sanitizer = preprocessing.NetworkFlowSanitizer()
    data = np.array([[1.0, np.inf], [-np.inf, 4.0]])
    transformed = sanitizer.transform(data)
    assert np.isnan(transformed[0, 1])
    assert np.isnan(transformed[1, 0])
    assert transformed[0, 0] == 1.0
    assert transformed[1, 1] == 4.0


def test_feature_pipeline_fit_transform():
    pipe = preprocessing.build_feature_pipeline()
    data = np.array([
        [10.0, np.inf, 5.0],
        [20.0, 30.0, np.nan],
        [15.0, 25.0, 5.0]
    ])
    result = pipe.fit_transform(data)
    assert result.shape == (3, 3)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
