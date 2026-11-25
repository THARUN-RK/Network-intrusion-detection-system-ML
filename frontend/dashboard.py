"""
Frontend Streamlit Dashboard Entry Point
Proxies directly to root dashboard.py for multi-runtime compatibility.
"""

from pathlib import Path
import sys

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the main dashboard
import dashboard
