"""
Unified Launcher for Network Intrusion Detection System (NIDS)
Provides a menu to run Backend, Dashboard, Training, or Full Test Suite.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable


def run_command(cmd, desc):
    print(f"\n[NIDS Launcher] {desc}...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error running command: {e}")


def main():
    print("=" * 65)
    print("  🛡️  NETWORK INTRUSION DETECTION SYSTEM (NIDS) - ML & SVM")
    print("=" * 65)
    print("1. Launch Streamlit SOC Dashboard")
    print("2. Launch FastAPI REST Backend")
    print("3. Train SVM Model on Dataset")
    print("4. Evaluate Trained Model & Generate Charts")
    print("5. Run Pytest Test Suite")
    print("6. Run Scapy Live Packet Capture Monitor")
    print("7. Exit")
    print("=" * 65)
    
    choice = input("Select an option (1-7): ").strip()
    
    if choice == "1":
        run_command([PYTHON_EXE, "-m", "streamlit", "run", "frontend/dashboard.py"], "Starting Streamlit Dashboard")
    elif choice == "2":
        run_command([PYTHON_EXE, "-m", "uvicorn", "backend.app.main:app", "--reload", "--port", "8000"], "Starting FastAPI Backend")
    elif choice == "3":
        run_command([PYTHON_EXE, "train_model.py"], "Running SVM Model Training")
    elif choice == "4":
        run_command([PYTHON_EXE, "evaluation.py"], "Evaluating Model Performance")
    elif choice == "5":
        run_command([PYTHON_EXE, "-m", "pytest", "-v"], "Executing Pytest Test Suite")
    elif choice == "6":
        run_command([PYTHON_EXE, "monitoring/live_detection.py"], "Starting Scapy Live Traffic Monitor")
    elif choice == "7":
        print("Exiting NIDS Launcher. Goodbye!")
    else:
        print("Invalid selection.")


if __name__ == "__main__":
    main()
