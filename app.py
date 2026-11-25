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
    print("1. Launch FastAPI Web Application & SOC Console (http://127.0.0.1:8000)")
    print("2. Train SVM Model on Dataset")
    print("3. Evaluate Trained Model & Generate Charts")
    print("4. Run Pytest Test Suite")
    print("5. Run Scapy Live Packet Capture Monitor")
    print("6. Exit")
    print("=" * 65)
    
    choice = input("Select an option (1-6): ").strip()
    
    if choice == "1":
        run_command([PYTHON_EXE, "-m", "uvicorn", "backend.app.main:app", "--reload", "--port", "8000"], "Starting FastAPI Web Application")
    elif choice == "2":
        run_command([PYTHON_EXE, "train_model.py"], "Running SVM Model Training")
    elif choice == "3":
        run_command([PYTHON_EXE, "evaluation.py"], "Evaluating Model Performance")
    elif choice == "4":
        run_command([PYTHON_EXE, "-m", "pytest", "-v"], "Executing Pytest Test Suite")
    elif choice == "5":
        run_command([PYTHON_EXE, "monitoring/live_detection.py"], "Starting Scapy Live Traffic Monitor")
    elif choice == "6":
        print("Exiting NIDS Launcher. Goodbye!")
    else:
        print("Invalid selection.")


if __name__ == "__main__":
    main()
