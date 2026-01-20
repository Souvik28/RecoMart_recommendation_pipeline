import subprocess
import sys
import logging
import os
import time
import signal
from datetime import datetime

# --- Setup Orchestration Logging ---
LOG_DIR = "../recomart_lake/logs/orchestration"
os.makedirs(LOG_DIR, exist_ok=True)
run_id = datetime.now().strftime('%Y%m%d_%H%M')
orchestration_log = os.path.join(LOG_DIR, f"run_{run_id}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(orchestration_log, encoding='utf-8'), 
        logging.StreamHandler(sys.stdout)
    ]
)

def run_batch_stage(script_name, description):
    """Executes a standard batch script and waits for completion."""
    logging.info(f"STARTING BATCH STAGE: {description}")
    try:
        # Use the virtual environment's Python interpreter
        venv_python = os.path.join(os.getcwd(), "..", "recomart_env", "Scripts", "python.exe")
        if os.path.exists(venv_python):
            python_exe = venv_python
            logging.info(f"Using virtual environment Python: {python_exe}")
        else:
            python_exe = sys.executable
            logging.info(f"Using system Python: {python_exe}")
        
        # Change to src directory for execution
        subprocess.run([python_exe, script_name], check=True, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        logging.info(f"COMPLETED: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FAILED: {description}")
        logging.error(f"Error: {e.stderr}")
        return False

def main():
    logging.info("=== RECOMART LAMBDA PIPELINE ORCHESTRATOR ===")

    # --- STEP 1: Start the Speed Layer (Real-Time Stream) ---
    logging.info("LAUNCHING SPEED LAYER: stream_simulator.py | speed_layer.py")
    logging.info("Speed Layer will run for 5 minutes to generate streaming data...")
    
    # We use shell=True to support the pipe '|' operator
    # We use start_new_session so it doesn't block our batch execution
    venv_python = os.path.join(os.getcwd(), "..", "recomart_env", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        python_exe = venv_python
    else:
        python_exe = sys.executable
    
    cmd = f"{python_exe} stream_simulator.py | {python_exe} speed_layer.py"
    stream_process = subprocess.Popen(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)), preexec_fn=os.setsid if os.name != 'nt' else None)
    
    # Let the Speed Layer run for 4 minutes to generate more streaming data
    logging.info("Waiting 2 minutes for Speed Layer to generate streaming data...")
    time.sleep(240)  # 4 minutes = 240 seconds

    # --- STEP 2: Execute the Batch Pipeline Sequence ---
    # These are the tasks that run once and finish
    batch_steps = [
        ("ingest_master.py", "Data Ingestion (Batch & API)"),
        ("04_validate_data.py", "Data Validation & DQ Reporting"),
        ("05_prepare_and_eda.py", "Data Preparation & EDA"),
        ("06_feature_engineering.py", "Feature Engineering"),
        ("07_feature_store.py", "Feature Store Management"),
        ("08_data_lineage.py", "Data Lineage Tracking & Versioning"),
        ("09_train_and_evaluate_model.py", "Model Training & MLflow Tracking")
    ]

    pipeline_success = True
    for script, desc in batch_steps:
        if not run_batch_stage(script, desc):
            logging.critical(f"Pipeline halted due to error in {desc}.")
            pipeline_success = False
            break

    # --- STEP 3: Finalization ---
    if pipeline_success:
        logging.info("=== FULL LAMBDA PIPELINE COMPLETED SUCCESSFULLY ===")
    
    logging.info("Terminating Speed Layer process...")
    # Cleanly stop the background streaming process
    if os.name == 'nt':
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(stream_process.pid)])
    else:
        os.killpg(os.getpgid(stream_process.pid), signal.SIGTERM)
        
    logging.info("Orchestration Finished.")

if __name__ == "__main__":
    main()