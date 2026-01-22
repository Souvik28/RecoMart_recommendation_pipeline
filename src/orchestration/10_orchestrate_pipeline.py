import subprocess
import sys
import logging
import os
import time
import signal
from datetime import datetime

# --- Setup Orchestration Logging ---
LOG_DIR = "../../recomart_lake/logs/orchestration"
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
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        venv_python = os.path.join(project_root, "recomart_env", "Scripts", "python.exe")
        if os.path.exists(venv_python):
            python_exe = venv_python
            logging.info(f"Using virtual environment Python: {python_exe}")
        else:
            python_exe = sys.executable
            logging.info(f"Using system Python: {python_exe}")
        
        # Handle special case for recommendation generation
        if script_name == "modeling/generate_recommendations.py":
            # Run as subprocess to avoid import issues with BiasedSVD class
            src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            result = subprocess.run([python_exe, script_name], capture_output=True, text=True, cwd=src_dir)
            
            if result.returncode == 0:
                logging.info(f"COMPLETED: {description}")
                if result.stdout:
                    logging.info(f"Output: {result.stdout.strip()}")
                return True
            else:
                logging.error(f"FAILED: {description}")
                if result.stderr:
                    logging.error(f"Error: {result.stderr.strip()}")
                if result.stdout:
                    logging.error(f"Output: {result.stdout.strip()}")
                return False
        else:
            # Change to src directory for execution
            src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            result = subprocess.run([python_exe, script_name], capture_output=True, text=True, cwd=src_dir)
            
            if result.returncode == 0:
                logging.info(f"COMPLETED: {description}")
                if result.stdout:
                    logging.info(f"Output: {result.stdout.strip()}")
                return True
            else:
                logging.error(f"FAILED: {description}")
                if result.stderr:
                    logging.error(f"Error: {result.stderr.strip()}")
                if result.stdout:
                    logging.error(f"Output: {result.stdout.strip()}")
                return False
            
    except subprocess.CalledProcessError as e:
        logging.error(f"FAILED: {description}")
        logging.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"FAILED: {description}")
        logging.error(f"Error: {str(e)}")
        return False

def main():
    logging.info("=== RECOMART LAMBDA PIPELINE ORCHESTRATOR ===")

    # --- STEP 0: Data Generation (Optional) ---
    logging.info("=== STEP 0: DATA GENERATION ===")
    if not run_batch_stage("utils/generate_source_data.py", "Generate Source Data"):
        logging.warning("Data generation failed, continuing with existing data...")

    # --- STEP 1: Start the Speed Layer (Real-Time Stream) ---
    logging.info("LAUNCHING SPEED LAYER: stream_simulator.py | speed_layer.py")
    logging.info("Speed Layer will run for 5 minutes to generate streaming data...")
    
    # We use shell=True to support the pipe '|' operator
    # We use start_new_session so it doesn't block our batch execution
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(project_root, "recomart_env", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        python_exe = venv_python
    else:
        python_exe = sys.executable
    
    cmd = f"{python_exe} ingestion/stream_simulator.py | {python_exe} ingestion/speed_layer.py"
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    stream_process = subprocess.Popen(cmd, shell=True, cwd=src_dir, preexec_fn=os.setsid if os.name != 'nt' else None)
    
    # Let the Speed Layer run for 1 minute to generate more streaming data
    logging.info("Waiting 1 minute for Speed Layer to generate streaming data...")
    time.sleep(60)  # 1 minutes = 60 seconds

    # --- STEP 2: Execute the Batch Pipeline Sequence ---
    # These are the tasks that run once and finish
    batch_steps = [
        ("ingestion/ingest_master.py", "Data Ingestion (Batch & API)"),
        ("processing/merge_streaming_data.py", "Merge Streaming Data with Batch Data"),
        ("processing/04_validate_data.py", "Data Validation & DQ Reporting"),
        ("processing/05_prepare_and_eda.py", "Data Preparation & EDA"),
        ("processing/06_feature_engineering.py", "Feature Engineering"),
        ("processing/07_feature_store.py", "Feature Store Management"),
        ("processing/08_data_lineage.py", "Data Lineage Tracking & Versioning"),
        ("modeling/09_train_and_evaluate_model.py", "Model Training & MLflow Tracking"),
        ("modeling/generate_recommendations.py", "Generate Product Recommendations")
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