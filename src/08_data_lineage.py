import hashlib
import json
import os
from datetime import datetime

LINEAGE_LOG = "../recomart_lake/metadata/data_lineage.json"
os.makedirs("../recomart_lake/metadata", exist_ok=True)

def get_file_hash(file_path):
    """Generates an MD5 hash to version the specific state of the data."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def track_lineage(step_name, input_file, output_file, params=None):
    lineage_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step_name,
        "input_hash": get_file_hash(input_file),
        "output_hash": get_file_hash(output_file),
        "parameters": params or {}
    }
    
    # Load existing or create new
    if os.path.exists(LINEAGE_LOG):
        with open(LINEAGE_LOG, 'r') as f:
            data = json.load(f)
    else:
        data = []
        
    data.append(lineage_entry)
    with open(LINEAGE_LOG, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Lineage tracked for: {step_name}")

if __name__ == "__main__":
    # Example tracking for the transformation step
    track_lineage(
        step_name="Feature_Engineering",
        input_file="../recomart_lake/processed/prepared_transactions.csv",
        output_file="../recomart_lake/feature_store/recomart_features.db"
    )