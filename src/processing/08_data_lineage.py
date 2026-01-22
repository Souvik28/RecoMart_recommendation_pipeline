import hashlib
import json
import os
import glob
from datetime import datetime

LINEAGE_LOG = "../../recomart_lake/metadata/data_lineage.json"
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
LINEAGE_LOG = os.path.join(project_root, "recomart_lake", "metadata", "data_lineage.json")
os.makedirs(os.path.join(project_root, "recomart_lake", "metadata"), exist_ok=True)

def get_file_hash(file_path):
    """Generates an MD5 hash to version the specific state of the data."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def track_product_metadata_lineage():
    """Track lineage for product metadata integration"""
    # Find product metadata files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    metadata_pattern = os.path.join(project_root, "recomart_lake", "raw", "external_api", "products", "*", "*.json")
    metadata_files = glob.glob(metadata_pattern)
    
    if metadata_files:
        latest_metadata = max(metadata_files, key=os.path.getctime)
        
        # Track API metadata ingestion (only if output exists)
        output_file = os.path.join(project_root, "recomart_lake", "processed", "prepared_products.csv")
        if os.path.exists(output_file):
            track_lineage(
                step_name="API_Product_Metadata_Ingestion",
                input_file=latest_metadata,
                output_file=output_file,
                params={
                    "source": "external_api",
                    "data_type": "product_metadata",
                    "content_features_enabled": True
                }
            )
        
        # Track content feature creation (only if both files exist)
        input_file = os.path.join(project_root, "recomart_lake", "processed", "prepared_products.csv")
        output_file = os.path.join(project_root, "recomart_lake", "feature_store", "recomart_features.db")
        if os.path.exists(input_file) and os.path.exists(output_file):
            track_lineage(
                step_name="Content_Feature_Engineering",
                input_file=input_file,
                output_file=output_file,
                params={
                    "feature_types": ["category_encoding", "brand_encoding", "price_normalization", "popularity_scoring"],
                    "cold_start_enabled": True,
                    "hybrid_model_ready": True
                }
            )
    else:
        print("No product metadata found for lineage tracking")

def track_lineage(step_name, input_file, output_file, params=None):
    # Only track if both files exist
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        print(f"Skipping lineage for {step_name}: files not found")
        return
        
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
    # Setup absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Ensure directories exist
    os.makedirs(os.path.join(project_root, "recomart_lake", "metadata"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "recomart_lake", "processed"), exist_ok=True)
    
    print("Starting data lineage tracking...")
    
    # Track product metadata lineage first
    track_product_metadata_lineage()
    
    # Track additional pipeline steps
    raw_pattern = os.path.join(project_root, "recomart_lake", "raw", "internal", "transactions", "*", "*.csv")
    raw_files = glob.glob(raw_pattern)
    
    if raw_files:
        latest_raw = max(raw_files, key=os.path.getctime)
        
        # Track data ingestion step
        combined_path = os.path.join(project_root, "recomart_lake", "processed", "combined_transactions.csv")
        if os.path.exists(combined_path):
            track_lineage(
                step_name="Data_Ingestion_and_Streaming_Merge",
                input_file=latest_raw,
                output_file=combined_path,
                params={
                    "includes_streaming": True,
                    "lambda_architecture": True,
                    "data_source": "batch_and_streaming"
                }
            )
        
        # Track data preparation step
        prepared_data_path = os.path.join(project_root, "recomart_lake", "processed", "prepared_transactions.csv")
        if os.path.exists(prepared_data_path):
            input_file = combined_path if os.path.exists(combined_path) else latest_raw
            track_lineage(
                step_name="Data_Preparation_and_Aggregation",
                input_file=input_file,
                output_file=prepared_data_path,
                params={
                    "aggregation_applied": True,
                    "enhanced_ratings": True,
                    "interaction_weighting": "70% explicit + 30% frequency"
                }
            )
    
    # Track feature engineering step
    prepared_data_path = os.path.join(project_root, "recomart_lake", "processed", "prepared_transactions.csv")
    feature_store_path = os.path.join(project_root, "recomart_lake", "feature_store", "recomart_features.db")
    
    if os.path.exists(prepared_data_path) and os.path.exists(feature_store_path):
        data_source = "combined_streaming_batch" if os.path.exists(combined_path) else "batch_only"
        track_lineage(
            step_name="Enhanced_Feature_Engineering",
            input_file=prepared_data_path,
            output_file=feature_store_path,
            params={
                "data_source": data_source, 
                "includes_streaming": os.path.exists(combined_path),
                "product_metadata_integrated": True,
                "hybrid_features_created": True,
                "cold_start_supported": True
            }
        )
    
    print("Data lineage tracking completed.")