import sqlite3
import pandas as pd
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
DB_PATH = os.path.join(project_root, "recomart_lake", "feature_store", "recomart_features.db")
os.makedirs(os.path.join(project_root, "recomart_lake", "feature_store"), exist_ok=True)

def update_feature_store(user_df, item_df):
    conn = sqlite3.connect(DB_PATH)
    
    # Add metadata columns for versioning
    user_df['updated_at'] = datetime.now().isoformat()
    item_df['updated_at'] = datetime.now().isoformat()
    user_df['version'] = 'v1.0_combined'  # Indicate combined data
    item_df['version'] = 'v1.0_combined'

    # Store in SQLite
    user_df.to_sql("user_features", conn, if_exists="replace", index=False)
    item_df.to_sql("item_features", conn, if_exists="replace", index=False)
    
    # Also save as CSV for optimal model training performance
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    feature_store_dir = os.path.join(project_root, "recomart_lake", "feature_store")
    
    # Create user-item matrix for model training
    combined_path = os.path.join(project_root, "recomart_lake", "processed", "combined_transactions.csv")
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
    else:
        df = pd.read_csv(os.path.join(project_root, "recomart_lake", "processed", "prepared_transactions.csv"))
    
    # Save training-ready data
    df.to_csv(os.path.join(feature_store_dir, "user_item_features.csv"), index=False)
    
    conn.close()
    print(f"Feature Store Updated: {DB_PATH}")
    print(f"Training data saved: {os.path.join(feature_store_dir, 'user_item_features.csv')}")

def fetch_features(entity_id, entity_type="user"):
    conn = sqlite3.connect(DB_PATH)
    table = "user_features" if entity_type == "user" else "item_features"
    id_col = "user_id" if entity_type == "user" else "product_id"
    
    query = f"SELECT * FROM {table} WHERE {id_col} = ?"
    feature_row = pd.read_sql_query(query, conn, params=(entity_id,))
    conn.close()
    return feature_row

if __name__ == "__main__":
    # Setup absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Ensure directories exist
    os.makedirs(os.path.join(project_root, "recomart_lake", "feature_store"), exist_ok=True)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("feature_engineering", os.path.join(script_dir, "06_feature_engineering.py"))
        feature_eng = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_eng)
        
        u, i = feature_eng.engineering()
        update_feature_store(u, i)
        
        # Sample Retrieval Demonstration
        print("\n--- Feature Retrieval Demo ---")
        print(fetch_features("U001", "user"))
    except Exception as e:
        print(f"Error loading feature engineering module: {e}")