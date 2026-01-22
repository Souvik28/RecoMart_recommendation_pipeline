import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = "../recomart_lake/feature_store/recomart_features.db"
os.makedirs("../recomart_lake/feature_store", exist_ok=True)

def update_feature_store(user_df, item_df, interaction_df=None):
    conn = sqlite3.connect(DB_PATH)
    
    # Add metadata columns for versioning
    user_df['updated_at'] = datetime.now().isoformat()
    item_df['updated_at'] = datetime.now().isoformat()
    user_df['version'] = 'v1.0_combined'  # Indicate combined data
    item_df['version'] = 'v1.0_combined'

    # Store in SQLite
    user_df.to_sql("user_features", conn, if_exists="replace", index=False)
    item_df.to_sql("item_features", conn, if_exists="replace", index=False)
    
    # Store interaction features if provided
    if interaction_df is not None:
        interaction_df['updated_at'] = datetime.now().isoformat()
        interaction_df['version'] = 'v1.0_combined'
        interaction_df.to_sql("interaction_features", conn, if_exists="replace", index=False)
        print(f"✅ Feature Store Updated with interaction features: {DB_PATH}")
    
    conn.close()
    print(f"✅ Feature Store Updated: {DB_PATH}")

def fetch_features(entity_id, entity_type="user"):
    conn = sqlite3.connect(DB_PATH)
    table = "user_features" if entity_type == "user" else "item_features"
    id_col = "user_id" if entity_type == "user" else "product_id"
    
    query = f"SELECT * FROM {table} WHERE {id_col} = ?"
    feature_row = pd.read_sql_query(query, conn, params=(entity_id,))
    conn.close()
    return feature_row

if __name__ == "__main__":
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("feature_engineering", "06_feature_engineering.py")
        feature_eng = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_eng)
        
        u, i, int_feat = feature_eng.engineering()
        update_feature_store(u, i, int_feat)
        
        # Sample Retrieval Demonstration
        print("\n--- Feature Retrieval Demo ---")
        print(fetch_features("U001", "user"))
    except Exception as e:
        print(f"Error loading feature engineering module: {e}")