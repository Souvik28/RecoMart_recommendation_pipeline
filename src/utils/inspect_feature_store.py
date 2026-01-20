import sqlite3
import pandas as pd
import os

# Path to the feature store
DB_PATH = "../../recomart_lake/feature_store/recomart_features.db"

def inspect_features():
    # Debug: Show current working directory and absolute path
    current_dir = os.getcwd()
    abs_path = os.path.abspath(DB_PATH)
    print(f"Current directory: {current_dir}")
    print(f"Looking for database at: {abs_path}")
    
    db_path = DB_PATH
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        # Try alternative path
        alt_path = os.path.join(current_dir, "recomart_lake", "feature_store", "recomart_features.db")
        if os.path.exists(alt_path):
            print(f"Found database at alternative path: {alt_path}")
            db_path = alt_path
        else:
            print("\nTo create the feature store, run the pipeline first:")
            print("1. cd src")
            print("2. python 10_orchestrate_pipeline.py")
            print("\nOr run individual steps:")
            print("1. python utils/generate_source_data.py")
            print("2. python ingest_master.py")
            print("3. python 05_prepare_and_eda.py")
            print("4. python 06_feature_engineering.py")
            print("5. python 07_feature_store.py")
            return

    try:
        conn = sqlite3.connect(db_path)
        
        print("\n" + "="*50)
        print("RECOMART FEATURE STORE INSPECTOR")
        print("="*50)

        # 1. Inspect User Features
        print("\n[Table: user_features]")
        user_df = pd.read_sql_query("SELECT * FROM user_features", conn)
        if not user_df.empty:
            print(f"Total Users Tracked: {len(user_df)}")
            print(user_df.head(10)) # Show first 10 users
        else:
            print("Table is empty.")

        print("\n" + "-"*30)

        # 2. Inspect Item Features
        print("\n[Table: item_features]")
        item_df = pd.read_sql_query("SELECT * FROM item_features", conn)
        if not item_df.empty:
            print(f"Total Products Tracked: {len(item_df)}")
            print(item_df.head(10)) # Show first 10 products
        else:
            print("Table is empty.")

        # 3. Check for Versioning Consistency
        unique_versions = user_df['version'].unique()
        last_update = user_df['updated_at'].max()
        print("\n" + "="*50)
        print(f"Metadata Check:")
        print(f" - Data Versions found: {unique_versions}")
        print(f" - Last Updated: {last_update}")
        print("="*50 + "\n")

        conn.close()
        
    except Exception as e:
        print(f"Error reading feature store: {e}")

if __name__ == "__main__":
    inspect_features()