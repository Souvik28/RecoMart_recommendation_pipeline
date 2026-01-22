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
            print(f"User Feature Columns: {list(user_df.columns)}")
            print("\nSample User Features:")
            print(user_df[['user_id', 'user_total_interactions', 'user_avg_rating', 'user_engagement_score']].head(5))
        else:
            print("Table is empty.")

        print("\n" + "-"*30)

        # 2. Inspect Item Features
        print("\n[Table: item_features]")
        item_df = pd.read_sql_query("SELECT * FROM item_features", conn)
        if not item_df.empty:
            print(f"Total Products Tracked: {len(item_df)}")
            print(f"Item Feature Columns: {list(item_df.columns)}")
            print("\nSample Item Features:")
            print(item_df[['product_id', 'item_popularity_score', 'item_avg_rating', 'item_quality_score']].head(5))
        else:
            print("Table is empty.")
            
        print("\n" + "-"*30)
        
        # 3. Inspect Interaction Features (NEW)
        print("\n[Table: interaction_features]")
        try:
            interaction_df = pd.read_sql_query("SELECT * FROM interaction_features", conn)
            if not interaction_df.empty:
                print(f"Total User-Item Interactions Tracked: {len(interaction_df)}")
                print(f"Interaction Feature Columns: {list(interaction_df.columns)}")
                print("\nSample Interaction Features:")
                print(interaction_df[['user_id', 'product_id', 'interaction_count', 'interaction_avg_rating']].head(5))
            else:
                print("Table is empty.")
        except Exception as e:
            print(f"Interaction features table not found (older version): {e}")

        # 4. Feature Statistics Summary
        print("\n" + "="*50)
        print("FEATURE STATISTICS SUMMARY")
        print("="*50)
        
        if not user_df.empty:
            print("\nUser Feature Statistics:")
            print(f"- Avg Engagement Score: {user_df['user_engagement_score'].mean():.2f}")
            print(f"- Most Active User: {user_df.loc[user_df['user_total_interactions'].idxmax(), 'user_id']} ({user_df['user_total_interactions'].max()} interactions)")
            print(f"- Avg User Rating: {user_df['user_avg_rating'].mean():.2f}")
            
        if not item_df.empty:
            print("\nItem Feature Statistics:")
            print(f"- Avg Quality Score: {item_df['item_quality_score'].mean():.2f}")
            print(f"- Most Popular Item: {item_df.loc[item_df['item_popularity_score'].idxmax(), 'product_id']} ({item_df['item_popularity_score'].max()} interactions)")
            print(f"- Avg Item Rating: {item_df['item_avg_rating'].mean():.2f}")
            
        try:
            if not interaction_df.empty:
                print("\nInteraction Feature Statistics:")
                print(f"- Total Unique User-Item Pairs: {len(interaction_df)}")
                print(f"- Avg Interactions per Pair: {interaction_df['interaction_count'].mean():.2f}")
                print(f"- Most Frequent Pair: {interaction_df.loc[interaction_df['interaction_count'].idxmax(), 'user_id']} -> {interaction_df.loc[interaction_df['interaction_count'].idxmax(), 'product_id']} ({interaction_df['interaction_count'].max()} times)")
        except:
            pass
            
        # 5. Check for Versioning Consistency
        unique_versions = user_df['version'].unique() if not user_df.empty else []
        last_update = user_df['updated_at'].max() if not user_df.empty else "N/A"
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