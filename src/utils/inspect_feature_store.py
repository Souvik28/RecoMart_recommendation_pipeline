import sqlite3
import pandas as pd
import os
from datetime import datetime

def inspect_features():
    """Comprehensive feature store inspection and monitoring"""
    
    # Use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    db_path = os.path.join(project_root, "recomart_lake", "feature_store", "recomart_features.db")
    
    if not os.path.exists(db_path):
        print("❌ Feature Store database not found.")
        print(f"Expected location: {db_path}")
        print("\n🚀 To create the feature store, run:")
        print("   cd src && python 10_orchestrate_pipeline.py")
        print("\n🔧 Or run individual steps:")
        print("   python 06_feature_engineering.py")
        print("   python 07_feature_store.py")
        return

    try:
        conn = sqlite3.connect(db_path)
        
        print("🏦 FEATURE STORE INSPECTOR")
        print("=" * 50)
        print(f"📅 Database: {db_path}")
        print(f"⏰ Inspected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check tables exist
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = pd.read_sql_query(tables_query, conn)['name'].tolist()
        print(f"📋 Tables Found: {', '.join(tables)}")
        
        # Inspect User Features
        if 'user_features' in tables:
            print("\n👥 USER FEATURES:")
            print("-" * 30)
            user_df = pd.read_sql_query("SELECT * FROM user_features LIMIT 5", conn)
            user_count = pd.read_sql_query("SELECT COUNT(*) as count FROM user_features", conn)['count'][0]
            
            print(f"Total Users: {user_count}")
            print(f"Columns: {list(user_df.columns)}")
            
            if not user_df.empty:
                print("\nSample Data:")
                for _, row in user_df.iterrows():
                    print(f"  {row['user_id']}: {row.get('user_total_interactions', 'N/A')} interactions")
        
        # Inspect Item Features  
        if 'item_features' in tables:
            print("\n📎 ITEM FEATURES:")
            print("-" * 30)
            item_df = pd.read_sql_query("SELECT * FROM item_features LIMIT 5", conn)
            item_count = pd.read_sql_query("SELECT COUNT(*) as count FROM item_features", conn)['count'][0]
            
            print(f"Total Items: {item_count}")
            print(f"Columns: {list(item_df.columns)}")
            
            if not item_df.empty:
                print("\nSample Data:")
                for _, row in item_df.iterrows():
                    interactions = row.get('item_total_interactions', row.get('item_popularity_score', 'N/A'))
                    print(f"  {row['product_id']}: {interactions} interactions")
        
        # Check for content features
        if 'item_features' in tables:
            content_cols = [col for col in item_df.columns if col.startswith(('cat_', 'brand_', 'avg_rating', 'price_norm'))]
            if content_cols:
                print(f"\n🏷️ Content Features: {len(content_cols)} attributes")
                print(f"   Categories: {len([c for c in content_cols if c.startswith('cat_')])}")
                print(f"   Brands: {len([c for c in content_cols if c.startswith('brand_')])}")
            else:
                print("\n⚠️ No content features found (API metadata missing)")
        
        # Version info
        if 'user_features' in tables:
            version_info = pd.read_sql_query("SELECT version, updated_at FROM user_features LIMIT 1", conn)
            if not version_info.empty:
                print(f"\n📊 Version: {version_info['version'][0]}")
                print(f"🔄 Last Updated: {version_info['updated_at'][0]}")
        
        conn.close()
        print("\n✅ Feature Store is ready for model training")
        
    except Exception as e:
        print(f"❌ Error inspecting feature store: {e}")

if __name__ == "__main__":
    inspect_features()