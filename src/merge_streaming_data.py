import pandas as pd
import sqlite3
import os
import glob

def merge_streaming_with_batch():
    """
    Merge real-time streaming data with batch data for model training.
    This implements proper Lambda Architecture data integration.
    """
    
    # 1. Load batch data from raw ingestion
    batch_files = glob.glob("../recomart_lake/raw/internal/transactions/*/*.csv")
    if not batch_files:
        print("Error: No batch data found. Run ingest_master.py first.")
        return None
    
    latest_batch_file = max(batch_files, key=os.path.getctime)
    batch_df = pd.read_csv(latest_batch_file)
    print(f"Loaded {len(batch_df)} batch transactions from raw data")
    
    # 2. Load streaming data
    stream_db_path = "../recomart_lake/speed_layer/real_time_events.db"
    if not os.path.exists(stream_db_path):
        print("Warning: No streaming data found. Using batch data only.")
        return batch_df
    
    conn = sqlite3.connect(stream_db_path)
    stream_df = pd.read_sql_query("""
        SELECT user_id, product_id, action, ts as timestamp
        FROM realtime_clicks 
        WHERE action IN ('view', 'click', 'add_to_cart')
        ORDER BY ts DESC
    """, conn)
    conn.close()
    
    if len(stream_df) == 0:
        print("No relevant streaming events found. Using batch data only.")
        return batch_df
    
    # 3. Convert streaming events to implicit ratings with better context
    # Higher ratings for more engaged actions
    rating_map = {
        'view': 2,        # Low engagement
        'click': 3,       # Medium engagement  
        'add_to_cart': 4  # High engagement (purchase intent)
    }
    
    stream_df['rating'] = stream_df['action'].map(rating_map)
    stream_df['amount'] = 0.0  # No transaction amount for streaming events
    stream_df['transaction_id'] = 'STREAM_' + stream_df.index.astype(str)
    
    # 4. Align columns and merge
    stream_df = stream_df[['transaction_id', 'user_id', 'product_id', 'amount', 'rating', 'timestamp']]
    
    # 5. Combine datasets
    combined_df = pd.concat([batch_df, stream_df], ignore_index=True)
    
    print(f"Added {len(stream_df)} streaming events")
    print(f"Total combined dataset: {len(combined_df)} interactions")
    
    # 6. Save combined dataset
    output_path = "../recomart_lake/processed/combined_transactions.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined dataset to: {output_path}")
    
    return combined_df

if __name__ == "__main__":
    merge_streaming_with_batch()