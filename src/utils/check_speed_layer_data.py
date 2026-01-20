import sqlite3
import pandas as pd
import os

# Path to the speed layer database
DB_PATH = "../../recomart_lake/speed_layer/real_time_events.db"

def check_speed_layer():
    # Debug: Show current working directory and absolute path
    current_dir = os.getcwd()
    abs_path = os.path.abspath(DB_PATH)
    print(f"Current directory: {current_dir}")
    print(f"Looking for database at: {abs_path}")
    
    db_path = DB_PATH
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        # Try alternative path
        alt_path = os.path.join(current_dir, "recomart_lake", "speed_layer", "real_time_events.db")
        if os.path.exists(alt_path):
            print(f"Found database at alternative path: {alt_path}")
            db_path = alt_path
        else:
            print("\nSpeed Layer database not found. Run the pipeline first:")
            print("cd src && python 10_orchestrate_pipeline.py")
            return
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Query the most recent 10 events
        query = "SELECT * FROM realtime_clicks ORDER BY id DESC LIMIT 10"
        df = pd.read_sql_query(query, conn)
        
        print("--- Latest 10 Real-Time Events ---")
        print(df)
        
        # Get total record count
        count = pd.read_sql_query("SELECT COUNT(*) as total FROM realtime_clicks", conn)
        print(f"\nTotal events in Speed Layer: {count['total'][0]}")
        
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    check_speed_layer()