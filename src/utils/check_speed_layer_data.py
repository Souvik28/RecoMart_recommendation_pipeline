import sqlite3
import pandas as pd

# Path to the speed layer database
DB_PATH = "../../recomart_lake/speed_layer/real_time_events.db"

def check_speed_layer():
    try:
        conn = sqlite3.connect(DB_PATH)
        
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