import sqlite3
import pandas as pd
import os
from datetime import datetime

def check_speed_layer():
    """Monitor real-time streaming data and performance"""
    
    # Use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    db_path = os.path.join(project_root, "recomart_lake", "speed_layer", "real_time_events.db")
    
    if not os.path.exists(db_path):
        print("❌ Speed Layer database not found.")
        print(f"Expected location: {db_path}")
        print("\n🚀 To generate streaming data, run:")
        print("   cd src && python 10_orchestrate_pipeline.py")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        
        print("🔄 SPEED LAYER MONITORING")
        print("=" * 40)
        
        # Get total events
        total_query = "SELECT COUNT(*) as total FROM realtime_clicks"
        total_count = pd.read_sql_query(total_query, conn)['total'][0]
        
        # Get events by action type
        action_query = "SELECT action, COUNT(*) as count FROM realtime_clicks GROUP BY action"
        action_stats = pd.read_sql_query(action_query, conn)
        
        # Get recent events
        recent_query = "SELECT * FROM realtime_clicks ORDER BY id DESC LIMIT 5"
        recent_events = pd.read_sql_query(recent_query, conn)
        
        print(f"📊 Total Events: {total_count}")
        print(f"📅 Database: {db_path}")
        print(f"⏰ Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📈 Event Breakdown:")
        for _, row in action_stats.iterrows():
            print(f"   {row['action']}: {row['count']} events")
        
        print("\n🕐 Recent Events:")
        if not recent_events.empty:
            for _, event in recent_events.iterrows():
                print(f"   {event['user_id']} → {event['product_id']} ({event['action']})")
        else:
            print("   No events found")
        
        conn.close()
        print("\n✅ Speed Layer is active and collecting data")
        
    except Exception as e:
        print(f"❌ Error reading speed layer: {e}")

if __name__ == "__main__":
    check_speed_layer()