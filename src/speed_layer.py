import sys, json, sqlite3, os

# Set environment variable for UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

DB_PATH = "../recomart_lake/speed_layer/real_time_events.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS realtime_clicks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            product_id TEXT,
            action TEXT,
            ts TEXT
        )
    """)
    conn.commit()
    return conn

if __name__ == "__main__":
    conn = init_db()
    print("Speed Layer Listening...")
    # This reads the piped output from stream_simulator.py [cite: 7, 8]
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                conn.execute(
                    "INSERT INTO realtime_clicks (user_id, product_id, action, ts) VALUES (?,?,?,?)",
                    (event['user_id'], event['product_id'], event['action'], event['timestamp'])
                )
                conn.commit()
                print("Speed Layer: Processed {}".format(event['event_id']))
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Speed Layer stopped.")
        conn.close()