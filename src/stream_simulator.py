import time, json, random, uuid
from datetime import datetime, timezone

PRODUCTS = [f"P{i}" for i in range(101, 151)]
USERS = [f"U{i:03d}" for i in range(1, 201)]

def gen_clickstream_event():
    return {
        "event_id": str(uuid.uuid4())[:8],
        "user_id": random.choice(USERS),
        "product_id": random.choice(PRODUCTS),
        "action": random.choice(["view", "click", "add_to_cart"]),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import sys
    # Only print to stderr so it doesn't interfere with JSON output
    print("Clickstream Stream Started (Press Ctrl+C to stop)...", file=sys.stderr)
    try:
        while True:
            event = gen_clickstream_event()
            # In a real setup, this would go to Kafka. Here we print for the Speed Layer to consume.
            print(json.dumps(event), flush=True)
            time.sleep(1.5) 
    except KeyboardInterrupt:
        print("Stream Stopped.", file=sys.stderr)