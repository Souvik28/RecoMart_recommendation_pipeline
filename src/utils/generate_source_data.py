import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_transactions(n=1200):
    # Use absolute paths to avoid working directory issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    source_dir = os.path.join(project_root, "source_data")
    file_path = os.path.join(source_dir, "transactions.csv")
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Creating directory: {source_dir}")
    print(f"Will save file to: {file_path}")
    
    # Ensure directory exists
    os.makedirs(source_dir, exist_ok=True)
    
    # 1. Generate clean base data
    data = {
        "transaction_id": [f"TXN-{1000 + i}" for i in range(n)],
        "user_id": [f"U{random.randint(1, 200):03}" for _ in range(n)],
        "product_id": [f"P{random.randint(101, 150)}" for _ in range(n)],
        "amount": [round(random.uniform(10.0, 500.0), 2) for _ in range(n)],
        "rating": [random.randint(1, 5) for _ in range(n)], # Expected scale 1-5
        "timestamp": [(datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d %H:%M:%S') for _ in range(n)]
    }
    
    df = pd.DataFrame(data)

    # 2. Inject 2% Missing Values (~24 rows)
    for col in ["amount", "user_id"]:
        null_indices = random.sample(range(n), int(n * 0.01))
        df.loc[null_indices, col] = np.nan

    # 3. Inject 2% Duplicate Entries (~24 rows)
    dup_indices = random.sample(range(n), int(n * 0.02))
    df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

    # 4. Inject Range & Format Checks (Invalid Ratings)
    # Adding some ratings as 0, 6, or 99 to test validation logic
    range_error_indices = random.sample(range(len(df)), 15)
    df.loc[range_error_indices, "rating"] = random.choice([0, 6, 99])

    # 5. Inject Schema Mismatch
    # Occasionally insert a string where a float is expected in 'amount'
    mismatch_indices = random.sample(range(len(df)), 10)
    df["amount"] = df["amount"].astype(object)
    df.loc[mismatch_indices, "amount"] = "ERROR_VAL"

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Generated {len(df)} records in {file_path}")
    print(f"Summary: Includes missing values, duplicates, and range errors (ratings > 5).")

if __name__ == "__main__":
    generate_transactions(10000)