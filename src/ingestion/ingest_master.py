import pandas as pd
import requests
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# Setup logging for audit trails 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_PATH = os.path.join(project_root, "recomart_lake", "logs", "ingestion.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ingest_api_with_retry(url, retries=3):
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"API Attempt {i+1} failed: {e}")
            time.sleep(2)
    return None

def store_raw_data(data, source, data_type, format="csv"):
    # Partitioned by source, type, and timestamp 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_partition = datetime.now().strftime("%Y-%m-%d")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = Path(project_root) / "recomart_lake" / "raw" / source / data_type / date_partition
    path.mkdir(parents=True, exist_ok=True)
    
    file_name = f"{source}_{timestamp}.{format}"
    full_path = path / file_name
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(full_path, index=False)
    else:
        with open(full_path, 'w') as f:
            import json
            json.dump(data, f)
    
    logging.info(f"Successfully stored {source} data at {full_path}")

if __name__ == "__main__":
    # 1. Ingest Transactional CSV (Batch)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    source_data_path = os.path.join(project_root, "source_data", "transactions.csv")
    
    if os.path.exists(source_data_path):
        tx_df = pd.read_csv(source_data_path)
        store_raw_data(tx_df, "internal", "transactions")
        print(f"Ingested {len(tx_df)} transaction records")
    else:
        print("No transaction data found. Please run data generation first.")
    
    # 2. Ingest Product Metadata (API) - with fallback
    product_data = ingest_api_with_retry("http://127.0.0.1:5000/api/products")
    if product_data:
        store_raw_data(product_data, "external_api", "products", format="json")
        print(f"Ingested {len(product_data)} product records from API")
    else:
        print("API not available, creating fallback product metadata")
        # Create fallback product data
        fallback_products = [
            {"product_id": f"P{i:03d}", "category": "Electronics", "price": 100.0 + i*10, 
             "brand": f"Brand{i%5}", "avg_rating": 3.5 + (i%3)*0.5, "popularity_score": 50 + i*2}
            for i in range(101, 151)
        ]
        store_raw_data(fallback_products, "external_api", "products", format="json")
        print(f"Created {len(fallback_products)} fallback product records")