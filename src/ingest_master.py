import pandas as pd
import requests
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# Setup logging for audit trails 
LOG_PATH = "../recomart_lake/logs/ingestion.log"
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
    path = Path(f"../recomart_lake/raw/{source}/{data_type}/{date_partition}")
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
    tx_df = pd.read_csv("../source_data/transactions.csv")
    store_raw_data(tx_df, "internal", "transactions")
    
    # 2. Ingest Product Metadata (API)
    product_data = ingest_api_with_retry("http://127.0.0.1:5000/api/products")
    if product_data:
        store_raw_data(product_data, "external_api", "products", format="json")