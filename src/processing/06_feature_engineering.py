import pandas as pd
import sqlite3
import json
import glob
import os

def load_product_metadata():
    """Load prepared product metadata"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    prepared_path = os.path.join(project_root, "recomart_lake", "processed", "prepared_products.csv")
    if os.path.exists(prepared_path):
        product_df = pd.read_csv(prepared_path)
        print(f"Loaded {len(product_df)} prepared products")
        return product_df
    
    # Fallback to raw metadata
    metadata_pattern = os.path.join(project_root, "recomart_lake", "raw", "external_api", "products", "*", "*.json")
    metadata_files = glob.glob(metadata_pattern)
    if not metadata_files:
        print("No product metadata found, creating basic item features only")
        return None
    
    latest_file = max(metadata_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        products = json.load(f)
    
    product_df = pd.DataFrame(products)
    print(f"Loaded {len(product_df)} products from raw metadata")
    return product_df

def create_content_features(product_df):
    """Create content-based features for cold start"""
    if product_df is None:
        return None
    
    # One-hot encode categories
    category_dummies = pd.get_dummies(product_df['category'], prefix='cat')
    brand_dummies = pd.get_dummies(product_df['brand'], prefix='brand')
    
    # Normalize numerical features
    product_df['price_norm'] = (product_df['price'] - product_df['price'].mean()) / product_df['price'].std()
    product_df['popularity_norm'] = (product_df['popularity_score'] - product_df['popularity_score'].mean()) / product_df['popularity_score'].std()
    
    # Combine features
    content_features = pd.concat([
        product_df[['product_id', 'avg_rating', 'price_norm', 'popularity_norm']],
        category_dummies,
        brand_dummies
    ], axis=1)
    
    return content_features

def engineering():
    # Use prepared data (which is now aggregated streaming + batch)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Try multiple data sources in order of preference
    prepared_path = os.path.join(project_root, "recomart_lake", "processed", "prepared_transactions.csv")
    combined_path = os.path.join(project_root, "recomart_lake", "processed", "combined_transactions.csv")
    
    print(f"Checking for prepared data at: {prepared_path}")
    print(f"Checking for combined data at: {combined_path}")
    
    if os.path.exists(prepared_path):
        df = pd.read_csv(prepared_path)
        print(f"Feature engineering on {len(df)} prepared transactions")
    elif os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        print(f"Feature engineering on {len(df)} combined transactions")
        # Add interaction_count column if missing
        if 'interaction_count' not in df.columns:
            df['interaction_count'] = 1
    else:
        # Fallback to raw data
        raw_pattern = os.path.join(project_root, "recomart_lake", "raw", "internal", "transactions", "*", "*.csv")
        raw_files = glob.glob(raw_pattern)
        print(f"Checking for raw data with pattern: {raw_pattern}")
        print(f"Found raw files: {raw_files}")
        if raw_files:
            latest_file = max(raw_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            df['interaction_count'] = 1  # Add missing column
            print(f"Feature engineering on {len(df)} raw transactions (fallback)")
        else:
            print("ERROR: No transaction data found in any location")
            print(f"Checked paths:")
            print(f"  - Prepared: {prepared_path}")
            print(f"  - Combined: {combined_path}")
            print(f"  - Raw pattern: {raw_pattern}")
            raise FileNotFoundError("No transaction data found for feature engineering")

    # Feature 1: User Activity Frequency (enhanced with interaction counts)
    if 'interaction_count' in df.columns:
        user_features = df.groupby('user_id').agg(
            user_total_interactions=('interaction_count', 'sum'),  # Total clicks across all products
            user_avg_rating=('rating', 'mean'),
            user_unique_products=('product_id', 'nunique'),  # Product diversity
            user_max_interactions=('interaction_count', 'max')  # Max clicks on single product
        ).reset_index()
    else:
        # Fallback for data without interaction_count
        user_features = df.groupby('user_id').agg(
            user_avg_rating=('rating', 'mean'),
            user_unique_products=('product_id', 'nunique'),  # Product diversity
            user_total_transactions=('rating', 'count')  # Total transactions
        ).reset_index()

    # Feature 2: Enhanced Item Features (with interaction data)
    if 'interaction_count' in df.columns:
        item_features = df.groupby('product_id').agg(
            item_avg_rating=('rating', 'mean'),
            item_total_interactions=('interaction_count', 'sum'),  # Total clicks received
            item_unique_users=('user_id', 'nunique'),  # User reach
            item_avg_interactions_per_user=('interaction_count', 'mean')  # Engagement depth
        ).reset_index()
    else:
        # Fallback for data without interaction_count
        item_features = df.groupby('product_id').agg(
            item_avg_rating=('rating', 'mean'),
            item_unique_users=('user_id', 'nunique'),  # User reach
            item_total_transactions=('rating', 'count')  # Total transactions
        ).reset_index()
    
    # Feature 3: Product Content Features (from API metadata)
    product_metadata = load_product_metadata()
    content_features = create_content_features(product_metadata)
    
    if content_features is not None:
        # Merge with enhanced item features
        item_features = item_features.merge(content_features, on='product_id', how='left')
        print(f"Enhanced item features with {content_features.shape[1]-1} content attributes + interaction metrics")
    
    return user_features, item_features

if __name__ == "__main__":
    # Setup absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Ensure directories exist
    os.makedirs(os.path.join(project_root, "recomart_lake", "processed"), exist_ok=True)
    
    u_feat, i_feat = engineering()
    print("Features Engineered: User and Item profiles created.")