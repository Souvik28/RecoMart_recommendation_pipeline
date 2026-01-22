import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import json

# Create directory for plots
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
os.makedirs(os.path.join(project_root, "recomart_lake", "eda_plots"), exist_ok=True)

def prepare_product_metadata():
    """Prepare and clean product metadata"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    metadata_pattern = os.path.join(project_root, "recomart_lake", "raw", "external_api", "products", "*", "*.json")
    metadata_files = glob.glob(metadata_pattern)
    if not metadata_files:
        print("No product metadata found to prepare")
        return None
    
    latest_file = max(metadata_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        products = json.load(f)
    
    product_df = pd.DataFrame(products)
    
    # Clean product data
    product_df['price'] = pd.to_numeric(product_df['price'], errors='coerce')
    product_df = product_df.dropna()
    
    # Save prepared product metadata
    output_path = os.path.join(project_root, "recomart_lake", "processed", "prepared_products.csv")
    product_df.to_csv(output_path, index=False)
    print(f"Prepared {len(product_df)} products metadata")
    
    return product_df

def prepare_data():
    # 1. Load latest data - prioritize combined data if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    combined_path = os.path.join(project_root, "recomart_lake", "processed", "combined_transactions.csv")
    
    if os.path.exists(combined_path):
        print("Using combined streaming + batch data for preparation")
        df = pd.read_csv(combined_path)
    else:
        print("Using batch data only for preparation")
        raw_pattern = os.path.join(project_root, "recomart_lake", "raw", "internal", "transactions", "*", "*.csv")
        csv_files = glob.glob(raw_pattern)
        if not csv_files:
            print("No CSV files found. Please run ingest_master.py first.")
            return None
        latest_file = max(csv_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
    
    print(f"Processing {len(df)} total interactions")

    # 2. Aggregate multiple clicks/interactions per user-product pair
    print("Aggregating multiple user-product interactions...")
    
    # Count interactions and calculate weighted rating
    interaction_agg = df.groupby(['user_id', 'product_id']).agg({
        'rating': 'mean',  # Average rating
        'transaction_id': 'count',  # Interaction count
        'amount': 'sum',  # Total amount spent
        'timestamp': 'max'  # Latest interaction
    }).reset_index()
    
    # Rename columns
    interaction_agg.columns = ['user_id', 'product_id', 'avg_rating', 'interaction_count', 'total_amount', 'timestamp']
    
    # Create enhanced rating based on interaction frequency
    # More interactions = higher implicit rating
    interaction_agg['enhanced_rating'] = (
        interaction_agg['avg_rating'] * 0.7 +  # 70% explicit rating
        np.log1p(interaction_agg['interaction_count']) * 0.3  # 30% interaction frequency
    ).clip(1, 5)
    
    # Use enhanced rating as final rating
    df_aggregated = interaction_agg[['user_id', 'product_id', 'enhanced_rating', 'interaction_count', 'timestamp']].copy()
    df_aggregated.rename(columns={'enhanced_rating': 'rating'}, inplace=True)
    
    print(f"Aggregated to {len(df_aggregated)} unique user-product pairs")
    print(f"Average interactions per pair: {interaction_agg['interaction_count'].mean():.2f}")

    # 3. Cleaning
    # Handle Schema Mismatch: Convert amount to numeric, forcing errors (like 'ERROR_VAL') to NaN
    df_aggregated['rating'] = pd.to_numeric(df_aggregated['rating'], errors='coerce')
    
    # Handle Missing Values: Fill missing ratings with global mean
    df_aggregated['rating'] = df_aggregated['rating'].fillna(df_aggregated['rating'].mean())
    df_aggregated = df_aggregated.dropna(subset=['user_id', 'product_id'])

    # Handle Range Checks: Clip ratings to [1, 5]
    df_aggregated['rating'] = df_aggregated['rating'].clip(1, 5)

    # Handle Duplicates (shouldn't exist after aggregation, but safety check)
    df_aggregated = df_aggregated.drop_duplicates(subset=['user_id', 'product_id'])

    # 4. Exploratory Data Analysis (EDA) on aggregated data
    # Plot 1: Interaction Distribution (Enhanced Ratings)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_aggregated, x='rating', hue='rating', palette='viridis', legend=False)
    plt.title('Distribution of Enhanced User Ratings (Aggregated)')
    plt.savefig(os.path.join(project_root, "recomart_lake", "eda_plots", "enhanced_rating_dist.png"))

    # Plot 2: Interaction Count Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df_aggregated['interaction_count'], bins=20, alpha=0.7, color='skyblue')
    plt.title('Distribution of User-Product Interaction Counts')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(project_root, "recomart_lake", "eda_plots", "interaction_count_dist.png"))

    # Plot 3: Item Popularity (Top 10 Products)
    plt.figure(figsize=(10, 5))
    df_aggregated['product_id'].value_counts().head(10).plot(kind='bar', color='orange')
    plt.title('Top 10 Most Popular Products (Aggregated)')
    plt.savefig(os.path.join(project_root, "recomart_lake", "eda_plots", "item_popularity_aggregated.png"))

    # 5. Save Prepared Data
    output_path = os.path.join(project_root, "recomart_lake", "processed", "prepared_transactions.csv")
    os.makedirs(os.path.join(project_root, "recomart_lake", "processed"), exist_ok=True)
    df_aggregated.to_csv(output_path, index=False)
    print(f"Aggregated data prepared & EDA plots saved to {os.path.join(project_root, 'recomart_lake', 'eda_plots')}")
    return df_aggregated

if __name__ == "__main__":
    # Setup absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Ensure directories exist
    os.makedirs(os.path.join(project_root, "recomart_lake", "processed"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "recomart_lake", "eda_plots"), exist_ok=True)
    
    # Prepare product metadata first
    prepare_product_metadata()
    
    # Then prepare transaction data
    result = prepare_data()
    if result is None:
        print("Data preparation failed - no transaction data available")
    else:
        print("Data preparation completed successfully")