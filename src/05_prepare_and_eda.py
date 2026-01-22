import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Create directory for plots
os.makedirs("../recomart_lake/eda_plots", exist_ok=True)

def prepare_data():
    # 1. Load latest data - prioritize combined data if available
    combined_path = "../recomart_lake/processed/combined_transactions.csv"
    
    if os.path.exists(combined_path):
        print("Using combined streaming + batch data for preparation")
        df = pd.read_csv(combined_path)
    else:
        print("Using batch data only for preparation")
        csv_files = glob.glob("../recomart_lake/raw/internal/transactions/*/*.csv")
        if not csv_files:
            print("No CSV files found. Please run ingest_master.py first.")
            return
        latest_file = max(csv_files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
    
    print(f"Processing {len(df)} total interactions")

    # 2. Cleaning
    # Handle Schema Mismatch: Convert amount to numeric, forcing errors (like 'ERROR_VAL') to NaN
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    
    # Handle Missing Values: Fill missing ratings with mean, drop rows without user_id
    df['rating'] = df['rating'].fillna(df['rating'].mean())
    df = df.dropna(subset=['user_id', 'product_id'])

    # Handle Range Checks: Clip ratings to [1, 5]
    df['rating'] = df['rating'].clip(1, 5)

    # Handle Duplicates
    df = df.drop_duplicates()

    # 3. Exploratory Data Analysis (EDA)
    # Plot 1: Interaction Distribution (Ratings)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='rating', hue='rating', palette='viridis', legend=False)
    plt.title('Distribution of User Ratings')
    plt.savefig("../recomart_lake/eda_plots/rating_dist.png")

    # Plot 2: Item Popularity (Top 10 Products)
    plt.figure(figsize=(10, 5))
    df['product_id'].value_counts().head(10).plot(kind='bar', color='orange')
    plt.title('Top 10 Most Popular Products')
    plt.savefig("../recomart_lake/eda_plots/item_popularity.png")

    # 4. Save Prepared Data
    output_path = "../recomart_lake/processed/prepared_transactions.csv"
    os.makedirs("../recomart_lake/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data Prepared & EDA plots saved to ../recomart_lake/eda_plots/")
    return df

if __name__ == "__main__":
    prepare_data()