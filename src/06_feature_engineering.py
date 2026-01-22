import pandas as pd
import sqlite3

def engineering():
    # Use prepared data (which is now combined streaming + batch)
    df = pd.read_csv("../recomart_lake/processed/prepared_transactions.csv")
    print(f"Feature engineering on {len(df)} transactions (combined streaming + batch data)")

    # Feature 1: User Activity Frequency (How many times has a user interacted?)
    user_features = df.groupby('user_id').agg(
        user_total_interactions=('transaction_id', 'count'),
        user_avg_rating=('rating', 'mean')
    ).reset_index()

    # Feature 2: Item Popularity Features (Average rating and count per product)
    item_features = df.groupby('product_id').agg(
        item_avg_rating=('rating', 'mean'),
        item_popularity_score=('transaction_id', 'count')
    ).reset_index()

    return user_features, item_features

if __name__ == "__main__":
    u_feat, i_feat = engineering()
    print("Features Engineered: User and Item profiles created.")