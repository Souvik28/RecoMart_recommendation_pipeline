import pandas as pd
import numpy as np
from datetime import datetime

def engineering():
    # Use prepared data (which is now combined streaming + batch)
    df = pd.read_csv("../recomart_lake/processed/prepared_transactions.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True).dt.tz_localize(None)
    print(f"Feature engineering on {len(df)} transactions (combined streaming + batch data)")

    # === USER FEATURES ===
    user_features = df.groupby('user_id').agg(
        # Basic activity metrics
        user_total_interactions=('transaction_id', 'count'),
        user_avg_rating=('rating', 'mean'),
        user_total_spent=('amount', 'sum'),
        user_avg_spent=('amount', 'mean'),
        
        # Behavioral patterns
        user_rating_std=('rating', 'std'),
        user_unique_products=('product_id', 'nunique'),
        
        # Temporal features
        user_first_interaction=('timestamp', 'min'),
        user_last_interaction=('timestamp', 'max'),
    ).reset_index()
    
    # Calculate user tenure and recency
    now = pd.Timestamp.now()
    user_features['user_tenure_days'] = (now - user_features['user_first_interaction']).dt.days
    user_features['user_recency_days'] = (now - user_features['user_last_interaction']).dt.days
    user_features['user_activity_rate'] = user_features['user_total_interactions'] / (user_features['user_tenure_days'] + 1)
    
    # User engagement level based on rating patterns
    user_features['user_engagement_score'] = (
        user_features['user_avg_rating'] * 0.4 + 
        np.log1p(user_features['user_total_interactions']) * 0.3 +
        (5 - user_features['user_recency_days'].clip(0, 30) / 6) * 0.3
    )
    
    # === ITEM FEATURES ===
    item_features = df.groupby('product_id').agg(
        # Basic popularity metrics
        item_avg_rating=('rating', 'mean'),
        item_popularity_score=('transaction_id', 'count'),
        item_total_revenue=('amount', 'sum'),
        item_avg_price=('amount', lambda x: x[x > 0].mean()),  # Exclude streaming events (amount=0)
        
        # Quality metrics
        item_rating_std=('rating', 'std'),
        item_unique_users=('user_id', 'nunique'),
        
        # Temporal features
        item_first_interaction=('timestamp', 'min'),
        item_last_interaction=('timestamp', 'max'),
    ).reset_index()
    
    # Calculate item lifecycle metrics
    item_features['item_age_days'] = (now - item_features['item_first_interaction']).dt.days
    item_features['item_recency_days'] = (now - item_features['item_last_interaction']).dt.days
    item_features['item_velocity'] = item_features['item_popularity_score'] / (item_features['item_age_days'] + 1)
    
    # Item quality score
    item_features['item_quality_score'] = (
        item_features['item_avg_rating'] * 0.5 +
        np.log1p(item_features['item_popularity_score']) * 0.3 +
        (item_features['item_unique_users'] / item_features['item_popularity_score']) * 0.2  # Diversity factor
    )
    
    # === INTERACTION FEATURES ===
    # Create user-item interaction matrix for collaborative features
    interaction_features = df.groupby(['user_id', 'product_id']).agg(
        interaction_count=('transaction_id', 'count'),
        interaction_avg_rating=('rating', 'mean'),
        interaction_total_spent=('amount', 'sum'),
        interaction_first=('timestamp', 'min'),
        interaction_last=('timestamp', 'max')
    ).reset_index()
    
    # Calculate interaction recency and frequency
    interaction_features['interaction_recency_days'] = (now - interaction_features['interaction_last']).dt.days
    interaction_features['interaction_span_days'] = (interaction_features['interaction_last'] - interaction_features['interaction_first']).dt.days
    
    # Fill NaN values
    user_features = user_features.fillna(0)
    item_features = item_features.fillna(0)
    interaction_features = interaction_features.fillna(0)
    
    print(f"Generated {len(user_features)} user features, {len(item_features)} item features, {len(interaction_features)} interaction features")
    
    return user_features, item_features, interaction_features

if __name__ == "__main__":
    u_feat, i_feat, int_feat = engineering()
    print("Advanced Features Engineered:")
    print(f"- User features: {u_feat.shape[1]-1} features for {len(u_feat)} users")
    print(f"- Item features: {i_feat.shape[1]-1} features for {len(i_feat)} items")
    print(f"- Interaction features: {int_feat.shape[1]-2} features for {len(int_feat)} user-item pairs")