import pickle
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, RegressorMixin

class BiasedSVD(BaseEstimator, RegressorMixin):
    """Enhanced SVD with bias handling and regularization"""
    
    def __init__(self, n_components=20, regularization=0.01, learning_rate=0.01, n_epochs=100):
        self.n_components = n_components
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
    def fit(self, X, y=None):
        self.global_mean_ = np.mean(X[X > 0])
        self.user_bias_ = np.zeros(X.shape[0])
        self.item_bias_ = np.zeros(X.shape[1])
        
        # Calculate initial biases
        for i in range(X.shape[0]):
            user_ratings = X[i, X[i] > 0]
            if len(user_ratings) > 0:
                self.user_bias_[i] = np.mean(user_ratings) - self.global_mean_
                
        for j in range(X.shape[1]):
            item_ratings = X[X[:, j] > 0, j]
            if len(item_ratings) > 0:
                self.item_bias_[j] = np.mean(item_ratings) - self.global_mean_
        
        # Mean center the matrix
        X_centered = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] > 0:
                    X_centered[i, j] = X[i, j] - self.global_mean_ - self.user_bias_[i] - self.item_bias_[j]
        
        # Apply SVD
        self.svd_ = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.U_ = self.svd_.fit_transform(X_centered)
        self.Vt_ = self.svd_.components_
        
        return self
    
    def predict(self, X):
        # Reconstruct matrix
        X_pred = np.dot(self.U_, self.Vt_)
        
        # Add biases back
        for i in range(X_pred.shape[0]):
            for j in range(X_pred.shape[1]):
                X_pred[i, j] += self.global_mean_ + self.user_bias_[i] + self.item_bias_[j]
        
        return np.clip(X_pred, 1, 5)
    
    @property
    def explained_variance_ratio_(self):
        return self.svd_.explained_variance_ratio_

def setup_logging():
    """Setup logging for recommendation generation"""
    # Use absolute path for log directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    log_dir = os.path.join(project_root, "recomart_lake", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def generate_recommendations_for_orchestrator(logger=None):
    """Main function for orchestrator to call"""
    if logger is None:
        logger = logging.getLogger('recommendations')
        logger.setLevel(logging.INFO)
    
    logger.info("=== STEP 11: RECOMMENDATION GENERATION ===")
    
    try:
        # Show model performance
        show_model_performance(logger)
        
        # Generate recommendations for sample users + cold start test
        sample_users = ["U001", "U002", "U003", "U010", "U020", "NEW_USER_COLD_START"]
        all_recommendations = {}
        
        for user in sample_users:
            recommendations = recommend_products(user, top_k=5, logger=logger)
            all_recommendations[user] = recommendations
        
        logger.info(f"Generated recommendations for {len(all_recommendations)} users")
        return True
        
    except Exception as e:
        logger.error(f"Error in recommendation generation: {str(e)}")
        return False

def load_trained_model():
    """Load the trained SVD model and metadata"""
    # Use absolute path to avoid working directory issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(project_root, "recomart_lake", "models")
    
    # Load model
    model_path = os.path.join(model_dir, "svd_model.pkl")
    with open(model_path, "rb") as f:
        svd_model = pickle.load(f)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.pkl")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return svd_model, metadata

def recommend_products(user_id, top_k=5, logger=None):
    """Generate hybrid product recommendations for a specific user"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Generating hybrid recommendations for user {user_id}")
        svd_model, metadata = load_trained_model()
        
        # Reconstruct user-item matrix structure
        columns = metadata["user_item_matrix_columns"]
        index = metadata["user_item_matrix_index"]
        reconstructed_matrix = metadata["reconstructed_matrix"]
        content_features = metadata.get("content_features")
        
        # Convert content features back to DataFrame if available
        content_df = None
        if content_features:
            content_df = pd.DataFrame(content_features)
        
        # Use hybrid recommendation
        if user_id not in index:
            logger.info(f"Cold start scenario for user {user_id}")
            if content_df is not None:
                # Safe cold start recommendations with proper scoring
                try:
                    if 'popularity_norm' in content_df.columns:
                        # Use normalized popularity scores
                        top_products = content_df.nlargest(top_k, 'popularity_norm')
                        recommendations = [(row['product_id'], row['popularity_norm']) for _, row in top_products.iterrows()]
                    elif 'item_popularity_score' in content_df.columns:
                        # Use item popularity scores
                        top_products = content_df.nlargest(top_k, 'item_popularity_score')
                        recommendations = [(row['product_id'], row['item_popularity_score']) for _, row in top_products.iterrows()]
                    elif 'avg_rating' in content_df.columns:
                        # Fallback to average ratings
                        top_products = content_df.nlargest(top_k, 'avg_rating')
                        recommendations = [(row['product_id'], row['avg_rating'] / 5.0) for _, row in top_products.iterrows()]
                    else:
                        # Last resort: use first products with default score
                        top_products = content_df.head(top_k)
                        recommendations = [(row['product_id'], 0.5) for _, row in top_products.iterrows()]
                except Exception as e:
                    logger.warning(f"Cold start fallback failed: {e}")
                    recommendations = []
            else:
                logger.warning(f"No content features available for cold start user {user_id}")
                return []
        else:
            # Hybrid recommendations for existing users
            user_idx = index.index(user_id)
            user_scores = reconstructed_matrix[user_idx]
            
            # Create recommendations with hybrid scores
            product_scores = []
            for i, prod_id in enumerate(columns):
                cf_score = user_scores[i]
                cb_score = 0
                
                if content_df is not None and prod_id in content_df['product_id'].values:
                    try:
                        product_row = content_df[content_df['product_id'] == prod_id].iloc[0]
                        
                        # Safe column access with multiple fallbacks
                        rating_score = 3.0  # Default
                        if 'avg_rating' in product_row:
                            rating_score = product_row['avg_rating']
                        elif 'item_avg_rating' in product_row:
                            rating_score = product_row['item_avg_rating']
                        
                        popularity_score = 0  # Default
                        if 'popularity_norm' in product_row:
                            popularity_score = product_row['popularity_norm']
                        elif 'item_popularity_score' in product_row:
                            popularity_score = product_row['item_popularity_score']
                        
                        cb_score = rating_score * 0.3 + popularity_score * 0.7
                    except Exception as e:
                        logger.debug(f"Content score calculation failed for {prod_id}: {e}")
                        cb_score = 0
                
                # Hybrid score (70% collaborative, 30% content)
                hybrid_score = 0.7 * cf_score + 0.3 * cb_score
                product_scores.append((prod_id, hybrid_score))
            
            product_scores.sort(key=lambda x: x[1], reverse=True)
            recommendations = product_scores[:top_k]
        
        logger.info(f"Generated {len(recommendations)} hybrid recommendations for user {user_id}")
        
        print(f"\nTop {top_k} hybrid recommendations for user {user_id}:")
        print("-" * 50)
        for i, (product_id, score) in enumerate(recommendations, 1):
            rec_type = "Cold Start" if user_id not in index else "Hybrid"
            print(f"{i}. Product {product_id}: Score {score:.3f} ({rec_type})")
        
        return recommendations
        
    except Exception as e:
        error_msg = f"Error generating recommendations: {str(e)}"
        if logger:
            logger.error(error_msg)
        print(error_msg)
        return []

def show_model_performance(logger=None):
    """Display model performance metrics"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        _, metadata = load_trained_model()
        
        logger.info("Displaying model performance metrics")
        
        print("\nModel Performance Metrics:")
        print("=" * 30)
        print(f"RMSE: {metadata['rmse']:.4f}")
        
        if 'quality_metrics' in metadata:
            quality = metadata['quality_metrics']
            print(f"Precision@5: {quality['precision_at_5']:.4f}")
            print(f"Recall@5: {quality['recall_at_5']:.4f}")
            print(f"F1-Score@5: {quality['f1_at_5']:.4f}")
            
            logger.info(f"Model metrics - RMSE: {metadata['rmse']:.4f}, Precision@5: {quality['precision_at_5']:.4f}")
        
        if 'sample_recommendations' in metadata:
            print(f"\nSample recommendations available for {len(metadata['sample_recommendations'])} users")
            
    except FileNotFoundError:
        error_msg = "Model not found. Please run model training first."
        if logger:
            logger.error(error_msg)
        print(error_msg)

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    logger.info("Starting recommendation generation")
    
    # Show model performance
    show_model_performance(logger)
    
    # Generate recommendations for sample users
    sample_users = ["U001", "U002", "U003"]
    
    for user in sample_users:
        recommend_products(user, top_k=5, logger=logger)
        print()
    
    logger.info("Recommendation generation completed")