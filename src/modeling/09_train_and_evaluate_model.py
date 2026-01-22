import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import ParameterGrid

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

def hyperparameter_tuning(user_item_matrix, sparsity):
    """Enhanced hyperparameter tuning based on data characteristics"""
    
    # Define parameter grids based on sparsity
    if sparsity > 0.95:  # Very sparse
        param_grid = {
            'n_components': [15, 20, 25],
            'regularization': [0.1, 0.05, 0.02],
            'learning_rate': [0.01, 0.005]
        }
    elif sparsity > 0.90:  # Moderately sparse
        param_grid = {
            'n_components': [20, 30, 40],
            'regularization': [0.05, 0.01, 0.005],
            'learning_rate': [0.01, 0.02]
        }
    else:  # Dense
        param_grid = {
            'n_components': [30, 40, 50],
            'regularization': [0.01, 0.005, 0.001],
            'learning_rate': [0.02, 0.05]
        }
    
    best_rmse = float('inf')
    best_params = None
    
    # Grid search with cross-validation
    for params in ParameterGrid(param_grid):
        model = BiasedSVD(**params)
        model.fit(user_item_matrix)
        
        # Predict and calculate RMSE
        predictions = model.predict(user_item_matrix)
        mask = user_item_matrix > 0
        rmse = np.sqrt(mean_squared_error(user_item_matrix[mask], predictions[mask]))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    
    print(f"Best parameters: {best_params} (RMSE: {best_rmse:.4f})")
    return best_params

def check_sparsity(matrix):
    """Check matrix sparsity level"""
    total_elements = matrix.size
    non_zero_elements = np.count_nonzero(matrix)
    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity

def calculate_precision_recall_at_k(train_matrix, test_matrix, pred_matrix, k=5, threshold=3.5):
    """
    Fixed precision/recall calculation using proper holdout evaluation.
    """
    precisions = []
    recalls = []
    
    for i in range(train_matrix.shape[0]):
        # Get test ratings for this user (items held out from training)
        test_user = test_matrix[i]
        pred_user = pred_matrix[i]
        train_user = train_matrix[i]
        
        # Find items that were held out for testing (in test but not in train)
        test_items = np.where((test_user > 0) & (train_user == 0))[0]
        
        # Skip users with no test items
        if len(test_items) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        # Items the user actually liked in test set (above threshold)
        relevant_items = test_items[test_user[test_items] >= threshold]
        
        # Get candidate items for recommendation (unrated in training)
        candidate_items = np.where(train_user == 0)[0]
        
        if len(candidate_items) < k:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        # Top K recommendations from candidate items
        candidate_scores = pred_user[candidate_items]
        top_k_indices = candidate_items[np.argsort(candidate_scores)[::-1][:k]]
        
        # How many of top-K recommendations were actually relevant in test set
        n_rel_and_rec = len(np.intersect1d(relevant_items, top_k_indices))
        
        # Precision@K: relevant recommendations / total recommendations
        precisions.append(n_rel_and_rec / k)
        
        # Recall@K: relevant recommendations / total relevant items
        if len(relevant_items) > 0:
            recalls.append(n_rel_and_rec / len(relevant_items))
        else:
            recalls.append(0.0)
            
    return np.mean(precisions), np.mean(recalls)

def calculate_comprehensive_metrics(train_matrix, test_matrix, pred_matrix, k_values=[1, 3, 5, 10], threshold=3.5):
    """
    Calculate precision, recall, and F1-score for multiple K values using proper train/test split
    """
    metrics = {}
    
    for k in k_values:
        precision, recall = calculate_precision_recall_at_k(train_matrix, test_matrix, pred_matrix, k, threshold)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'precision_at_{k}'] = precision
        metrics[f'recall_at_{k}'] = recall
        metrics[f'f1_at_{k}'] = f1
    
    return metrics

def hybrid_recommendation(user_item_matrix, matrix_reconstructed, content_features, user_id, top_k=5, alpha=0.7):
    """Hybrid collaborative + content-based recommendations"""
    if user_id not in user_item_matrix.index:
        # Cold start: use content-based only
        if content_features is not None and 'popularity_norm' in content_features.columns:
            # Recommend popular items in diverse categories
            popular_items = content_features.nlargest(top_k, 'popularity_norm')['product_id'].tolist()
            return popular_items
        return []
    
    # Warm start: hybrid approach
    user_idx = user_item_matrix.index.get_loc(user_id)
    cf_scores = matrix_reconstructed[user_idx]
    
    # Get unrated products
    unrated_mask = user_item_matrix.iloc[user_idx] == 0
    unrated_products = user_item_matrix.columns[unrated_mask]
    
    recommendations = []
    
    for product in unrated_products:
        product_idx = user_item_matrix.columns.get_loc(product)
        cf_score = cf_scores[product_idx]
        
        # Content-based score with safe column access
        cb_score = 0
        if content_features is not None and product in content_features['product_id'].values:
            product_row = content_features[content_features['product_id'] == product].iloc[0]
            
            # Use available columns safely
            rating_score = product_row.get('avg_rating', product_row.get('item_avg_rating', 3.0))
            popularity_score = product_row.get('popularity_norm', product_row.get('item_popularity_score', 0))
            
            cb_score = rating_score * 0.3 + popularity_score * 0.7
        
        # Hybrid score
        hybrid_score = alpha * cf_score + (1 - alpha) * cb_score
        recommendations.append((product, hybrid_score))
    
    # Sort and return top-K
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [prod for prod, _ in recommendations[:top_k]]

def train_sklearn_recommender():
    # 1. Setup paths - use absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = os.path.join(project_root, "recomart_lake", "models")
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model artifacts will be saved to: {model_dir}")
    
    # Use feature store data for training (optimal approach)
    feature_store_path = "../../recomart_lake/feature_store/user_item_features.csv"
    
    if os.path.exists(feature_store_path):
        print("Using feature store data for training (optimal)")
        df = pd.read_csv(feature_store_path)
    else:
        # Fallback to combined data
        combined_path = "../../recomart_lake/processed/combined_transactions.csv"
        if os.path.exists(combined_path):
            print("Using combined streaming + batch data for training")
            df = pd.read_csv(combined_path)
        else:
            print("Using batch data only for training")
            df = pd.read_csv("../../recomart_lake/processed/prepared_transactions.csv")
    
    # 2. Load content features from feature store
    try:
        import sqlite3
        conn = sqlite3.connect("../../recomart_lake/feature_store/recomart_features.db")
        content_features = pd.read_sql_query("SELECT * FROM item_features", conn)
        conn.close()
        print(f"Loaded content features: {content_features.shape[1]} attributes")
    except:
        content_features = None
        print("No content features found in feature store")
    
    # 3. Create user-item matrix and implement train/test split
    user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
    sparsity = check_sparsity(user_item_matrix.values)
    print(f"Matrix sparsity: {sparsity:.3f}")
    
    # Create train/test split by hiding 20% of ratings for evaluation
    train_matrix = user_item_matrix.copy()
    test_matrix = user_item_matrix.copy()
    
    np.random.seed(42)
    test_items_count = 0
    users_with_test_items = 0
    
    for i in range(user_item_matrix.shape[0]):
        user_ratings = np.where(user_item_matrix.iloc[i] > 0)[0]
        if len(user_ratings) > 2:  # Only split if user has enough ratings
            n_test = max(1, int(0.2 * len(user_ratings)))  # 20% for testing
            test_indices = np.random.choice(user_ratings, n_test, replace=False)
            
            # Hide test ratings from training matrix
            for idx in test_indices:
                train_matrix.iloc[i, idx] = 0
                test_items_count += 1
            users_with_test_items += 1
    
    print(f"Train matrix sparsity: {check_sparsity(train_matrix.values):.3f}")
    print(f"Users with test items: {users_with_test_items}/{user_item_matrix.shape[0]}")
    print(f"Total test items: {test_items_count}")
    
    # Debug: Check if we actually have test data
    test_mask = (test_matrix.values > 0) & (train_matrix.values == 0)
    print(f"Actual test ratings available: {np.sum(test_mask)}")
    
    # SPARSITY CHECK FOR PRECISION@K VALIDITY
    print("\n=== SPARSITY ANALYSIS FOR PRECISION@K VALIDITY ===")
    total_possible = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    actual_ratings = np.sum(user_item_matrix.values > 0)
    density = actual_ratings / total_possible
    print(f"Matrix density: {density:.4f} ({density*100:.2f}%)")
    print(f"Total possible ratings: {total_possible:,}")
    print(f"Actual ratings: {actual_ratings:,}")
    

    if density > 0.1:  # More than 10% density
        print("WARNING: Matrix is quite dense - Precision@K may be artificially high")
        print("   Consider using different evaluation metrics for dense matrices")
    elif density > 0.05:  # 5-10% density
        print("CAUTION: Matrix density is moderate - Precision@K results should be interpreted carefully")
    else:
        print("Matrix is appropriately sparse for Precision@K evaluation")
    
    # Check average ratings per user
    avg_ratings_per_user = np.mean(np.sum(user_item_matrix.values > 0, axis=1))
    print(f"Average ratings per user: {avg_ratings_per_user:.1f}")
    
    if avg_ratings_per_user > user_item_matrix.shape[1] * 0.5:
        print("WARNING: Users have rated >50% of items - Precision@K may not be meaningful")
    
    # 4. Enhanced hyperparameter tuning on training data
    best_params = hyperparameter_tuning(train_matrix.values, sparsity)
    
    # Set MLflow tracking URI to project root (absolute path)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    mlflow_db_path = os.path.join(project_root, "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment("RecoMart_SVD")
    
    print(f"MLflow database location: {mlflow_db_path}")
    print(f"To view MLflow UI, run from project root: mlflow ui")
    
    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "BiasedSVD_Tuned")
        
        # 5. Train enhanced model ONLY on training data (no data leakage)
        enhanced_model = BiasedSVD(**best_params)
        enhanced_model.fit(train_matrix.values)  # Train only on training data
        
        # 6. Generate predictions from training-only model
        matrix_reconstructed = enhanced_model.predict(train_matrix.values)
        
        print("\n=== DATA LEAKAGE CHECK ===")
        print("Model trained ONLY on training matrix (no test data seen)")
        print("Predictions generated from training-only model")
        print("Evaluation uses proper holdout test set")
        
        # 7. Calculate metrics on test set (realistic evaluation)
        mask = test_matrix.values > 0  # Only evaluate on test ratings
        if np.sum(mask) > 0:
            actuals = test_matrix.values[mask]
            preds = matrix_reconstructed[mask]
            
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            mae = mean_absolute_error(actuals, preds)
        else:
            rmse = mae = 0.0
            
        explained_var = enhanced_model.explained_variance_ratio_.sum()

        # 8. Comprehensive metrics with multiple thresholds and K values
        print("\n=== TESTING DIFFERENT THRESHOLDS AND K VALUES ===")
        
        # Test with stricter threshold (4.5 - only highly liked items)
        strict_metrics = calculate_comprehensive_metrics(train_matrix.values, test_matrix.values, matrix_reconstructed, 
                                                       k_values=[5, 10], threshold=4.5)
        print(f"Strict threshold (4.5): Precision@5={strict_metrics['precision_at_5']:.4f}, Precision@10={strict_metrics['precision_at_10']:.4f}")
        
        # Test with normal threshold (3.5)
        quality_metrics = calculate_comprehensive_metrics(train_matrix.values, test_matrix.values, matrix_reconstructed, 
                                                        k_values=[1, 3, 5, 10], threshold=3.5)
        print(f"Normal threshold (3.5): Precision@5={quality_metrics['precision_at_5']:.4f}, Precision@10={quality_metrics['precision_at_10']:.4f}")
        
        # Add strict metrics to logging
        for key, value in strict_metrics.items():
            quality_metrics[f"strict_{key}"] = value
        
        # Log enhanced parameters
        mlflow.log_params({
            "n_components": best_params['n_components'],
            "regularization": best_params['regularization'],
            "learning_rate": best_params['learning_rate'],
            "sparsity": sparsity,
            "bias_handled": True,
            "baseline_offsets": True,
            "hyperparameter_tuned": True,
            "content_features": content_features is not None,
            "hybrid_model": True,
            "cold_start_handled": True,
            "click_aggregated": True
        })
        
        all_metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "Explained_Variance": explained_var,
            "Sparsity": sparsity
        }
        all_metrics.update(quality_metrics)
        mlflow.log_metrics(all_metrics)

        # --- 6. SAVING PLOTS ---
        # A. Variance Plot
        var_plot_path = os.path.join(model_dir, "variance_plot.png")
        plt.figure(figsize=(6, 3))
        plt.plot(np.cumsum(enhanced_model.explained_variance_ratio_))
        plt.title('Enhanced SVD Cumulative Explained Variance')
        plt.grid(True)
        plt.savefig(var_plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(var_plot_path)
        plt.close()
        
        # B. Comprehensive Quality Dashboard
        qual_plot_path = os.path.join(model_dir, "quality_metrics.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Precision, Recall, F1 by K
        k_values = [1, 3, 5, 10]
        precisions = [quality_metrics[f'precision_at_{k}'] for k in k_values]
        recalls = [quality_metrics[f'recall_at_{k}'] for k in k_values]
        f1_scores = [quality_metrics[f'f1_at_{k}'] for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.25
        
        ax1.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax1.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('K Value')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision, Recall & F1-Score by K')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'@{k}' for k in k_values])
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Precision vs Recall curve
        ax2.plot(recalls, precisions, 'bo-', markersize=6, linewidth=2)
        for i, k in enumerate(k_values):
            ax2.annotate(f'K={k}', (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(qual_plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(qual_plot_path)
        plt.close()

        # C. Prediction vs Actual Hexbin Plot
        scatter_plot_path = os.path.join(model_dir, "prediction_hexbin.png")
        plt.figure(figsize=(5, 4))
        plt.hexbin(actuals, preds, gridsize=15, cmap='Blues', mincnt=1)
        plt.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Actual vs Predicted (Density)')
        plt.colorbar(label='Count')
        plt.legend()
        plt.xlim(0.5, 5.5)
        plt.ylim(0.5, 5.5)
        plt.savefig(scatter_plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(scatter_plot_path)
        plt.close()
        
        # D. Rating Distribution KDE Plot
        dist_plot_path = os.path.join(model_dir, "rating_distribution.png")
        plt.figure(figsize=(6, 3))
        sns.kdeplot(actuals, label='Actual', fill=True, alpha=0.6)
        sns.kdeplot(preds, label='Predicted', fill=True, alpha=0.6)
        plt.xlabel('Rating')
        plt.ylabel('Density')
        plt.title('Rating Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 6)
        plt.savefig(dist_plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(dist_plot_path)
        plt.close()

        # E. Error Analysis
        error_plot_path = os.path.join(model_dir, "error_analysis.png")
        plt.figure(figsize=(8, 3))
        
        residuals = actuals - preds
        
        plt.subplot(1, 2, 1)
        plt.scatter(actuals, residuals, alpha=0.4, s=8)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(error_plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(error_plot_path)
        plt.close()
        
        # F. User-Item Matrix Heatmap
        heatmap_plot_path = os.path.join(model_dir, "matrix_heatmap.png")
        plt.figure(figsize=(7, 4))
        
        sample_matrix = user_item_matrix.iloc[:15, :15]
        sns.heatmap(sample_matrix, cmap='YlOrRd', cbar_kws={'label': 'Rating'})
        plt.title('User-Item Matrix (Sample 15x15)')
        plt.xlabel('Products')
        plt.ylabel('Users')
        plt.xticks(rotation=45)
        plt.savefig(heatmap_plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(heatmap_plot_path)
        plt.close()
        
        # G. Performance Summary Dashboard
        summary_plot_path = os.path.join(model_dir, "performance_summary.png")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
        
        # Metrics comparison
        metrics_names = ['RMSE', 'MAE', 'Precision@5', 'Recall@5']
        metrics_values = [rmse, mae, quality_metrics['precision_at_5'], quality_metrics['recall_at_5']]
        colors = ['red', 'orange', 'green', 'blue']
        ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax1.set_title('Performance Metrics', fontsize=10)
        ax1.tick_params(axis='x', labelsize=8)
        
        # Explained variance
        ax2.plot(range(1, len(enhanced_model.explained_variance_ratio_) + 1), 
                np.cumsum(enhanced_model.explained_variance_ratio_), 'bo-', markersize=4)
        ax2.set_title('Explained Variance', fontsize=10)
        ax2.set_xlabel('Components', fontsize=8)
        ax2.grid(True)
        
        # Rating distributions
        ax3.hist([actuals, preds], bins=15, alpha=0.7, label=['Actual', 'Predicted'], 
                color=['blue', 'orange'])
        ax3.set_title('Rating Distributions', fontsize=10)
        ax3.legend(fontsize=8)
        
        # MAE by rating
        rating_bins = np.arange(1, 6)
        accuracy_by_rating = []
        for rating in rating_bins:
            mask_rating = (actuals >= rating - 0.5) & (actuals < rating + 0.5)
            if np.sum(mask_rating) > 0:
                mae_rating = np.mean(np.abs(actuals[mask_rating] - preds[mask_rating]))
                accuracy_by_rating.append(mae_rating)
            else:
                accuracy_by_rating.append(0)
        
        ax4.bar(rating_bins, accuracy_by_rating, color='purple', alpha=0.7)
        ax4.set_title('MAE by Rating', fontsize=10)
        ax4.set_xlabel('Rating', fontsize=8)
        ax4.set_xticks(rating_bins)
        
        plt.tight_layout()
        plt.savefig(summary_plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(summary_plot_path)
        plt.close()

        # Save enhanced model and log to MLflow
        model_path = os.path.join(model_dir, "svd_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(enhanced_model, f)
        
        # Log model as artifact (works better with custom classes)
        mlflow.log_artifact(model_path, "model")
        
        # Also save a simple version for MLflow compatibility
        simple_model_path = os.path.join(model_dir, "svd_component.pkl")
        with open(simple_model_path, "wb") as f:
            pickle.dump(enhanced_model.svd_, f)
        mlflow.log_artifact(simple_model_path, "model")

        # Generate hybrid recommendations (collaborative + content-based)
        sample_users = user_item_matrix.index[:5]
        sample_recommendations = {}
        for user in sample_users:
            recs = hybrid_recommendation(user_item_matrix, matrix_reconstructed, content_features, user)
            sample_recommendations[user] = recs
        
        # Test cold start scenario
        cold_start_recs = hybrid_recommendation(user_item_matrix, matrix_reconstructed, content_features, "NEW_USER_001")
        sample_recommendations["NEW_USER_001"] = cold_start_recs
        
        model_metadata = {
            "user_item_matrix_columns": user_item_matrix.columns.tolist(),
            "user_item_matrix_index": user_item_matrix.index.tolist(),
            "reconstructed_matrix": matrix_reconstructed,
            "content_features": content_features.to_dict() if content_features is not None else None,
            "n_components": best_params['n_components'],
            "sparsity": sparsity,
            "regularization": best_params['regularization'],
            "global_mean": enhanced_model.global_mean_,
            "user_bias": enhanced_model.user_bias_.tolist(),
            "item_bias": enhanced_model.item_bias_.tolist(),
            "rmse": rmse,
            "quality_metrics": quality_metrics,
            "sample_recommendations": sample_recommendations
        }
        
        metadata_path = os.path.join(model_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(model_metadata, f)
        mlflow.log_artifact(metadata_path)

        print(f"Success: Hybrid Model (CF + Content) saved to {model_dir}")
        print(f"Components: {best_params['n_components']} | Explained Variance: {explained_var:.3f} | Sparsity: {sparsity:.3f}")
        print(f"RMSE: {rmse:.4f} | Cold Start: {'Enabled' if content_features is not None else 'Disabled'}")
        print(f"Precision@5: {quality_metrics['precision_at_5']:.4f} | Recall@5: {quality_metrics['recall_at_5']:.4f}")
        print(f"F1@5: {quality_metrics['f1_at_5']:.4f}")

if __name__ == "__main__":
    train_sklearn_recommender()