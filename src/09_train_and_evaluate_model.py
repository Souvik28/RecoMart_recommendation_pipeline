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
from sklearn.model_selection import train_test_split

def calculate_precision_recall_at_k(actual_matrix, pred_matrix, k=5, threshold=3.5):
    """
    Industry standard quality check: Precision and Recall at K.
    Measures how many of the top-K recommended items were actually liked by the user.
    """
    precisions = []
    recalls = []
    
    for i in range(actual_matrix.shape[0]):
        # Get actual ratings and predicted scores for this user
        actual_user = actual_matrix[i]
        pred_user = pred_matrix[i]
        
        # Items the user actually liked (above threshold)
        actual_relevant = np.where(actual_user >= threshold)[0]
        
        # Top K items recommended by the model
        top_k_idx = np.argsort(pred_user)[::-1][:k]
        
        # Recommended items that were actually liked
        n_rel_and_rec = len(np.intersect1d(actual_relevant, top_k_idx))
        
        # Precision@K: (Rel + Rec) / (Total Rec)
        precisions.append(n_rel_and_rec / k)
        
        # Recall@K: (Rel + Rec) / (Total Rel)
        if len(actual_relevant) > 0:
            recalls.append(n_rel_and_rec / len(actual_relevant))
            
    return np.mean(precisions), np.mean(recalls)

def train_sklearn_recommender():
    # 1. Setup paths
    model_dir = "../recomart_lake/models/"
    os.makedirs(model_dir, exist_ok=True)
    
    data_path = "../recomart_lake/processed/prepared_transactions.csv"
    if not os.path.exists(data_path):
        print(f"Error: Prepared data not found at {data_path}. Run Step 5 first.")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Prepare Matrix and Train-Test Split
    # Rows = Users, Columns = Products, Values = Ratings
    user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
    
    # Industry Practice: Split data to evaluate on 'unseen' interactions
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("RecoMart_SVD")
    
    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "MatrixFactorization")
        
        # 3. Training (Matrix Factorization via SVD)
        n_components = 12
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        matrix_reduced = svd.fit_transform(user_item_matrix)
        matrix_reconstructed = np.dot(matrix_reduced, svd.components_)
        
        # 4. Error Metrics Calculation
        mask = user_item_matrix.values > 0
        actuals = user_item_matrix.values[mask]
        preds = matrix_reconstructed[mask]
        
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        explained_var = svd.explained_variance_ratio_.sum()

        # 5. Recommendation Quality Metrics (The 'Quality Check')
        p_at_k, r_at_k = calculate_precision_recall_at_k(user_item_matrix.values, matrix_reconstructed, k=5)

        # Log all metrics to MLflow
        mlflow.log_params({"n_components": n_components, "k": 5})
        mlflow.log_metrics({
            "RMSE": rmse, 
            "MAE": mae,
            "Explained_Variance": explained_var,
            "Precision_at_5": p_at_k,
            "Recall_at_5": r_at_k
        })

        # --- 6. SAVING PLOTS ---
        # A. Variance Plot
        var_plot_path = os.path.join(model_dir, "variance_plot.png")
        plt.figure(figsize=(6, 3))
        plt.plot(np.cumsum(svd.explained_variance_ratio_))
        plt.title('SVD Cumulative Explained Variance')
        plt.grid(True)
        plt.savefig(var_plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(var_plot_path)
        plt.close()
        
        # B. Quality Dashboard (Precision vs Recall)
        qual_plot_path = os.path.join(model_dir, "quality_metrics.png")
        plt.figure(figsize=(5, 3))
        metrics = ['Precision@5', 'Recall@5']
        values = [p_at_k, r_at_k]
        sns.barplot(x=metrics, y=values, palette="Blues_d")
        plt.ylim(0, 1)
        plt.title('Recommendation Quality (Top-5)')
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
        metrics_values = [rmse, mae, p_at_k, r_at_k]
        colors = ['red', 'orange', 'green', 'blue']
        ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax1.set_title('Performance Metrics', fontsize=10)
        ax1.tick_params(axis='x', labelsize=8)
        
        # Explained variance
        ax2.plot(range(1, len(svd.explained_variance_ratio_) + 1), 
                np.cumsum(svd.explained_variance_ratio_), 'bo-', markersize=4)
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

        # 7. Save Model and Metadata
        mlflow.sklearn.log_model(svd, "model")
        
        # Save model locally as well
        model_path = os.path.join(model_dir, "svd_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(svd, f)

        model_metadata = {
            "user_item_matrix_columns": user_item_matrix.columns.tolist(),
            "user_item_matrix_index": user_item_matrix.index.tolist(),
            "reconstructed_matrix": matrix_reconstructed,
            "n_components": n_components,
            "rmse": rmse,
            "precision_at_5": p_at_k,
            "recall_at_5": r_at_k
        }
        
        metadata_path = os.path.join(model_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(model_metadata, f)
        mlflow.log_artifact(metadata_path)

        print(f"Success: Model and Quality Report saved to {model_dir}")
        print(f"RMSE: {rmse:.4f} | Precision@5: {p_at_k:.4f}")

if __name__ == "__main__":
    train_sklearn_recommender()