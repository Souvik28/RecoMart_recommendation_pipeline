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

def train_sklearn_recommender():
    # 1. Setup paths
    model_dir = "../recomart_lake/models/"
    os.makedirs(model_dir, exist_ok=True)
    
    data_path = "../recomart_lake/processed/prepared_transactions.csv"
    df = pd.read_csv(data_path)
    
    # 2. Prepare Matrix
    user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
    
    # Set MLflow tracking URI to root directory (including database)
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("RecoMart_SVD")
    
    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", "MatrixFactorization")
        
        # 3. Training
        n_components = 12
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        matrix_reduced = svd.fit_transform(user_item_matrix)
        matrix_reconstructed = np.dot(matrix_reduced, svd.components_)
        
        # 4. Metrics
        mask = user_item_matrix.values > 0
        actuals = user_item_matrix.values[mask]
        preds = matrix_reconstructed[mask]
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        explained_var = svd.explained_variance_ratio_.sum()

        mlflow.log_metrics({"RMSE": rmse, "Explained_Variance": explained_var})

        # --- 5. SAVING PLOTS TO recomart_lake/models/ ---
        
        # A. Variance Plot
        var_plot_path = os.path.join(model_dir, "variance_plot.png")
        plt.figure(figsize=(8, 4))
        plt.plot(np.cumsum(svd.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('SVD Explained Variance')
        plt.grid(True)
        plt.savefig(var_plot_path)
        mlflow.log_artifact(var_plot_path) # Still log to MLflow for the UI
        plt.close()
        
        # B. Distribution Comparison
        dist_plot_path = os.path.join(model_dir, "distribution_comparison.png")
        plt.figure(figsize=(8, 4))
        sns.kdeplot(actuals, label='Actual Ratings', fill=True)
        sns.kdeplot(preds, label='Predicted Ratings', fill=True)
        plt.legend()
        plt.title('Rating Distribution: Actual vs Predicted')
        plt.savefig(dist_plot_path)
        mlflow.log_artifact(dist_plot_path)
        plt.close()

        # 6. Save Model Artifacts
        mlflow.sklearn.log_model(svd, "model")

        model_metadata = {
            "user_item_matrix_columns": user_item_matrix.columns,
            "user_item_matrix_index": user_item_matrix.index,
            "reconstructed_matrix": matrix_reconstructed
        }
        
        metadata_path = os.path.join(model_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(model_metadata, f)
        mlflow.log_artifact(metadata_path)

        print(f"Model & Plots saved to {model_dir}")
        print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train_sklearn_recommender()