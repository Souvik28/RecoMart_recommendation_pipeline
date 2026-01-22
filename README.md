# RecoMart: End-to-End Recommendation Data Pipeline

## Project Overview
This project implements a complete data management pipeline for **RecoMart**, an e-commerce startup. The goal is to build a scalable and maintainable infrastructure that supports a recommendation engine by processing both historical batch data and real-time user behavior.

The architecture follows a **Lambda Architecture** pattern, splitting data processing into a **Batch Layer** for master datasets and a **Speed Layer** for near-real-time event processing.

## Key Features
- **Hybrid Recommendation System**: Combines collaborative filtering (BiasedSVD) with content-based filtering
- **Cold Start Handling**: Uses product metadata for new users/items
- **Real-time Processing**: Streaming data integration with batch processing
- **Comprehensive Evaluation**: Proper train/test split with realistic metrics (RMSE ~0.52, Precision@5 ~0.11)
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Organized Architecture**: Semantic folder structure for maintainability

---

## Setup Instructions

### 1. Environment Setup

#### Create Virtual Environment
```bash
python -m venv recomart_env
```

#### Activate Virtual Environment
```bash
# Windows
recomart_env\Scripts\activate

# Linux/Mac
source recomart_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Requirements.txt Contents
```
# Core data processing
pandas

# HTTP requests for API calls
requests

# Data validation and schema enforcement
pandera

# PDF generation for reports
fpdf2

# Data visualization and plotting
matplotlib
seaborn

# Machine learning and recommendation systems
surprise
mlflow

# Note: The following are part of Python standard library
# but listed here for documentation purposes:
# - sqlite3 (database operations)
# - json (JSON data handling)
# - logging (audit trail logging)
# - datetime (timestamp operations)
# - pathlib (file path operations)
# - uuid (unique ID generation)
# - time (delays and timing)
# - random (data simulation)
# - sys (system operations)
# - os (operating system interface)
```

---

## Project Structure

```
RecoMart_recommendation_pipeline/
├── recomart_env/                    # Virtual environment
├── src/                            # Source code directory
│   ├── ingestion/                  # Data ingestion scripts
│   │   ├── ingest_master.py        # Batch & API data ingestion
│   │   ├── mock_api.py             # Product metadata API
│   │   ├── speed_layer.py          # Real-time data processing
│   │   └── stream_simulator.py     # Stream event simulation
│   ├── processing/                 # Data processing scripts
│   │   ├── 04_validate_data.py     # Data validation & quality
│   │   ├── 05_prepare_and_eda.py   # Data preparation & EDA
│   │   ├── 06_feature_engineering.py # Feature engineering
│   │   ├── 07_feature_store.py     # Feature store management
│   │   ├── 08_data_lineage.py      # Data lineage tracking
│   │   └── merge_streaming_data.py # Lambda architecture merge
│   ├── modeling/                   # Machine learning scripts
│   │   ├── 09_train_and_evaluate_model.py # BiasedSVD training
│   │   ├── generate_recommendations.py    # Hybrid recommendations
│   │   └── load_best_model.py      # Load specific MLflow models
│   ├── orchestration/              # Pipeline orchestration
│   │   └── 10_orchestrate_pipeline.py # Complete pipeline runner
│   └── utils/                      # Utility scripts
│       ├── generate_source_data.py # Data generation utility
│       ├── check_speed_layer_data.py # Speed layer verification
│       └── inspect_feature_store.py # Feature store inspection
├── recomart_lake/                  # Data lake (auto-generated)
│   ├── logs/                       # Audit logs
│   ├── raw/                        # Raw data storage
│   ├── processed/                  # Processed data
│   ├── feature_store/              # Feature store
│   ├── models/                     # Trained models
│   ├── reports/                    # Data quality reports
│   ├── eda_plots/                  # EDA visualizations
│   ├── metadata/                   # Data lineage metadata
│   └── speed_layer/                # Real-time data store
├── source_data/                    # Source data files
├── mlflow.db                       # MLflow tracking database
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── DEMO_SCRIPTS.md                 # Demo and testing scripts
```

---

## Complete Pipeline Execution

### Automated Execution (Recommended)
Run the complete pipeline with orchestration:
```bash
cd src/orchestration
python 10_orchestrate_pipeline.py
```

This will:
1. **Generate synthetic data** (if needed)
2. **Start Speed Layer** (runs for 2 minutes to generate streaming data)
3. **Execute batch processing** in sequence:
   - Data ingestion from CSV and API sources
   - **Merge streaming data with batch data** (Lambda Architecture)
   - **Data validation** on combined dataset
   - **Data preparation and EDA** with click aggregation
   - **Feature engineering** using combined data sources
   - **Feature store** with versioning
   - **Data lineage tracking** with streaming integration
   - **BiasedSVD model training** with proper train/test split
   - **Hybrid recommendation generation** (CF + Content-based)
4. **Handle errors** and provide detailed logging
5. **Clean up processes** upon completion

---

## Model Performance

### Achieved Metrics
- **RMSE**: ~0.52 (excellent prediction accuracy)
- **Precision@5**: ~0.11 (realistic and good for sparse data)
- **Recall@5**: Varies based on user behavior
- **F1-Score@5**: Balanced precision-recall performance

### Model Features
- **BiasedSVD**: Enhanced SVD with user/item bias handling
- **Hyperparameter Tuning**: Sparsity-aware parameter optimization
- **Proper Evaluation**: Train/test split prevents data leakage
- **Hybrid Approach**: 70% collaborative filtering + 30% content-based
- **Cold Start Handling**: Content-based recommendations for new users

---

## MLflow Integration

### Viewing Results
```bash
# From project root directory
mlflow ui
```

Access MLflow UI at: `http://localhost:5000`

### What You'll Find
- **Experiments**: All training runs with metrics and parameters
- **Models**: Registered BiasedSVD models with versioning
- **Artifacts**: Model files, plots, and metadata
- **Metrics Tracking**: RMSE, Precision@K, Recall@K, F1-Score@K

---

## Output Artifacts

After successful execution, you'll find:

- **Data Lake**: `recomart_lake/` with organized data storage
- **Logs**: Comprehensive audit trails in `recomart_lake/logs/`
- **Reports**: Data quality PDF reports in `recomart_lake/reports/`
- **Visualizations**: EDA plots in `recomart_lake/eda_plots/`
- **Models**: Trained models and metadata in `recomart_lake/models/`
- **MLflow Database**: `mlflow.db` with experiment tracking
- **Feature Store**: SQLite database with user/item features

---

## Architecture Highlights

### Lambda Architecture
- **Batch Layer**: Historical transaction processing
- **Speed Layer**: Real-time clickstream processing
- **Serving Layer**: Combined data for recommendations

### Recommendation System
- **Collaborative Filtering**: BiasedSVD with regularization
- **Content-Based**: Product metadata integration
- **Hybrid Scoring**: Weighted combination of both approaches
- **Cold Start**: Popularity-based recommendations for new users

### Data Quality
- **Validation**: Schema enforcement with Pandera
- **Monitoring**: Comprehensive logging and error handling
- **Lineage**: Full data transformation tracking
- **Evaluation**: Proper train/test methodology

---

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure virtual environment is activated
2. **Path errors**: Run scripts from correct directories
3. **MLflow UI issues**: Run `mlflow ui` from project root
4. **Unicode errors**: Ensure terminal supports UTF-8 encoding

### Verification Commands

```bash
# Check Speed Layer data
cd src/utils
python check_speed_layer_data.py

# Inspect feature store
python inspect_feature_store.py

# Load specific model
cd ../modeling
python load_best_model.py
```

---

## Next Steps

1. **Production Deployment**: Containerize with Docker
2. **Scaling**: Implement Apache Kafka for streaming
3. **Monitoring**: Add comprehensive alerting
4. **API Development**: Build REST API for model serving
5. **CI/CD**: Implement automated testing and deployment
6. **A/B Testing**: Framework for recommendation experiments