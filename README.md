# RecoMart: End-to-End Recommendation Data Pipeline

## Project Overview
This project implements a complete data management pipeline for **RecoMart**, an e-commerce startup. The goal is to build a scalable and maintainable infrastructure that supports a recommendation engine by processing both historical batch data and real-time user behavior.

The architecture follows a **Lambda Architecture** pattern, splitting data processing into a **Batch Layer** for master datasets and a **Speed Layer** for near-real-time event processing.

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
│   ├── utils/                      # Utility scripts
│   │   ├── generate_source_data.py # Data generation utility
│   │   └── check_speed_layer_data.py # Speed layer verification
│   ├── 04_validate_data.py         # Step 4: Data validation
│   ├── 05_prepare_and_eda.py       # Step 5: Data preparation & EDA
│   ├── 06_feature_engineering.py   # Step 6: Feature engineering
│   ├── 07_feature_store.py         # Step 7: Feature store management
│   ├── 08_data_lineage.py          # Step 8: Data lineage tracking
│   ├── 09_train_and_evaluate_model.py           # Step 9: Model training
│   ├── 10_orchestrate_pipeline.py  # Pipeline orchestration
│   ├── ingest_master.py            # Step 2: Data ingestion
│   ├── speed_layer.py              # Step 2: Real-time processing
│   └── stream_simulator.py         # Step 2: Stream simulation
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
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Pipeline Steps

### Step 1: Data Generation (Optional)
Generate synthetic data for testing:
```bash
cd src/utils
python generate_source_data.py
```

### Step 2: Data Ingestion & Real-time Processing
This step includes three components that work together:

#### 2a. Stream Simulator (`stream_simulator.py`)
- Generates real-time clickstream events
- Simulates user interactions (views, clicks, add-to-cart)
- Outputs JSON events every 1.5 seconds

#### 2b. Speed Layer (`speed_layer.py`)
- Processes real-time events from stream simulator
- Stores events in SQLite database
- Handles real-time data ingestion

#### 2c. Batch Ingestion (`ingest_master.py`)
- Ingests transactional CSV data
- Fetches product metadata from API
- Stores data in partitioned data lake structure

**Manual Execution:**
```bash
# Terminal 1: Start real-time processing
cd src
python stream_simulator.py | python speed_layer.py

# Terminal 2: Run batch ingestion
python ingest_master.py
```

### Step 3: Mock API (Run separately if needed)
```bash
python mock_api.py
```

### Step 4: Data Validation (`04_validate_data.py`)
- Validates data against predefined schemas
- Generates data quality reports in PDF format
- Identifies missing values, duplicates, and range errors

```bash
cd src
python 04_validate_data.py
```

### Step 5: Data Preparation & EDA (`05_prepare_and_eda.py`)
- Cleans and prepares data for analysis
- Handles missing values and outliers
- Generates exploratory data analysis plots

```bash
python 05_prepare_and_eda.py
```

### Step 6: Feature Engineering (`06_feature_engineering.py`)
- Creates user and item features
- Calculates aggregated metrics
- Prepares features for machine learning

```bash
python 06_feature_engineering.py
```

### Step 7: Feature Store (`07_feature_store.py`)
- Stores engineered features in SQLite database
- Provides versioning and metadata tracking
- Enables feature retrieval for model training

```bash
python 07_feature_store.py
```

### Step 8: Data Lineage Tracking (`08_data_lineage.py`)
- Tracks data transformations and dependencies
- Maintains audit trail of data processing
- Generates lineage metadata

```bash
python 08_data_lineage.py
```

### Step 9: Model Training (`09_train_and_evaluate_model.py`)
- Trains recommendation model using scikit-learn
- Implements matrix factorization with TruncatedSVD
- Tracks experiments with MLflow
- Generates model performance visualizations

```bash
python 09_train_and_evaluate_model.py
```

### Step 10: Pipeline Orchestration (`10_orchestrate_pipeline.py`)
- Orchestrates the entire pipeline execution
- Manages Speed Layer and Batch Layer coordination
- Provides comprehensive logging and error handling

```bash
python 10_orchestrate_pipeline.py
```

---

## Complete Pipeline Execution

### Automated Execution (Recommended)
Run the complete pipeline with orchestration:
```bash
cd src
python 10_orchestrate_pipeline.py
```

This will:
1. Start the Speed Layer (runs for 2 minutes to generate streaming data)
2. Execute all batch processing steps in sequence
3. Handle errors and provide detailed logging
4. Clean up processes upon completion

### Manual Step-by-Step Execution
If you prefer to run steps individually:

1. **Generate test data:**
   ```bash
   cd src/utils
   python generate_source_data.py
   ```

2. **Start real-time processing:**
   ```bash
   cd src
   python stream_simulator.py | python speed_layer.py
   ```

3. **Run batch steps in order:**
   ```bash
   python ingest_master.py
   python 04_validate_data.py
   python 05_prepare_and_eda.py
   python 06_feature_engineering.py
   python 07_feature_store.py
   python 08_data_lineage.py
   python 09_train_and_evaluate_model.py
   ```

---

## Output Artifacts

After successful execution, you'll find:

- **Data Lake**: `recomart_lake/` with organized data storage
- **Logs**: Comprehensive audit trails in `recomart_lake/logs/`
- **Reports**: Data quality PDF reports in `recomart_lake/reports/`
- **Visualizations**: EDA plots in `recomart_lake/eda_plots/`
- **Models**: Trained models and metadata in `recomart_lake/models/`
- **MLflow Tracking**: Experiment tracking in SQLite database (`mlflow.db`)

---

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure virtual environment is activated and requirements are installed
2. **Path errors**: Run scripts from the correct directory (usually `src/`)
3. **Port conflicts**: Ensure port 5000 is available for mock API
4. **Unicode errors**: Ensure terminal supports UTF-8 encoding

### Verification Commands

Check Speed Layer data:
```bash
cd src/utils
python check_speed_layer_data.py
```

View MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Architecture Notes

- **Lambda Architecture**: Combines batch and real-time processing
- **Data Lake**: Partitioned storage by source, type, and timestamp
- **Feature Store**: Centralized feature management with versioning
- **MLflow Integration**: Comprehensive experiment tracking
- **Data Quality**: Automated validation and reporting
- **Lineage Tracking**: Full data transformation audit trail

---

## Next Steps

1. **Deployment**: Containerize with Docker for production deployment
2. **Scaling**: Implement with Apache Kafka for real-time streaming
3. **Monitoring**: Add comprehensive monitoring and alerting
4. **API**: Build REST API for model serving
5. **CI/CD**: Implement automated testing and deployment pipelines