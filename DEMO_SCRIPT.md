# RecoMart Pipeline Demo Script
## Duration: 3-5 minutes

---

## Introduction (30 seconds)
**[Show project folder structure]**

"Hello! Today I'll demonstrate RecoMart - an end-to-end recommendation data pipeline that I built using Lambda Architecture. This project processes both real-time user behavior and historical batch data to support a recommendation engine for an e-commerce startup."

**[Open README.md briefly]**

"The pipeline follows industry best practices with data validation, feature engineering, model training, and comprehensive logging. It achieves realistic metrics: RMSE ~0.52 and Precision@5 ~0.11."

---

## Project Structure Overview (45 seconds)
**[Navigate through folder structure]**

"Let me show you the organized project structure:
- **src/ingestion/** handles data ingestion and real-time streaming
- **src/processing/** contains data validation, preparation, and feature engineering
- **src/modeling/** includes BiasedSVD training and hybrid recommendations
- **src/orchestration/** orchestrates the complete pipeline
- **src/utils/** has utility scripts for data generation and inspection
- **recomart_lake/** stores all processed data using a data lake pattern
- **mlflow.db** tracks experiments and model performance"

**[Show key files]**
- "stream_simulator.py and speed_layer.py handle real-time streaming"
- "merge_streaming_data.py implements true Lambda Architecture"
- "09_train_and_evaluate_model.py includes proper train/test split and comprehensive evaluation"
- "10_orchestrate_pipeline.py orchestrates everything with error handling"

---

## Environment Setup (30 seconds)
**[Open terminal]**

"First, let's activate our virtual environment and ensure dependencies are installed:"

```bash
recomart_env\Scripts\activate
pip install -r requirements.txt
```

**[Show requirements.txt briefly]**
"We're using pandas for data processing, MLflow for experiment tracking, pandera for data validation, surprise for recommendations, and matplotlib for visualizations."

---

## Data Generation (30 seconds)
**[Navigate to src/utils]**

"Let's generate some synthetic e-commerce data for our demo:"

```bash
cd src/utils
python generate_source_data.py
```

**[Show generated files]**
"This creates realistic transaction data with intentional quality issues like missing values and duplicates - perfect for testing our data validation pipeline."

---

## Pipeline Execution (90 seconds)
**[Navigate to src/orchestration directory]**

"Now for the main event - let's run our complete Lambda Architecture pipeline:"

```bash
cd ../orchestration
python 10_orchestrate_pipeline.py
```

**[While pipeline runs, explain what's happening]**

"Watch the orchestration in action:

1. **Speed Layer Launch**: The pipeline starts real-time stream simulation
   - stream_simulator.py generates clickstream events every 1.5 seconds
   - speed_layer.py processes these events into SQLite database
   - Runs for 1 minute to generate streaming data

2. **Batch Processing**: Sequential execution of all pipeline stages:
   - **Data Ingestion**: CSV and API data ingestion
   - **Lambda Architecture**: Combines real-time events with batch data
   - **Data Validation**: Schema enforcement with PDF quality reports
   - **Data Preparation**: EDA with click aggregation (70% rating + 30% interaction)
   - **Feature Engineering**: User/item features with content metadata
   - **Feature Store**: SQLite database with versioned features
   - **Data Lineage**: Complete transformation tracking
   - **Model Training**: BiasedSVD with proper train/test split and hyperparameter tuning
   - **Recommendations**: Hybrid collaborative filtering + content-based with cold start handling

Each step processes combined streaming + batch data for true Lambda Architecture."

---

## Results Showcase (60 seconds)
**[Show generated artifacts]**

"Let's explore what the pipeline created:

**[Navigate to recomart_lake]**
- **Data Lake Structure**: Organized by source, type, and date partitions
- **Logs**: Comprehensive audit trails in logs/orchestration/
- **Reports**: PDF data quality reports with validation results
- **EDA Plots**: Rating distributions, user patterns, interaction analysis
- **Feature Store**: SQLite database with user/item features
- **Models**: BiasedSVD model with comprehensive evaluation plots

**[Show MLflow UI]**
```bash
mlflow ui
```
"MLflow tracks experiments with realistic metrics:
- RMSE: ~0.52 (excellent prediction accuracy)
- Precision@5: ~0.11 (realistic for sparse data)
- Multiple K values (1, 3, 5, 10) for comprehensive evaluation
- Artifacts include 7 evaluation plots: variance, quality metrics, prediction scatter, rating distributions, error analysis, matrix heatmap, and performance summary"

---

## Model Performance Demo (45 seconds)
**[Show model training output or MLflow results]**

"Key model achievements:
- **No Data Leakage**: Proper train/test split with holdout evaluation
- **Sparsity Analysis**: Matrix density checks for meaningful Precision@K
- **Hyperparameter Tuning**: Sparsity-aware parameter optimization
- **Hybrid Approach**: 70% collaborative filtering + 30% content-based
- **Cold Start Handling**: Content-based recommendations for new users
- **Comprehensive Evaluation**: Multiple thresholds (3.5, 4.5) and K values

**[Show recommendation generation]**
```bash
cd src/modeling
python generate_recommendations.py
```
"Generates recommendations for existing users (hybrid scores 3-5) and new users (content-based scores 0.5-0.9)."

---

## Architecture Highlights (30 seconds)
**[Show key architectural decisions]**

"Key architectural features:
- **Lambda Architecture**: Real-time streaming automatically merged throughout pipeline
- **Organized Structure**: Semantic folders (ingestion/, processing/, modeling/, orchestration/)
- **Data Quality**: Automated validation with sparsity checks
- **Feature Store**: Centralized management with content metadata integration
- **Experiment Tracking**: MLflow with comprehensive model versioning
- **Proper Evaluation**: Train/test split prevents overfitting (no more Precision=1.0)
- **Cold Start Solution**: Product metadata API for new user recommendations
- **Error Handling**: Unicode-safe logging and comprehensive error management"

---

## Conclusion (15 seconds)
**[Show final project structure]**

"This demonstrates a production-ready recommendation pipeline with:
- Realistic evaluation metrics (RMSE 0.52, Precision@5 0.11)
- Proper train/test methodology preventing data leakage
- Hybrid recommendations solving cold start problems
- Complete Lambda Architecture with streaming integration
- MLflow experiment tracking with comprehensive artifacts
- Industry-standard data quality and lineage tracking

Perfect foundation for scaling an e-commerce recommendation system!"

---

## Demo Tips:

### Before Recording:
1. **Clean environment**: Delete existing recomart_lake/ and mlflow.db
2. **Test run**: Execute pipeline once to ensure everything works
3. **Check MLflow**: Verify experiments show realistic metrics (not 1.0 precision)
4. **Prepare windows**: Terminal, file explorer, MLflow UI ready

### During Recording:
1. **Highlight realistic metrics**: Emphasize RMSE ~0.52, Precision@5 ~0.11
2. **Show train/test split**: Point out "Model trained ONLY on training matrix"
3. **Demonstrate cold start**: Show non-zero scores for new users
4. **Navigate organized structure**: Show semantic folder organization
5. **MLflow integration**: Display experiments, artifacts, and evaluation plots

### Key Technical Points:
- **No Data Leakage**: Proper holdout evaluation with realistic metrics
- **Sparsity Analysis**: Matrix density checks for meaningful evaluation
- **Hybrid Recommendations**: CF + content-based with cold start handling
- **Lambda Architecture**: Streaming data integrated throughout pipeline
- **Comprehensive Evaluation**: 7 evaluation plots with multiple metrics
- **Error Handling**: Unicode-safe, Windows-compatible logging

### Expected Outputs:
- **Pipeline Runtime**: 3-5 minutes total
- **RMSE**: 0.4-0.6 range (excellent performance)
- **Precision@5**: 0.08-0.15 range (realistic for sparse data)
- **Cold Start Scores**: 0.5-0.9 (content-based recommendations)
- **MLflow Experiments**: Multiple runs with comprehensive artifacts

### Troubleshooting:
- If Precision=1.0: Check for "Data Leakage Check" output
- If Cold Start=0.0: Verify content features loaded
- If Unicode errors: Ensure terminal encoding set properly
- If MLflow UI empty: Run from project root directory