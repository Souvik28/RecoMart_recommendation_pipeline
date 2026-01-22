# RecoMart Pipeline Demo Script
## Duration: 3-5 minutes

---

## Introduction (30 seconds)
**[Show project folder structure]**

"Hello! Today I'll demonstrate RecoMart - an end-to-end recommendation data pipeline that I built using Lambda Architecture. This project processes both real-time user behavior and historical batch data to support a recommendation engine for an e-commerce startup."

**[Open README.md briefly]**

"The pipeline follows industry best practices with data validation, feature engineering, model training, and comprehensive logging."

---

## Project Structure Overview (45 seconds)
**[Navigate through folder structure]**

"Let me show you the project organization:
- **src/** contains all our Python pipeline steps
- **src/utils/** has utility scripts for data generation
- **recomart_lake/** will store all our processed data using a data lake pattern
- **requirements.txt** lists all dependencies
- **Virtual environment** keeps everything isolated"

**[Show key files]**
- "Step 2 handles real-time streaming with stream_simulator.py and speed_layer.py"
- "Steps 4-9 are our main batch processing pipeline"
- "Step 10 orchestrates everything together"

---

## Environment Setup (30 seconds)
**[Open terminal]**

"First, let's activate our virtual environment and ensure dependencies are installed:"

```bash
recomart_env\Scripts\activate
pip install -r requirements.txt
```

**[Show requirements.txt briefly]**
"We're using pandas for data processing, MLflow for experiment tracking, pandera for data validation, and scikit-learn for machine learning."

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
**[Navigate to src directory]**

"Now for the main event - let's run our complete Lambda Architecture pipeline:"

```bash
cd ../
python 10_orchestrate_pipeline.py
```

**[While pipeline runs, explain what's happening]**

"Watch the orchestration in action:

1. **Speed Layer Launch**: The pipeline starts real-time stream simulation and processing
   - stream_simulator.py generates clickstream events every 1.5 seconds
   - speed_layer.py processes these events into SQLite database

2. **Batch Processing**: After 2 minutes of streaming data collection, batch steps execute:
   - **Step 2**: Data ingestion from CSV and API sources
   - **Step 4**: Data validation with schema enforcement and PDF reporting
   - **Step 5**: Data preparation and exploratory data analysis
   - **Step 6**: Feature engineering and feature store management
   - **Step 7**: Feature store management with SQLite database
   - **Step 8**: Data lineage tracking for audit trails
   - **Step 9**: Model training and evaluation with comprehensive plots

Each step processes data and stores results in our partitioned data lake structure."

---

## Results Showcase (60 seconds)
**[Show generated artifacts]**

"Let's explore what the pipeline created:

**[Navigate to recomart_lake]**
- **Data Lake Structure**: Organized by source, type, and date partitions
- **Logs**: Comprehensive audit trails for every operation
- **Reports**: PDF data quality reports with validation results
- **EDA Plots**: Visualizations showing data distributions
- **Feature Store**: SQLite database with versioned features
- **Models**: Trained SVD recommendation model with performance plots and metadata

**[Open MLflow UI if time permits]**
```bash
mlflow ui
```
"MLflow tracks our experiments with metrics like RMSE, Precision@5, and model artifacts including performance visualizations."

---

## Architecture Highlights (30 seconds)
**[Show key architectural decisions]**

"Key architectural features:
- **Lambda Architecture**: Combines real-time and batch processing
- **Data Lake**: Partitioned storage with proper organization
- **Feature Store**: Centralized feature management with versioning
- **Data Quality**: Automated validation and reporting
- **Experiment Tracking**: MLflow integration for model lifecycle
- **Audit Trail**: Complete data lineage tracking"

---

## Conclusion (15 seconds)
**[Show final project structure]**

"This demonstrates a production-ready data pipeline that handles:
- Real-time streaming data
- Batch data processing
- Data quality validation
- Feature engineering
- Model training and tracking
- Comprehensive logging and monitoring

Perfect foundation for scaling an e-commerce recommendation system!"

---

## Demo Tips:

### Before Recording:
1. **Clean environment**: Delete any existing recomart_lake/ and mlruns/ folders
2. **Test run**: Execute the pipeline once to ensure everything works
3. **Prepare windows**: Have terminal, file explorer, and code editor ready
4. **Check timing**: Practice to stay within 3-5 minutes

### During Recording:
1. **Speak clearly** and at moderate pace
2. **Show, don't just tell** - navigate through actual files and outputs
3. **Highlight key concepts** - Lambda Architecture, data lake, feature store
4. **Keep terminal visible** - let viewers see the pipeline execution logs
5. **Point out industry practices** - data validation, experiment tracking, audit trails

### Updated Pipeline Features:
- **Enhanced Model Training**: Now includes 7 comprehensive evaluation plots
- **Improved Visualizations**: Hexbin plots, KDE distributions, error analysis, and performance dashboards
- **Better Screen Fit**: All plots optimized for standard screen sizes
- **Expanded User/Product Range**: 200 users (U001-U200) and 50 products (P101-P150)
- **Comprehensive Evaluation**: Precision@K, Recall@K, residual analysis, and rating distribution comparisons
- Pipeline takes ~3-4 minutes to complete
- Speed Layer generates ~200-300 events in 2 minutes
- All outputs are automatically organized in recomart_lake/
- MLflow UI can be shown at the end if time permits

### Backup Plan:
If pipeline fails during recording, you can show pre-generated results and explain the architecture using the folder structure and code files.