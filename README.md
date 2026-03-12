# E-Commerce Customer Churn Prediction: End-to-End ML Pipeline with XGBoost, MLflow & Gradio

A complete machine learning and data engineering project demonstrating modern MLOps best practices. This project implements a full ML pipeline from raw data ingestion through feature engineering to interactive predictions.

## Project Overview

This project predicts customer churn for an e-commerce platform using real customer behavior data. It showcases:

- **Data Engineering** with Pandas & Great Expectations
- **Feature Engineering** with scikit-learn
- **Model Training** with XGBoost
- **Experiment Tracking** with MLflow
- **Interactive Predictions** with FastAPI + Gradio

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            PROJECT ARCHITECTURE                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Raw Data          Data Pipeline        ML Training           Model Serving     │
│   (CSV)             (Pandas, GX)         (XGBoost)             (FastAPI+Gradio)  │
│                                                                                  │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐      ┌──────────────┐  │
│   │E-Commerce    │   │Load & Clean  │   │Feature       │      │REST API      │  │
│   │Customer Data │──>│Validate      │──>│Engineering  │─────>│Endpoint      │  │
│   │(5,630 rows)  │   │Preprocess    │   │(30 features)│      │              │  │
│   └──────────────┘   └──────────────┘   └──────┬──────┘      │Web UI        │  │
│                                                 │             │(Interactive) │  │
│                                                 v             │              │  │
│                                        ┌──────────────────┐   │MLflow        │  │
│                                        │XGBoost Classifier│──>│Tracking      │  │
│                                        │(Trained Model)   │   └──────────────┘  │
│                                        │ROC AUC: 0.998    │                     │
│                                        └──────────────────┘                     │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core language & scripting | 3.12 |
| **Pandas** | Data manipulation & analysis | 2.1.4 |
| **scikit-learn** | Feature engineering & utilities | 1.5.2 |
| **XGBoost** | Gradient boosting classifier | 3.0.3 |
| **MLflow** | Experiment tracking & model registry | 2.14.1 |
| **Great Expectations** | Data quality validation | 1.5.8 |
| **FastAPI** | REST API framework | 0.115.0 |
| **Gradio** | Interactive web interface | 6.9.0 |
| **Uvicorn** | ASGI server | 0.30.5 |

---

## Data Source

The E-Commerce Customer Churn dataset is a real-world e-commerce customer analytics dataset from Kaggle.

### Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Records** | 5,630 customers |
| **Features** | 20 raw attributes |
| **Target Variable** | Churn (binary: 0/1) |
| **Churn Rate** | 16.84% (948 churned customers) |
| **Missing Values** | Handled in preprocessing |

### Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Demographics** | Gender, MaritalStatus | Customer personal attributes |
| **Engagement** | Tenure, HourSpendOnApp, NumberOfDeviceRegistered | Platform usage metrics |
| **Location** | CityTier, WarehouseToHome | Geographic & logistics data |
| **Purchase Behavior** | OrderCount, OrderAmountHike, CouponUsed, DaySinceLastOrder | Transaction patterns |
| **Satisfaction** | SatisfactionScore, Complain | Customer feedback |
| **Preferences** | PreferredLoginDevice, PreferredPaymentMode, PreferedOrderCat | User choices |
| **Monetization** | CashbackAmount | Loyalty rewards |

### Data Quality Validation

All source data includes automated quality checks:
- **Schema Validation**: Required columns and data types
- **Business Logic Tests**: Valid ranges for tenure (0-61 months), satisfaction (1-5), etc.
- **Consistency Checks**: Logical relationships between features
- **Results**: 21/21 validation checks pass ✅

---

## Data Engineering Pipeline

### Pipeline Architecture

```
RAW DATA INPUT
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA LOADING & VALIDATION                          │
├─────────────────────────────────────────────────────────────┤
│ • Load CSV dataset (5,630 rows × 20 columns)                │
│ • Run Great Expectations quality checks                      │
│ • Validate schema, business logic, numeric ranges            │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: DATA PREPROCESSING                                 │
├─────────────────────────────────────────────────────────────┤
│ • Handle missing values (median imputation for numeric)      │
│ • Mode imputation for categorical features                  │
│ • Remove unnecessary ID columns                             │
│ • Standardize column names and data types                   │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: FEATURE ENGINEERING                                │
├─────────────────────────────────────────────────────────────┤
│ • Identify binary vs multi-category features                │
│ • Binary encoding: Gender (Female=0, Male=1)                │
│ • One-hot encoding for 4 categorical features               │
│ • Generate 30 final ML-ready features                       │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: MODEL TRAINING                                     │
├─────────────────────────────────────────────────────────────┤
│ • Train/test split (80/20)                                  │
│ • Handle class imbalance (scale_pos_weight = 4.94)          │
│ • Train XGBoost classifier                                  │
│ • Evaluate with precision, recall, F1, ROC-AUC             │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 5: MODEL LOGGING TO MLflow                            │
├─────────────────────────────────────────────────────────────┤
│ • Log model artifacts (model.pkl, feature_columns.txt)      │
│ • Log hyperparameters and metrics                           │
│ • Track experiment lineage and reproducibility              │
└─────────────────────────────────────────────────────────────┘
      ↓
TRAINED MODEL READY FOR SERVING
```

### Key Transformations

#### Binary Feature Encoding
```
Gender: {'Female': 0, 'Male': 1}
Complain: {0: 0, 1: 1}
```

#### Categorical Feature One-Hot Encoding
```
PreferredLoginDevice → PreferredLoginDevice_Phone, PreferredLoginDevice_Computer
PreferredPaymentMode → PaymentMode_UPI, PaymentMode_CC, PaymentMode_E wallet
PreferedOrderCat → OrderCat_Mobile, OrderCat_Others, OrderCat_Grocery
MaritalStatus → MaritalStatus_Single, MaritalStatus_Married
```

#### Missing Value Handling
```
Numeric columns: Median imputation (robust to outliers)
Categorical columns: Mode imputation (most frequent value)
Critical fields: Checked for NULL values (none allowed)
```

---

## Machine Learning Model

### Model Performance

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **Precision** | 0.855 | 85.5% of predicted churners actually churn |
| **Recall** | 0.995 | Catches 99.5% of actual churners |
| **F1 Score** | 0.920 | Excellent balance between precision & recall |
| **ROC AUC** | 0.998 | Outstanding model discrimination |
| **Accuracy** | 97.1% | Correct predictions on test set |

### Classification Report

```
              precision    recall  f1-score   support

Not Churn         0.999     0.966     0.982       936
Churn             0.855     0.995     0.920       190

accuracy                              0.971      1126
macro avg         0.927     0.980     0.951      1126
weighted avg      0.975     0.971     0.972      1126
```

### Model Configuration

```python
XGBClassifier(
    n_estimators=301,           # Number of boosting rounds
    learning_rate=0.034,        # Shrinkage parameter
    max_depth=7,                # Tree depth
    scale_pos_weight=4.94,      # Handle class imbalance
    objective='binary:logistic',
    eval_metric='logloss'
)
```

**Training Performance**:
- Training Time: 0.59 seconds
- Inference Time: 0.0041 seconds per sample
- Throughput: ~272,254 predictions/second

### Feature Importance (Top 10)

The model learns that customer churn is driven by:

1. **DaySinceLastOrder** - Inactivity is strongest churn indicator
2. **Tenure** - Newer customers at higher risk
3. **SatisfactionScore** - Low satisfaction = high churn
4. **OrderCount** - Frequent buyers stay longer
5. **CashbackAmount** - Engagement metric
6. **HourSpendOnApp** - App usage indicates stickiness
7. **OrderAmountHikeFromlastYear** - Spending trends matter
8. **CityTier** - Geographic location impact
9. **Complain** - Customer issues correlate with churn
10. **WarehouseToHome** - Logistics performance matters

---

## Model Serving & Interactive Predictions

### Serving Architecture

The model is served through a dual-interface system:

```
┌────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────────┐         ┌──────────────────────┐  │
│  │   REST API          │         │   Gradio Web UI      │  │
│  │   /predict          │         │   /ui                │  │
│  │   (JSON)            │         │   (Interactive Form) │  │
│  └──────────┬──────────┘         └──────────┬───────────┘  │
│             │                                │              │
│             └────────────────┬───────────────┘              │
│                              │                              │
│                    ┌─────────▼────────┐                     │
│                    │ Inference Pipeline│                     │
│                    │ (Feature Transform│                     │
│                    │  + XGBoost Model) │                     │
│                    └─────────────────┘                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### REST API Endpoints

#### Health Check
```bash
GET /
Response: {"status": "ok"}
```

#### Prediction Endpoint
```bash
POST /predict
Content-Type: application/json

{
    "Gender": "Male",
    "MaritalStatus": "Single",
    "Tenure": 10.0,
    "HourSpendOnApp": 2.5,
    "NumberOfDeviceRegistered": 2,
    "NumberOfAddress": 3,
    "CityTier": 2,
    "WarehouseToHome": 30.0,
    "PreferedOrderCat": "Laptop & Accessory",
    "OrderCount": 5.0,
    "OrderAmountHikeFromlastYear": 15.0,
    "CouponUsed": 2.0,
    "DaySinceLastOrder": 10.0,
    "CashbackAmount": 100.0,
    "PreferredLoginDevice": "Mobile Phone",
    "PreferredPaymentMode": "Debit Card",
    "SatisfactionScore": 4,
    "Complain": 0
}

Response: {"prediction": "Not likely to churn"}
```

### Gradio Web Interface

Interactive prediction form with:

- **18 input fields** matching customer attributes
- **Dropdown selections** for categorical features (Gender, Marital Status, Device, etc.)
- **Numeric inputs** for continuous features (Tenure, HourSpendOnApp, etc.)
- **Pre-filled examples** for high-risk and low-risk customers
- **Real-time predictions** with business-friendly output

**URL**: `http://localhost:8000/ui`
<img width="1426" height="785" alt="image" src="https://github.com/user-attachments/assets/b273fd24-cc03-4b60-a117-9b34c7ea67dd" />


---

## MLflow Experiment Tracking

### Tracked Artifacts

For each training run, MLflow logs:

| Artifact | Description |
|----------|-------------|
| **model.pkl** | Serialized XGBoost model |
| **feature_columns.txt** | Exact feature order (ensures train/serve consistency) |
| **preprocessing.pkl** | Preprocessing pipeline |

### Tracked Metrics

| Metric | Purpose |
|--------|---------|
| precision | Positive prediction accuracy |
| recall | Coverage of actual churners |
| f1 | Harmonic mean of precision & recall |
| roc_auc | Area under ROC curve |
| train_time | Training execution time |
| pred_time | Prediction latency |
| data_quality_pass | Validation check results |

### Experiment Navigation

```bash
# View all experiments and runs
mlflow ui --backend-store-uri file:./mlruns

# Access at: http://localhost:5000
```

---

## Project Structure

```
E-Commerce-Customer-Churn-ML/
│
├── data/
│   ├── raw/
│   │   └── E-Commerce-Customer-Churn.csv    # Source dataset
│   └── processed/
│       └── telco_churn_processed.csv        # Cleaned dataset
│
├── src/
│   ├── data/
│   │   ├── load_data.py                     # CSV loading with error handling
│   │   └── preprocess.py                    # Data cleaning & imputation
│   ├── features/
│   │   └── build_features.py                # Feature engineering pipeline
│   ├── utils/
│   │   └── validate_data.py                 # Great Expectations validation
│   ├── app/
│   │   └── main.py                          # FastAPI + Gradio server
│   └── serving/
│       └── inference.py                     # Model loading & predictions
│
├── scripts/
│   ├── run_pipeline.py                      # Complete training orchestration
│   ├── test_pipeline_phase1_data_features.py # Data/feature tests
│   ├── test_pipeline_phase2_modeling.py     # Model tests
│   └── test_fastapi.py                      # API endpoint tests
│
├── mlruns/                                  # MLflow experiment tracking
│   └── [experiment_id]/
│       └── [run_id]/
│           ├── artifacts/
│           │   ├── model/
│           │   │   ├── model.pkl
│           │   │   ├── MLmodel
│           │   │   └── requirements.txt
│           │   └── feature_columns.txt
│           └── metrics/
│
├── .venv/                                   # Python virtual environment
├── requirements.txt                         # Python dependencies
│
├── start_ui.py                              # Quick start script (cross-platform)
├── start_ui.sh                              # Quick start script (macOS/Linux)
├── start_ui.bat                             # Quick start script (Windows)
│
├── README.md                                # This file
└── CLAUDE.md                                # Development guide
```

---

## Project Statistics

| Metric | Count |
|--------|-------|
| **Python Scripts** | 8 |
| **Source Features** | 20 |
| **Engineered Features** | 30 |
| **Data Validation Checks** | 21 |
| **Training Samples** | 4,504 |
| **Test Samples** | 1,126 |
| **Model Hyperparameters** | 6 |
| **Training Time** | 0.59s |
| **Model Performance** | ROC AUC: 0.998 |

---

## Key Technical Achievements

### 1. End-to-End Pipeline Automation

Complete automated ML pipeline with single command:
```bash
python scripts/run_pipeline.py
```

Orchestrates:
- Data loading and validation
- Preprocessing and feature engineering
- Model training and evaluation
- MLflow logging and artifact storage

### 2. Train/Serve Consistency

Identical feature transformations at training and serving time:
- Binary encoding with deterministic mappings
- One-hot encoding with `drop_first=True`
- Feature column order enforcement via `feature_columns.txt`

```python
# Ensures serving predictions match training predictions
df = df.reindex(columns=FEATURE_COLS, fill_value=0)
```

### 3. Data Quality Validation

Great Expectations integration with 21 validation checks:
- Schema validation (required columns, types)
- Business logic constraints (valid ranges, categories)
- Numeric range validation (Tenure 0-61, Score 1-5)
- Consistency checks (data relationships)

### 4. Class Imbalance Handling

Addressed skewed target distribution (16.84% churn):
```python
scale_pos_weight = len(no_churn) / len(churn)  # 4.94
```

Achieved 99.5% recall on minority class without sacrificing precision.

### 5. Flexible Model Loading

Intelligent model discovery across environments:
```python
# Auto-discover latest MLflow model from local runs
latest_model = sorted(glob.glob("./mlruns/*/models/*/artifacts"),
                     key=os.path.getmtime)[-1]
```

---

## How to Run

### Prerequisites

- Python 3.12+
- Git
- ~900MB disk space for virtual environment

### Setup

1. **Clone and navigate to project**
```bash
cd E-Commerce-Customer-Churn-ML
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Training

4. **Run complete training pipeline**
```bash
# Automatically loads E-Commerce dataset
python scripts/run_pipeline.py

# Or use custom dataset
python scripts/run_pipeline.py --input path/to/data.csv --experiment "My Experiment"
```

**Output**:
- Trained model logged to MLflow
- Feature columns saved for serving
- Validation metrics printed to console

### Serving & Predictions

5. **Start the web application (choose one option)**

#### Option A: Quick Start (Recommended) 🚀
```bash
python start_ui.py
```
This automatically:
- Creates virtual environment if needed
- Installs dependencies
- Activates venv
- Launches the application

Also available:
- **macOS/Linux**: `./start_ui.sh`
- **Windows**: `start_ui.bat`

#### Option B: Manual Start
```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

6. **Access interactive interface**
- **Web UI**: http://localhost:8000/ui ← **Make predictions here**
- **REST API Docs**: http://localhost:8000/docs
- **API Server**: http://localhost:8000

### Experiment Tracking

7. **View MLflow dashboard** (optional)
```bash
mlflow ui --backend-store-uri file:./mlruns
# Access at: http://localhost:5000
```

---

## Usage Examples

### Python API

```python
from src.serving.inference import predict

customer_data = {
    'Gender': 'Female',
    'MaritalStatus': 'Single',
    'Tenure': 5.0,
    'HourSpendOnApp': 1.0,
    'NumberOfDeviceRegistered': 1,
    'NumberOfAddress': 2,
    'CityTier': 3,
    'WarehouseToHome': 50.0,
    'PreferedOrderCat': 'Mobile',
    'OrderCount': 1.0,
    'OrderAmountHikeFromlastYear': 12.0,
    'CouponUsed': 0.0,
    'DaySinceLastOrder': 30.0,
    'CashbackAmount': 50.0,
    'PreferredLoginDevice': 'Mobile Phone',
    'PreferredPaymentMode': 'Cash on Delivery',
    'SatisfactionScore': 2,
    'Complain': 1,
}

prediction = predict(customer_data)
print(prediction)  # Output: "Likely to churn"
```

### REST API (curl)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "MaritalStatus": "Married",
    "Tenure": 40.0,
    ...
}'

# Response: {"prediction": "Not likely to churn"}
```

---

## Web UI Examples & Interactive Predictions

### Prediction Interface

The Gradio web interface provides an intuitive form for making predictions on new customers. Fill in the customer attributes and instantly receive churn predictions.

**Access**: `http://localhost:8000/ui`

### Example 1: High-Risk Customer (Likely to Churn)

**Scenario**: New customer with low engagement and no satisfaction

| Feature | Value |
|---------|-------|
| Gender | Female |
| Marital Status | Single |
| Tenure | 5 months |
| Hour Spend On App | 1.0 hours |
| Number of Devices | 1 |
| Number of Addresses | 2 |
| City Tier | 3 |
| Warehouse to Home | 50 km |
| Preferred Order Category | Mobile |
| Order Count | 1 |
| Order Amount Hike | 12% |
| Coupon Used | 0 |
| Days Since Last Order | **30 days** ⚠️ |
| Cashback Amount | $50 |
| Preferred Login Device | Mobile Phone |
| Preferred Payment Mode | Cash on Delivery |
| Satisfaction Score | **2/5** ⚠️ |
| Customer Complained | **Yes** ⚠️ |

**Model Prediction**: 🔴 **"Likely to churn"**

**Why**: This customer is at high risk because:
- Only 1 order (low engagement)
- 30+ days of inactivity
- Low satisfaction score (2/5)
- Filed a complaint
- New customer (5 months tenure)
- Single transaction pattern

**Recommended Action**:
- Send immediate re-engagement offer
- Resolve complaint with follow-up
- Offer loyalty reward or discount
- Personalized product recommendations

---

### Example 2: Low-Risk Customer (Not Likely to Churn)

**Scenario**: Established customer with high engagement and satisfaction

| Feature | Value |
|---------|-------|
| Gender | Male |
| Marital Status | Married |
| Tenure | **40 months** ✅ |
| Hour Spend On App | **4.0 hours** ✅ |
| Number of Devices | **4** ✅ |
| Number of Addresses | **8** ✅ |
| City Tier | 1 |
| Warehouse to Home | 20 km |
| Preferred Order Category | Laptop & Accessory |
| Order Count | **12** ✅ |
| Order Amount Hike | **20%** ✅ |
| Coupon Used | **5** ✅ |
| Days Since Last Order | **5 days** ✅ |
| Cashback Amount | **$200** ✅ |
| Preferred Login Device | Computer |
| Preferred Payment Mode | Credit Card |
| Satisfaction Score | **5/5** ✅ |
| Customer Complained | **No** ✅ |

**Model Prediction**: 🟢 **"Not likely to churn"**

**Why**: This customer is loyal because:
- Long tenure (40 months) - established customer
- High app engagement (4 hours daily)
- Frequent orders (12 total)
- Very recent purchase (5 days ago)
- Highest satisfaction score (5/5)
- No complaints
- Multiple devices and addresses (multi-location user)
- Significant cashback earned

**Recommended Action**:
- Recognize as VIP customer
- Offer exclusive premium products
- Invite to loyalty program perks
- Send early access to new collections

---

### Feature Importance in Predictions

Based on model analysis, these features have the strongest influence on churn predictions:

| Rank | Feature | Impact | Threshold |
|------|---------|--------|-----------|
| 🥇 | Days Since Last Order | **Very High** | >20 days = high risk |
| 🥈 | Tenure | **Very High** | <10 months = higher risk |
| 🥉 | Satisfaction Score | **High** | <3 = churn indicator |
| 4️⃣ | Order Count | **High** | <5 orders = less sticky |
| 5️⃣ | Hour Spend On App | **Medium** | <2 hours = low engagement |

---

## Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Machine Learning** | Classification, feature engineering, hyperparameter tuning, model evaluation |
| **Data Engineering** | Data loading, validation, preprocessing, pipeline orchestration |
| **Python** | Pandas, scikit-learn, XGBoost, MLflow, Pydantic |
| **Web Development** | FastAPI, Gradio, REST APIs, interactive interfaces |
| **Data Quality** | Great Expectations, validation testing, data profiling |
| **MLOps** | Experiment tracking, model versioning, reproducibility |
| **Software Engineering** | Modular code, error handling, documentation, testing |

---

## Future Enhancements

- [ ] Implement feature store for production features
- [ ] Add A/B testing framework for model updates
- [ ] Create data drift detection pipeline
- [ ] Build customer retention recommendation engine
- [ ] Deploy to cloud (AWS SageMaker / GCP Vertex AI)
- [ ] Add explainability with SHAP values
- [ ] Implement automated retraining schedule
- [ ] Create performance monitoring dashboards

---

## Author & Attribution

**Nikolas** - Data Scientist & ML Engineer

### Project Credits

This project was **adapted** to use a real-world E-Commerce Customer Churn dataset, building upon the foundational MLOps architecture and pipeline design created by **Anas Riad**. The core concepts of:
- End-to-end ML pipeline orchestration
- MLflow experiment tracking integration
- FastAPI + Gradio serving architecture
- Feature engineering patterns
- Model evaluation and validation

...were inspired by Anas Riad's original work and adapted here with:
- A different but similar e-commerce dataset (replacing Telco Churn)
- Customized feature engineering for e-commerce domain
- Updated validation rules and business logic constraints
- E-commerce specific API schema and web interface

The objective was to create a **unique portfolio project** by applying proven ML engineering patterns to a new domain, demonstrating the ability to adapt and transfer MLOps best practices across different datasets and problem spaces.

---

## License

This project is for educational and portfolio purposes.
