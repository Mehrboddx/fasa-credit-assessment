# 💰 Fasanara Credit Assessment System

An AI-powered credit risk modeling platform that combines structured financial metrics with narrative analysis using sentence embeddings and Claude-generated credit memos.

## 🎯 Overview

This system provides:
- **Automated credit risk scoring** using machine learning (Logistic Regression, Random Forest, XGBoost)
- **Narrative analysis** via 384D sentence embeddings (all-MiniLM-L6-v2)
- **AI-generated credit memos** using Claude Sonnet API
- **Interactive web interface** built with Streamlit
- **Nested cross-validation** with optional hyperparameter tuning (Optuna)
- **Embedding fine-tuning** with contrastive learning

## 📋 Features

### Data Processing
- ✅ Loads company financials, narratives, and default labels
- ✅ Parses risk/benefit signals from text descriptions
- ✅ Engineers 7+ quantitative features (debt service risk, profitability, leverage, etc.)
- ✅ Target encodes categorical variables (country, sector)
- ✅ Generates 384D narrative embeddings with optional fine-tuning

### Model Training
- ✅ **3 model families**: Logistic Regression, Random Forest, XGBoost
- ✅ **Nested cross-validation**: Outer loop for model selection, inner loop for HP tuning
- ✅ **Threshold optimization**: F1-based cutoff on out-of-fold predictions
- ✅ **Optional Optuna HP search**: TPE sampler with stratified CV
- ✅ **Optional embedding fine-tuning**: Cosine similarity or MNR losses
- ✅ **Leakage-free design**: Proper train/validation splits at every stage

### Predictions & Scoring
- ✅ Batch scoring of new companies
- ✅ Default probability + risk rating (Very Low/Low/Medium/High/Very High)
- ✅ Claude-generated analyst memos (requires `ANTHROPIC_API_KEY`)
- ✅ Challenge-compliant output format (`outputs/predictions.csv`)

### Interactive Interface (6 Pages)
- 📊 **Dashboard**: Overall statistics, sector/country risk profiles, feature correlations
- 🏢 **Company Analysis**: Browse training companies and their financial metrics
- 💼 **Credit Assessment**: Submit company data for instant risk evaluation with AI memo
- 🤖 **Model Training**: Configure training parameters and view terminal commands
- 📈 **Results Explorer**: Filter and analyze scored predictions
- ❓ **Help**: Full documentation and terminal command reference

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd f:\new_fasanara\fasa-credit-assessment

# Create virtual environment with uv
uv venv --python 3.12

# Activate
.\.venv\Scripts\activate

# Install dependencies
uv sync
```

### 2. Train Model

```bash
# Basic training (2-3 minutes)
.\.venv\Scripts\python Train\train.py

# With hyperparameter tuning (15-30 minutes)
.\.venv\Scripts\python Train\train.py --hptune --hptune_trials 50

# With embedding fine-tuning
.\.venv\Scripts\python Train\train.py --finetune --finetune_epochs 3

# With AI-generated memos (requires ANTHROPIC_API_KEY)
.\.venv\Scripts\python Train\train.py --memos

# All options combined
.\.venv\Scripts\python Train\train.py --finetune --hptune --memos --hptune_trials 100
```

### 3. Start Interactive App

```bash
# IMPORTANT: Must use .venv Python!
.\.venv\Scripts\python -m streamlit run app/app.py
```

Opens at `http://localhost:8501`

## 📁 Project Structure

```
fasa-credit-assessment/
├── Train/
│   └── train.py                    # Core training pipeline (700+ lines)
├── app/
│   ├── app.py                      # Streamlit interface (6 pages)
│   ├── utils.py                    # Helper functions
│   ├── requirements.txt            # App dependencies
│   ├── README.md                   # App-specific docs
│   └── CONFIGURATION.md            # Advanced setup
├── data/
│   ├── train_companies.csv         # 1000 companies with financials
│   ├── train_narratives.csv        # Business descriptions
│   ├── train_outcomes.csv          # Default labels (17.4% rate)
│   └── scoring_companies.csv       # Companies to score
├── results/
│   ├── best_model.pkl             # Trained classifier
│   ├── feature_columns.pkl        # Feature list
│   ├── target_encoders.pkl        # Category encoders
│   ├── scored_companies.csv       # Predictions + probabilities
│   ├── scored_companies.json      # Same in JSON format
│   ├── scored_companies_memos.txt # Credit analysis memos
│   └── scored_companies_hp_report.txt # HP tuning details
├── outputs/
│   └── predictions.csv            # Challenge format output
├── models/
│   └── finetuned_embedder/        # Fine-tuned SBERT (if --finetune)
├── Analytics/
│   └── Analyze.py                 # EDA script
├── pyproject.toml                 # Dependencies
├── LEAKAGE_ANALYSIS.md            # Info leakage analysis
├── LEAKAGE_FIX_SUMMARY.md         # Leakage-free design
└── README.md                       # This file
```

## 🔧 Training Parameters

```bash
# Embedding Fine-Tuning
--finetune                  # Enable embedding fine-tuning
--finetune_loss cosine|mnr  # Loss function (default: cosine)
--finetune_frac 0.40        # Fraction for fine-tuning (default: 0.40)
--finetune_epochs 3         # Epochs (default: 3)
--finetune_batch 16         # Batch size (default: 16)

# Hyperparameter Tuning (Optuna)
--hptune                    # Enable Optuna search
--hptune_trials 50          # Trials per model (default: 50)
--hptune_inner_folds 3      # Inner CV folds (default: 3)

# Claude API
--memos                     # Generate credit memos
--api_key YOUR_KEY          # Anthropic API key (or set env var)

# Output
--train_companies data/train_companies.csv
--train_narratives data/train_narratives.csv
--train_outcomes data/train_outcomes.csv
--scoring_companies data/scoring_companies.csv
--output results/scored_companies.csv
```

## 🌐 Environment Variables

```bash
# For Claude memo generation
set ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
```

## 📊 Model Architecture

### Feature Engineering (13 features total)

**Quantitative (7)**:
- Revenue (€M)
- EBITDA margin
- Debt ratio
- Interest coverage
- Cash ratio
- Revenue growth YoY
- Distress score (0-5)

**Narrative (3 from 384D embeddings)**:
- Similarity to defaulted companies centroid
- Similarity to safe companies centroid
- Difference (defaulted - safe similarity)

**Categorical (2, target-encoded)**:
- Country (smoothed default rates)
- Sector (smoothed default rates)

**Additional**: Risk count, benefit count, years in operation, employee count

### Model Selection (3 Families)

1. **Logistic Regression**: Interpretable baseline
   - C=0.5, balanced class weights, max_iter=1000

2. **Random Forest**: Feature importance
   - 300 estimators, max_depth=5, balanced weights

3. **XGBoost**: Gradient boosting
   - 300 estimators, max_depth=4, learning_rate=0.1, AUCPR metric

**Selection**: Best PR-AUC on outer CV fold

### Nested Cross-Validation (Leakage-Free)

```
Outer CV: 5-fold StratifiedKFold (model selection)
  └─ Fold 1:
     ├─ Train (80%): Inner CV + HP search
     │   └─ Inner CV: 3-fold (Optuna TPE sampler)
     │       ├─ Train: Run HP trials
     │       └─ Validation: Score
     ├─ Test (20%): Evaluate selected model
     
  ├─ Fold 2-5: ...
  
  └─ Report: Unbiased metrics for each model family
```

### Threshold Optimization

- **Metric**: F1 score
- **Data**: Out-of-fold predictions from nested CV
- **Method**: Exhaustive grid search over [0.0, 1.0]
- **Output**: Optimal probability cutoff for binary decision

## 💾 Output Files

### Predictions
- `results/scored_companies.csv` - Full results (ID, probabilities, risks, benefits)
- `outputs/predictions.csv` - **Challenge format** (ID, probability, risk_rating, explanation)
- `results/scored_companies.json` - JSON predictions

### Analysis & Reports
- `results/scored_companies_memos.txt` - Full credit memos (analyst-style analysis)
- `results/scored_companies_hp_report.txt` - Hyperparameter tuning details
- `results/scored_companies_embedding_comparison.txt` - Fine-tuning metrics

### Model Artifacts (Auto-saved)
- `results/best_model.pkl` - Trained pipeline (LR/RF/XGB)
- `results/feature_columns.pkl` - Exact feature order
- `results/target_encoders.pkl` - Category encoding mappings

## 🎓 Usage Examples

### Via Web Interface (Credit Assessment)
1. Navigate to **http://localhost:8501**
2. Go to **💼 Credit Assessment** page
3. Enter company metrics and description
4. Click **"🔍 Run Credit Assessment"**
5. Get instant decision + AI-generated memo

### Via Terminal (Batch Scoring)
```bash
# Score all companies + generate memos
.\.venv\Scripts\python Train\train.py --memos

# Output: results/scored_companies.csv + memos.txt
```

### Via Python (Programmatic)
```python
from Train.train import (
    load_data, engineer_quant_features, 
    build_narrative_features, train_model, 
    score_companies
)
import pandas as pd

# Load
train_df, scoring_df = load_data(...)

# Engineer features
train_df = engineer_quant_features(train_df)
train_df, scoring_df = build_narrative_features(train_df, scoring_df)

# Train
model, cols, name, threshold, _ = train_model(train_df)

# Score
results = score_companies(model, cols, scoring_df, threshold, generate_memos=False)

# View
print(results[["company_name", "default_probability", "risk_rating"]])
```

## 🔍 Key Algorithms

### Embedding Fine-Tuning (Optional)
- **Strategy**: Contrastive learning on 40% of training data
- **Losses**:
  - `cosine`: Same outcome → label 1.0, cross outcome → label 0.0
  - `mnr`: Label-free (risk text similar to benefit text)
- **Validation**: Leakage-free evaluation on held-out set (30% / 35% / 35% split)
- **Comparison**: Pretrained vs Fine-tuned on same validation set

### Target Encoding (Categorical)
- **Formula**: `(count * mean + smoothing * global_mean) / (count + smoothing)`
- **Smoothing**: 10.0 (balances rare categories)
- **Fit on**: Training/ML set only
- **Apply to**: New companies

### Threshold Optimization
- **Goal**: Maximize F1 score for binary decision
- **Search**: Range [0.0, 1.0], step 0.01
- **Data**: Out-of-fold predictions (no leakage)
- **Result**: Single probability cutoff for all companies

## 🐛 Troubleshooting

### Missing Module (optuna, anthropic, etc.)
```bash
# Reinstall all dependencies
uv sync

# Or install specific package
uv pip install optuna anthropic transformers torch
```

### Streamlit Won't Start
```bash
# ❌ WRONG (uses system Python)
streamlit run app/app.py

# ✅ CORRECT (uses .venv Python)
.\.venv\Scripts\python -m streamlit run app/app.py
```

### Module Error When Importing Train.train
```bash
# Make sure you're in the project directory
cd f:\new_fasanara\fasa-credit-assessment

# Then run Streamlit
.\.venv\Scripts\python -m streamlit run app/app.py
```

### API Key Issues
```bash
# Set environment variable
set ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx

# Or pass via CLI
.\.venv\Scripts\python Train\train.py --memos --api_key sk-ant-xxxxxxxxxxxxx
```

### Out of Memory
```bash
# Reduce fine-tuning batch size
.\.venv\Scripts\python Train\train.py --finetune --finetune_batch 8

# Or skip fine-tuning
.\.venv\Scripts\python Train\train.py
```

## ⚡ Performance

| Task | Time | Notes |
|------|------|-------|
| Training (no tuning) | 2-3 min | 5-fold CV, 1000 companies |
| Training (HP tuning) | 15-30 min | 50 trials × 3 families |
| Training (fine-tuning) | 5-10 min | 3 epochs on 400 narratives |
| Scoring (100 companies) | <1 sec | Model inference only |
| All together | 30-60 min | --finetune --hptune --memos |
| Model size | ~50 MB | Pickle file |
| Embedding size | 384 dimensions | Per narrative |

## 📚 Documentation

- **train.py**: Core functions with full docstrings
- **app/app.py**: Streamlit pages and interactions
- **app/README.md**: App-specific setup
- **app/CONFIGURATION.md**: Advanced configuration
- **LEAKAGE_ANALYSIS.md**: Information leakage analysis
- **LEAKAGE_FIX_SUMMARY.md**: Cross-validation design

## 🎯 Data Format

### Training Input
```csv
company_id,company_name,revenue_m,ebitda_margin,debt_ratio,interest_coverage,cash_ratio,years_in_operation,employee_count,revenue_growth,country,sector
1,Acme Corp,25.5,0.15,0.45,3.2,0.12,5,50,0.08,UK,Technology
```

### Narratives Input
```csv
company_id,business_description
1,"Acme develops cloud software... key risk factors include market saturation... strengths include strong team..."
```

### Outcomes Input
```csv
company_id,defaulted
1,0
```

### Predictions Output
```csv
company_id,predicted_default_probability,risk_rating,explanation
1,0.1234,Low,"Strong competitive position with healthy cash flow. EBITDA margin at 15% above sector average..."
```

## 🚀 Next Steps

1. Train the model: `python Train\train.py`
2. Start Streamlit: `python -m streamlit run app\app.py`
3. Explore Dashboard page
4. Try Credit Assessment with sample company data
5. View training companies in Company Analysis page

## 📝 License

Internal use - Fasanara Challenge

---

**Last Updated**: 2026-06-10  
**Status**: ✅ Production Ready  
**Training Data**: 1,000 companies | 17.4% default rate  
**Test Results**: PR-AUC ~0.70-0.75 (typical)