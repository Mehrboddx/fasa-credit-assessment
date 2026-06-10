# Fasanara Credit Assessment App

Interactive Streamlit web application for credit risk modeling, company analysis, and scoring.

## Features

✅ **Dashboard**
- Overview statistics (total companies, default rate, sectors)
- Sector risk analysis with default rates
- Country risk analysis
- Feature correlation analysis

✅ **Company Analysis**
- Browse training companies
- View financial metrics
- Read business descriptions
- See default status

✅ **Model Training**
- Configure training parameters
- Enable embedding fine-tuning
- Configure hyperparameter search
- View data summary

✅ **Results Explorer**
- Filter and view scoring results
- Sort by default probability
- Download predictions
- View credit memos (if generated)

✅ **Help & Documentation**
- Quick start guide
- Feature descriptions
- Model options
- Terminal commands reference
- Output file guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or use the main project venv (if already installed):
```bash
.\.venv\Scripts\pip install streamlit>=1.28.0
```

## Running the App

**From the app directory:**
```bash
streamlit run app.py
```

**Or from project root:**
```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

## Typical Workflow

1. **Launch the app** → View Dashboard to understand data
2. **Run training** (in separate terminal):
   ```bash
   .\.venv\Scripts\python Train\train.py --hptune
   ```
3. **Refresh Results Explorer** → View scored companies
4. **Download results** → Export predictions for downstream use

## Training Commands

**Standard (no hyperparameter search):**
```bash
.\.venv\Scripts\python Train\train.py
```

**With hyperparameter optimization:**
```bash
.\.venv\Scripts\python Train\train.py --hptune
```

**With embedding fine-tuning:**
```bash
.\.venv\Scripts\python Train\train.py --finetune
```

**With credit memo generation (requires ANTHROPIC_API_KEY):**
```bash
.\.venv\Scripts\python Train\train.py --memos
```

**All features:**
```bash
.\.venv\Scripts\python Train\train.py --finetune --hptune --memos
```

## Architecture

```
app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file

../Train/
├── train.py           # Training pipeline (run from terminal)
└── [training code]    # Narrative embeddings, model training, etc.

../Analytics/
├── Analyze.py         # EDA and visualization scripts
└── [outputs]          # Correlation analysis, embedding plots

../data/
├── train_companies.csv
├── train_narratives.csv
├── train_outcomes.csv
└── scoring_companies.csv

../results/
├── scored_companies.csv
├── scored_companies.json
└── scored_companies_memos.txt
```

## Features Used

- **Structured**: Revenue, debt ratio, EBITDA margin, interest coverage, cash ratio, years in operation, employee count, revenue growth
- **Narrative**: 384D sentence embeddings + parsed risk/benefit signals
- **Engineered**: Debt service risk, profitability-liquidity score, leverage-adjusted coverage, revenue per employee, distress score
- **Categorical**: Target-encoded country and sector

## Models Trained

- Logistic Regression (baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting)

All with:
- 5-fold cross-validation for model selection
- Optional Optuna hyperparameter search (nested CV)
- F1-based threshold optimization
- Feature importance tracking

## Output Files

After training (`Train/train.py`), check these directories:

**results/** - Scoring outputs
- `scored_companies.csv` - Predictions
- `scored_companies_memos.txt` - Credit analyses
- `scored_companies_hp_report.txt` - Tuning details

**outputs/** - Analytics
- `sector_analysis.csv/png` - Sector risk profiles
- `country_analysis.csv` - Country risk profiles
- `feature_correlations.csv` - Correlation matrix
- `narrative_embeddings_*.png` - Embedding visualizations

## Troubleshooting

**App won't load:**
```bash
streamlit cache clear
streamlit run app.py
```

**Training is slow:**
- Skip `--hptune` for faster runs
- Reduce `--hptune_trials` (default: 50)
- Skip `--memos` if you don't need Claude analysis

**Memory issues:**
- Reduce batch sizes in train.py
- Run analytics and training in separate sessions

## API Requirements

**Optional - for credit memo generation:**
Set environment variable:
```bash
set ANTHROPIC_API_KEY=sk-...
```

Then run with `--memos` flag.

## Next Steps

- Explore sector/country risk profiles in Dashboard
- Train a model with `python Train\train.py`
- Review results in Results Explorer
- Refine model parameters and retrain
- Export predictions for downstream systems
