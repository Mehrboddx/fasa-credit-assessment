# Advanced Configuration Guide

## Installation & Setup

### 1. Install Streamlit Package

If you already have the project venv activated, just install streamlit:

```bash
pip install streamlit>=1.28.0
```

Or install all app dependencies:

```bash
pip install -r app/requirements.txt
```

### 2. Run the App

**Quick start (Windows):**
```bash
app\run_app.bat
```

**Quick start (Mac/Linux):**
```bash
chmod +x app/run_app.sh
./app/run_app.sh
```

**Manual (any OS):**
```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

## Configuration Files

### .streamlit/config.toml
Located at `app/.streamlit/config.toml`

Customize theme and behavior:
```toml
[theme]
primaryColor = "#2E86AB"        # Header color
backgroundColor = "#FFFFFF"     # Page background
secondaryBackgroundColor = "#F0F2F6"  # Cards/sidebars
textColor = "#262730"          # Text color
font = "sans serif"            # Font choice

[client]
showErrorDetails = true        # Show error messages
toolbarMode = "developer"      # Show developer menu

[server]
port = 8501                    # Change port if needed
headless = true               # Run without browser
maxUploadSize = 200           # Max file upload (MB)
```

### Streamlit Cache Settings

The app caches data loading to improve performance:

```python
@st.cache_resource
def load_data_files():  # Cached function
    # Loads once, reuses until cache cleared
    ...
```

**Clear cache:**
```bash
streamlit cache clear
streamlit run app/app.py
```

## Environment Variables

### ANTHROPIC_API_KEY (Optional)

For credit memo generation:

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-your-key-here"
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-your-key-here
```

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY="sk-your-key-here"
```

Or create a `.env` file in project root:
```
ANTHROPIC_API_KEY=sk-your-key-here
```

## Training from the App

### 1. Configure Training in Model Training Tab

- Enable/disable fine-tuning
- Set hyperparameter search trials
- View data summary

### 2. Open Terminal Window

While app is running, open a **new terminal** (don't close Streamlit):

```bash
# Activate venv first (if not already active)
.\.venv\Scripts\activate

# Run training
python Train\train.py --hptune

# Or with all features
python Train\train.py --finetune --hptune --memos
```

### 3. Monitor Progress

Training logs appear in terminal. This takes 10-30 minutes depending on configuration.

### 4. Refresh Results Tab

Once training completes, go to Results Explorer and view scored companies.

## Performance Tuning

### Speed Up Training

**Skip hyperparameter search:**
```bash
python Train\train.py
```
Expected time: 5-10 minutes

**Reduce HP tuning trials:**
```bash
python Train\train.py --hptune --hptune_trials 20
```
Default is 50 trials per model

**Reduce inner CV folds:**
```bash
python Train\train.py --hptune_inner_folds 2
```
Default is 3 folds

### Reduce Memory Usage

In `Train/train.py`, you can modify:
```python
# Lower batch size for embeddings
batch_size = 16  # Default: 32

# Skip fine-tuning
# Don't use --finetune flag

# Reduce model families
# Edit MODEL_CONFIGS dict
```

## Data Path Configuration

The app expects data in relative paths:

```
fasa-credit-assessment/
├── data/
│   ├── train_companies.csv
│   ├── train_narratives.csv
│   ├── train_outcomes.csv
│   └── scoring_companies.csv
├── results/
│   ├── scored_companies.csv
│   └── scored_companies_memos.txt
└── app/
    └── app.py
```

If your data is elsewhere, modify in `app/app.py`:

```python
# Change this line in load_data_files()
data_dir = Path("../data")  # Change to your path
```

## Troubleshooting

### App Won't Start

**Check Python version:**
```bash
python --version  # Should be 3.9+
```

**Reinstall dependencies:**
```bash
pip install -r app/requirements.txt --force-reinstall
```

**Clear Streamlit cache:**
```bash
streamlit cache clear
```

### Slow Performance

1. Clear browser cache (Ctrl+Shift+Delete)
2. Close other apps using memory
3. Skip HP tuning: `python Train\train.py` (no --hptune)
4. Reduce data size for testing

### Data Not Loading

1. Check file paths match workspace structure
2. Verify CSV files have proper headers
3. Check for encoding issues (should be UTF-8)

**Debug data loading:**
```bash
python -c "import pandas as pd; print(pd.read_csv('data/train_companies.csv').head())"
```

### Credits/Memos Not Generating

1. Verify ANTHROPIC_API_KEY is set
2. Check API key is valid (no extra spaces)
3. Try from terminal directly:

```bash
set ANTHROPIC_API_KEY=sk-your-key
python Train\train.py --memos
```

## Browser & Display

### Recommended Settings

- **Browser**: Chrome, Firefox, or Edge (latest)
- **Resolution**: 1920x1080 or higher
- **JavaScript**: Must be enabled

### Full-Screen Mode

Press **Ctrl+F** in Streamlit app for full-screen

### Dark Mode

Custom CSS in `app.py` forces light theme. To enable dark:

Edit `app.py`:
```python
st.markdown("""
<style>
    html {
        color-scheme: dark;  # Add this line
    }
</style>
""", unsafe_allow_html=True)
```

## Logs & Debugging

### View Streamlit Logs

```bash
streamlit run app/app.py --logger.level=debug
```

### View Training Logs

```bash
python Train/train.py 2>&1 | tee training.log
```

This saves logs to `training.log` file

### Check Results Files

```bash
# View scored companies
head -10 results/scored_companies.csv

# Count predictions
grep -c "1" results/scored_companies.csv  # Count default=1

# View memos
type results/scored_companies_memos.txt  # Windows
cat results/scored_companies_memos.txt   # Mac/Linux
```

## Advanced Features

### Custom Visualizations

Add to Dashboard page:
```python
import plotly.graph_objects as go

fig = go.Figure(data=[go.Scatter(...)])
st.plotly_chart(fig, use_container_width=True)
```

### Export Pipeline

Add to Results Explorer:
```python
st.download_button(
    "Download as Excel",
    data=filtered.to_excel(index=False),
    file_name="results.xlsx",
    mime="application/vnd.ms-excel"
)
```

### Custom Model Parameters

Edit `Train/train.py` MODEL_CONFIGS:
```python
MODEL_CONFIGS = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0],  # Regularization strength
        "max_iter": [1000],
    },
    ...
}
```

## Support

For issues:
1. Check this guide's Troubleshooting section
2. Review Streamlit logs
3. Check training logs
4. Verify data files exist
5. Test with `python -c "import streamlit; print(streamlit.__version__)"`

## Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **scikit-learn**: https://scikit-learn.org
- **XGBoost**: https://xgboost.readthedocs.io
- **Sentence Transformers**: https://www.sbert.net
- **Optuna**: https://optuna.org
