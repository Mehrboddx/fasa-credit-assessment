# 🚀 Full Experiment Suite Guide

## Overview

The **Fasanara Credit Assessment** system includes a comprehensive experiment orchestrator that systematically evaluates different configurations:

1. **Baseline** - Statistical features only (no NLP embeddings)
2. **Transformers (Pretrained)** - Three models without fine-tuning or HP tuning
3. **Transformers (Pretrained + HP Tuning)** - Three models with hyperparameter optimization
4. **Best Transformer** - Fine-tuned embedder (without HP tuning)
5. **Best Transformer (Full Optimization)** - Fine-tuned + Hyperparameter tuning

---

## 🎯 Quick Start

### Run All 9 Experiments

```bash
python run_full_experiment.py
```

This will:
- ✅ Run 9 comprehensive experiments (~60-90 minutes total)
- ✅ Save all results to `experiments/` directory
- ✅ Generate consolidated comparison report
- ✅ Display summary statistics to console

### View Results in Streamlit

After experiments complete:

```bash
streamlit run app/app.py
```

Then navigate to: **🔬 Experiment Comparison** page

---

## 📊 Experiment Details

### Configuration Breakdown

| # | Experiment | Embedder | Fine-tune | HP-Tune | Category | Est. Time |
|---|-----------|----------|-----------|---------|----------|-----------|
| 1 | Baseline | None | ❌ | ❌ | Baseline | ~5 min |
| 2 | all-MiniLM (P) | all-MiniLM-L6-v2 | ❌ | ❌ | Pretrained | ~10 min |
| 3 | all-mpnet (P) | all-mpnet-base-v2 | ❌ | ❌ | Pretrained | ~12 min |
| 4 | paraphrase-MiniLM (P) | paraphrase-MiniLM-L6-v2 | ❌ | ❌ | Pretrained | ~12 min |
| 5 | all-MiniLM (P+HP) | all-MiniLM-L6-v2 | ❌ | ✅ | Pretrained + HP | ~15 min |
| 6 | all-mpnet (P+HP) | all-mpnet-base-v2 | ❌ | ✅ | Pretrained + HP | ~18 min |
| 7 | paraphrase (P+HP) | paraphrase-MiniLM-L6-v2 | ❌ | ✅ | Pretrained + HP | ~18 min |
| 8 | all-MiniLM (FT) | all-MiniLM-L6-v2 | ✅ | ❌ | Fine-tuned | ~12 min |
| 9 | all-MiniLM (FT+HP) | all-MiniLM-L6-v2 | ✅ | ✅ | Fine-tuned + HP | ~20 min |

**Total Estimated Time**: 60-90 minutes (depending on hardware)

---

## 🔍 Understanding Results

### Per-Experiment Artifacts

Each experiment creates files in `experiments/{experiment_name}/`:

```
experiments/01_baseline_no_embeddings/
├── scores.csv                    # Predictions for scoring companies
├── model.pkl                      # Trained classifier
├── feature_columns.pkl            # Feature names used
├── target_encoders.pkl            # Country/sector encoders
└── metrics.json                   # CV metrics, ROC-AUC, PR-AUC
```

### Consolidated Results

After all experiments complete:

```
experiments/
├── CONSOLIDATED_RESULTS.csv       # ✅ Main results file (for Streamlit)
├── CONSOLIDATED_RESULTS.txt       # Human-readable report
└── CONSOLIDATED_RESULTS.json      # Programmatic access
```

### CSV Columns

The consolidated CSV contains:

```
experiment,category,description,best_model,roc_auc,roc_auc_std,pr_auc,pr_auc_std,threshold,finetune,hptune
01_baseline_no_embeddings,Baseline,"Statistical features only (no NLP embeddings)",Logistic Regression,0.7824,0.0312,0.4521,0.0456,0.5123,False,False
02_transformer_all-minilm-l6-v2_pretrained,Transformers (Pretrained),all-MiniLM-L6-v2 - Pretrained,Logistic Regression,0.8121,0.0234,0.5253,0.0412,0.6147,False,False
...
```

---

## 📈 Streamlit Dashboard

### Experiment Comparison Page Features

1. **Performance Summary** - Quick metrics overview
   - Best ROC-AUC across all experiments
   - Average ROC-AUC
   - Best PR-AUC
   - Total experiments run

2. **Results by Category** - Expandable sections for each experiment type
   - Baseline results
   - Pretrained transformer results
   - HP-tuned transformer results
   - Fine-tuned results

3. **Comparative Charts**
   - ROC-AUC bar chart with error bars (±std)
   - PR-AUC bar chart with error bars (±std)

4. **Export Options**
   - Download CSV for further analysis
   - Download text report for presentation

---

## 🛠️ Custom Experiment Runs

### Run Single Experiment

```bash
# Baseline only
python Train/train.py --experiment_name "my_baseline" --no_embeddings

# Specific transformer
python Train/train.py --experiment_name "my_experiment" --embedder_model "all-mpnet-base-v2"

# With fine-tuning
python Train/train.py --experiment_name "my_finetuned" --finetune

# With HP tuning
python Train/train.py --experiment_name "my_hptuned" --hptune --hptune_trials 50
```

### Run Bulk Comparison (Original Behavior)

```bash
python Train/train.py --compare_embedders --compare_embedders_list "all-MiniLM-L6-v2" "all-mpnet-base-v2"
```

---

## 📊 Interpreting the Results

### Key Metrics

- **ROC-AUC**: Area under Receiver Operating Characteristic curve
  - Range: 0 to 1 (higher is better)
  - 0.5 = random classifier, 1.0 = perfect classifier
  - Good baseline: 0.7+, Excellent: 0.85+

- **PR-AUC**: Area under Precision-Recall curve
  - More useful for imbalanced datasets
  - Range: 0 to 1 (higher is better)
  - Focuses on minority class performance

- **Standard Deviation** (±Std)
  - Measures consistency across CV folds
  - Lower std = more stable model
  - High std = variable performance

### What to Look For

1. **Baseline Impact**: Compare baseline (row 1) to pretrained models
   - How much does embedding help?

2. **Transformer Comparison**: Rows 2-4 vs 5-7
   - Which transformer is best?
   - Does HP tuning help?

3. **Fine-tuning Impact**: Rows 8-9
   - Does fine-tuning improve pretrained?
   - Is HP tuning worth the extra time?

### Example Decision Making

```
If best ROC-AUC is row 9 (FT+HP):
  → Fine-tuning + HP tuning is valuable
  → Worth the 20-minute training time
  
If best ROC-AUC is row 2 (Pretrained):
  → Simple approach is optimal
  → Additional complexity doesn't help
  
If rows 2-4 are similar:
  → Transformer choice doesn't matter much
  → Pick smallest/fastest for production
```

---

## 🔧 Troubleshooting

### Experiments Fail Midway

**Check**: 
- Available disk space for model files
- Python environment has all dependencies
- No concurrent runs of train.py

**Solution**:
```bash
# Install/update dependencies
pip install -r app/requirements.txt

# Delete any incomplete experiments
rmdir experiments\failed_experiment
```

### Memory Issues

If you hit memory limits during HP tuning:

```bash
# Reduce trials
python Train/train.py --hptune --hptune_trials 20

# Reduce batch size
python Train/train.py --finetune --finetune_batch 8
```

### Missing Results in Streamlit

**Check**: 
- `experiments/CONSOLIDATED_RESULTS.csv` exists
- Experiments ran without errors

**Solution**:
```bash
# Check experiment status
ls experiments/*/metrics.json

# Re-generate report manually
python run_full_experiment.py
```

---

## 📝 Output Files

### In `experiments/` Directory

- **`{exp_name}/scores.csv`** - Scored companies for this experiment
- **`{exp_name}/model.pkl`** - Trained classifier
- **`{exp_name}/metrics.json`** - Cross-validation metrics
- **`CONSOLIDATED_RESULTS.csv`** - All experiments comparison (main file)
- **`CONSOLIDATED_RESULTS.txt`** - Human-readable report
- **`CONSOLIDATED_RESULTS.json`** - Machine-readable format

### In `results/` Directory

- **`scored_companies.csv`** - Latest scoring results
- **`scored_companies_hp_report.txt`** - Latest HP tuning details

---

## 🎓 Learning Path

1. **Start with Baseline** (`experiment 1`)
   - Understand feature importance without embeddings
   
2. **Add Embeddings** (`experiments 2-4`)
   - See impact of different transformers
   
3. **Optimize Parameters** (`experiments 5-7`)
   - Understand HP tuning benefits
   
4. **Fine-tune** (`experiments 8-9`)
   - Measure embedding fine-tuning impact

5. **Make Decision**
   - Choose best model for production
   - Balance performance vs inference speed

---

## 📞 Support

For detailed model information, see:
- `README.md` - Project overview
- `Train/train.py` - Implementation details
- `CONFIGURATION.md` - Configuration options

For UI questions, see:
- `app/README.md` - Streamlit app documentation
