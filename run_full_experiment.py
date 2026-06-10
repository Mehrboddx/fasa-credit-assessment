#!/usr/bin/env python3
"""
Comprehensive Experiment Orchestrator
Runs all experiment configurations and generates a consolidated comparison report
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import sys

# Configuration
TRANSFORMERS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
]

EXPERIMENTS = {
    # 1. BASELINE: No embeddings (statistical features only)
    "01_baseline_no_embeddings": {
        "description": "Statistical features only (no NLP embeddings)",
        "train_args": [
            "python Train/train.py",
            "--experiment_name", "01_baseline_no_embeddings",
            "--no_embeddings",  # Skip narrative embeddings
        ],
        "category": "Baseline",
    },
    
    # 2. TRANSFORMERS: Pretrained (no fine-tuning, no HP tuning)
    **{
        f"02_transformer_{t.split('/')[-1]}_pretrained": {
            "description": f"{t.split('/')[-1]} - Pretrained (no tuning)",
            "train_args": [
                "python Train/train.py",
                "--experiment_name", f"02_transformer_{t.split('/')[-1]}_pretrained",
                "--embedder_model", t,
            ],
            "category": "Transformers (Pretrained, No Tuning)",
        }
        for t in TRANSFORMERS
    },
    
    # 3. TRANSFORMERS + HP TUNING: Pretrained with hyperparameter tuning
    **{
        f"03_transformer_{t.split('/')[-1]}_pretrained_hptune": {
            "description": f"{t.split('/')[-1]} - Pretrained + HP Tuning",
            "train_args": [
                "python Train/train.py",
                "--experiment_name", f"03_transformer_{t.split('/')[-1]}_pretrained_hptune",
                "--embedder_model", t,
                "--hptune",
                "--hptune_trials", "30",  # Reduced for speed
            ],
            "category": "Transformers (Pretrained + HP Tuning)",
        }
        for t in TRANSFORMERS
    },
    
    # 4. BEST TRANSFORMER (all-MiniLM): Fine-tuned only
    "04_best_transformer_finetuned": {
        "description": "all-MiniLM-L6-v2 - Fine-tuned (no HP tuning)",
        "train_args": [
            "python Train/train.py",
            "--experiment_name", "04_best_transformer_finetuned",
            "--embedder_model", "all-MiniLM-L6-v2",
            "--finetune",
        ],
        "category": "Best Transformer (Fine-tuned)",
    },
    
    # 5. BEST TRANSFORMER + BOTH: Fine-tuned AND HP-tuned
    "05_best_transformer_finetuned_hptune": {
        "description": "all-MiniLM-L6-v2 - Fine-tuned + HP Tuning (BEST)",
        "train_args": [
            "python Train/train.py",
            "--experiment_name", "05_best_transformer_finetuned_hptune",
            "--embedder_model", "all-MiniLM-L6-v2",
            "--finetune",
            "--hptune",
            "--hptune_trials", "30",
        ],
        "category": "Best Transformer (Fine-tuned + HP Tuning)",
    },
}


def run_experiment(exp_name: str, exp_config: Dict) -> bool:
    """Run a single experiment"""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  {exp_config['description']}")
    print(f"{'='*70}")
    
    cmd = exp_config["train_args"]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ FAILED: {exp_name}")
            return False
        
        print(f"✅ SUCCESS: {exp_name}")
        return True
    
    except Exception as e:
        print(f"❌ ERROR in {exp_name}: {e}")
        return False


def load_experiment_metrics(exp_name: str) -> Dict:
    """Load metrics.json from an experiment"""
    metrics_path = Path("experiments") / exp_name.replace(" ", "_").lower() / "metrics.json"
    
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, "r") as f:
        return json.load(f)


def load_experiment_scores(exp_name: str) -> pd.DataFrame:
    """Load scores.csv from an experiment"""
    scores_path = Path("experiments") / exp_name.replace(" ", "_").lower() / "scores.csv"
    
    if not scores_path.exists():
        return None
    
    return pd.read_csv(scores_path)


def generate_consolidated_report() -> pd.DataFrame:
    """Generate consolidated metrics comparison across all experiments"""
    
    all_metrics = []
    
    for exp_name in EXPERIMENTS.keys():
        metrics = load_experiment_metrics(exp_name)
        if not metrics:
            print(f"⚠️  No metrics.json found for {exp_name}")
            continue
        
        cv_metrics = metrics.get("cross_validation_metrics", {})
        best_model = metrics.get("model_name", "Unknown")
        
        # Get metrics for best model
        if best_model in cv_metrics:
            best_metrics = cv_metrics[best_model]
            row = {
                "experiment": exp_name,
                "category": EXPERIMENTS[exp_name].get("category", ""),
                "description": EXPERIMENTS[exp_name]["description"],
                "best_model": best_model,
                "roc_auc": best_metrics.get("roc_auc", 0),
                "roc_auc_std": best_metrics.get("roc_auc_std", 0),
                "pr_auc": best_metrics.get("pr_auc", 0),
                "pr_auc_std": best_metrics.get("pr_auc_std", 0),
                "threshold": metrics.get("threshold", 0),
                "finetune": metrics.get("finetune", False),
                "hptune": metrics.get("hptune", False),
            }
            all_metrics.append(row)
    
    df = pd.DataFrame(all_metrics)
    
    # Sort by ROC-AUC descending
    if not df.empty:
        df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    
    return df


def save_consolidated_report(df: pd.DataFrame):
    """Save consolidated report in multiple formats"""
    
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)
    
    # CSV format (easy for Streamlit)
    csv_path = output_dir / "CONSOLIDATED_RESULTS.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved: {csv_path}")
    
    # Pretty text format
    text_path = output_dir / "CONSOLIDATED_RESULTS.txt"
    with open(text_path, "w") as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE EXPERIMENT COMPARISON\n")
        f.write("="*100 + "\n\n")
        
        # Group by category
        categories = df["category"].unique()
        for cat in categories:
            f.write(f"\n{cat}\n")
            f.write("-" * 100 + "\n")
            
            cat_df = df[df["category"] == cat].copy()
            
            for idx, row in cat_df.iterrows():
                f.write(f"\n#{idx+1} {row['description']}\n")
                f.write(f"   Best Model: {row['best_model']}\n")
                f.write(f"   ROC-AUC:    {row['roc_auc']:.4f} ± {row['roc_auc_std']:.4f}\n")
                f.write(f"   PR-AUC:     {row['pr_auc']:.4f} ± {row['pr_auc_std']:.4f}\n")
                f.write(f"   Threshold:  {row['threshold']:.4f}\n")
                f.write(f"   Fine-tuned: {'Yes' if row['finetune'] else 'No'}\n")
                f.write(f"   HP-Tuned:   {'Yes' if row['hptune'] else 'No'}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Total Experiments: {len(df)}\n")
        f.write(f"Best ROC-AUC: {df['roc_auc'].max():.4f} ({df.iloc[0]['description']})\n")
        f.write(f"Worst ROC-AUC: {df['roc_auc'].min():.4f}\n")
        f.write(f"Average ROC-AUC: {df['roc_auc'].mean():.4f}\n")
    
    print(f"✅ Saved: {text_path}")
    
    # JSON format (for programmatic use)
    json_path = output_dir / "CONSOLIDATED_RESULTS.json"
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print(f"✅ Saved: {json_path}")


def print_summary(df: pd.DataFrame):
    """Print consolidated results to console"""
    print("\n" + "="*100)
    print("CONSOLIDATED RESULTS (Sorted by ROC-AUC)")
    print("="*100 + "\n")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    display_df = df[[
        "description", 
        "best_model",
        "roc_auc",
        "roc_auc_std",
        "pr_auc",
        "finetune",
        "hptune"
    ]].copy()
    
    display_df["roc_auc"] = display_df.apply(
        lambda r: f"{r['roc_auc']:.4f}±{r['roc_auc_std']:.4f}", axis=1
    )
    display_df["pr_auc"] = display_df.apply(
        lambda r: f"{r['pr_auc']:.4f}±{r['pr_auc_std']:.4f}", axis=1
    )
    
    display_df = display_df.drop(columns=["roc_auc_std"])
    display_df.columns = ["Experiment", "Best Model", "ROC-AUC", "PR-AUC", "Fine-tuned", "HP-Tuned"]
    
    print(display_df.to_string(index=False))
    print("\n" + "="*100)


def main():
    """Main orchestration loop"""
    print("\n" + "="*70)
    print("FASANARA CREDIT ASSESSMENT - FULL EXPERIMENT SUITE")
    print("="*70)
    print(f"\n📋 Total experiments to run: {len(EXPERIMENTS)}")
    print(f"   - 1 Baseline (no embeddings)")
    print(f"   - 3 Transformers (pretrained, no tuning)")
    print(f"   - 3 Transformers (pretrained + HP tuning)")
    print(f"   - 1 Best transformer (fine-tuned)")
    print(f"   - 1 Best transformer (fine-tuned + HP tuning)")
    
    results = {}
    successful = 0
    failed = 0
    
    # Run all experiments
    for exp_name, exp_config in EXPERIMENTS.items():
        success = run_experiment(exp_name, exp_config)
        results[exp_name] = success
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT EXECUTION SUMMARY")
    print("="*70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(EXPERIMENTS)}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for exp_name, success in results.items():
            if not success:
                print(f"  - {exp_name}")
    
    # Generate consolidated report
    print("\n" + "="*70)
    print("GENERATING CONSOLIDATED REPORT")
    print("="*70)
    
    df = generate_consolidated_report()
    
    if not df.empty:
        print_summary(df)
        save_consolidated_report(df)
        print("\n✅ Consolidated results saved to experiments/")
        print("   - CONSOLIDATED_RESULTS.csv (for Streamlit)")
        print("   - CONSOLIDATED_RESULTS.txt (human-readable)")
        print("   - CONSOLIDATED_RESULTS.json (programmatic)")
    else:
        print("❌ No metrics found. Check experiment logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
