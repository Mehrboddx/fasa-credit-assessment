"""
Utility functions for Fasanara Credit Assessment App
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import pickle
import json


def load_results(results_path: str = "results/scored_companies.csv") -> Optional[pd.DataFrame]:
    """Load scoring results if they exist"""
    path = Path(results_path)
    if path.exists():
        return pd.read_csv(path)
    return None


def load_memos(memos_path: str = "results/scored_companies_memos.txt") -> dict:
    """Load credit memos and parse by company"""
    memos = {}
    path = Path(memos_path)
    
    if not path.exists():
        return memos
    
    try:
        with open(path, 'r') as f:
            content = f.read()
        
        # Simple split by company header
        sections = content.split("=" * 70)
        
        for section in sections[1:]:  # Skip first empty split
            lines = section.strip().split('\n')
            if len(lines) >= 3:
                company_line = lines[0].replace("Company      : ", "").strip()
                memo_text = '\n'.join(lines[2:]).strip()
                if company_line and memo_text:
                    memos[company_line] = memo_text
    except Exception as e:
        print(f"Error parsing memos: {e}")
    
    return memos


def load_analytics_data() -> dict:
    """Load pre-computed analytics"""
    analytics = {}
    outputs_dir = Path("outputs")
    
    try:
        # Load correlation analysis
        corr_path = outputs_dir / "feature_correlations.csv"
        if corr_path.exists():
            analytics["correlations"] = pd.read_csv(corr_path)
        
        # Load sector analysis
        sector_path = outputs_dir / "sector_analysis.csv"
        if sector_path.exists():
            analytics["sectors"] = pd.read_csv(sector_path)
        
        # Load country analysis
        country_path = outputs_dir / "country_analysis.csv"
        if country_path.exists():
            analytics["countries"] = pd.read_csv(country_path)
        
        # Load summary statistics
        summary_path = outputs_dir / "summary_statistics.csv"
        if summary_path.exists():
            analytics["summary"] = pd.read_csv(summary_path)
    
    except Exception as e:
        print(f"Error loading analytics: {e}")
    
    return analytics


def get_company_risk_level(default_prob: float) -> Tuple[str, str]:
    """Classify risk level based on default probability"""
    if default_prob >= 0.5:
        return "🔴 Very High Risk", "#e74c3c"
    elif default_prob >= 0.3:
        return "🟠 High Risk", "#e67e22"
    elif default_prob >= 0.15:
        return "🟡 Medium Risk", "#f39c12"
    elif default_prob >= 0.05:
        return "🟢 Low Risk", "#2ecc71"
    else:
        return "🟢 Very Low Risk", "#27ae60"


def format_currency(value: float, currency: str = "€") -> str:
    """Format number as currency"""
    return f"{currency}{value:,.2f}"


def format_percentage(value: float) -> str:
    """Format number as percentage"""
    return f"{value:.1%}"


def format_ratio(value: float) -> str:
    """Format number as ratio"""
    return f"{value:.2f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    if denominator == 0:
        return default
    return numerator / denominator


def highlight_high_values(val: float, threshold: float = 0.5) -> str:
    """Return color for high values"""
    if val >= threshold:
        return "🔴"
    return "🟢"


class ResultsManager:
    """Manage scoring results"""
    
    def __init__(self, results_path: str = "results/scored_companies.csv"):
        self.results_path = Path(results_path)
        self.results = None
        self.load()
    
    def load(self):
        """Load results from file"""
        if self.results_path.exists():
            self.results = pd.read_csv(self.results_path)
            self.results = self.results.sort_values("default_probability", ascending=False)
    
    def has_results(self) -> bool:
        """Check if results exist"""
        return self.results is not None and len(self.results) > 0
    
    def get_high_risk(self, threshold: float = 0.3) -> pd.DataFrame:
        """Get high risk companies"""
        if not self.has_results():
            return pd.DataFrame()
        return self.results[self.results["default_probability"] >= threshold]
    
    def get_by_sector(self, sector: str) -> pd.DataFrame:
        """Filter by sector"""
        if not self.has_results():
            return pd.DataFrame()
        return self.results[self.results["sector"] == sector]
    
    def get_by_country(self, country: str) -> pd.DataFrame:
        """Filter by country"""
        if not self.has_results():
            return pd.DataFrame()
        return self.results[self.results["country"] == country]
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics"""
        if not self.has_results():
            return {}
        
        return {
            "total": len(self.results),
            "predicted_default": self.results["predicted_default"].sum(),
            "predicted_safe": (self.results["predicted_default"] == 0).sum(),
            "avg_prob": self.results["default_probability"].mean(),
            "median_prob": self.results["default_probability"].median(),
            "max_prob": self.results["default_probability"].max(),
            "min_prob": self.results["default_probability"].min(),
        }


class TrainingConfig:
    """Manage training configuration"""
    
    def __init__(self):
        self.config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "finetune": False,
            "finetune_frac": 0.4,
            "finetune_epochs": 3,
            "hptune": False,
            "hptune_trials": 50,
            "hptune_inner_folds": 3,
            "generate_memos": False,
        }
    
    def to_command(self) -> str:
        """Generate training command"""
        cmd = ".\\venv\\Scripts\\python Train\\train.py"
        
        if self.config["finetune"]:
            cmd += f" --finetune --finetune_frac {self.config['finetune_frac']}"
            cmd += f" --finetune_epochs {self.config['finetune_epochs']}"
        
        if self.config["hptune"]:
            cmd += f" --hptune --hptune_trials {self.config['hptune_trials']}"
            cmd += f" --hptune_inner_folds {self.config['hptune_inner_folds']}"
        
        if self.config["generate_memos"]:
            cmd += " --memos"
        
        return cmd
    
    def update(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
