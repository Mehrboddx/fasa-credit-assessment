#!/usr/bin/env python3
"""Test script to verify the leakage fix works correctly."""

import pandas as pd
import numpy as np
from Train.train import _make_three_way_split

# Create dummy training data (mimics actual dataset) with unique identifiers
print("\n" + "="*60)
print("TESTING LEAKAGE FIX: Three-Way Split")
print("="*60)

np.random.seed(42)
df = pd.DataFrame({
    'id': range(1000),  # Unique ID to track rows
    'defaulted': [0]*800 + [1]*200,  # 80% safe, 20% default
})

print(f"\nOriginal dataset: {len(df)} companies, {df['defaulted'].mean():.1%} default")

# Test the three-way split
ft, ml, val = _make_three_way_split(df, finetune_frac=0.30, ml_frac=0.35)

print("\n✓ Three-way split created successfully!")
print(f"\n  Finetune : {len(ft):>4} ({ft['defaulted'].mean():.1%} default)")
print(f"  ML       : {len(ml):>4} ({ml['defaulted'].mean():.1%} default)")
print(f"  Val      : {len(val):>4} ({val['defaulted'].mean():.1%} default)")
print(f"  {'─'*35}")
print(f"  Total    : {len(ft)+len(ml)+len(val):>4} (should be {len(df)})")

# Verify split proportions
total = len(ft) + len(ml) + len(val)
ft_pct = len(ft) / total * 100
ml_pct = len(ml) / total * 100
val_pct = len(val) / total * 100

print(f"\nSplit proportions:")
print(f"  Finetune : {ft_pct:.1f}% (target: 30%)")
print(f"  ML       : {ml_pct:.1f}% (target: 35%)")
print(f"  Val      : {val_pct:.1f}% (target: 35%)")

# Verify no overlap using unique IDs
ft_ids = set(ft['id'].values)
ml_ids = set(ml['id'].values)
val_ids = set(val['id'].values)

print(f"\nOverlap check (using row IDs):")
ft_ml_overlap = len(ft_ids & ml_ids)
ft_val_overlap = len(ft_ids & val_ids)
ml_val_overlap = len(ml_ids & val_ids)
total_coverage = len(ft_ids | ml_ids | val_ids)

print(f"  FT ∩ ML  : {ft_ml_overlap} (should be 0)")
print(f"  FT ∩ Val : {ft_val_overlap} (should be 0)")
print(f"  ML ∩ Val : {ml_val_overlap} (should be 0)")
print(f"  Union    : {total_coverage} (should be {len(df)})")

# Check class balance maintenance
print(f"\nClass balance (stratification check):")
original_default_rate = df['defaulted'].mean()
print(f"  Original : {original_default_rate:.1%}")
print(f"  Finetune : {ft['defaulted'].mean():.1%} (diff: {abs(ft['defaulted'].mean() - original_default_rate):.2%})")
print(f"  ML       : {ml['defaulted'].mean():.1%} (diff: {abs(ml['defaulted'].mean() - original_default_rate):.2%})")
print(f"  Val      : {val['defaulted'].mean():.1%} (diff: {abs(val['defaulted'].mean() - original_default_rate):.2%})")

all_pass = (
    len(ft) + len(ml) + len(val) == len(df) and
    ft_ml_overlap == 0 and
    ft_val_overlap == 0 and
    ml_val_overlap == 0 and
    total_coverage == len(df)
)

if all_pass:
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - Leakage fix is working correctly!")
    print("="*60)
else:
    print("\n" + "="*60)
    print("❌ TESTS FAILED")
    print("="*60)
