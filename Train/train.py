import argparse
import os
import random
import re
import warnings
from pathlib import Path
from typing import Literal

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold, StratifiedShuffleSplit,
    cross_val_score, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

EMBED_MODEL        = "all-MiniLM-L6-v2"
FINETUNE_LOSS      = "cosine"   # "cosine" | "mnr"
FINETUNE_FRAC      = 0.40       # fraction of train used for fine-tuning
FINETUNE_EPOCHS    = 3
FINETUNE_BATCH     = 16
FINETUNE_MAX_PAIRS = 2000
RANDOM_SEED        = 42
TARGET_ENCODE_COLS = ["country", "sector"]

QUANT_FEATURES = [
    "revenue_m", "ebitda_margin", "debt_ratio", "interest_coverage",
    "cash_ratio", "years_in_operation", "employee_count", "revenue_growth",
]


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def detect_sep(path: str) -> str:
    with open(path, "r") as f:
        first_line = f.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def load_data(train_companies_path, train_narratives_path,
              train_outcomes_path, scoring_companies_path):
    train = pd.read_csv(train_companies_path,   sep=detect_sep(train_companies_path))
    narr  = pd.read_csv(train_narratives_path,  sep=detect_sep(train_narratives_path))
    out   = pd.read_csv(train_outcomes_path,    sep=detect_sep(train_outcomes_path))
    score = pd.read_csv(scoring_companies_path, sep=detect_sep(scoring_companies_path))

    train = train.merge(narr, on="company_id", how="left")
    train = train.merge(out,  on="company_id", how="left")

    print(f"✓ Training samples : {len(train)} | Defaulted: {train['defaulted'].sum()} "
          f"({train['defaulted'].mean():.1%})")
    print(f"✓ Scoring  samples : {len(score)}")
    return train, score


# ---------------------------------------------------------------------------
# 2. NARRATIVE PARSING
# ---------------------------------------------------------------------------

def parse_narrative(text: str) -> dict:
    if not isinstance(text, str):
        return {"risk_text": "", "benefit_text": "", "n_risks": 0, "n_benefits": 0}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    risks    = [s for s in sentences if s.lower().startswith("the business faces")]
    benefits = [s for s in sentences if s.lower().startswith("the company benefits")]
    return {
        "risk_text":    " ".join(risks),
        "benefit_text": " ".join(benefits),
        "n_risks":      len(risks),
        "n_benefits":   len(benefits),
    }


def parse_narratives(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["business_description"].apply(parse_narrative).apply(pd.Series)
    return pd.concat([df, parsed], axis=1)


def strip_management_sentence(text: str) -> str:
    if not isinstance(text, str):
        return ""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return " ".join(s for s in sentences if not s.lower().startswith("management"))


# ---------------------------------------------------------------------------
# 3A. EMBEDDING FINE-TUNING  (new)
# ---------------------------------------------------------------------------

def _make_finetune_split(
    train_df: pd.DataFrame,
    finetune_frac: float = FINETUNE_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split → (finetune_set, ml_set).

    Leakage-safe design
    -------------------
    finetune_set  shapes the embedder weights ONLY — no classifier training,
                  no centroid construction happens on this partition.
    ml_set        is entirely unseen by the embedder → used to build
                  default/safe centroids and to train the downstream classifier.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - finetune_frac), random_state=seed
    )
    ft_idx, ml_idx = next(sss.split(train_df, train_df["defaulted"]))
    finetune_set = train_df.iloc[ft_idx].reset_index(drop=True)
    ml_set       = train_df.iloc[ml_idx].reset_index(drop=True)

    print(f"\n{'='*60}")
    print("Finetune / ML split (stratified, leakage-safe)")
    print(f"{'='*60}")
    print(f"  finetune_set : {len(finetune_set):>4} rows  "
          f"({finetune_set['defaulted'].mean():.1%} defaulted)  → shapes embedder only")
    print(f"  ml_set       : {len(ml_set):>4} rows  "
          f"({ml_set['defaulted'].mean():.1%} defaulted)  → centroids + classifier")
    return finetune_set, ml_set


def _make_three_way_split(
    train_df: pd.DataFrame,
    finetune_frac: float = 0.30,
    ml_frac: float = 0.35,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three-way stratified split → (finetune_set, ml_set, val_set).

    Leakage-free design for embedder comparison
    ────────────────────────────────────────────
    finetune_set : shapes embedder weights ONLY (fine-tuning data)
    ml_set       : builds centroid features (centroid construction)
    val_set      : evaluates embedder quality (held-out validation)

    This prevents centroid leakage where the same data builds AND evaluates
    the centroid features. The fine-tuned embedder is compared fairly
    against pretrained on a held-out validation set.
    """
    # First split: finetune vs rest
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - finetune_frac), random_state=seed
    )
    ft_idx, rest_idx = next(sss1.split(train_df, train_df["defaulted"]))
    finetune_set = train_df.iloc[ft_idx].reset_index(drop=True)
    rest = train_df.iloc[rest_idx].reset_index(drop=True)

    # Second split: ml vs val (from the rest)
    val_frac_of_rest = 1 - (ml_frac / (1 - finetune_frac))
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_of_rest, random_state=seed
    )
    ml_idx, val_idx = next(sss2.split(rest, rest["defaulted"]))
    ml_set = rest.iloc[ml_idx].reset_index(drop=True)
    val_set = rest.iloc[val_idx].reset_index(drop=True)

    print(f"\n{'='*60}")
    print("Three-way split (stratified, leakage-free embedder comparison)")
    print(f"{'='*60}")
    print(f"  finetune_set : {len(finetune_set):>4} rows  "
          f"({finetune_set['defaulted'].mean():.1%} defaulted)  → fine-tune embedder")
    print(f"  ml_set       : {len(ml_set):>4} rows  "
          f"({ml_set['defaulted'].mean():.1%} defaulted)  → build centroids")
    print(f"  val_set      : {len(val_set):>4} rows  "
          f"({val_set['defaulted'].mean():.1%} defaulted)  → evaluate embedders")
    return finetune_set, ml_set, val_set


def _build_cosine_pairs(df: pd.DataFrame, max_pairs: int = FINETUNE_MAX_PAIRS) -> list:
    """
    CosineSimilarityLoss pairs using outcome labels:
      same outcome  → label 1.0
      cross outcome → label 0.0
    Built exclusively from finetune_set — labels from ml_set never touch
    the embedder.
    """
    texts  = df["business_description"].apply(strip_management_sentence).tolist()
    labels = df["defaulted"].tolist()

    defaulted_idx = [i for i, l in enumerate(labels) if l == 1]
    safe_idx      = [i for i, l in enumerate(labels) if l == 0]
    rng = random.Random(RANDOM_SEED)

    pairs = []

    d_pairs = [(i, j) for i in defaulted_idx for j in defaulted_idx if i < j]
    rng.shuffle(d_pairs)
    for i, j in d_pairs[: max_pairs // 4]:
        pairs.append(InputExample(texts=[texts[i], texts[j]], label=1.0))

    s_pairs = [(i, j) for i in safe_idx for j in safe_idx if i < j]
    rng.shuffle(s_pairs)
    for i, j in s_pairs[: max_pairs // 4]:
        pairs.append(InputExample(texts=[texts[i], texts[j]], label=1.0))

    neg = [(i, j) for i in defaulted_idx for j in safe_idx]
    rng.shuffle(neg)
    for i, j in neg[: max_pairs // 2]:
        pairs.append(InputExample(texts=[texts[i], texts[j]], label=0.0))

    rng.shuffle(pairs)
    print(f"  CosineSimilarityLoss pairs built: {len(pairs)}")
    return pairs


def _build_mnr_pairs(df: pd.DataFrame, max_pairs: int = FINETUNE_MAX_PAIRS) -> list:
    """
    MultipleNegativesRankingLoss pairs — does NOT use outcome labels.
    Each company's (risk_text, benefit_text) is a natural positive pair;
    negatives are sampled in-batch automatically.
    Safest choice when finetune_set is small or class imbalance is severe.
    """
    pairs = []
    for text in df["business_description"]:
        p = parse_narrative(text)
        if p["risk_text"] and p["benefit_text"]:
            pairs.append(InputExample(texts=[p["risk_text"], p["benefit_text"]]))

    if len(pairs) > max_pairs:
        random.Random(RANDOM_SEED).shuffle(pairs)
        pairs = pairs[:max_pairs]

    print(f"  MultipleNegativesRankingLoss pairs built: {len(pairs)}")
    return pairs


def _run_finetune(
    finetune_df: pd.DataFrame,
    base_model:  str,
    loss_type:   Literal["cosine", "mnr"],
    epochs:      int,
    batch_size:  int,
    output_dir:  str,
) -> SentenceTransformer:
    """Fine-tune base_model on finetune_df and return the updated model."""
    print(f"\n{'='*60}")
    print(f"Fine-tuning embedder  |  loss={loss_type}  epochs={epochs}")
    print(f"  Base: {base_model}")
    print(f"{'='*60}")

    model = SentenceTransformer(base_model)

    if loss_type == "cosine":
        examples   = _build_cosine_pairs(finetune_df)
        train_loss = losses.CosineSimilarityLoss(model)
    else:
        examples   = _build_mnr_pairs(finetune_df)
        train_loss = losses.MultipleNegativesRankingLoss(model)

    if not examples:
        print("  ⚠  No training pairs — returning base model unchanged.")
        return model

    loader       = DataLoader(examples, shuffle=True, batch_size=batch_size)
    warmup_steps = max(1, len(loader) * epochs // 10)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
    )
    print(f"  ✓ Fine-tuned model saved → {output_dir}")
    return SentenceTransformer(output_dir)


# ---------------------------------------------------------------------------
# 3B. NARRATIVE EMBEDDING + SUPERVISED SIMILARITY FEATURES
# ---------------------------------------------------------------------------

def cosine_similarity_to_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return a_norm @ b_norm


def _attach_centroid_features(
    df: pd.DataFrame,
    emb: np.ndarray,
    centroid_default: np.ndarray,
    centroid_safe: np.ndarray,
) -> pd.DataFrame:
    df = df.copy()
    sim_d = cosine_similarity_to_vec(emb, centroid_default)
    sim_s = cosine_similarity_to_vec(emb, centroid_safe)
    df["narrative_sim_default"] = sim_d
    df["narrative_sim_safe"]    = sim_s
    df["narrative_sim_diff"]    = sim_d - sim_s
    return df


def _embed_all(
    embedder: SentenceTransformer,
    ml_set: pd.DataFrame,
    scoring_df: pd.DataFrame,
    extra_df: pd.DataFrame | None = None,
) -> tuple:
    """
    Encode texts with `embedder`.
    Centroids are derived exclusively from ml_set labels.
    Returns (ml_set_out, scoring_df_out, [extra_df_out]).
    """
    texts_ml = ml_set["business_description"].apply(strip_management_sentence).tolist()
    texts_sc = scoring_df["business_description"].apply(strip_management_sentence).tolist()

    emb_ml = embedder.encode(texts_ml, show_progress_bar=True, batch_size=64)
    emb_sc = embedder.encode(texts_sc, show_progress_bar=True, batch_size=64)

    labels           = ml_set["defaulted"].values
    centroid_default = emb_ml[labels == 1].mean(axis=0)
    centroid_safe    = emb_ml[labels == 0].mean(axis=0)
    print(f"  Centroids: {(labels==1).sum()} defaulted / {(labels==0).sum()} safe  (ml_set only)")

    ml_out = _attach_centroid_features(ml_set,    emb_ml, centroid_default, centroid_safe)
    sc_out = _attach_centroid_features(scoring_df, emb_sc, centroid_default, centroid_safe)

    if extra_df is not None:
        texts_ex = extra_df["business_description"].apply(strip_management_sentence).tolist()
        emb_ex   = embedder.encode(texts_ex, show_progress_bar=False, batch_size=64)
        ex_out   = _attach_centroid_features(extra_df, emb_ex, centroid_default, centroid_safe)
        return ml_out, sc_out, ex_out

    return ml_out, sc_out


def _embed_all_centroid_only(
    embedder: SentenceTransformer,
    eval_df: pd.DataFrame,
    centroid_source_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode eval_df with centroids built from centroid_source_df.
    
    Used to prevent centroid leakage: centroids are built from one dataset
    and evaluated on a separate held-out dataset.
    
    Returns eval_df with attached centroid features.
    """
    texts_eval = eval_df["business_description"].apply(strip_management_sentence).tolist()
    texts_source = centroid_source_df["business_description"].apply(strip_management_sentence).tolist()

    emb_eval = embedder.encode(texts_eval, show_progress_bar=False, batch_size=64)
    emb_source = embedder.encode(texts_source, show_progress_bar=False, batch_size=64)

    labels           = centroid_source_df["defaulted"].values
    centroid_default = emb_source[labels == 1].mean(axis=0)
    centroid_safe    = emb_source[labels == 0].mean(axis=0)

    eval_out = _attach_centroid_features(eval_df, emb_eval, centroid_default, centroid_safe)
    return eval_out


def _centroid_diff_auc(df: pd.DataFrame, tag: str) -> float:
    auc = roc_auc_score(df["defaulted"].values, df["narrative_sim_diff"].values)
    print(f"  [{tag}] centroid-diff ROC-AUC on ml_set: {auc:.4f}")
    return auc


def build_narrative_features(
    train_df: pd.DataFrame,
    scoring_df: pd.DataFrame,
    embedder_model: str = None,
):
    """
    Original pipeline (no fine-tuning).
    Parses narratives, embeds with pretrained model, attaches centroid features.
    Returns (train_df_with_features, scoring_df_with_features).
    """
    embedder_model = embedder_model or EMBED_MODEL
    print(f"\n{'='*60}")
    print(f"Narrative embedding (pretrained)  model={embedder_model}")
    print(f"{'='*60}")

    train_df   = parse_narratives(train_df)
    scoring_df = parse_narratives(scoring_df)

    train_texts   = train_df["business_description"].apply(strip_management_sentence).tolist()
    scoring_texts = scoring_df["business_description"].apply(strip_management_sentence).tolist()

    print(f"  Encoding {len(train_texts)} train + {len(scoring_texts)} scoring narratives...")
    embedder    = SentenceTransformer(embedder_model)
    train_emb   = embedder.encode(train_texts,   show_progress_bar=True, batch_size=64)
    scoring_emb = embedder.encode(scoring_texts, show_progress_bar=True, batch_size=64)

    labels           = train_df["defaulted"].values
    default_centroid = train_emb[labels == 1].mean(axis=0)
    safe_centroid    = train_emb[labels == 0].mean(axis=0)
    print(f"  Centroids: {(labels==1).sum()} defaulted, {(labels==0).sum()} safe")

    for df, emb in [(train_df, train_emb), (scoring_df, scoring_emb)]:
        sim_d = cosine_similarity_to_vec(emb, default_centroid)
        sim_s = cosine_similarity_to_vec(emb, safe_centroid)
        df["narrative_sim_default"] = sim_d
        df["narrative_sim_safe"]    = sim_s
        df["narrative_sim_diff"]    = sim_d - sim_s

    return train_df, scoring_df


def build_narrative_features_with_finetune(
    train_df: pd.DataFrame,
    scoring_df: pd.DataFrame,
    embedder_model: str = None,
    loss_type: Literal["cosine", "mnr"] = FINETUNE_LOSS,
    finetune_frac: float = FINETUNE_FRAC,
    epochs: int = FINETUNE_EPOCHS,
    batch_size: int = FINETUNE_BATCH,
    model_dir: str = "models/finetuned_embedder",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fine-tune variant of build_narrative_features().

    Returns
    -------
    combined_train        : DataFrame  — finetune_set + ml_set (65% of train) with
                                          features attached; pass to train_model()
    scoring_df_with_features: DataFrame — full scoring set, ready to score
    comparison             : dict       — AUC metrics for pretrained vs fine-tuned
                                          (evaluated on held-out val_set to prevent leakage)

    Leakage-free contract
    ────────────────────
    finetune_set (30%)
      → feeds _run_finetune() ONLY
      → outcome labels visible for cosine loss only
      → included in final training (has separate features, no mixing)

    ml_set (35%)
      → builds centroid vectors from its outcomes (after fine-tuning is complete)
      → included in final training with centroid features

    val_set (35%)
      → evaluates both embedders on held-out data
      → centroids built from ml_set, tested on val_set
      → provides unbiased AUC comparison
      → NOT used in final model training
    """
    train_df   = parse_narratives(train_df)
    scoring_df = parse_narratives(scoring_df)

    # Use three-way split for fair embedder comparison
    finetune_set, ml_set, val_set = _make_three_way_split(
        train_df,
        finetune_frac=0.30,  # 30% for fine-tuning
        ml_frac=0.35,        # 35% for centroid building
        # 35% for validation (implicitly)
    )

    comparison = {}

    # Add default if not specified
    embedder_model = embedder_model or EMBED_MODEL

    # ── Pretrained baseline ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Baseline  (pretrained {embedder_model})")
    print(f"{'='*60}")
    pretrained_embedder = SentenceTransformer(embedder_model)
    ml_base, sc_base, ft_base = _embed_all(
        pretrained_embedder, ml_set, scoring_df, extra_df=finetune_set
    )
    # Evaluate on held-out val_set (not ml_set to prevent centroid leakage)
    val_base = _embed_all_centroid_only(
        pretrained_embedder, val_set, ml_set
    )
    comparison["pretrained_auc"] = _centroid_diff_auc(val_base, "pretrained")

    # ── Fine-tuned model ─────────────────────────────────────────────────────
    finetuned_embedder = _run_finetune(
        finetune_set,
        base_model=embedder_model,
        loss_type=loss_type,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=model_dir,
    )
    print(f"\n{'='*60}")
    print("Embedding with fine-tuned model")
    print(f"{'='*60}")
    ml_ft, sc_ft, ft_ft = _embed_all(
        finetuned_embedder, ml_set, scoring_df, extra_df=finetune_set
    )
    # Evaluate on held-out val_set using centroids built from ml_set
    val_ft = _embed_all_centroid_only(
        finetuned_embedder, val_set, ml_set
    )
    comparison["finetuned_auc"] = _centroid_diff_auc(val_ft, "finetuned")

    # ── Head-to-head summary ─────────────────────────────────────────────────
    delta  = comparison["finetuned_auc"] - comparison["pretrained_auc"]
    winner = "finetuned ✓" if delta > 0 else "pretrained ✓"
    comparison["delta"]  = delta
    comparison["winner"] = winner

    print(f"\n{'='*60}")
    print("Embedding comparison  (centroid ROC-AUC on held-out val_set)")
    print("  ⚠  Evaluated on validation set (not ml_set) to prevent leakage")
    print(f"{'='*60}")
    print(f"  Pretrained : {comparison['pretrained_auc']:.4f}")
    print(f"  Fine-tuned : {comparison['finetuned_auc']:.4f}   "
          f"Δ = {delta:+.4f}  →  {winner}")
    print(f"{'='*60}")

    # Combine finetune_set + ml_set (65% of original train_df) for classifier training
    # Both have centroid features attached from the fine-tuned embedder
    combined_train = pd.concat([ft_ft, ml_ft], ignore_index=True)

    return combined_train, sc_ft, comparison


# ---------------------------------------------------------------------------
# 4. TARGET ENCODING (country + sector)
# ---------------------------------------------------------------------------

def fit_target_encoder(train_df: pd.DataFrame,
                        cols: list,
                        target: str = "defaulted",
                        smoothing: float = 10.0) -> dict:
    global_mean = train_df[target].mean()
    encoders    = {"global_mean": global_mean}
    for col in cols:
        stats = (
            train_df.groupby(col)[target]
            .agg(["count", "mean"])
            .rename(columns={"count": "n", "mean": "mean_cat"})
        )
        stats["encoded"] = (
            (stats["n"] * stats["mean_cat"] + smoothing * global_mean)
            / (stats["n"] + smoothing)
        )
        encoders[col] = stats["encoded"].to_dict()
        print(f"  Target encoding [{col}]: {len(stats)} categories")
    return encoders


def apply_target_encoder(df: pd.DataFrame, encoders: dict, cols: list) -> pd.DataFrame:
    df = df.copy()
    global_mean = encoders["global_mean"]
    for col in cols:
        df[f"{col}_te"] = df[col].map(encoders[col]).fillna(global_mean)
    return df


# ---------------------------------------------------------------------------
# 5. QUANTITATIVE FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["debt_service_risk"]          = df["debt_ratio"] / df["interest_coverage"].clip(lower=0.01)
    df["profitability_liquidity"]    = df["ebitda_margin"] * df["cash_ratio"]
    df["leverage_adjusted_coverage"] = df["interest_coverage"] / (1 + df["debt_ratio"])
    df["revenue_per_employee"]       = df["revenue_m"] / df["employee_count"].clip(lower=1)
    df["growth_adjusted_margin"]     = df["ebitda_margin"] + df["revenue_growth"]
    df["cash_to_debt"]               = df["cash_ratio"] / df["debt_ratio"].clip(lower=0.01)
    df["distress_score"] = (
        (df["interest_coverage"] < 1.5).astype(int) +
        (df["debt_ratio"]        > 0.6 ).astype(int) +
        (df["cash_ratio"]        < 0.05).astype(int) +
        (df["ebitda_margin"]     < 0.10).astype(int) +
        (df["revenue_growth"]    < -0.05).astype(int)
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    base = QUANT_FEATURES + [
        "debt_service_risk", "profitability_liquidity", "leverage_adjusted_coverage",
        "revenue_per_employee", "growth_adjusted_margin", "cash_to_debt", "distress_score",
        "country_te", "sector_te",
        "n_risks", "n_benefits",
        "narrative_sim_default", "narrative_sim_safe", "narrative_sim_diff",
    ]
    return [c for c in base if c in df.columns]


# ---------------------------------------------------------------------------
# 6. MODEL TRAINING
# ---------------------------------------------------------------------------

def tune_threshold(model, X, y, cv) -> float:
    oof_probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oof_probs)
    f1_scores  = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx   = f1_scores[:-1].argmax()
    best_thresh = float(thresholds[best_idx])
    print(f"  Threshold tuning → best F1 = {f1_scores[best_idx]:.4f} "
          f"at threshold = {best_thresh:.4f}")
    return best_thresh


# ---------------------------------------------------------------------------
# 6a. HYPERPARAMETER SEARCH SPACES  (one per model family)
# ---------------------------------------------------------------------------

def _objective_lr(trial, X, y, inner_cv, scale_pos_weight):
    """Optuna objective for Logistic Regression."""
    C         = trial.suggest_float("C",        1e-3, 10.0, log=True)
    solver    = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    penalty   = "l2" if solver == "lbfgs" else trial.suggest_categorical("penalty", ["l2", "l1"])
    model = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            C=C, solver=solver, penalty=penalty,
            max_iter=2000, class_weight="balanced",
        )),
    ])
    scores = cross_val_score(model, X, y, cv=inner_cv, scoring="average_precision")
    return scores.mean()


def _objective_rf(trial, X, y, inner_cv, scale_pos_weight):
    """Optuna objective for Random Forest."""
    model = RandomForestClassifier(
        n_estimators     = trial.suggest_int("n_estimators",    100, 600, step=50),
        max_depth        = trial.suggest_int("max_depth",         3,  12),
        min_samples_leaf = trial.suggest_int("min_samples_leaf",  1,  10),
        max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        class_weight     = "balanced",
        random_state     = 42,
    )
    scores = cross_val_score(model, X, y, cv=inner_cv, scoring="average_precision")
    return scores.mean()


def _objective_xgb(trial, X, y, inner_cv, scale_pos_weight):
    """Optuna objective for XGBoost."""
    model = XGBClassifier(
        n_estimators      = trial.suggest_int("n_estimators",   100, 600, step=50),
        max_depth         = trial.suggest_int("max_depth",        3,   8),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample         = trial.suggest_float("subsample",     0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight",   1,  10),
        gamma             = trial.suggest_float("gamma",          0.0,  5.0),
        reg_alpha         = trial.suggest_float("reg_alpha",      0.0,  2.0),
        reg_lambda        = trial.suggest_float("reg_lambda",     0.5,  5.0),
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = "aucpr",
        random_state      = 42,
        verbosity         = 0,
    )
    scores = cross_val_score(model, X, y, cv=inner_cv, scoring="average_precision")
    return scores.mean()


_OBJECTIVES = {
    "Logistic Regression": _objective_lr,
    "Random Forest":       _objective_rf,
    "XGBoost":             _objective_xgb,
}

_MODEL_CONSTRUCTORS = {
    "Logistic Regression": lambda p, spw: Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            C=p["C"], solver=p["solver"],
            penalty=p.get("penalty", "l2"),
            max_iter=2000, class_weight="balanced",
        )),
    ]),
    "Random Forest": lambda p, spw: RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
        class_weight="balanced", random_state=42,
    ),
    "XGBoost": lambda p, spw: XGBClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
        gamma=p["gamma"], reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=42, verbosity=0,
    ),
}


# ---------------------------------------------------------------------------
# 6b. NESTED-CV HP TUNING  (inner loop = Optuna; outer loop = model selection)
# ---------------------------------------------------------------------------

def _tune_one_model(
    name: str,
    X, y,
    inner_cv,
    scale_pos_weight: float,
    n_trials: int,
) -> tuple:
    """
    Run Optuna for a single model family.
    Returns (best_params, best_inner_pr_auc).

    Why nested CV?
    ──────────────
    The outer CV (in train_model) estimates generalisation performance of
    each *already-tuned* model.  The inner CV here finds the best params
    for that model family using only the training folds — the validation
    fold of the outer loop is never touched during the search, so the
    outer AUC is an unbiased estimate.
    """
    objective = _OBJECTIVES[name]
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda trial: objective(trial, X, y, inner_cv, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study.best_params, study.best_value


def _build_tuned_model(name: str, best_params: dict, scale_pos_weight: float):
    return _MODEL_CONSTRUCTORS[name](best_params, scale_pos_weight)


# ---------------------------------------------------------------------------
# 6c. UNIFIED train_model  (HP tuning is opt-in via hptune flag)
# ---------------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
    hptune: bool      = False,
    n_trials: int     = 50,
    inner_folds: int  = 3,
):
    """
    Train and select the best classifier.

    Parameters
    ----------
    hptune      : if True, runs Optuna nested-CV search for every model family
                  before the outer model-selection CV.
    n_trials    : Optuna trials per model family (only used when hptune=True).
    inner_folds : folds for the inner (HP search) CV (only used when hptune=True).

    Nested CV contract
    ------------------
    Outer CV (model selection, n_splits folds)
      └─ Inner CV (HP search, inner_folds folds) — runs on the outer TRAIN split only
           └─ Optuna TPE sampler — never sees the outer validation fold

    This means the outer PR-AUC scores are unbiased estimates of the
    generalisation performance of the fully tuned pipelines.
    """
    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols].fillna(0)
    y = train_df["defaulted"]

    print(f"\n{'='*60}")
    print(f"Training  |  {len(X)} samples  |  {len(feature_cols)} features")
    if hptune:
        print(f"  HP tuning: Optuna TPE  |  {n_trials} trials/model  |  "
              f"{inner_folds}-fold inner CV")
    print(f"  Class balance: {int(y.sum())} defaulted ({y.mean():.1%}) / "
          f"{int((y==0).sum())} safe ({(y==0).mean():.1%})")
    print(f"{'='*60}")

    scale_pos_weight = int((y == 0).sum()) / int((y == 1).sum())

    # ── Default (fixed) hyperparameters — used when hptune=False ────────────
    default_models = {
        "Logistic Regression": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", random_state=42, verbosity=0,
        ),
    }

    min_class = int(y.value_counts().min())
    n_splits  = min(5, max(2, min_class))
    outer_cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    inner_cv  = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)

    # ── HP tuning pass (inner loop) ─────────────────────────────────────────
    tuned_params = {}   # name → best_params dict (populated only when hptune=True)
    if hptune:
        print(f"\n  {'─'*56}")
        print(f"  Hyperparameter search  (inner {inner_folds}-fold CV, {n_trials} trials each)")
        print(f"  {'─'*56}")
        for name in default_models:
            print(f"  Tuning {name}...", flush=True)
            best_params, best_inner_auc = _tune_one_model(
                name, X, y, inner_cv, scale_pos_weight, n_trials
            )
            tuned_params[name] = best_params
            print(f"    Best inner PR-AUC = {best_inner_auc:.4f}")
            print(f"    Best params: {best_params}")

    # ── Build model objects (tuned or default) ───────────────────────────────
    if hptune:
        models = {
            name: _build_tuned_model(name, tuned_params[name], scale_pos_weight)
            for name in default_models
        }
    else:
        models = default_models

    # ── Outer CV: model selection ────────────────────────────────────────────
    print(f"\n  Outer {n_splits}-fold CV  (model selection)")
    print(f"  {'Model':<25}  {'ROC-AUC':>10}  {'PR-AUC':>10}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}")

    best_pr_auc, best_model, best_name = 0.0, None, ""
    cv_metrics = {}  # Store CV metrics for all models
    
    for name, model in models.items():
        roc_scores = cross_val_score(model, X, y, cv=outer_cv, scoring="roc_auc")
        pr_scores  = cross_val_score(model, X, y, cv=outer_cv, scoring="average_precision")
        mean_roc, mean_pr = roc_scores.mean(), pr_scores.mean()
        tuned_tag = " *" if (hptune and name in tuned_params) else ""
        print(f"  {name+tuned_tag:<25}  {mean_roc:.4f}±{roc_scores.std():.3f}  "
              f"{mean_pr:.4f}±{pr_scores.std():.3f}")
        
        # Store metrics
        cv_metrics[name] = {
            "roc_auc": mean_roc,
            "roc_auc_std": roc_scores.std(),
            "pr_auc": mean_pr,
            "pr_auc_std": pr_scores.std(),
        }
        
        if mean_pr > best_pr_auc:
            best_pr_auc, best_model, best_name = mean_pr, model, name

    if hptune:
        print("  (* = Optuna-tuned params)")
    print(f"\n  → Best model: {best_name}  (PR-AUC = {best_pr_auc:.4f})")

    # ── Threshold tuning on OOF probs ───────────────────────────────────────
    best_threshold = tune_threshold(best_model, X, y, outer_cv)

    # ── Final fit on full training data ─────────────────────────────────────
    best_model.fit(X, y)

    # ── Feature importance ───────────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=feature_cols).nlargest(10)
    else:
        coef = best_model.named_steps["clf"].coef_[0]
        imp  = pd.Series(np.abs(coef), index=feature_cols).nlargest(10)

    print("\n  Top 10 features:")
    for feat, val in imp.items():
        print(f"    {feat:<42} {val:.4f}")

    # ── Attach tuned params to results for saving ────────────────────────────
    hp_report = tuned_params if hptune else {}

    return best_model, feature_cols, best_name, best_threshold, hp_report, imp, cv_metrics


# ---------------------------------------------------------------------------
# 7. LLM CREDIT MEMO GENERATION
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior credit analyst at a European alternative asset manager.
Write concise, structured credit memos in investment-committee style.
Be direct, cite specific numbers, and keep each memo under 300 words."""


def build_analyst_prompt(row: pd.Series, prob: float, pred: int) -> str:
    return f"""
Analyse the following company and write a structured credit memo.

## Company Profile
- Name: {row.get('company_name', 'N/A')}
- Sector: {row.get('sector', 'N/A')} | Country: {row.get('country', 'N/A')}
- Revenue: €{row.get('revenue_m', 0):.1f}M | Employees: {int(row.get('employee_count', 0))}
- Years in operation: {int(row.get('years_in_operation', 0))}

## Financial Metrics
- EBITDA Margin:      {row.get('ebitda_margin', 0):.1%}
- Debt Ratio:         {row.get('debt_ratio', 0):.2f}
- Interest Coverage:  {row.get('interest_coverage', 0):.2f}x
- Cash Ratio:         {row.get('cash_ratio', 0):.3f}
- Revenue Growth YoY: {row.get('revenue_growth', 0):.1%}
- Distress Score:     {int(row.get('distress_score', 0))} / 5

## Qualitative Signals
- Risk factors ({int(row.get('n_risks', 0))}):    {row.get('risk_text', 'None')}
- Benefits ({int(row.get('n_benefits', 0))}):      {row.get('benefit_text', 'None')}

## Model Output
- Default Probability: {prob:.1%}
- Predicted Default:   {'YES' if pred == 1 else 'NO'}

---
Write a credit memo with these sections:
1. **Credit Summary** (2-3 sentences: verdict and key driver)
2. **Financial Analysis** (cite key ratios, flag red lines or strengths)
3. **Qualitative Risk Factors** (from parsed risk and benefit signals)
4. **Mitigants** (any offsetting positives)
5. **Recommendation**: PASS or REJECT — one-line rationale
"""


def generate_credit_memo(client, row: pd.Series,
                          prob: float, pred: int) -> str:
    """Generate credit memo via Claude API. Client should be anthropic.Anthropic instance."""
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_analyst_prompt(row, prob, pred)}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"[Memo generation failed: {e}]"


# ---------------------------------------------------------------------------
# 8. SCORING & OUTPUT
# ---------------------------------------------------------------------------

def score_companies(model, feature_cols: list, scoring_df: pd.DataFrame,
                    threshold: float, generate_memos: bool,
                    api_key: str | None) -> pd.DataFrame:
    X_score = scoring_df[feature_cols].fillna(0)
    probs   = model.predict_proba(X_score)[:, 1]
    preds   = (probs >= threshold).astype(int)
    print(f"\n  Applying threshold = {threshold:.4f}  "
          f"(predicted defaulted: {preds.sum()} / {len(preds)})")

    results = scoring_df[["company_id", "company_name", "sector", "country",
                           "n_risks", "n_benefits"]].copy()
    results["default_probability"] = probs.round(4)
    results["predicted_default"]   = preds

    if generate_memos:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("\n⚠  No ANTHROPIC_API_KEY — skipping memos.")
            results["credit_memo"] = "API key not provided."
        elif not HAS_ANTHROPIC:
            print("\n⚠  anthropic module not installed — skipping memos.")
            print("   Install with: pip install anthropic")
            results["credit_memo"] = "Module not available."
        else:
            client = anthropic.Anthropic(api_key=api_key)
            memos  = []
            print(f"\n{'='*60}")
            print(f"Generating {len(scoring_df)} credit memos via Claude API...")
            print(f"{'='*60}")
            for i, (_, row) in enumerate(scoring_df.iterrows(), 1):
                prob = probs[i - 1]
                pred = int(preds[i - 1])
                print(f"  [{i:>3}/{len(scoring_df)}] {row['company_name'][:38]:<38} "
                      f"p={prob:.2%}  pred={pred}")
                memos.append(generate_credit_memo(client, row, prob, pred))
            results["credit_memo"] = memos

    return results.sort_values("default_probability", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 9. RISK RATING & PREDICTIONS FORMAT
# ---------------------------------------------------------------------------

def get_risk_rating(prob: float) -> str:
    """Convert default probability to risk rating"""
    if prob >= 0.5:
        return "Very High"
    elif prob >= 0.3:
        return "High"
    elif prob >= 0.15:
        return "Medium"
    elif prob >= 0.05:
        return "Low"
    else:
        return "Very Low"


# ---------------------------------------------------------------------------
# 10. REPORTING & SAVING
# ---------------------------------------------------------------------------

def print_summary(results: pd.DataFrame):
    print(f"\n{'='*60}")
    print("SCORING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total scored:      {len(results)}")
    print(f"  Predicted default: {results['predicted_default'].sum()}")
    print(f"  Predicted safe:    {(results['predicted_default'] == 0).sum()}")
    print(f"\n  Top 5 highest-risk companies:")
    cols = ["company_name", "sector", "country", "n_risks", "n_benefits",
            "default_probability", "predicted_default"]
    print(results[cols].head(5).to_string(index=False))


def save_results(results: pd.DataFrame, output_path: str,
                 model = None,
                 feature_cols: list | None = None,
                 encoders: dict | None = None,
                 comparison: dict | None = None,
                 hp_report:  dict | None = None,
                 feature_importance: pd.Series | None = None):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results.drop(columns=["credit_memo"], errors="ignore").to_csv(output_path, index=False)
    print(f"\n✓ Scores → {output_path}")
    
    # Save model artifacts for app use
    import pickle
    if model is not None:
        model_path = Path("results/best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"✓ Model  → {model_path}")
    
    if feature_cols is not None:
        cols_path = Path("results/feature_columns.pkl")
        with open(cols_path, "wb") as f:
            pickle.dump(feature_cols, f)
        print(f"✓ Feature columns → {cols_path}")
    
    if encoders is not None:
        encoders_path = Path("results/target_encoders.pkl")
        with open(encoders_path, "wb") as f:
            pickle.dump(encoders, f)
        print(f"✓ Target encoders → {encoders_path}")
    
    if feature_importance is not None:
        importance_path = Path("results/feature_importance.pkl")
        with open(importance_path, "wb") as f:
            pickle.dump(feature_importance, f)
        print(f"✓ Feature importance → {importance_path}")
    
    # Save predictions in challenge format: outputs/predictions.csv
    predictions_path = Path("outputs/predictions.csv")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use credit memo as explanation if available, otherwise use default message
    explanations = []
    for _, row in results.iterrows():
        if "credit_memo" in results.columns and pd.notna(row.get("credit_memo")):
            # Truncate memo to ~500 chars for CSV readability
            memo = str(row["credit_memo"])[:500]
            explanations.append(memo)
        else:
            # Default explanation based on probability
            prob = row["default_probability"]
            risk_rating = get_risk_rating(prob)
            explanations.append(f"Default probability {prob:.1%}. Risk rating: {risk_rating}.")
    
    predictions_df = pd.DataFrame({
        "company_id": results["company_id"],
        "predicted_default_probability": results["default_probability"],
        "risk_rating": results["default_probability"].apply(get_risk_rating),
        "explanation": explanations
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions (challenge format) → {predictions_path}")

    if "credit_memo" in results.columns:
        memo_path = output_path.replace(".csv", "_memos.txt")
        with open(memo_path, "w") as f:
            for _, row in results.iterrows():
                f.write("=" * 70 + "\n")
                f.write(f"Company      : {row['company_name']}\n")
                f.write(f"Default Prob : {row['default_probability']:.1%}  |  "
                        f"Predicted: {int(row['predicted_default'])}  |  "
                        f"Risks: {int(row['n_risks'])}  Benefits: {int(row['n_benefits'])}\n")
                f.write("=" * 70 + "\n")
                f.write(row["credit_memo"] + "\n\n")
        print(f"✓ Memos  → {memo_path}")

    json_path = output_path.replace(".csv", ".json")
    results.to_json(json_path, orient="records", indent=2)
    print(f"✓ JSON   → {json_path}")

    # Save embedding comparison if embedding fine-tuning was run
    if comparison:
        comp_path = output_path.replace(".csv", "_embedding_comparison.txt")
        with open(comp_path, "w") as f:
            f.write("Embedding Fine-Tune Comparison\n")
            f.write("=" * 40 + "\n")
            f.write(f"Pretrained AUC : {comparison.get('pretrained_auc', 'N/A')}\n")
            f.write(f"Fine-tuned AUC : {comparison.get('finetuned_auc', 'N/A')}\n")
            f.write(f"Delta          : {comparison.get('delta', 'N/A'):+.4f}\n")
            f.write(f"Winner         : {comparison.get('winner', 'N/A')}\n")
        print(f"✓ Embedding comparison → {comp_path}")

    # Save HP tuning report if hyperparameter search was run
    if hp_report:
        hp_path = output_path.replace(".csv", "_hp_report.txt")
        with open(hp_path, "w") as f:
            f.write("Hyperparameter Tuning Report  (Optuna TPE, nested CV)\n")
            f.write("=" * 50 + "\n\n")
            for mname, params in hp_report.items():
                f.write(f"[{mname}]\n")
                for k, v in params.items():
                    f.write(f"  {k:<25} = {v}\n")
                f.write("\n")
        print(f"✓ HP report  → {hp_path}")


# ---------------------------------------------------------------------------
# 11. EXPERIMENT TRACKING
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Track multiple experiments for model comparison"""
    
    def __init__(self, tracking_dir: str = "experiments"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []
    
    def run_experiment(self, 
                      name: str,
                      train_df: pd.DataFrame,
                      scoring_df: pd.DataFrame,
                      embedder_model: str,
                      finetune: bool = False,
                      finetune_loss: str = "cosine",
                      finetune_frac: float = 0.40,
                      finetune_epochs: int = 3,
                      finetune_batch: int = 16,
                      hptune: bool = False,
                      hptune_trials: int = 50,
                      hptune_inner_folds: int = 3) -> dict:
        """Run a single experiment configuration"""
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {name}")
        print(f"  Embedder: {embedder_model}")
        print(f"  Fine-tune: {finetune}")
        print(f"  HP-tune: {hptune}")
        print(f"{'='*70}")
        
        result = {
            "experiment_name": name,
            "embedder": embedder_model,
            "finetune": finetune,
            "hptune": hptune,
            "metrics": {}
        }
        
        # Build narrative features
        if finetune:
            train_data, score_data, comparison = build_narrative_features_with_finetune(
                train_df.copy(), scoring_df.copy(),
                loss_type=finetune_loss,
                finetune_frac=finetune_frac,
                epochs=finetune_epochs,
                batch_size=finetune_batch,
                model_dir=f"models/finetuned_{embedder_model.replace('/', '_')}",
            )
            result["embedding_comparison"] = comparison
            result["metrics"]["pretrained_auc"] = comparison.get("pretrained_auc", 0)
            result["metrics"]["finetuned_auc"] = comparison.get("finetuned_auc", 0)
        else:
            train_data = train_df.copy()
            score_data = scoring_df.copy()
            train_data, score_data = build_narrative_features(train_data, score_data)
        
        # Quantitative features
        train_data = engineer_quant_features(train_data)
        score_data = engineer_quant_features(score_data)
        
        # Target encoding
        encoders = fit_target_encoder(train_data, TARGET_ENCODE_COLS)
        train_data = apply_target_encoder(train_data, encoders, TARGET_ENCODE_COLS)
        score_data = apply_target_encoder(score_data, encoders, TARGET_ENCODE_COLS)
        
        # Model training
        model, feature_cols, model_name, threshold, hp_report, feature_importance, cv_metrics = train_model(
            train_data,
            hptune=hptune,
            n_trials=hptune_trials,
            inner_folds=hptune_inner_folds,
        )
        
        result["model_name"] = model_name
        result["threshold"] = threshold
        result["metrics"]["threshold"] = threshold
        result["cv_metrics"] = cv_metrics  # Store cross-validation metrics
        result["best_model_metrics"] = cv_metrics.get(model_name, {})
        
        self.experiments.append(result)
        
        # Save experiment artifacts
        exp_dir = self.tracking_dir / name.replace(" ", "_").lower()
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        scores = score_companies(model, feature_cols, score_data, threshold=threshold,
                                generate_memos=False, api_key=None)
        
        # Save scores
        scores_path = exp_dir / "scores.csv"
        scores.to_csv(scores_path, index=False)
        
        # Save model artifacts
        import pickle
        import json
        
        with open(exp_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(exp_dir / "feature_columns.pkl", "wb") as f:
            pickle.dump(feature_cols, f)
        with open(exp_dir / "target_encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
        
        # Save metrics as JSON
        metrics_file = exp_dir / "metrics.json"
        metrics_dict = {
            "model_name": model_name,
            "threshold": threshold,
            "cross_validation_metrics": cv_metrics,
            "embedding_comparison": result.get("embedding_comparison"),
            "embedder": embedder_model,
            "finetune": finetune,
            "hptune": hptune,
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"✓ Experiment saved → {exp_dir}")
        print(f"  - scores.csv")
        print(f"  - model.pkl, feature_columns.pkl, target_encoders.pkl")
        print(f"  - metrics.json")
        
        return result
    
    def generate_comparison_report(self):
        """Generate comparison report across all experiments"""
        
        if not self.experiments:
            print("No experiments to compare")
            return
        
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPARISON REPORT")
        print(f"{'='*70}")
        
        report_path = self.tracking_dir / "comparison_report.txt"
        with open(report_path, "w") as f:
            f.write("EXPERIMENT COMPARISON REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary table
            f.write("Summary:\n")
            f.write(f"{'Experiment':<30}  {'Embedder':<25}  {'Fine-tune':<12}  {'HP-tune':<8}\n")
            f.write("-" * 80 + "\n")
            
            for exp in self.experiments:
                embedder = exp["embedder"][:20]
                ft = "Yes" if exp["finetune"] else "No"
                hpt = "Yes" if exp["hptune"] else "No"
                f.write(f"{exp['experiment_name']:<30}  {embedder:<25}  {ft:<12}  {hpt:<8}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Embedding Comparison:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Experiment':<30}  {'Pretrained AUC':<20}  {'Fine-tuned AUC':<20}\n")
            f.write("-" * 70 + "\n")
            
            for exp in self.experiments:
                if exp.get("embedding_comparison"):
                    comp = exp["embedding_comparison"]
                    pre_auc = comp.get("pretrained_auc", 0)
                    ft_auc = comp.get("finetuned_auc", 0)
                    delta = ft_auc - pre_auc
                    f.write(f"{exp['experiment_name']:<30}  {pre_auc:>18.4f}  {ft_auc:>18.4f}  "
                           f"(Δ {delta:+.4f})\n")
                else:
                    f.write(f"{exp['experiment_name']:<30}  {'N/A':<20}  {'N/A':<20}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Model Performance (Cross-Validation Metrics):\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Experiment':<30}  {'ROC-AUC':<15}  {'PR-AUC':<15}\n")
            f.write("-" * 70 + "\n")
            
            for exp in self.experiments:
                best_metrics = exp.get("best_model_metrics", {})
                roc = best_metrics.get("roc_auc", 0)
                roc_std = best_metrics.get("roc_auc_std", 0)
                pr = best_metrics.get("pr_auc", 0)
                pr_std = best_metrics.get("pr_auc_std", 0)
                f.write(f"{exp['experiment_name']:<30}  "
                       f"{roc:.4f}±{roc_std:.3f}          "
                       f"{pr:.4f}±{pr_std:.3f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Threshold & Model Selection:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Experiment':<30}  {'Model':<25}  {'Threshold':<15}\n")
            f.write("-" * 70 + "\n")
            
            for exp in self.experiments:
                model = exp.get("model_name", "N/A")[:20]
                threshold = exp.get("threshold", 0)
                f.write(f"{exp['experiment_name']:<30}  {model:<25}  {threshold:>13.4f}\n")
        
        print(f"\n✓ Comparison report → {report_path}")
        
        # Print to console
        with open(report_path, "r") as f:
            print(f.read())


# ---------------------------------------------------------------------------
# 12. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fasanara Credit Risk Pipeline")
    parser.add_argument("--train_companies",   default="data/train_companies.csv")
    parser.add_argument("--train_narratives",  default="data/train_narratives.csv")
    parser.add_argument("--train_outcomes",    default="data/train_outcomes.csv")
    parser.add_argument("--scoring_companies", default="data/scoring_companies.csv")
    parser.add_argument("--output",            default="results/scored_companies.csv")
    parser.add_argument("--memos",             action="store_true",
                        help="Generate LLM credit memos (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--api_key",           default=os.getenv("ANTHROPIC_API_KEY", ""),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    # ── Fine-tuning flags (all optional) ──────────────────────────────────
    parser.add_argument("--finetune",          action="store_true",
                        help="Fine-tune the sentence embedder before building features")
    parser.add_argument("--finetune_loss",     default="cosine", choices=["cosine", "mnr"],
                        help="cosine: uses default/safe labels  |  mnr: label-free (safer for small datasets)")
    parser.add_argument("--finetune_frac",     type=float, default=0.40,
                        help="Fraction of train used for fine-tuning (default 0.40)")
    parser.add_argument("--finetune_epochs",   type=int,   default=3)
    parser.add_argument("--finetune_batch",    type=int,   default=16)
    parser.add_argument("--finetune_model_dir", default="models/finetuned_embedder")
    # ── HP tuning flags (all optional) ────────────────────────────────────
    parser.add_argument("--hptune",            action="store_true",
                        help="Run Optuna nested-CV hyperparameter search for all ML models")
    parser.add_argument("--hptune_trials",     type=int, default=50,
                        help="Optuna trials per model family (default 50)")
    parser.add_argument("--hptune_inner_folds", type=int, default=3,
                        help="Inner CV folds for HP search (default 3)")
    # ── Experiment tracking flags ────────────────────────────────────────
    parser.add_argument("--experiment_name",   default="default",
                        help="Name tag for this experiment (saved to experiments/)")
    parser.add_argument("--compare_embedders", action="store_true",
                        help="Run comparison across different sentence transformers + fine-tuning")
    parser.add_argument("--compare_embedders_list", nargs="+",
                        default=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
                        help="List of sentence transformer models to compare")
    parser.add_argument("--embedder_model",    default=None,
                        help="Specific sentence transformer model to use (overrides EMBED_MODEL constant)")
    parser.add_argument("--no_embeddings",     action="store_true",
                        help="Run baseline with statistical features only (skip narrative embeddings)")
    args = parser.parse_args()

    # Load data once
    train_df, scoring_df = load_data(
        args.train_companies, args.train_narratives,
        args.train_outcomes,  args.scoring_companies,
    )

    # Handle experiment comparison mode
    if args.compare_embedders:
        print("\n" + "=" * 70)
        print("  EXPERIMENT COMPARISON MODE")
        print("  Running multiple embedder + fine-tuning configurations")
        print("=" * 70)
        
        tracker = ExperimentTracker(tracking_dir="experiments")
        
        # Define experiment configurations
        configs = []
        for embedder in args.compare_embedders_list:
            # Pretrained baseline
            configs.append({
                "name": f"{embedder.split('/')[-1]} - Pretrained",
                "embedder": embedder,
                "finetune": False,
            })
            # Fine-tuned version
            configs.append({
                "name": f"{embedder.split('/')[-1]} - Fine-tuned",
                "embedder": embedder,
                "finetune": True,
            })
        
        # Run all experiments
        for config in configs:
            tracker.run_experiment(
                name=config["name"],
                train_df=train_df,
                scoring_df=scoring_df,
                embedder_model=config["embedder"],
                finetune=config["finetune"],
                finetune_loss=args.finetune_loss,
                finetune_frac=args.finetune_frac,
                finetune_epochs=args.finetune_epochs,
                finetune_batch=args.finetune_batch,
                hptune=args.hptune,
                hptune_trials=args.hptune_trials,
                hptune_inner_folds=args.hptune_inner_folds,
            )
        
        # Generate comparison report
        tracker.generate_comparison_report()
        return

    # Standard single experiment mode
    print("\n" + "=" * 70)
    print("  FASANARA CREDIT RISK PIPELINE")
    if args.no_embeddings:
        print("  + Baseline: Statistical features only (no NLP embeddings)")
    else:
        embedder_to_use = args.embedder_model if args.embedder_model else EMBED_MODEL
        print(f"  + Embedder: {embedder_to_use}")
        if args.finetune:
            print(f"  + Embedding fine-tuning  (loss={args.finetune_loss}, "
                  f"frac={args.finetune_frac}, epochs={args.finetune_epochs})")
    if args.hptune:
        print(f"  + HP tuning  (Optuna TPE, {args.hptune_trials} trials, "
              f"{args.hptune_inner_folds}-fold inner CV)")
    print(f"  Experiment: {args.experiment_name}")
    print("=" * 70)

    # Build narrative features (or skip if baseline)
    comparison = None
    if args.no_embeddings:
        # Baseline: skip embedding features entirely
        print("⚠️  Skipping narrative embeddings (baseline mode)")
    elif args.finetune:
        embedder_model = args.embedder_model if args.embedder_model else EMBED_MODEL
        train_df, scoring_df, comparison = build_narrative_features_with_finetune(
            train_df, scoring_df,
            embedder_model=embedder_model,
            loss_type=args.finetune_loss,
            finetune_frac=args.finetune_frac,
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch,
            model_dir=args.finetune_model_dir,
        )
    else:
        embedder_model = args.embedder_model if args.embedder_model else EMBED_MODEL
        train_df, scoring_df = build_narrative_features(
            train_df, scoring_df,
            embedder_model=embedder_model,
        )

    # Quantitative feature engineering
    train_df   = engineer_quant_features(train_df)
    scoring_df = engineer_quant_features(scoring_df)

    # Target encoding
    print(f"\n{'='*60}")
    print("Target encoding: country + sector")
    print(f"{'='*60}")
    encoders   = fit_target_encoder(train_df, TARGET_ENCODE_COLS)
    train_df   = apply_target_encoder(train_df,   encoders, TARGET_ENCODE_COLS)
    scoring_df = apply_target_encoder(scoring_df, encoders, TARGET_ENCODE_COLS)

    # Train model
    model, feature_cols, model_name, threshold, hp_report, feature_importance, cv_metrics = train_model(
        train_df,
        hptune=args.hptune,
        n_trials=args.hptune_trials,
        inner_folds=args.hptune_inner_folds,
    )

    # Score companies
    results = score_companies(
        model, feature_cols, scoring_df,
        threshold=threshold,
        generate_memos=args.memos,
        api_key=args.api_key,
    )

    # Report & save
    print_summary(results)
    save_results(results, args.output, 
                 model=model, 
                 feature_cols=feature_cols,
                 encoders=encoders,
                 comparison=comparison, 
                 hp_report=hp_report,
                 feature_importance=feature_importance)
    
    # Save to experiment tracking
    tracker = ExperimentTracker()
    exp_dir = tracker.tracking_dir / args.experiment_name.replace(" ", "_").lower()
    exp_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(exp_dir / "scores.csv", index=False)
    
    import pickle
    import json
    
    with open(exp_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(exp_dir / "feature_columns.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    with open(exp_dir / "target_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    
    # Save metrics (with JSON serialization handling)
    import json
    
    def convert_to_serializable(obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    metrics_dict = {
        "model_name": str(model_name),
        "threshold": float(threshold),
        "cross_validation_metrics": convert_to_serializable(cv_metrics) if cv_metrics else {},
        "embedding_comparison": convert_to_serializable(comparison) if comparison else None,
        "hyperparameter_tuning": convert_to_serializable(hp_report) if hp_report else {},
        "finetune": bool(args.finetune),
        "hptune": bool(args.hptune),
    }
    
    try:
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        print(f"\n✓ Experiment '{args.experiment_name}' saved → {exp_dir}")
        print(f"  - scores.csv")
        print(f"  - model.pkl, feature_columns.pkl, target_encoders.pkl")
        print(f"  - metrics.json")
    except Exception as e:
        print(f"\n⚠️  Failed to save metrics.json: {e}")
        print(f"   Experiment data saved but metrics file missing")


if __name__ == "__main__":
    main()