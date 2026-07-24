
r"""
CARE-IoT++ ALL-IN-ONE NOVEL ARCHITECTURE PIPELINE
=================================================

Proposed model:
RCS-MoE = Rarity-Conditioned Selective Mixture-of-Experts

Novel trainable components
--------------------------
1. Feature-group tokenization for tabular IoT flow features.
2. Rarity-conditioned multi-head self-attention.
3. Sparse specialist expert bank.
4. Learned three-way router:
      edge head / specialist bank / cloud head.
5. Auxiliary family, rare-class, reconstruction, calibration,
   load-balancing, and cost objectives.
6. Safe deferral and software-only edge/cloud measurements.
7. True component ablations trained from scratch.

The script creates one fixed stratified subset using data-split seed 42:
20% of CICIoT23 training data, 10% of validation data, and 10% of test
data. Every method, ablation, and model seed reuses the exact same rows.

Expected paths
--------------
D:\other\CARE-IoT\Datasets\CICIOT23\train\train.csv
D:\other\CARE-IoT\Datasets\CICIOT23\validation\validation.csv
D:\other\CARE-IoT\Datasets\CICIOT23\test\test.csv

Run
---
conda activate care-iot
cd /d D:\other\CARE-IoT\Code
python careiot_rcsmoe_all_in_one.py

Outputs
-------
D:\other\CARE-IoT\Models\RCSMOE_JOURNAL
D:\other\CARE-IoT\Results\RCSMOE_JOURNAL

Important
---------
This code can provide experimental evidence. It cannot guarantee acceptance
by a journal. Claims must match the measured results.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    recall_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(r"D:\other\CARE-IoT")
DATA_DIR = ROOT / "Datasets" / "CICIOT23"
TRAIN_PATH = DATA_DIR / "train" / "train.csv"
VAL_PATH = DATA_DIR / "validation" / "validation.csv"
TEST_PATH = DATA_DIR / "test" / "test.csv"

RESULT_DIR = ROOT / "Results" / "RCSMOE_JOURNAL"
MODEL_DIR = ROOT / "Models" / "RCSMOE_JOURNAL"
TABLE_DIR = RESULT_DIR / "tables"
FIG_DIR = RESULT_DIR / "figures"
PRED_DIR = RESULT_DIR / "predictions"
LOG_DIR = RESULT_DIR / "logs"

for directory in [RESULT_DIR, MODEL_DIR, TABLE_DIR, FIG_DIR, PRED_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.20
VAL_FRAC = 0.10
TEST_FRAC = 0.10

# The data subset is sampled once and reused for every method and run.
DATA_SPLIT_SEED = 42
MODEL_SEEDS = [11, 22]
SEEDS = MODEL_SEEDS  # backward-compatible alias

FIXED_SPLIT_DIR = ROOT / "Datasets" / "CICIOT23_FIXED_20_10_10"
FIXED_TRAIN_PATH = FIXED_SPLIT_DIR / "train_20pct_seed42.csv"
FIXED_VAL_PATH = FIXED_SPLIT_DIR / "validation_10pct_seed42.csv"
FIXED_TEST_PATH = FIXED_SPLIT_DIR / "test_10pct_seed42.csv"
FIXED_SPLIT_META = FIXED_SPLIT_DIR / "split_metadata.json"
FIXED_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Set False for a smoke test. True trains all ablations across all seeds.
JOURNAL_MODE = True

FULL_EPOCHS = 8 if JOURNAL_MODE else 2
ABLATION_EPOCHS = 6 if JOURNAL_MODE else 1
PATIENCE = 2
BATCH_SIZE = 4096
NUM_WORKERS = 0
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

D_MODEL = 48
N_HEADS = 4
N_LAYERS = 2
N_EXPERTS = 6
TOP_K_EXPERTS = 2
DROPOUT = 0.15

RARE_THRESHOLD = 1000
DPI = 600

# Classical models are included as references, not as the proposed method.
RUN_RANDOM_FOREST = False
RUN_XGBOOST = False

ABLATIONS = {
    # Retrain only the corrected complete model. Existing ablation checkpoints
    # are left untouched.
    "full_rcsmoe": {},
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.2,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.8,
})


# =============================================================================
# REPRODUCIBILITY AND UTILITIES
# =============================================================================

def log(message: str) -> None:
    print(f"[CARE-IoT++ RCS-MoE] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


def save_table(df: pd.DataFrame, stem: str, caption: str, label: str) -> None:
    df.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    df.to_excel(TABLE_DIR / f"{stem}.xlsx", index=False)

    latex = df.to_latex(
        index=False,
        escape=False,
        na_rep="-",
        float_format=lambda x: f"{x:.4f}",
    )
    latex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + latex
        + "}\n\\end{table*}\n"
    )
    (TABLE_DIR / f"{stem}.tex").write_text(latex, encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def confidence_interval(values: Iterable[float]) -> Tuple[float, float, float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if len(x) == 1:
        return mean, std, mean, mean
    margin = stats.t.ppf(0.975, len(x) - 1) * std / math.sqrt(len(x))
    return mean, std, mean - margin, mean + margin


def holm_adjust(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def process_memory_mb() -> float:
    if not HAS_PSUTIL:
        return np.nan
    return psutil.Process().memory_info().rss / (1024 ** 2)


# =============================================================================
# DATA
# =============================================================================

def detect_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "label", "Label", "class", "Class",
        "attack", "Attack", "category", "Category"
    ]
    for column in candidates:
        if column in df.columns:
            return column
    objects = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if objects:
        return objects[-1]
    raise ValueError("Could not detect the label column.")


def stratified_fraction(
    df: pd.DataFrame,
    label_col: str,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    if fraction >= 1.0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    pieces = []
    for _, group in df.groupby(label_col, sort=False):
        n = max(1, int(round(len(group) * fraction)))
        n = min(n, len(group))
        pieces.append(group.sample(n=n, random_state=seed))

    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def family_label(label: str) -> str:
    label = str(label)
    if label == "BenignTraffic":
        return "Benign"
    if label.startswith("DDoS"):
        return "DDoS"
    if label.startswith("DoS"):
        return "DoS"
    if label.startswith("Mirai"):
        return "Mirai"
    if label.startswith("Recon") or label == "VulnerabilityScan":
        return "Recon"
    if "Spoofing" in label:
        return "Spoofing"
    if label in {
        "BrowserHijacking", "CommandInjection", "SqlInjection",
        "XSS", "Uploading_Attack"
    }:
        return "Web"
    if label == "DictionaryBruteForce":
        return "BruteForce"
    if label == "Backdoor_Malware":
        return "Malware"
    return "Other"


def infer_feature_groups(feature_names: List[str]) -> np.ndarray:
    """Assign features to semantic IoT traffic groups using names."""
    groups = []
    for name in feature_names:
        n = name.lower()
        if any(k in n for k in ["duration", "time", "iat"]):
            group = 0  # timing
        elif any(k in n for k in ["rate", "pps", "bps"]):
            group = 1  # rates
        elif any(k in n for k in ["syn", "ack", "fin", "rst", "psh", "urg"]):
            group = 2  # flags
        elif any(k in n for k in ["protocol", "tcp", "udp", "icmp", "http", "dns"]):
            group = 3  # protocol
        elif any(k in n for k in ["header", "payload", "byte", "length", "size"]):
            group = 4  # size/content
        else:
            group = 5  # other statistical
        groups.append(group)
    return np.asarray(groups, dtype=np.int64)


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    family_train: np.ndarray
    family_val: np.ndarray
    family_test: np.ndarray
    rare_train: np.ndarray
    rare_val: np.ndarray
    rare_test: np.ndarray
    feature_names: List[str]
    feature_groups: np.ndarray
    label_encoder: LabelEncoder
    family_encoder: LabelEncoder
    imputer: SimpleImputer
    scaler: StandardScaler
    class_counts: np.ndarray



def fixed_split_files_exist() -> bool:
    return (
        FIXED_TRAIN_PATH.exists()
        and FIXED_VAL_PATH.exists()
        and FIXED_TEST_PATH.exists()
        and FIXED_SPLIT_META.exists()
    )


def create_fixed_stratified_subsets(force_rebuild: bool = False) -> None:
    """
    Create the 20%/10%/10% subsets exactly once using DATA_SPLIT_SEED.

    All model seeds and all ablations subsequently read these same cached CSV
    files. This prevents data-resampling variation from being mixed with
    model-initialization variation.
    """
    if fixed_split_files_exist() and not force_rebuild:
        log(f"Using cached fixed split from: {FIXED_SPLIT_DIR}")
        return

    log("Creating the fixed stratified 20%/10%/10% data split")
    log(f"Data split seed: {DATA_SPLIT_SEED}")

    raw_train = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw_val = pd.read_csv(VAL_PATH, low_memory=False)
    raw_test = pd.read_csv(TEST_PATH, low_memory=False)

    label_col = detect_label_column(raw_train)

    fixed_train = stratified_fraction(
        raw_train, label_col, TRAIN_FRAC, DATA_SPLIT_SEED
    )
    fixed_val = stratified_fraction(
        raw_val, label_col, VAL_FRAC, DATA_SPLIT_SEED
    )
    fixed_test = stratified_fraction(
        raw_test, label_col, TEST_FRAC, DATA_SPLIT_SEED
    )

    log(f"Fixed train shape: {fixed_train.shape}")
    log(f"Fixed validation shape: {fixed_val.shape}")
    log(f"Fixed test shape: {fixed_test.shape}")

    fixed_train.to_csv(FIXED_TRAIN_PATH, index=False)
    fixed_val.to_csv(FIXED_VAL_PATH, index=False)
    fixed_test.to_csv(FIXED_TEST_PATH, index=False)

    metadata = {
        "data_split_seed": DATA_SPLIT_SEED,
        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VAL_FRAC,
        "test_fraction": TEST_FRAC,
        "label_column": label_col,
        "source_paths": {
            "train": str(TRAIN_PATH),
            "validation": str(VAL_PATH),
            "test": str(TEST_PATH),
        },
        "source_shapes": {
            "train": list(raw_train.shape),
            "validation": list(raw_val.shape),
            "test": list(raw_test.shape),
        },
        "fixed_shapes": {
            "train": list(fixed_train.shape),
            "validation": list(fixed_val.shape),
            "test": list(fixed_test.shape),
        },
        "fixed_paths": {
            "train": str(FIXED_TRAIN_PATH),
            "validation": str(FIXED_VAL_PATH),
            "test": str(FIXED_TEST_PATH),
        },
        "class_counts": {
            "train": fixed_train[label_col].astype(str).value_counts().to_dict(),
            "validation": fixed_val[label_col].astype(str).value_counts().to_dict(),
            "test": fixed_test[label_col].astype(str).value_counts().to_dict(),
        },
    }
    save_json(metadata, FIXED_SPLIT_META)

    del (
        raw_train, raw_val, raw_test,
        fixed_train, fixed_val, fixed_test
    )
    gc.collect()
    log("Fixed split created and cached successfully.")


def load_and_prepare(seed: int) -> PreparedData:
    log(
        f"Loading the fixed cached subset for model seed {seed}; "
        f"data split seed remains {DATA_SPLIT_SEED}"
    )
    create_fixed_stratified_subsets(force_rebuild=False)

    train = pd.read_csv(FIXED_TRAIN_PATH, low_memory=False)
    val = pd.read_csv(FIXED_VAL_PATH, low_memory=False)
    test = pd.read_csv(FIXED_TEST_PATH, low_memory=False)

    label_col = detect_label_column(train)

    log(f"Fixed shapes: train={train.shape}, val={val.shape}, test={test.shape}")

    for column in ["Telnet", "IRC"]:
        train.drop(columns=[column], errors="ignore", inplace=True)
        val.drop(columns=[column], errors="ignore", inplace=True)
        test.drop(columns=[column], errors="ignore", inplace=True)

    y_train_text = train[label_col].astype(str).to_numpy()
    y_val_text = val[label_col].astype(str).to_numpy()
    y_test_text = test[label_col].astype(str).to_numpy()

    X_train_df = train.drop(columns=[label_col])
    X_val_df = val.drop(columns=[label_col])
    X_test_df = test.drop(columns=[label_col])

    numeric = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    common = [
        c for c in numeric
        if c in X_val_df.columns and c in X_test_df.columns
    ]

    zero_variance = [
        c for c in common
        if X_train_df[c].replace([np.inf, -np.inf], np.nan).nunique(dropna=True) <= 1
    ]
    feature_names = [c for c in common if c not in zero_variance]

    X_train_df = X_train_df[feature_names].replace([np.inf, -np.inf], np.nan)
    X_val_df = X_val_df[feature_names].replace([np.inf, -np.inf], np.nan)
    X_test_df = X_test_df[feature_names].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train_df)
    X_val = imputer.transform(X_val_df)
    X_test = imputer.transform(X_test_df)

    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    label_encoder = LabelEncoder()
    label_encoder.fit(
        np.concatenate([y_train_text, y_val_text, y_test_text])
    )
    y_train = label_encoder.transform(y_train_text).astype(np.int64)
    y_val = label_encoder.transform(y_val_text).astype(np.int64)
    y_test = label_encoder.transform(y_test_text).astype(np.int64)

    family_encoder = LabelEncoder()
    family_train_text = np.asarray([family_label(x) for x in y_train_text])
    family_val_text = np.asarray([family_label(x) for x in y_val_text])
    family_test_text = np.asarray([family_label(x) for x in y_test_text])
    family_encoder.fit(
        np.concatenate([family_train_text, family_val_text, family_test_text])
    )
    family_train = family_encoder.transform(family_train_text).astype(np.int64)
    family_val = family_encoder.transform(family_val_text).astype(np.int64)
    family_test = family_encoder.transform(family_test_text).astype(np.int64)

    train_counts_text = pd.Series(y_train_text).value_counts()
    rare_labels = set(train_counts_text[train_counts_text < RARE_THRESHOLD].index)
    rare_train = np.asarray([x in rare_labels for x in y_train_text], dtype=np.float32)
    rare_val = np.asarray([x in rare_labels for x in y_val_text], dtype=np.float32)
    rare_test = np.asarray([x in rare_labels for x in y_test_text], dtype=np.float32)

    class_counts = np.bincount(
        y_train,
        minlength=len(label_encoder.classes_)
    ).astype(np.float64)

    feature_groups = infer_feature_groups(feature_names)

    prep_path = MODEL_DIR / f"preprocess_seed_{seed}.joblib"
    joblib.dump({
        "imputer": imputer,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "family_encoder": family_encoder,
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "rare_labels": sorted(rare_labels),
    }, prep_path)

    save_json({
        "model_seed": seed,
        "data_split_seed": DATA_SPLIT_SEED,
        "fixed_split_directory": str(FIXED_SPLIT_DIR),
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "num_classes": len(label_encoder.classes_),
        "num_families": len(family_encoder.classes_),
        "rare_labels": sorted(rare_labels),
        "zero_variance_removed": zero_variance,
        "feature_names": feature_names,
    }, LOG_DIR / f"dataset_audit_seed_{seed}.json")

    del train, val, test, X_train_df, X_val_df, X_test_df
    gc.collect()

    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        family_train=family_train,
        family_val=family_val,
        family_test=family_test,
        rare_train=rare_train,
        rare_val=rare_val,
        rare_test=rare_test,
        feature_names=feature_names,
        feature_groups=feature_groups,
        label_encoder=label_encoder,
        family_encoder=family_encoder,
        imputer=imputer,
        scaler=scaler,
        class_counts=class_counts,
    )


class FlowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family: np.ndarray,
        rare: np.ndarray,
    ):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.family = torch.from_numpy(family)
        self.rare = torch.from_numpy(rare)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            self.X[index],
            self.y[index],
            self.family[index],
            self.rare[index],
        )


# =============================================================================
# PROPOSED ARCHITECTURE
# =============================================================================

class FeatureGroupTokenizer(nn.Module):
    """
    Converts each scalar feature into a learnable token.

    token_i = value_i * W_i + B_i + feature_embedding_i + group_embedding_g(i)
    """

    def __init__(
        self,
        num_features: int,
        feature_groups: np.ndarray,
        d_model: int,
        num_groups: int = 6,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        self.value_weight = nn.Parameter(torch.empty(num_features, d_model))
        self.value_bias = nn.Parameter(torch.zeros(num_features, d_model))
        self.feature_embedding = nn.Embedding(num_features, d_model)
        self.group_embedding = nn.Embedding(num_groups, d_model)

        self.register_buffer(
            "feature_ids",
            torch.arange(num_features, dtype=torch.long),
        )
        self.register_buffer(
            "group_ids",
            torch.as_tensor(feature_groups, dtype=torch.long),
        )

        nn.init.xavier_uniform_(self.value_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value_tokens = x.unsqueeze(-1) * self.value_weight.unsqueeze(0)
        value_tokens = value_tokens + self.value_bias.unsqueeze(0)
        feature_tokens = self.feature_embedding(self.feature_ids).unsqueeze(0)
        group_tokens = self.group_embedding(self.group_ids).unsqueeze(0)
        return value_tokens + feature_tokens + group_tokens


class RarityConditionedEncoder(nn.Module):
    """
    Multi-layer feature attention conditioned on a learned rarity query.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_rarity: bool,
    ):
        super().__init__()
        self.use_rarity = use_rarity

        self.pre_rarity = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.rarity_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
        )

        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norms1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
            )
            for _ in range(n_layers)
        ])
        self.norms2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled0 = tokens.mean(dim=1)
        rarity_logit = self.pre_rarity(pooled0)
        rarity_prob = torch.sigmoid(rarity_logit)

        if self.use_rarity:
            rarity_context = self.rarity_embedding(rarity_prob).unsqueeze(1)
            tokens = tokens + rarity_context

        attention_maps = []
        for attn, norm1, ffn, norm2 in zip(
            self.layers, self.norms1, self.ffns, self.norms2
        ):
            attended, weights = attn(
                tokens, tokens, tokens,
                need_weights=True,
                average_attn_weights=False,
            )
            tokens = norm1(tokens + attended)
            tokens = norm2(tokens + ffn(tokens))
            attention_maps.append(weights)

        pooled = tokens.mean(dim=1)
        stacked_attention = torch.stack(attention_maps, dim=1)
        return pooled, rarity_logit.squeeze(-1), stacked_attention


class SpecialistExpert(nn.Module):
    def __init__(self, d_model: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.net(representation)


class RCSMoE(nn.Module):
    """
    Rarity-Conditioned Selective Mixture-of-Experts.

    Three inference paths:
    0: compact edge head
    1: sparse specialist expert bank
    2: deeper cloud head
    """

    def __init__(
        self,
        num_features: int,
        feature_groups: np.ndarray,
        n_classes: int,
        n_families: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        n_experts: int = N_EXPERTS,
        top_k: int = TOP_K_EXPERTS,
        dropout: float = DROPOUT,
        use_rarity: bool = True,
        use_moe: bool = True,
        use_reconstruction: bool = True,
        force_route: Optional[str] = None,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.use_moe = use_moe
        self.use_reconstruction = use_reconstruction
        self.force_route = force_route

        self.tokenizer = FeatureGroupTokenizer(
            num_features=num_features,
            feature_groups=feature_groups,
            d_model=d_model,
        )
        self.encoder = RarityConditionedEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_rarity=use_rarity,
        )

        self.edge_head = nn.Linear(d_model, n_classes)

        self.experts = nn.ModuleList([
            SpecialistExpert(d_model, n_classes, dropout)
            for _ in range(n_experts)
        ])
        self.expert_gate = nn.Linear(d_model + 3, n_experts)

        self.cloud_head = nn.Sequential(
            nn.Linear(d_model, 3 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, n_classes),
        )

        self.family_head = nn.Linear(d_model, n_families)
        self.reconstruction_head = nn.Linear(d_model, num_features)

        # Router inputs: representation + rarity + uncertainty + OOD score
        self.route_head = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    @staticmethod
    def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=-1)
        return entropy / math.log(logits.shape[-1])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(x)
        representation, rarity_logit, attention = self.encoder(tokens)

        edge_logits = self.edge_head(representation)
        uncertainty = self.normalized_entropy(edge_logits).unsqueeze(-1)

        reconstruction = self.reconstruction_head(representation)
        reconstruction_error = (
            (reconstruction - x).pow(2).mean(dim=-1, keepdim=True)
            if self.use_reconstruction
            else torch.zeros(
                x.shape[0], 1, device=x.device, dtype=x.dtype
            )
        )

        rarity_prob = torch.sigmoid(rarity_logit).unsqueeze(-1)
        route_features = torch.cat(
            [representation, rarity_prob, uncertainty, reconstruction_error],
            dim=-1,
        )

        route_logits = self.route_head(route_features)
        route_prob = torch.softmax(route_logits, dim=-1)

        if self.force_route == "edge":
            route_prob = torch.zeros_like(route_prob)
            route_prob[:, 0] = 1.0
        elif self.force_route == "cloud":
            route_prob = torch.zeros_like(route_prob)
            route_prob[:, 2] = 1.0

        if self.use_moe:
            gate_logits = self.expert_gate(route_features)
            top_values, top_indices = torch.topk(
                gate_logits, k=self.top_k, dim=-1
            )
            top_weights = torch.softmax(top_values, dim=-1)

            expert_outputs = torch.stack(
                [expert(representation) for expert in self.experts],
                dim=1,
            )

            gather_index = top_indices.unsqueeze(-1).expand(
                -1, -1, self.n_classes
            )
            selected = torch.gather(
                expert_outputs, dim=1, index=gather_index
            )
            moe_logits = (
                selected * top_weights.unsqueeze(-1)
            ).sum(dim=1)

            gate_prob = torch.softmax(gate_logits, dim=-1)
        else:
            moe_logits = edge_logits
            gate_prob = torch.full(
                (x.shape[0], self.n_experts),
                1.0 / self.n_experts,
                device=x.device,
                dtype=x.dtype,
            )

        cloud_logits = self.cloud_head(representation)

        final_logits = (
            route_prob[:, 0:1] * edge_logits
            + route_prob[:, 1:2] * moe_logits
            + route_prob[:, 2:3] * cloud_logits
        )

        family_logits = self.family_head(representation)

        return {
            "logits": final_logits,
            "edge_logits": edge_logits,
            "moe_logits": moe_logits,
            "cloud_logits": cloud_logits,
            "family_logits": family_logits,
            "rarity_logit": rarity_logit,
            "route_logits": route_logits,
            "route_prob": route_prob,
            "gate_prob": gate_prob,
            "reconstruction": reconstruction,
            "reconstruction_error": reconstruction_error.squeeze(-1),
            "uncertainty": uncertainty.squeeze(-1),
            "attention": attention,
        }


# =============================================================================
# OBJECTIVES
# =============================================================================

class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: np.ndarray,
        beta: float = 0.9999,
        gamma: float = 2.0,
    ):
        super().__init__()
        counts = np.maximum(class_counts, 1.0)
        effective = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective
        weights = weights / weights.sum() * len(weights)
        self.register_buffer(
            "weights",
            torch.as_tensor(weights, dtype=torch.float32),
        )
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        target_log_prob = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(1, target.unsqueeze(1)).squeeze(1)
        target_weight = self.weights[target]
        loss = -target_weight * (1.0 - target_prob).pow(self.gamma) * target_log_prob
        return loss.mean()


@dataclass
class ObjectiveConfig:
    use_rarity: bool = True
    use_moe: bool = True
    use_cost_loss: bool = True
    use_family_loss: bool = True
    use_reconstruction: bool = True
    force_route: Optional[str] = None

    fine_weight: float = 1.0
    family_weight: float = 0.25
    rare_weight: float = 0.20
    route_weight: float = 0.30
    edge_aux_weight: float = 0.20
    specialist_aux_weight: float = 0.10
    cloud_aux_weight: float = 0.10
    route_balance_weight: float = 0.10
    calibration_weight: float = 0.05
    reconstruction_weight: float = 0.03
    load_balance_weight: float = 0.02
    cost_weight: float = 0.02

    # Among common samples, the most confident fraction is explicitly taught
    # to use the edge path. This prevents the edge route from receiving no
    # targets during early training.
    edge_common_fraction: float = 0.35


class JointObjective(nn.Module):
    def __init__(
        self,
        class_counts: np.ndarray,
        config: ObjectiveConfig,
    ):
        super().__init__()
        self.config = config
        self.focal = ClassBalancedFocalLoss(class_counts)

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        family: torch.Tensor,
        rare: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config

        fine_loss = self.focal(output["logits"], y)
        loss = cfg.fine_weight * fine_loss

        components = {"fine": float(fine_loss.detach())}

        if cfg.use_family_loss:
            family_loss = F.cross_entropy(output["family_logits"], family)
            loss = loss + cfg.family_weight * family_loss
            components["family"] = float(family_loss.detach())

        if cfg.use_rarity:
            rare_loss = F.binary_cross_entropy_with_logits(
                output["rarity_logit"], rare
            )
            loss = loss + cfg.rare_weight * rare_loss
            components["rare"] = float(rare_loss.detach())

        # Train each path directly. Without these auxiliary losses, the edge
        # head can receive almost no useful gradient after the router starts
        # preferring the specialist path.
        edge_aux_loss = self.focal(output["edge_logits"], y)
        specialist_aux_loss = self.focal(output["moe_logits"], y)
        cloud_aux_loss = self.focal(output["cloud_logits"], y)

        loss = loss + cfg.edge_aux_weight * edge_aux_loss
        loss = loss + cfg.specialist_aux_weight * specialist_aux_loss
        loss = loss + cfg.cloud_aux_weight * cloud_aux_loss
        components["edge_aux"] = float(edge_aux_loss.detach())
        components["specialist_aux"] = float(specialist_aux_loss.detach())
        components["cloud_aux"] = float(cloud_aux_loss.detach())

        # Supervised routing policy:
        #   rare samples -> cloud
        #   easiest common samples -> edge
        #   remaining common samples -> specialist
        #
        # The edge threshold is a batch quantile rather than a fixed 0.80
        # confidence threshold. A fixed threshold is unsafe early in training,
        # because an untrained multiclass edge head rarely reaches 0.80 and
        # therefore receives zero route assignments.
        with torch.no_grad():
            edge_conf = torch.softmax(
                output["edge_logits"], dim=-1
            ).max(dim=-1).values
            common_mask = rare < 0.5
            route_target = torch.full_like(y, 2)  # rare defaults to cloud

            if common_mask.any():
                common_conf = edge_conf[common_mask]
                quantile = max(0.0, min(1.0, 1.0 - cfg.edge_common_fraction))
                edge_threshold = torch.quantile(common_conf, quantile)
                edge_mask = common_mask & (edge_conf >= edge_threshold)
                specialist_mask = common_mask & ~edge_mask
                route_target[edge_mask] = 0
                route_target[specialist_mask] = 1

        if cfg.force_route is None:
            route_loss = F.cross_entropy(
                output["route_logits"], route_target
            )
            loss = loss + cfg.route_weight * route_loss
            components["route"] = float(route_loss.detach())

            # Match the soft mean route allocation to the supervised allocation
            # in the current batch. This discourages route collapse while still
            # allowing the data to determine the actual proportions.
            target_mix = F.one_hot(route_target, num_classes=3).float().mean(dim=0)
            predicted_mix = output["route_prob"].mean(dim=0)
            route_balance_loss = (predicted_mix - target_mix).pow(2).mean()
            loss = loss + cfg.route_balance_weight * route_balance_loss
            components["route_balance"] = float(route_balance_loss.detach())

        # Brier-style calibration term.
        probabilities = torch.softmax(output["logits"], dim=-1)
        one_hot = F.one_hot(
            y, num_classes=probabilities.shape[-1]
        ).float()
        calibration_loss = (probabilities - one_hot).pow(2).mean()
        loss = loss + cfg.calibration_weight * calibration_loss
        components["calibration"] = float(calibration_loss.detach())

        if cfg.use_reconstruction:
            reconstruction_loss = F.mse_loss(output["reconstruction"], x)
            loss = loss + cfg.reconstruction_weight * reconstruction_loss
            components["reconstruction"] = float(reconstruction_loss.detach())

        if cfg.use_moe:
            mean_gate = output["gate_prob"].mean(dim=0)
            uniform = torch.full_like(
                mean_gate, 1.0 / mean_gate.numel()
            )
            load_loss = (mean_gate - uniform).pow(2).mean()
            loss = loss + cfg.load_balance_weight * load_loss
            components["load_balance"] = float(load_loss.detach())

        if cfg.use_cost_loss and cfg.force_route is None:
            # Normalized path costs: edge 0.0, experts 0.35, cloud 1.0
            cost_vector = torch.tensor(
                [0.0, 0.35, 1.0],
                device=output["route_prob"].device,
            )
            sample_cost = (
                output["route_prob"] * cost_vector.unsqueeze(0)
            ).sum(dim=-1)

            # Avoid penalizing costly routing for rare samples.
            cost_weight = 1.0 - 0.75 * rare
            cost_loss = (sample_cost * cost_weight).mean()
            loss = loss + cfg.cost_weight * cost_loss
            components["cost"] = float(cost_loss.detach())

        components["total"] = float(loss.detach())
        return loss, components


# =============================================================================
# TRAINING AND INFERENCE
# =============================================================================

def create_loaders(data: PreparedData) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = FlowDataset(
        data.X_train, data.y_train, data.family_train, data.rare_train
    )
    val_ds = FlowDataset(
        data.X_val, data.y_val, data.family_val, data.rare_val
    )
    test_ds = FlowDataset(
        data.X_test, data.y_test, data.family_test, data.rare_test
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    objective: JointObjective,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x, y, family, rare in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        family = family.to(DEVICE, non_blocking=True)
        rare = rare.to(DEVICE, non_blocking=True)

        output = model(x)
        loss, _ = objective(output, x, y, family, rare)
        total += float(loss) * len(y)
        count += len(y)
    return total / max(count, 1)


def train_neural_variant(
    variant: str,
    seed: int,
    data: PreparedData,
    config_overrides: dict,
    epochs: int,
) -> Tuple[RCSMoE, pd.DataFrame, float]:
    set_seed(seed)
    train_loader, val_loader, _ = create_loaders(data)

    config = ObjectiveConfig(**config_overrides)

    model = RCSMoE(
        num_features=data.X_train.shape[1],
        feature_groups=data.feature_groups,
        n_classes=len(data.label_encoder.classes_),
        n_families=len(data.family_encoder.classes_),
        use_rarity=config.use_rarity,
        use_moe=config.use_moe,
        use_reconstruction=config.use_reconstruction,
        force_route=config.force_route,
    ).to(DEVICE)

    objective = JointObjective(data.class_counts, config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    history = []
    best_val = float("inf")
    best_state = None
    stale = 0

    start_training = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        sample_count = 0
        component_accumulator: Dict[str, float] = {}

        for x, y, family, rare in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            family = family.to(DEVICE, non_blocking=True)
            rare = rare.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss, components = objective(output, x, y, family, rare)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in {variant}, seed {seed}, epoch {epoch}"
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            batch_n = len(y)
            epoch_total += float(loss.detach()) * batch_n
            sample_count += batch_n

            for key, value in components.items():
                component_accumulator[key] = (
                    component_accumulator.get(key, 0.0)
                    + value * batch_n
                )

        train_loss = epoch_total / max(sample_count, 1)
        val_loss = evaluate_loss(model, val_loader, objective)
        scheduler.step(val_loss)

        row = {
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for key, value in component_accumulator.items():
            row[f"train_{key}"] = value / max(sample_count, 1)
        history.append(row)

        log(
            f"{variant} seed={seed} epoch={epoch}/{epochs} "
            f"train={train_loss:.5f} val={val_loss:.5f}"
        )

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                log(f"Early stopping {variant}, seed {seed}")
                break

    train_seconds = time.perf_counter() - start_training

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(DEVICE)

    checkpoint = {
        "variant": variant,
        "seed": seed,
        "state_dict": best_state,
        "objective_config": asdict(config),
        "model_config": {
            "num_features": data.X_train.shape[1],
            "feature_groups": data.feature_groups.tolist(),
            "n_classes": len(data.label_encoder.classes_),
            "n_families": len(data.family_encoder.classes_),
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "n_experts": N_EXPERTS,
            "top_k": TOP_K_EXPERTS,
            "dropout": DROPOUT,
        },
    }
    checkpoint_path = MODEL_DIR / f"{variant}_seed_{seed}.pt"
    torch.save(checkpoint, checkpoint_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        LOG_DIR / f"training_history_{variant}_seed_{seed}.csv",
        index=False,
    )

    return model, history_df, train_seconds


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = prediction == y_true

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            ece += (
                mask.mean()
                * abs(correctness[mask].mean() - confidence[mask].mean())
            )
    return float(ece)


@torch.no_grad()
def predict_neural(
    model: RCSMoE,
    loader: DataLoader,
) -> Dict[str, np.ndarray]:
    model.eval()

    outputs = {
        "y": [],
        "family": [],
        "rare": [],
        "probabilities": [],
        "family_probabilities": [],
        "route_prob": [],
        "uncertainty": [],
        "reconstruction_error": [],
        "attention": [],
    }

    start = time.perf_counter()
    memory_before = process_memory_mb()

    for batch_index, (x, y, family, rare) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=True)
        output = model(x)

        outputs["y"].append(y.numpy())
        outputs["family"].append(family.numpy())
        outputs["rare"].append(rare.numpy())
        outputs["probabilities"].append(
            torch.softmax(output["logits"], dim=-1).cpu().numpy()
        )
        outputs["family_probabilities"].append(
            torch.softmax(output["family_logits"], dim=-1).cpu().numpy()
        )
        outputs["route_prob"].append(
            output["route_prob"].cpu().numpy()
        )
        outputs["uncertainty"].append(
            output["uncertainty"].cpu().numpy()
        )
        outputs["reconstruction_error"].append(
            output["reconstruction_error"].cpu().numpy()
        )

        # Save attention only from first few batches to control storage.
        if batch_index < 3:
            outputs["attention"].append(
                output["attention"].cpu().numpy()
            )

    elapsed = time.perf_counter() - start
    memory_after = process_memory_mb()

    result = {}
    for key, value in outputs.items():
        if not value:
            result[key] = np.empty((0,))
        else:
            result[key] = np.concatenate(value, axis=0)

    result["elapsed_seconds"] = np.asarray([elapsed])
    result["memory_delta_mb"] = np.asarray([memory_after - memory_before])
    return result


def evaluate_predictions(
    method: str,
    seed: int,
    data: PreparedData,
    prediction: Dict[str, np.ndarray],
    train_seconds: float,
    model_path: Path,
) -> Dict[str, float]:
    y_true = prediction["y"]
    probabilities = prediction["probabilities"]
    y_pred = probabilities.argmax(axis=1)

    family_true = prediction["family"]
    family_pred = prediction["family_probabilities"].argmax(axis=1)

    rare_mask = prediction["rare"] > 0.5
    rare_recall = (
        recall_score(
            y_true[rare_mask],
            y_pred[rare_mask],
            average="macro",
            zero_division=0,
        )
        if rare_mask.any()
        else np.nan
    )

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    route_prob = prediction["route_prob"]
    route_choice = route_prob.argmax(axis=1)
    edge_rate = float((route_choice == 0).mean())
    specialist_rate = float((route_choice == 1).mean())
    cloud_rate = float((route_choice == 2).mean())

    elapsed = float(prediction["elapsed_seconds"][0])
    latency_ms = elapsed / len(y_true) * 1000.0
    throughput = len(y_true) / elapsed

    metrics = {
        "method": method,
        "seed": seed,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "family_macro_f1": f1_score(
            family_true,
            family_pred,
            average="macro",
            zero_division=0,
        ),
        "rare_class_recall": rare_recall,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "ece": expected_calibration_error(probabilities, y_true),
        "latency_ms_per_sample": latency_ms,
        "throughput_samples_sec": throughput,
        "train_time_sec": train_seconds,
        "model_size_mb": model_path.stat().st_size / (1024 ** 2),
        "memory_delta_mb": float(prediction["memory_delta_mb"][0]),
        "edge_route_rate": edge_rate,
        "specialist_route_rate": specialist_rate,
        "cloud_route_rate": cloud_rate,
        "mean_uncertainty": float(prediction["uncertainty"].mean()),
        "mean_reconstruction_error": float(
            prediction["reconstruction_error"].mean()
        ),
    }

    labels = data.label_encoder.classes_
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_DIR / f"classification_report_{method}_seed_{seed}.csv"
    )

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        RESULT_DIR / f"confusion_matrix_{method}_seed_{seed}.csv"
    )

    true_text = data.label_encoder.inverse_transform(y_true)
    pred_text = data.label_encoder.inverse_transform(y_pred)
    true_family_text = data.family_encoder.inverse_transform(family_true)
    pred_family_text = data.family_encoder.inverse_transform(family_pred)

    pd.DataFrame({
        "true_label": true_text,
        "predicted_label": pred_text,
        "true_family": true_family_text,
        "predicted_family": pred_family_text,
        "confidence": probabilities.max(axis=1),
        "edge_route_probability": route_prob[:, 0],
        "specialist_route_probability": route_prob[:, 1],
        "cloud_route_probability": route_prob[:, 2],
        "uncertainty": prediction["uncertainty"],
        "reconstruction_error": prediction["reconstruction_error"],
        "is_rare": rare_mask.astype(int),
    }).to_csv(
        PRED_DIR / f"predictions_{method}_seed_{seed}.csv",
        index=False,
    )

    if prediction["attention"].size:
        np.save(
            RESULT_DIR / f"attention_{method}_seed_{seed}.npy",
            prediction["attention"],
        )

    return metrics


# =============================================================================
# CLASSICAL REFERENCES
# =============================================================================

def train_evaluate_classical(
    method: str,
    seed: int,
    data: PreparedData,
) -> Dict[str, float]:
    set_seed(seed)

    if method == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=seed,
        )
    elif method == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("XGBoost is not installed.")
        estimator = XGBClassifier(
            n_estimators=250,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(method)

    memory_before = process_memory_mb()
    start_train = time.perf_counter()
    estimator.fit(data.X_train, data.y_train)
    train_seconds = time.perf_counter() - start_train

    model_path = MODEL_DIR / f"{method}_seed_{seed}.joblib"
    joblib.dump(estimator, model_path)

    start_predict = time.perf_counter()
    probabilities = estimator.predict_proba(data.X_test)
    predict_seconds = time.perf_counter() - start_predict
    memory_after = process_memory_mb()

    y_pred = probabilities.argmax(axis=1)

    true_family_text = np.asarray([
        family_label(x)
        for x in data.label_encoder.inverse_transform(data.y_test)
    ])
    pred_family_text = np.asarray([
        family_label(x)
        for x in data.label_encoder.inverse_transform(y_pred)
    ])

    rare_mask = data.rare_test > 0.5
    rare_recall = (
        recall_score(
            data.y_test[rare_mask],
            y_pred[rare_mask],
            average="macro",
            zero_division=0,
        )
        if rare_mask.any()
        else np.nan
    )

    metrics = {
        "method": method,
        "seed": seed,
        "accuracy": accuracy_score(data.y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(data.y_test, y_pred),
        "macro_precision": precision_recall_fscore_support(
            data.y_test, y_pred, average="macro", zero_division=0
        )[0],
        "macro_recall": precision_recall_fscore_support(
            data.y_test, y_pred, average="macro", zero_division=0
        )[1],
        "macro_f1": f1_score(
            data.y_test, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            data.y_test, y_pred, average="weighted", zero_division=0
        ),
        "family_macro_f1": f1_score(
            true_family_text,
            pred_family_text,
            average="macro",
            zero_division=0,
        ),
        "rare_class_recall": rare_recall,
        "mcc": matthews_corrcoef(data.y_test, y_pred),
        "ece": expected_calibration_error(probabilities, data.y_test),
        "latency_ms_per_sample": predict_seconds / len(data.y_test) * 1000.0,
        "throughput_samples_sec": len(data.y_test) / predict_seconds,
        "train_time_sec": train_seconds,
        "model_size_mb": model_path.stat().st_size / (1024 ** 2),
        "memory_delta_mb": memory_after - memory_before,
        "edge_route_rate": 0.0,
        "specialist_route_rate": 0.0,
        "cloud_route_rate": 1.0 if method == "xgboost" else 0.0,
        "mean_uncertainty": float(
            (-probabilities * np.log(probabilities + 1e-8)).sum(axis=1).mean()
        ),
        "mean_reconstruction_error": np.nan,
    }

    pd.DataFrame(classification_report(
        data.y_test,
        y_pred,
        target_names=data.label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )).transpose().to_csv(
        RESULT_DIR / f"classification_report_{method}_seed_{seed}.csv"
    )

    pd.DataFrame(
        confusion_matrix(data.y_test, y_pred),
        index=data.label_encoder.classes_,
        columns=data.label_encoder.classes_,
    ).to_csv(
        RESULT_DIR / f"confusion_matrix_{method}_seed_{seed}.csv"
    )

    return metrics


# =============================================================================
# AGGREGATION AND STATISTICS
# =============================================================================

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "family_macro_f1",
    "rare_class_recall",
    "mcc",
    "ece",
    "latency_ms_per_sample",
    "throughput_samples_sec",
    "train_time_sec",
    "model_size_mb",
    "memory_delta_mb",
    "edge_route_rate",
    "specialist_route_rate",
    "cloud_route_rate",
]


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in results.groupby("method"):
        row = {"method": method, "runs": len(group)}
        for metric in METRICS:
            mean, std, low, high = confidence_interval(
                group[metric].dropna()
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        "macro_f1_mean", ascending=False
    )
    summary.to_csv(RESULT_DIR / "summary_confidence_intervals.csv", index=False)
    return summary


def significance_tests(results: pd.DataFrame) -> pd.DataFrame:
    pivot = results.pivot_table(
        index="seed",
        columns="method",
        values="macro_f1",
        aggfunc="mean",
    )
    order = pivot.mean().sort_values(ascending=False).index.tolist()
    reference = "full_rcsmoe" if "full_rcsmoe" in order else order[0]

    rows = []
    p_values = []
    for method in order:
        if method == reference:
            continue
        pair = pivot[[reference, method]].dropna()
        a = pair[reference].to_numpy(dtype=float)
        b = pair[method].to_numpy(dtype=float)
        diff = a - b

        if len(diff) >= 2:
            t_stat, t_p = stats.ttest_rel(a, b)
            try:
                w_stat, w_p = stats.wilcoxon(a, b)
            except ValueError:
                w_stat, w_p = np.nan, 1.0
            std_diff = np.std(diff, ddof=1)
            dz = np.mean(diff) / std_diff if std_diff > 0 else np.inf
        else:
            t_stat = t_p = w_stat = w_p = dz = np.nan

        p_values.append(w_p if np.isfinite(w_p) else 1.0)
        rows.append({
            "reference": reference,
            "compared_method": method,
            "paired_runs": len(pair),
            "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
            "paired_t_p": t_p,
            "wilcoxon_p": w_p,
            "cohen_dz": dz,
        })

    adjusted = holm_adjust(p_values)
    for row, p_adj in zip(rows, adjusted):
        row["holm_adjusted_p"] = p_adj
        row["significant_0.05"] = p_adj < 0.05

    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "statistical_significance.csv", index=False)
    return df


# =============================================================================
# PAPER TABLES
# =============================================================================

def make_tables(
    summary: pd.DataFrame,
    significance: pd.DataFrame,
) -> None:
    performance_rows = []
    for _, row in summary.iterrows():
        performance_rows.append({
            "Method": row["method"],
            "Runs": int(row["runs"]),
            "Macro-F1": (
                f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
            ),
            "95% CI": (
                f"[{row['macro_f1_ci_low']:.4f}, "
                f"{row['macro_f1_ci_high']:.4f}]"
            ),
            "Family F1": row["family_macro_f1_mean"],
            "Rare recall": row["rare_class_recall_mean"],
            "Balanced Acc.": row["balanced_accuracy_mean"],
            "MCC": row["mcc_mean"],
            "ECE ↓": row["ece_mean"],
        })

    save_table(
        pd.DataFrame(performance_rows),
        "Table_I_MultiPerspective_Performance",
        "Multi-seed detection, hierarchy, minority-class, and calibration results.",
        "tab:rcsmoe_performance",
    )

    efficiency = summary[[
        "method",
        "latency_ms_per_sample_mean",
        "throughput_samples_sec_mean",
        "train_time_sec_mean",
        "model_size_mb_mean",
        "memory_delta_mb_mean",
        "edge_route_rate_mean",
        "specialist_route_rate_mean",
        "cloud_route_rate_mean",
    ]].copy()
    efficiency.columns = [
        "Method",
        "Latency (ms/sample)",
        "Throughput (samples/s)",
        "Training time (s)",
        "Size (MB)",
        "Memory Δ (MB)",
        "Edge rate",
        "Specialist rate",
        "Cloud rate",
    ]
    save_table(
        efficiency,
        "Table_II_Efficiency_Routing",
        "Computational efficiency and learned edge-specialist-cloud routing.",
        "tab:rcsmoe_efficiency",
    )

    ablation_methods = [
        method for method in ABLATIONS
        if method in set(summary["method"])
    ]
    ablation = summary[
        summary["method"].isin(ablation_methods)
    ][[
        "method",
        "macro_f1_mean",
        "macro_f1_std",
        "family_macro_f1_mean",
        "rare_class_recall_mean",
        "ece_mean",
        "latency_ms_per_sample_mean",
        "cloud_route_rate_mean",
    ]].copy()
    ablation.columns = [
        "Variant",
        "Macro-F1",
        "Std.",
        "Family F1",
        "Rare recall",
        "ECE ↓",
        "Latency",
        "Cloud rate",
    ]

    full_row = ablation[ablation["Variant"] == "full_rcsmoe"]
    if not full_row.empty:
        full_f1 = float(full_row.iloc[0]["Macro-F1"])
        ablation["Δ Macro-F1"] = ablation["Macro-F1"] - full_f1

    save_table(
        ablation,
        "Table_III_True_Ablation",
        "True component ablation of the proposed RCS-MoE architecture.",
        "tab:rcsmoe_ablation",
    )

    save_table(
        significance,
        "Table_IV_Statistical_Significance",
        "Paired significance tests against the complete RCS-MoE model.",
        "tab:rcsmoe_significance",
    )


# =============================================================================
# PAPER FIGURES
# =============================================================================

def make_figures(
    summary: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    # Figure 1: performance, rare classes, calibration
    d = summary.sort_values("macro_f1_mean", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))

    y = np.arange(len(d))
    axes[0].barh(
        y,
        d["macro_f1_mean"],
        xerr=d["macro_f1_std"],
        capsize=2,
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d["method"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Macro-F1")
    axes[0].set_title("(a) Fine-grained detection")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].scatter(
        d["family_macro_f1_mean"],
        d["rare_class_recall_mean"],
        s=35 + 180 * d["macro_f1_mean"],
    )
    for _, row in d.iterrows():
        axes[1].annotate(
            row["method"],
            (row["family_macro_f1_mean"], row["rare_class_recall_mean"]),
            fontsize=5.2,
            xytext=(2, 2),
            textcoords="offset points",
        )
    axes[1].set_xlabel("Family Macro-F1")
    axes[1].set_ylabel("Rare-class recall")
    axes[1].set_title("(b) Hierarchy and rarity")
    axes[1].grid(alpha=0.25)

    axes[2].scatter(
        d["ece_mean"],
        d["macro_f1_mean"],
        s=35 + 180 * d["rare_class_recall_mean"].fillna(0),
    )
    for _, row in d.iterrows():
        axes[2].annotate(
            row["method"],
            (row["ece_mean"], row["macro_f1_mean"]),
            fontsize=5.2,
            xytext=(2, 2),
            textcoords="offset points",
        )
    axes[2].set_xlabel("ECE (lower is better)")
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_title("(c) Accuracy-calibration")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    save_figure(fig, "Fig_1_Detection_Rarity_Calibration_1x3")

    # Figure 2: routing and cost
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))

    axes[0].scatter(
        d["latency_ms_per_sample_mean"],
        d["macro_f1_mean"],
        s=35 + 4 * np.sqrt(np.maximum(d["model_size_mb_mean"], 0)),
    )
    for _, row in d.iterrows():
        axes[0].annotate(
            row["method"],
            (row["latency_ms_per_sample_mean"], row["macro_f1_mean"]),
            fontsize=5.2,
            xytext=(2, 2),
            textcoords="offset points",
        )
    axes[0].set_xlabel("Latency (ms/sample)")
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("(a) Accuracy-latency")
    axes[0].grid(alpha=0.25)

    route = d.set_index("method")[[
        "edge_route_rate_mean",
        "specialist_route_rate_mean",
        "cloud_route_rate_mean",
    ]]
    route.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_ylabel("Routing fraction")
    axes[1].set_xlabel("")
    axes[1].set_title("(b) Learned route allocation")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(
        ["Edge", "Specialist", "Cloud"],
        fontsize=5.5,
        frameon=False,
    )

    ablation = d[d["method"].isin(ABLATIONS.keys())].copy()
    axes[2].barh(
        np.arange(len(ablation)),
        ablation["macro_f1_mean"],
    )
    axes[2].set_yticks(np.arange(len(ablation)))
    axes[2].set_yticklabels(ablation["method"], fontsize=5.5)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Macro-F1")
    axes[2].set_title("(c) Component ablation")
    axes[2].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    save_figure(fig, "Fig_2_Efficiency_Routing_Ablation_1x3")

    # Figure 3: convergence for full model
    history_files = sorted(
        LOG_DIR.glob("training_history_full_rcsmoe_seed_*.csv")
    )
    if history_files:
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
        all_history = []
        for path in history_files:
            h = pd.read_csv(path)
            all_history.append(h)
            axes[0].plot(h["epoch"], h["train_loss"], alpha=0.65)
            axes[1].plot(h["epoch"], h["val_loss"], alpha=0.65)

        history = pd.concat(all_history, ignore_index=True)
        axes[0].set_title("(a) Training loss")
        axes[1].set_title("(b) Validation loss")
        axes[0].set_xlabel("Epoch")
        axes[1].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)

        seed_f1 = results[
            results["method"] == "full_rcsmoe"
        ].sort_values("seed")
        axes[2].errorbar(
            seed_f1["seed"].astype(str),
            seed_f1["macro_f1"],
            yerr=np.zeros(len(seed_f1)),
            fmt="o-",
        )
        axes[2].set_xlabel("Seed")
        axes[2].set_ylabel("Macro-F1")
        axes[2].set_title("(c) Seed stability")
        axes[2].grid(alpha=0.25)

        fig.tight_layout()
        save_figure(fig, "Fig_3_Convergence_Stability_1x3")

    # Figure 4: best confusion matrix
    best = d.iloc[0]["method"]
    best_seed = int(
        results[results["method"] == best]
        .sort_values("macro_f1", ascending=False)
        .iloc[0]["seed"]
    )
    cm_path = RESULT_DIR / f"confusion_matrix_{best}_seed_{best_seed}.csv"
    if cm_path.exists():
        cm_df = pd.read_csv(cm_path, index_col=0)
        cm = cm_df.to_numpy(dtype=float)
        row_sum = cm.sum(axis=1, keepdims=True)
        norm = np.divide(
            cm, row_sum,
            out=np.zeros_like(cm),
            where=row_sum != 0,
        )

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        image = ax.imshow(norm, aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Normalized confusion matrix: {best}")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        step = max(1, len(cm_df.index) // 10)
        ticks = np.arange(0, len(cm_df.index), step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(
            [cm_df.columns[i] for i in ticks],
            rotation=55,
            ha="right",
            fontsize=5.2,
        )
        ax.set_yticklabels(
            [cm_df.index[i] for i in ticks],
            fontsize=5.2,
        )
        fig.colorbar(image, ax=ax, fraction=0.046)
        fig.tight_layout()
        save_figure(fig, "Fig_4_Best_Confusion_Matrix")

    # Figure 5: architecture
    make_architecture_figure()


def make_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 4.2))
    ax.axis("off")

    def box(x, y, w, h, title, subtitle=""):
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.1)
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h * 0.64,
            title,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
        )
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.25,
                subtitle,
                ha="center",
                va="center",
                fontsize=5.5,
            )

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.0),
        )
        if text:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2 + 0.015,
                text,
                fontsize=5.2,
                ha="center",
            )

    ax.text(
        0.50,
        0.96,
        "CARE-IoT++ RCS-MoE: Rarity-Conditioned Selective Edge-Cloud Experts",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    box(0.02, 0.70, 0.13, 0.16, "IoT flow", "standardized feature vector")
    box(0.19, 0.69, 0.17, 0.18, "Feature-group tokenizer",
        "value + identity + semantic group")
    box(0.40, 0.67, 0.19, 0.22, "Rarity-conditioned attention",
        "minority-sensitive feature interactions")
    box(0.64, 0.67, 0.16, 0.22, "Selective router",
        "uncertainty + rarity + OOD + cost")
    box(0.84, 0.71, 0.14, 0.15, "Final output",
        "subtype + family + confidence")

    arrow(0.15, 0.78, 0.19, 0.78)
    arrow(0.36, 0.78, 0.40, 0.78)
    arrow(0.59, 0.78, 0.64, 0.78)
    arrow(0.80, 0.78, 0.84, 0.78)

    box(0.08, 0.32, 0.18, 0.16, "Edge head", "fast common-flow decision")
    box(0.33, 0.29, 0.27, 0.22, "Sparse specialist bank",
        "top-k experts for rare and family-specific attacks")
    box(0.67, 0.32, 0.18, 0.16, "Cloud head",
        "hard / uncertain / shifted flows")
    box(0.87, 0.32, 0.11, 0.16, "Abstain",
        "unknown-safe output")

    arrow(0.70, 0.67, 0.17, 0.48, "low risk")
    arrow(0.72, 0.67, 0.46, 0.51, "specialized")
    arrow(0.74, 0.67, 0.76, 0.48, "high risk")
    arrow(0.76, 0.67, 0.925, 0.48, "OOD")

    ax.text(
        0.50,
        0.15,
        r"$\mathcal{L}=\mathcal{L}_{CB\!-\!focal}"
        r"+\lambda_f\mathcal{L}_{family}"
        r"+\lambda_r\mathcal{L}_{rare}"
        r"+\lambda_s\mathcal{L}_{route}"
        r"+\lambda_k\mathcal{L}_{calibration}"
        r"+\lambda_o\mathcal{L}_{reconstruction}"
        r"+\lambda_c\mathcal{L}_{cost}$",
        ha="center",
        va="center",
        fontsize=7.0,
    )

    ax.text(
        0.50,
        0.05,
        "Core contribution: jointly learned rare-class representation and "
        "resource-aware selective inference.",
        ha="center",
        fontsize=6.2,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save_figure(fig, "Fig_5_Proposed_RCSMoE_Architecture")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    log(f"Device: {DEVICE}")
    log(f"Journal mode: {JOURNAL_MODE}")
    log(f"XGBoost available: {HAS_XGB}")
    log(f"Fixed data split seed: {DATA_SPLIT_SEED}")
    log(f"Model initialization seeds: {MODEL_SEEDS}")
    log("Focused run: retraining only full_rcsmoe with corrected edge routing")

    create_fixed_stratified_subsets(force_rebuild=False)

    all_results: List[Dict[str, float]] = []

    for seed in SEEDS:
        set_seed(seed)
        data = load_and_prepare(seed)
        train_loader, val_loader, test_loader = create_loaders(data)

        for variant, overrides in ABLATIONS.items():
            epochs = FULL_EPOCHS if variant == "full_rcsmoe" else ABLATION_EPOCHS
            log(f"Training neural variant={variant}, seed={seed}")

            model, history, train_seconds = train_neural_variant(
                variant=variant,
                seed=seed,
                data=data,
                config_overrides=overrides,
                epochs=epochs,
            )

            prediction = predict_neural(model, test_loader)
            model_path = MODEL_DIR / f"{variant}_seed_{seed}.pt"
            metrics = evaluate_predictions(
                method=variant,
                seed=seed,
                data=data,
                prediction=prediction,
                train_seconds=train_seconds,
                model_path=model_path,
            )
            all_results.append(metrics)

            pd.DataFrame(all_results).to_csv(
                RESULT_DIR / "metrics_live.csv",
                index=False,
            )

            del model, prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if RUN_RANDOM_FOREST:
            log(f"Training Random Forest reference, seed={seed}")
            all_results.append(
                train_evaluate_classical(
                    "random_forest", seed, data
                )
            )

        if RUN_XGBOOST:
            log(f"Training XGBoost reference, seed={seed}")
            all_results.append(
                train_evaluate_classical(
                    "xgboost", seed, data
                )
            )

        pd.DataFrame(all_results).to_csv(
            RESULT_DIR / "metrics_live.csv",
            index=False,
        )

        del data, train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(all_results)
    results.to_csv(RESULT_DIR / "metrics_all_runs.csv", index=False)

    summary = aggregate_results(results)
    significance = significance_tests(results)

    make_tables(summary, significance)
    make_figures(summary, results)

    save_json({
        "device": str(DEVICE),
        "journal_mode": JOURNAL_MODE,
        "data_split_seed": DATA_SPLIT_SEED,
        "model_initialization_seeds": MODEL_SEEDS,
        "fixed_split_directory": str(FIXED_SPLIT_DIR),
        "fixed_split_metadata": str(FIXED_SPLIT_META),
        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VAL_FRAC,
        "test_fraction": TEST_FRAC,
        "ablations": ABLATIONS,
        "best_method": summary.iloc[0]["method"],
        "best_macro_f1": summary.iloc[0]["macro_f1_mean"],
        "best_rare_recall": summary.iloc[0]["rare_class_recall_mean"],
        "results_folder": str(RESULT_DIR),
        "models_folder": str(MODEL_DIR),
    }, RESULT_DIR / "final_run_report.json")

    log("All experiments completed.")
    log(f"Results: {RESULT_DIR}")
    log(f"Models: {MODEL_DIR}")


if __name__ == "__main__":
    main()
