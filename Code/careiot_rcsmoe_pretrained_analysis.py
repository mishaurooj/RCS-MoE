r"""
RCS-MoE PRETRAINED MODEL ANALYSIS AND PAPER-EVIDENCE GENERATOR
==============================================================

Purpose
-------
This script loads the already trained RCS-MoE checkpoints and creates the
additional tables and figures required to justify the architecture. It does
not retrain the neural models.

Place this file in the same Code folder as the original RCS-MoE training
script, then run:

    conda activate care-iot
    cd /d D:\other\CARE-IoT\Code
    python careiot_rcsmoe_pretrained_analysis.py

Expected original script names
------------------------------
The loader searches for one of these files in the current directory:

    careiot_rcsmoe_all_in_one.py
    careiot_rcsmoe.py
    careiot.py

Outputs
-------
D:\other\CARE-IoT\Results\RCSMOE_JOURNAL\extended_tables
D:\other\CARE-IoT\Results\RCSMOE_JOURNAL\extended_figures
D:\other\CARE-IoT\Results\RCSMOE_JOURNAL\extended_data

The script produces:
  1. Multi-seed summary with 95% confidence intervals.
  2. Component ablation deltas.
  3. Class-wise and rare-class performance.
  4. Attack-family performance.
  5. Routing by class, family, rarity, and difficulty.
  6. Expert utilization and expert specialization.
  7. Attention entropy and feature importance.
  8. Calibration, Brier score, negative log-likelihood, and reliability bins.
  9. Confidence-based failure analysis.
 10. Reconstruction-error analysis.
 11. Model parameter and checkpoint-size analysis.
 12. Post-hoc route-cost versus accuracy trade-off.
 13. Publication-ready PDF and 600-DPI PNG figures.

Important
---------
The post-hoc cost sweep changes route selection after training. Describe it as
an inference-policy sensitivity analysis, not as a new training experiment.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(r"D:\other\CARE-IoT")
RESULT_DIR = ROOT / "Results" / "RCSMOE_JOURNAL"

# The original code stores checkpoints here. The result-tree location is also
# checked because some projects keep model files under the result directory.
MODEL_CANDIDATES = [
    ROOT / "Models" / "RCSMOE_JOURNAL",
    RESULT_DIR / "models",
    RESULT_DIR / "Models",
]

TABLE_DIR = RESULT_DIR / "extended_tables"
FIG_DIR = RESULT_DIR / "extended_figures"
DATA_OUT_DIR = RESULT_DIR / "extended_data"
LOG_DIR = RESULT_DIR / "logs"
PRED_DIR = RESULT_DIR / "predictions"

for directory in [TABLE_DIR, FIG_DIR, DATA_OUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = [
    Path(__file__).resolve().parent / "careiot_rcsmoe_all_in_one.py",
    Path(__file__).resolve().parent / "careiot_rcsmoe.py",
    Path(__file__).resolve().parent / "careiot.py",
]

DPI = 600
BATCH_SIZE_OVERRIDE: Optional[int] = None
MAX_EMBEDDING_SAMPLES = 12000
MAX_ATTENTION_SAMPLES = 12000
RELIABILITY_BINS = 15
COST_SWEEP = np.linspace(0.0, 5.0, 21)
ROUTE_COSTS = np.asarray([0.0, 0.35, 1.0], dtype=np.float64)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.8,
})

# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def log(message: str) -> None:
    print(f"[RCS-MoE pretrained analysis] {message}", flush=True)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")


def save_df(df: pd.DataFrame, stem: str, caption: Optional[str] = None) -> None:
    csv_path = TABLE_DIR / f"{stem}.csv"
    xlsx_path = TABLE_DIR / f"{stem}.xlsx"
    tex_path = TABLE_DIR / f"{stem}.tex"

    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as exc:
        log(f"Excel export skipped for {stem}: {exc}")

    latex = df.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        float_format=lambda value: f"{value:.4f}",
    )
    if caption:
        latex = (
            "\\begin{table*}[t]\n"
            "\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{tab:{safe_name(stem).lower()}}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            + latex
            + "}\n\\end{table*}\n"
        )
    tex_path.write_text(latex, encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def mean_ci(values: Iterable[float]) -> Tuple[float, float, float, float]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if len(x) < 2:
        return mean, std, mean, mean
    margin = float(stats.t.ppf(0.975, len(x) - 1) * std / math.sqrt(len(x)))
    return mean, std, mean - margin, mean + margin


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    bins: int = RELIABILITY_BINS,
) -> Tuple[float, pd.DataFrame]:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if index == 0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence > left) & (confidence <= right)
        count = int(mask.sum())
        if count:
            mean_conf = float(confidence[mask].mean())
            mean_acc = float(correct[mask].mean())
            weight = count / len(y_true)
            ece += weight * abs(mean_acc - mean_conf)
        else:
            mean_conf = np.nan
            mean_acc = np.nan
            weight = 0.0
        rows.append({
            "bin": index + 1,
            "left": left,
            "right": right,
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": mean_acc,
            "weight": weight,
        })
    return float(ece), pd.DataFrame(rows)


def multiclass_brier(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=1)
    return entropy / math.log(probabilities.shape[1])


def import_training_module():
    existing = [path for path in SOURCE_CANDIDATES if path.exists() and path.resolve() != Path(__file__).resolve()]
    if not existing:
        names = "\n".join(str(path) for path in SOURCE_CANDIDATES)
        raise FileNotFoundError(
            "Could not locate the original RCS-MoE source file. Expected one of:\n" + names
        )
    source_path = existing[0]
    log(f"Importing architecture from {source_path}")
    spec = importlib.util.spec_from_file_location("rcsmoe_training_source", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def locate_model_dir() -> Path:
    scored = []
    for directory in MODEL_CANDIDATES:
        count = len(list(directory.glob("*.pt"))) if directory.exists() else 0
        scored.append((count, directory))
    scored.sort(reverse=True, key=lambda item: item[0])
    if not scored or scored[0][0] == 0:
        checked = "\n".join(str(path) for path in MODEL_CANDIDATES)
        raise FileNotFoundError(
            "No .pt checkpoints were found. Checked:\n" + checked
        )
    log(f"Using checkpoint directory {scored[0][1]}")
    return scored[0][1]


def discover_checkpoints(model_dir: Path) -> List[Tuple[str, int, Path]]:
    pattern = re.compile(r"(.+)_seed_(\d+)\.pt$")
    found = []
    for path in sorted(model_dir.glob("*.pt")):
        match = pattern.match(path.name)
        if match:
            found.append((match.group(1), int(match.group(2)), path))
    if not found:
        raise FileNotFoundError(f"No checkpoint filenames matched *_seed_<n>.pt in {model_dir}")
    return found


def build_model_from_checkpoint(module, checkpoint: Dict[str, Any]):
    config = checkpoint["model_config"]
    objective = checkpoint.get("objective_config", {})
    model = module.RCSMoE(
        num_features=int(config["num_features"]),
        feature_groups=np.asarray(config["feature_groups"], dtype=np.int64),
        n_classes=int(config["n_classes"]),
        n_families=int(config["n_families"]),
        d_model=int(config.get("d_model", module.D_MODEL)),
        n_heads=int(config.get("n_heads", module.N_HEADS)),
        n_layers=int(config.get("n_layers", module.N_LAYERS)),
        n_experts=int(config.get("n_experts", module.N_EXPERTS)),
        top_k=int(config.get("top_k", module.TOP_K_EXPERTS)),
        dropout=float(config.get("dropout", module.DROPOUT)),
        use_rarity=bool(objective.get("use_rarity", True)),
        use_moe=bool(objective.get("use_moe", True)),
        use_reconstruction=bool(objective.get("use_reconstruction", True)),
        force_route=objective.get("force_route"),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model

# =============================================================================
# PRETRAINED INFERENCE
# =============================================================================


@torch.no_grad()
def run_extended_inference(
    model,
    loader,
    max_embedding_samples: int = MAX_EMBEDDING_SAMPLES,
    max_attention_samples: int = MAX_ATTENTION_SAMPLES,
) -> Dict[str, np.ndarray]:
    outputs: Dict[str, List[np.ndarray]] = defaultdict(list)
    representation_holder: Dict[str, torch.Tensor] = {}

    def encoder_hook(_module, _inputs, result):
        representation_holder["representation"] = result[0].detach()

    hook = model.encoder.register_forward_hook(encoder_hook)
    total_embedding = 0
    total_attention = 0

    start = time.perf_counter()
    for x, y, family, rare in loader:
        x_device = x.to(DEVICE, non_blocking=True)
        result = model(x_device)

        final_prob = torch.softmax(result["logits"], dim=-1).cpu().numpy()
        edge_prob = torch.softmax(result["edge_logits"], dim=-1).cpu().numpy()
        specialist_prob = torch.softmax(result["moe_logits"], dim=-1).cpu().numpy()
        cloud_prob = torch.softmax(result["cloud_logits"], dim=-1).cpu().numpy()
        family_prob = torch.softmax(result["family_logits"], dim=-1).cpu().numpy()

        outputs["y"].append(y.numpy())
        outputs["family"].append(family.numpy())
        outputs["rare"].append(rare.numpy())
        outputs["final_prob"].append(final_prob)
        outputs["edge_prob"].append(edge_prob)
        outputs["specialist_prob"].append(specialist_prob)
        outputs["cloud_prob"].append(cloud_prob)
        outputs["family_prob"].append(family_prob)
        outputs["route_prob"].append(result["route_prob"].cpu().numpy())
        outputs["gate_prob"].append(result["gate_prob"].cpu().numpy())
        outputs["uncertainty"].append(result["uncertainty"].cpu().numpy())
        outputs["reconstruction_error"].append(result["reconstruction_error"].cpu().numpy())
        outputs["rarity_prob"].append(torch.sigmoid(result["rarity_logit"]).cpu().numpy())

        if total_embedding < max_embedding_samples:
            rep = representation_holder["representation"].cpu().numpy()
            take = min(len(rep), max_embedding_samples - total_embedding)
            outputs["representation"].append(rep[:take])
            outputs["embedding_y"].append(y.numpy()[:take])
            outputs["embedding_family"].append(family.numpy()[:take])
            outputs["embedding_rare"].append(rare.numpy()[:take])
            total_embedding += take

        if total_attention < max_attention_samples:
            attention = result["attention"].cpu().numpy()
            take = min(len(attention), max_attention_samples - total_attention)
            outputs["attention"].append(attention[:take])
            outputs["attention_rare"].append(rare.numpy()[:take])
            total_attention += take

    hook.remove()
    elapsed = time.perf_counter() - start

    merged: Dict[str, np.ndarray] = {}
    for key, pieces in outputs.items():
        merged[key] = np.concatenate(pieces, axis=0) if pieces else np.empty((0,))
    merged["elapsed_seconds"] = np.asarray([elapsed], dtype=np.float64)
    return merged


def save_extended_cache(data: Dict[str, np.ndarray], variant: str, seed: int) -> Path:
    path = DATA_OUT_DIR / f"extended_{safe_name(variant)}_seed_{seed}.npz"
    np.savez_compressed(path, **data)
    return path


def load_or_run_cache(
    model,
    loader,
    variant: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    path = DATA_OUT_DIR / f"extended_{safe_name(variant)}_seed_{seed}.npz"
    if path.exists():
        log(f"Loading cached extended inference: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    log(f"Running pretrained inference: {variant}, seed={seed}")
    data = run_extended_inference(model, loader)
    save_extended_cache(data, variant, seed)
    return data

# =============================================================================
# ANALYSIS TABLES
# =============================================================================


def compute_run_metrics(
    variant: str,
    seed: int,
    arrays: Dict[str, np.ndarray],
) -> Dict[str, float]:
    y_true = arrays["y"].astype(int)
    prob = arrays["final_prob"]
    pred = prob.argmax(axis=1)
    family_true = arrays["family"].astype(int)
    family_pred = arrays["family_prob"].argmax(axis=1)
    rare_mask = arrays["rare"] > 0.5
    ece, _ = expected_calibration_error(prob, y_true)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, pred, average="macro", zero_division=0
    )
    route = arrays["route_prob"].argmax(axis=1)
    elapsed = float(arrays["elapsed_seconds"][0])
    return {
        "method": variant,
        "seed": seed,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "family_macro_f1": f1_score(
            family_true, family_pred, average="macro", zero_division=0
        ),
        "rare_class_recall": (
            precision_recall_fscore_support(
                y_true[rare_mask], pred[rare_mask], average="macro", zero_division=0
            )[1] if rare_mask.any() else np.nan
        ),
        "mcc": matthews_corrcoef(y_true, pred),
        "ece": ece,
        "brier": multiclass_brier(prob, y_true),
        "nll": log_loss(y_true, prob, labels=np.arange(prob.shape[1])),
        "latency_ms_per_sample": elapsed / max(len(y_true), 1) * 1000.0,
        "throughput_samples_sec": len(y_true) / max(elapsed, 1e-12),
        "edge_route_rate": float((route == 0).mean()),
        "specialist_route_rate": float((route == 1).mean()),
        "cloud_route_rate": float((route == 2).mean()),
        "mean_uncertainty": float(arrays["uncertainty"].mean()),
        "mean_reconstruction_error": float(arrays["reconstruction_error"].mean()),
    }


def aggregate_runs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in run_metrics.columns if column not in {"method", "seed"}]
    rows = []
    for method, group in run_metrics.groupby("method", sort=False):
        row: Dict[str, Any] = {"Method": method, "Runs": len(group)}
        for metric in numeric:
            mean, std, low, high = mean_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def ablation_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    full = summary[summary["Method"] == "full_rcsmoe"]
    if full.empty:
        return pd.DataFrame()
    ref = full.iloc[0]
    selected = [
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "family_macro_f1_mean",
        "rare_class_recall_mean",
        "ece_mean",
        "latency_ms_per_sample_mean",
        "edge_route_rate_mean",
        "specialist_route_rate_mean",
        "cloud_route_rate_mean",
    ]
    rows = []
    for _, row in summary.iterrows():
        if row["Method"] in {"random_forest", "xgboost"}:
            continue
        item: Dict[str, Any] = {"Variant": row["Method"]}
        for metric in selected:
            item[metric.replace("_mean", "")] = row[metric]
            item[f"delta_{metric.replace('_mean', '')}"] = row[metric] - ref[metric]
        rows.append(item)
    return pd.DataFrame(rows)


def class_performance_table(
    arrays: Dict[str, np.ndarray],
    label_names: Sequence[str],
) -> pd.DataFrame:
    y_true = arrays["y"].astype(int)
    pred = arrays["final_prob"].argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=np.arange(len(label_names)),
        zero_division=0,
    )
    cm = confusion_matrix(y_true, pred, labels=np.arange(len(label_names)))
    false_negative = cm.sum(axis=1) - np.diag(cm)
    return pd.DataFrame({
        "Class": list(label_names),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support,
        "False negatives": false_negative,
    }).sort_values("F1")


def family_performance_table(
    arrays: Dict[str, np.ndarray],
    family_names: Sequence[str],
) -> pd.DataFrame:
    true = arrays["family"].astype(int)
    pred = arrays["family_prob"].argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=np.arange(len(family_names)),
        zero_division=0,
    )
    return pd.DataFrame({
        "Family": list(family_names),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support,
    }).sort_values("F1")


def routing_group_table(
    arrays: Dict[str, np.ndarray],
    group_values: np.ndarray,
    group_names: Sequence[str],
    group_title: str,
) -> pd.DataFrame:
    route = arrays["route_prob"].argmax(axis=1)
    y_true = arrays["y"].astype(int)
    pred = arrays["final_prob"].argmax(axis=1)
    rows = []
    for group_id, group_name in enumerate(group_names):
        mask = group_values == group_id
        if not mask.any():
            continue
        rows.append({
            group_title: group_name,
            "Samples": int(mask.sum()),
            "Accuracy": accuracy_score(y_true[mask], pred[mask]),
            "Mean confidence": float(arrays["final_prob"][mask].max(axis=1).mean()),
            "Mean uncertainty": float(arrays["uncertainty"][mask].mean()),
            "Mean reconstruction error": float(arrays["reconstruction_error"][mask].mean()),
            "Edge route": float((route[mask] == 0).mean()),
            "Specialist route": float((route[mask] == 1).mean()),
            "Cloud route": float((route[mask] == 2).mean()),
        })
    return pd.DataFrame(rows)


def rarity_routing_table(arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    route = arrays["route_prob"].argmax(axis=1)
    y_true = arrays["y"].astype(int)
    pred = arrays["final_prob"].argmax(axis=1)
    rows = []
    for name, mask in [
        ("Common", arrays["rare"] <= 0.5),
        ("Rare", arrays["rare"] > 0.5),
    ]:
        rows.append({
            "Group": name,
            "Samples": int(mask.sum()),
            "Accuracy": accuracy_score(y_true[mask], pred[mask]),
            "Macro F1": f1_score(y_true[mask], pred[mask], average="macro", zero_division=0),
            "Mean uncertainty": float(arrays["uncertainty"][mask].mean()),
            "Mean reconstruction error": float(arrays["reconstruction_error"][mask].mean()),
            "Edge route": float((route[mask] == 0).mean()),
            "Specialist route": float((route[mask] == 1).mean()),
            "Cloud route": float((route[mask] == 2).mean()),
        })
    return pd.DataFrame(rows)


def difficulty_routing_table(arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    confidence = arrays["edge_prob"].max(axis=1)
    quantiles = np.quantile(confidence, [0.0, 0.25, 0.50, 0.75, 1.0])
    route = arrays["route_prob"].argmax(axis=1)
    y_true = arrays["y"].astype(int)
    pred = arrays["final_prob"].argmax(axis=1)
    rows = []
    labels = ["Hardest quartile", "Hard", "Easy", "Easiest quartile"]
    for index in range(4):
        left, right = quantiles[index], quantiles[index + 1]
        if index == 0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence > left) & (confidence <= right)
        rows.append({
            "Difficulty": labels[index],
            "Edge-confidence range": f"[{left:.4f}, {right:.4f}]",
            "Samples": int(mask.sum()),
            "Accuracy": accuracy_score(y_true[mask], pred[mask]),
            "Edge route": float((route[mask] == 0).mean()),
            "Specialist route": float((route[mask] == 1).mean()),
            "Cloud route": float((route[mask] == 2).mean()),
            "Mean uncertainty": float(arrays["uncertainty"][mask].mean()),
        })
    return pd.DataFrame(rows)


def expert_utilization_table(
    arrays: Dict[str, np.ndarray],
    family_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gate = arrays["gate_prob"]
    top1 = gate.argmax(axis=1)
    overall = []
    for expert in range(gate.shape[1]):
        overall.append({
            "Expert": f"Expert {expert + 1}",
            "Top-1 assignment rate": float((top1 == expert).mean()),
            "Mean gate probability": float(gate[:, expert].mean()),
            "Mean probability when selected": (
                float(gate[top1 == expert, expert].mean()) if np.any(top1 == expert) else np.nan
            ),
        })
    overall_df = pd.DataFrame(overall)

    family_rows = []
    family = arrays["family"].astype(int)
    for family_id, family_name in enumerate(family_names):
        mask = family == family_id
        if not mask.any():
            continue
        row: Dict[str, Any] = {"Family": family_name, "Samples": int(mask.sum())}
        for expert in range(gate.shape[1]):
            row[f"Expert {expert + 1}"] = float((top1[mask] == expert).mean())
        family_rows.append(row)
    return overall_df, pd.DataFrame(family_rows)


def expert_class_specialization(
    arrays: Dict[str, np.ndarray],
    label_names: Sequence[str],
    top_n: int = 5,
) -> pd.DataFrame:
    gate = arrays["gate_prob"]
    top1 = gate.argmax(axis=1)
    labels = arrays["y"].astype(int)
    rows = []
    for expert in range(gate.shape[1]):
        mask = top1 == expert
        counts = pd.Series(labels[mask]).value_counts()
        total = max(int(mask.sum()), 1)
        for rank, (label_id, count) in enumerate(counts.head(top_n).items(), start=1):
            rows.append({
                "Expert": f"Expert {expert + 1}",
                "Rank": rank,
                "Class": label_names[int(label_id)],
                "Samples": int(count),
                "Share within expert": float(count / total),
            })
    return pd.DataFrame(rows)


def attention_statistics(
    arrays: Dict[str, np.ndarray],
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attention = arrays["attention"]
    # Shape: [samples, layers, heads, query_features, key_features]
    if attention.ndim != 5:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    eps = 1e-12
    normalized = attention / np.clip(attention.sum(axis=-1, keepdims=True), eps, None)
    entropy = -(normalized * np.log(normalized + eps)).sum(axis=-1) / math.log(normalized.shape[-1])
    rare = arrays["attention_rare"] > 0.5

    rows = []
    for layer in range(attention.shape[1]):
        for head in range(attention.shape[2]):
            values = entropy[:, layer, head, :].mean(axis=1)
            rows.append({
                "Layer": layer + 1,
                "Head": head + 1,
                "Mean attention entropy": float(values.mean()),
                "Rare entropy": float(values[rare].mean()) if rare.any() else np.nan,
                "Common entropy": float(values[~rare].mean()) if (~rare).any() else np.nan,
                "Rare-common difference": (
                    float(values[rare].mean() - values[~rare].mean())
                    if rare.any() and (~rare).any() else np.nan
                ),
            })
    entropy_df = pd.DataFrame(rows)

    # Incoming attention received by each key feature, averaged across samples,
    # layers, heads, and query features.
    received = attention.mean(axis=(0, 1, 2, 3))
    rare_received = attention[rare].mean(axis=(0, 1, 2, 3)) if rare.any() else np.full_like(received, np.nan)
    common_received = attention[~rare].mean(axis=(0, 1, 2, 3)) if (~rare).any() else np.full_like(received, np.nan)
    feature_df = pd.DataFrame({
        "Feature": list(feature_names),
        "Mean attention received": received,
        "Rare attention received": rare_received,
        "Common attention received": common_received,
        "Rare-common gain": rare_received - common_received,
    }).sort_values("Mean attention received", ascending=False)

    layer_matrix = attention.mean(axis=(0, 2, 3))
    # [layers, key_features]
    layer_rows = []
    for layer in range(layer_matrix.shape[0]):
        for feature, value in zip(feature_names, layer_matrix[layer]):
            layer_rows.append({
                "Layer": layer + 1,
                "Feature": feature,
                "Attention": float(value),
            })
    return entropy_df, feature_df, pd.DataFrame(layer_rows)


def calibration_table(
    arrays: Dict[str, np.ndarray],
    variant: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true = arrays["y"].astype(int)
    prob = arrays["final_prob"]
    ece, bins = expected_calibration_error(prob, y_true)
    summary = pd.DataFrame([{
        "Method": variant,
        "Seed": seed,
        "ECE": ece,
        "Brier score": multiclass_brier(prob, y_true),
        "NLL": log_loss(y_true, prob, labels=np.arange(prob.shape[1])),
        "Mean confidence": float(prob.max(axis=1).mean()),
        "Mean entropy": float(normalized_entropy(prob).mean()),
    }])
    bins.insert(0, "Seed", seed)
    bins.insert(0, "Method", variant)
    return summary, bins


def confidence_failure_table(arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    y_true = arrays["y"].astype(int)
    prob = arrays["final_prob"]
    pred = prob.argmax(axis=1)
    correct = pred == y_true
    confidence = prob.max(axis=1)
    rows = []
    for name, mask in [
        ("Correct", correct),
        ("Incorrect", ~correct),
        ("Rare correct", correct & (arrays["rare"] > 0.5)),
        ("Rare incorrect", (~correct) & (arrays["rare"] > 0.5)),
    ]:
        if not mask.any():
            continue
        rows.append({
            "Group": name,
            "Samples": int(mask.sum()),
            "Mean confidence": float(confidence[mask].mean()),
            "Median confidence": float(np.median(confidence[mask])),
            "Mean uncertainty": float(arrays["uncertainty"][mask].mean()),
            "Mean reconstruction error": float(arrays["reconstruction_error"][mask].mean()),
            "High-confidence share (>=0.9)": float((confidence[mask] >= 0.9).mean()),
        })
    return pd.DataFrame(rows)


def reconstruction_group_table(arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    y_true = arrays["y"].astype(int)
    pred = arrays["final_prob"].argmax(axis=1)
    correct = pred == y_true
    groups = [
        ("Common correct", (arrays["rare"] <= 0.5) & correct),
        ("Common incorrect", (arrays["rare"] <= 0.5) & (~correct)),
        ("Rare correct", (arrays["rare"] > 0.5) & correct),
        ("Rare incorrect", (arrays["rare"] > 0.5) & (~correct)),
    ]
    rows = []
    for name, mask in groups:
        values = arrays["reconstruction_error"][mask]
        if not len(values):
            continue
        rows.append({
            "Group": name,
            "Samples": len(values),
            "Mean": float(values.mean()),
            "Std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "Median": float(np.median(values)),
            "Q1": float(np.quantile(values, 0.25)),
            "Q3": float(np.quantile(values, 0.75)),
            "P95": float(np.quantile(values, 0.95)),
        })
    return pd.DataFrame(rows)


def model_complexity_row(model, checkpoint_path: Path, variant: str, seed: int) -> Dict[str, Any]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "Method": variant,
        "Seed": seed,
        "Parameters": parameters,
        "Trainable parameters": trainable,
        "Checkpoint size (MB)": checkpoint_path.stat().st_size / (1024 ** 2),
        "Parameter memory FP32 (MB)": parameters * 4 / (1024 ** 2),
    }


def route_cost_sweep(arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    y_true = arrays["y"].astype(int)
    route_prob = np.clip(arrays["route_prob"], 1e-12, 1.0)
    path_prob = np.stack([
        arrays["edge_prob"],
        arrays["specialist_prob"],
        arrays["cloud_prob"],
    ], axis=1)
    rows = []
    for penalty in COST_SWEEP:
        score = np.log(route_prob) - penalty * ROUTE_COSTS[None, :]
        route = score.argmax(axis=1)
        selected_prob = path_prob[np.arange(len(route)), route]
        pred = selected_prob.argmax(axis=1)
        rows.append({
            "Cost penalty": penalty,
            "Accuracy": accuracy_score(y_true, pred),
            "Balanced accuracy": balanced_accuracy_score(y_true, pred),
            "Macro F1": f1_score(y_true, pred, average="macro", zero_division=0),
            "Rare-class macro recall": (
                precision_recall_fscore_support(
                    y_true[arrays["rare"] > 0.5],
                    pred[arrays["rare"] > 0.5],
                    average="macro",
                    zero_division=0,
                )[1] if np.any(arrays["rare"] > 0.5) else np.nan
            ),
            "Mean route cost": float(ROUTE_COSTS[route].mean()),
            "Edge route": float((route == 0).mean()),
            "Specialist route": float((route == 1).mean()),
            "Cloud route": float((route == 2).mean()),
        })
    return pd.DataFrame(rows)

# =============================================================================
# FIGURES
# =============================================================================


def fig_multi_seed_performance(summary: pd.DataFrame) -> None:
    data = summary.sort_values("macro_f1_mean")
    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    y = np.arange(len(data))
    ax.errorbar(
        data["macro_f1_mean"],
        y,
        xerr=data["macro_f1_std"],
        fmt="o",
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(data["Method"])
    ax.set_xlabel("Macro F1-score")
    ax.set_title("Multi-seed classification performance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "Fig_01_MultiSeed_MacroF1")


def fig_ablation_delta(ablation: pd.DataFrame) -> None:
    if ablation.empty:
        return
    data = ablation[ablation["Variant"] != "full_rcsmoe"].sort_values("delta_macro_f1")
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    y = np.arange(len(data))
    ax.barh(y, data["delta_macro_f1"])
    ax.axvline(0.0, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["Variant"])
    ax.set_xlabel(r"$\Delta$ Macro F1 relative to full RCS-MoE")
    ax.set_title("Component contribution analysis")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "Fig_02_Ablation_Delta")


def fig_reliability_diagram(bins: pd.DataFrame, variant: str, seed: int) -> None:
    valid = bins[bins["count"] > 0]
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
    ax.plot(valid["mean_confidence"], valid["accuracy"], marker="o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability: {variant}, seed {seed}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, f"Fig_03_Reliability_{safe_name(variant)}_seed_{seed}")


def fig_confidence_histogram(arrays: Dict[str, np.ndarray], variant: str, seed: int) -> None:
    y = arrays["y"].astype(int)
    prob = arrays["final_prob"]
    pred = prob.argmax(axis=1)
    confidence = prob.max(axis=1)
    bins = np.linspace(0, 1, 31)
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ax.hist(confidence[pred == y], bins=bins, alpha=0.65, label="Correct")
    ax.hist(confidence[pred != y], bins=bins, alpha=0.65, label="Incorrect")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Samples")
    ax.set_title(f"Confidence distribution: {variant}")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, f"Fig_04_Confidence_{safe_name(variant)}_seed_{seed}")


def fig_class_f1(class_df: pd.DataFrame, variant: str, seed: int, rare_only: bool = False) -> None:
    data = class_df.copy()
    if rare_only:
        data = data[data["Is rare"] == True]
    data = data.sort_values("F1")
    height = max(3.0, 0.18 * len(data) + 1.0)
    fig, ax = plt.subplots(figsize=(7.16, height))
    y = np.arange(len(data))
    ax.barh(y, data["F1"])
    ax.set_yticks(y)
    ax.set_yticklabels(data["Class"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score")
    ax.set_title("Rare-class F1-score" if rare_only else "Fine-grained class F1-score")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    suffix = "Rare" if rare_only else "All"
    save_figure(fig, f"Fig_05_ClassF1_{suffix}_{safe_name(variant)}_seed_{seed}")


def fig_family_routing(family_route: pd.DataFrame, variant: str, seed: int) -> None:
    if family_route.empty:
        return
    data = family_route.set_index("Family")[["Edge route", "Specialist route", "Cloud route"]]
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Routing fraction")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title("Inference route by attack family")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save_figure(fig, f"Fig_06_Family_Routing_{safe_name(variant)}_seed_{seed}")


def fig_expert_heatmap(expert_family: pd.DataFrame, variant: str, seed: int) -> None:
    if expert_family.empty:
        return
    matrix = expert_family.set_index("Family").drop(columns=["Samples"]).to_numpy()
    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=max(float(np.nanmax(matrix)), 1e-6))
    ax.set_yticks(np.arange(len(expert_family)))
    ax.set_yticklabels(expert_family["Family"])
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(expert_family.columns[2:], rotation=35, ha="right")
    ax.set_title("Top-1 expert assignment by attack family")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    save_figure(fig, f"Fig_07_Expert_Specialization_{safe_name(variant)}_seed_{seed}")


def fig_attention_heatmap(layer_feature: pd.DataFrame, variant: str, seed: int) -> None:
    if layer_feature.empty:
        return
    pivot = layer_feature.pivot(index="Layer", columns="Feature", values="Attention")
    # Limit labels to the strongest features for readability.
    strongest = pivot.mean(axis=0).sort_values(ascending=False).head(20).index
    pivot = pivot[strongest]
    fig, ax = plt.subplots(figsize=(7.16, 2.7))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"Layer {value}" for value in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=55, ha="right")
    ax.set_title("Layer-wise attention received by the 20 strongest features")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    save_figure(fig, f"Fig_08_Attention_Features_{safe_name(variant)}_seed_{seed}")


def fig_reconstruction_distribution(arrays: Dict[str, np.ndarray], variant: str, seed: int) -> None:
    y = arrays["y"].astype(int)
    pred = arrays["final_prob"].argmax(axis=1)
    correct = pred == y
    groups = [
        ("Common correct", (arrays["rare"] <= 0.5) & correct),
        ("Common incorrect", (arrays["rare"] <= 0.5) & (~correct)),
        ("Rare correct", (arrays["rare"] > 0.5) & correct),
        ("Rare incorrect", (arrays["rare"] > 0.5) & (~correct)),
    ]
    values = [arrays["reconstruction_error"][mask] for _, mask in groups if mask.any()]
    labels = [name for name, mask in groups if mask.any()]
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    # Matplotlib 3.9+ renamed ``labels`` to ``tick_labels``.
    # Use the new argument first and retain compatibility with older releases.
    try:
        ax.boxplot(values, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_ylabel("Reconstruction error")
    ax.set_title("Reconstruction-error distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, f"Fig_09_Reconstruction_{safe_name(variant)}_seed_{seed}")


def fig_cost_tradeoff(cost_df: pd.DataFrame, variant: str, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    scatter = ax.scatter(cost_df["Mean route cost"], cost_df["Macro F1"], c=cost_df["Cost penalty"])
    ax.plot(cost_df["Mean route cost"], cost_df["Macro F1"], linewidth=0.8)
    ax.set_xlabel("Mean normalized route cost")
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Post-hoc cost-accuracy sensitivity")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Cost penalty")
    fig.tight_layout()
    save_figure(fig, f"Fig_10_Cost_Tradeoff_{safe_name(variant)}_seed_{seed}")


def fig_embedding_pca(
    arrays: Dict[str, np.ndarray],
    family_names: Sequence[str],
    variant: str,
    seed: int,
) -> None:
    representation = arrays["representation"]
    family = arrays["embedding_family"].astype(int)
    if len(representation) < 2:
        return
    pca = PCA(n_components=2, random_state=seed)
    embedded = pca.fit_transform(representation)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for family_id, family_name in enumerate(family_names):
        mask = family == family_id
        if mask.any():
            ax.scatter(embedded[mask, 0], embedded[mask, 1], s=4, alpha=0.45, label=family_name)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Shared representation by attack family")
    ax.legend(frameon=False, fontsize=5.2, ncol=2)
    fig.tight_layout()
    save_figure(fig, f"Fig_11_Embedding_PCA_{safe_name(variant)}_seed_{seed}")

# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main() -> None:
    log(f"Device: {DEVICE}")
    module = import_training_module()
    model_dir = locate_model_dir()
    checkpoints = discover_checkpoints(model_dir)
    log(f"Discovered {len(checkpoints)} neural checkpoints")

    # Restrict analyses to checkpoints that actually exist. This permits the
    # script to work with seeds 11 and 22 now, then automatically include later
    # seeds after those checkpoints are added.
    seeds = sorted({seed for _, seed, _ in checkpoints})
    variants = sorted({variant for variant, _, _ in checkpoints})
    log(f"Seeds: {seeds}")
    log(f"Variants: {variants}")

    run_metric_rows: List[Dict[str, Any]] = []
    complexity_rows: List[Dict[str, Any]] = []
    calibration_rows: List[pd.DataFrame] = []
    reliability_rows: List[pd.DataFrame] = []

    # Generate detailed figures for one representative full-model seed. Tables
    # remain multi-seed. Prefer seed 11 when available.
    full_seeds = sorted(seed for variant, seed, _ in checkpoints if variant == "full_rcsmoe")
    representative_seed = 11 if 11 in full_seeds else (full_seeds[0] if full_seeds else seeds[0])

    prepared_cache: Dict[int, Any] = {}
    loader_cache: Dict[int, Any] = {}

    for variant, seed, checkpoint_path in checkpoints:
        if seed not in prepared_cache:
            log(f"Preparing fixed test data for seed {seed}")
            prepared = module.load_and_prepare(seed)
            _, _, test_loader = module.create_loaders(prepared)
            if BATCH_SIZE_OVERRIDE is not None:
                test_loader = torch.utils.data.DataLoader(
                    test_loader.dataset,
                    batch_size=BATCH_SIZE_OVERRIDE,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                )
            prepared_cache[seed] = prepared
            loader_cache[seed] = test_loader

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = build_model_from_checkpoint(module, checkpoint)
        arrays = load_or_run_cache(model, loader_cache[seed], variant, seed)

        run_metric_rows.append(compute_run_metrics(variant, seed, arrays))
        complexity_rows.append(model_complexity_row(model, checkpoint_path, variant, seed))

        cal_summary, bins = calibration_table(arrays, variant, seed)
        calibration_rows.append(cal_summary)
        reliability_rows.append(bins)

        prepared = prepared_cache[seed]
        label_names = prepared.label_encoder.classes_.tolist()
        family_names = prepared.family_encoder.classes_.tolist()
        rare_labels = set(
            joblib.load(model_dir / f"preprocess_seed_{seed}.joblib").get("rare_labels", [])
        ) if (model_dir / f"preprocess_seed_{seed}.joblib").exists() else set()

        # Save detailed analysis for every full-model seed. This supports
        # multi-seed consistency checks without producing hundreds of figures.
        if variant == "full_rcsmoe":
            class_df = class_performance_table(arrays, label_names)
            class_df["Is rare"] = class_df["Class"].isin(rare_labels)
            save_df(
                class_df,
                f"Table_05_Classwise_full_rcsmoe_seed_{seed}",
                "Fine-grained class performance of the complete RCS-MoE model.",
            )
            save_df(
                class_df[class_df["Is rare"]].copy(),
                f"Table_06_RareClass_full_rcsmoe_seed_{seed}",
                "Rare-class performance of the complete RCS-MoE model.",
            )

            family_df = family_performance_table(arrays, family_names)
            save_df(
                family_df,
                f"Table_07_Family_full_rcsmoe_seed_{seed}",
                "Attack-family performance of the complete RCS-MoE model.",
            )

            class_route = routing_group_table(
                arrays,
                arrays["y"].astype(int),
                label_names,
                "Class",
            )
            family_route = routing_group_table(
                arrays,
                arrays["family"].astype(int),
                family_names,
                "Family",
            )
            save_df(class_route, f"Table_08_RoutingByClass_seed_{seed}")
            save_df(family_route, f"Table_09_RoutingByFamily_seed_{seed}")
            save_df(rarity_routing_table(arrays), f"Table_10_RoutingByRarity_seed_{seed}")
            save_df(difficulty_routing_table(arrays), f"Table_11_RoutingByDifficulty_seed_{seed}")

            expert_overall, expert_family = expert_utilization_table(arrays, family_names)
            save_df(expert_overall, f"Table_12_ExpertUtilization_seed_{seed}")
            save_df(expert_family, f"Table_13_ExpertByFamily_seed_{seed}")
            save_df(
                expert_class_specialization(arrays, label_names),
                f"Table_14_ExpertClassSpecialization_seed_{seed}",
            )

            attention_entropy, feature_attention, layer_attention = attention_statistics(
                arrays, prepared.feature_names
            )
            save_df(attention_entropy, f"Table_15_AttentionEntropy_seed_{seed}")
            save_df(feature_attention, f"Table_16_FeatureAttention_seed_{seed}")
            save_df(layer_attention, f"Table_17_LayerFeatureAttention_seed_{seed}")

            save_df(confidence_failure_table(arrays), f"Table_18_ConfidenceFailure_seed_{seed}")
            save_df(reconstruction_group_table(arrays), f"Table_19_ReconstructionGroups_seed_{seed}")

            cost_df = route_cost_sweep(arrays)
            save_df(
                cost_df,
                f"Table_20_CostAccuracySensitivity_seed_{seed}",
                "Post-hoc route-cost sensitivity of the complete RCS-MoE model.",
            )

            if seed == representative_seed:
                fig_reliability_diagram(bins, variant, seed)
                fig_confidence_histogram(arrays, variant, seed)
                fig_class_f1(class_df, variant, seed, rare_only=False)
                fig_class_f1(class_df, variant, seed, rare_only=True)
                fig_family_routing(family_route, variant, seed)
                fig_expert_heatmap(expert_family, variant, seed)
                fig_attention_heatmap(layer_attention, variant, seed)
                fig_reconstruction_distribution(arrays, variant, seed)
                fig_cost_tradeoff(cost_df, variant, seed)
                fig_embedding_pca(arrays, family_names, variant, seed)

        del model, arrays, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_metrics = pd.DataFrame(run_metric_rows)

    # Merge classical metrics already produced by the original run. Neural
    # rows from this script replace their older counterparts because these rows
    # add Brier score and NLL.
    metrics_path_candidates = [
        RESULT_DIR / "metrics_all_runs.csv",
        RESULT_DIR / "metrics_live.csv",
    ]
    existing_path = next((path for path in metrics_path_candidates if path.exists()), None)
    if existing_path is not None:
        existing = pd.read_csv(existing_path)
        classical = existing[existing["method"].isin(["random_forest", "xgboost"])].copy()
        for column in run_metrics.columns:
            if column not in classical.columns:
                classical[column] = np.nan
        for column in classical.columns:
            if column not in run_metrics.columns:
                run_metrics[column] = np.nan
        run_metrics = pd.concat([run_metrics, classical[run_metrics.columns]], ignore_index=True)

    save_df(run_metrics, "Table_01_AllRunMetrics")
    summary = aggregate_runs(run_metrics)
    save_df(
        summary,
        "Table_02_MultiSeedSummary",
        "Multi-seed classification, calibration, efficiency, and routing results.",
    )

    ablation = ablation_delta_table(summary)
    save_df(
        ablation,
        "Table_03_AblationDeltas",
        "Component ablation relative to the complete RCS-MoE model.",
    )

    complexity = pd.DataFrame(complexity_rows)
    save_df(
        complexity,
        "Table_04_ModelComplexity",
        "Parameter count and checkpoint storage of pretrained neural variants.",
    )

    calibration = pd.concat(calibration_rows, ignore_index=True)
    reliability = pd.concat(reliability_rows, ignore_index=True)
    save_df(calibration, "Table_21_CalibrationSummary")
    save_df(reliability, "Table_22_ReliabilityBins")

    fig_multi_seed_performance(summary)
    fig_ablation_delta(ablation)

    report = {
        "device": str(DEVICE),
        "source_script": str(next(path for path in SOURCE_CANDIDATES if path.exists() and path.resolve() != Path(__file__).resolve())),
        "checkpoint_directory": str(model_dir),
        "seeds": seeds,
        "variants": variants,
        "representative_seed_for_figures": representative_seed,
        "tables_directory": str(TABLE_DIR),
        "figures_directory": str(FIG_DIR),
        "cache_directory": str(DATA_OUT_DIR),
        "notes": [
            "All neural analyses load pretrained checkpoints and do not retrain the models.",
            "The cost-accuracy sweep is a post-hoc inference-policy sensitivity analysis.",
            "Classical baseline rows are imported from the existing metrics file when available.",
        ],
    }
    save_json(report, DATA_OUT_DIR / "pretrained_analysis_report.json")

    log("Pretrained analysis completed.")
    log(f"Tables: {TABLE_DIR}")
    log(f"Figures: {FIG_DIR}")
    log(f"Cached arrays: {DATA_OUT_DIR}")


if __name__ == "__main__":
    main()
