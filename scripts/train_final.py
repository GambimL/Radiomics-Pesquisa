#!/usr/bin/env python
"""
Treino final + avaliação no test set (une Trilha A e Trilha B treino final).

Lê o `task5_top_models.csv` (ou um CSV com sufixo `_k{K}`, gerado por
train_kfold.py) para saber, por dataset, qual foi o modelo vencedor do
k-fold (assume rank==1). Para cada dataset:
  1. Separa `--val-size` do trainval para calibrar o threshold de Youden.
  2. Retreina o pipeline vencedor no trainval completo.
  3. Avalia no val e no test set, nos thresholds 0.5 e Youden.

Saídas em --output-dir:
  - test_results.csv  uma linha por dataset com métricas de val/test (0.5 e Youden)
  - test_auc_barh.png, test_sens_spec_comparison.png, test_predicted_probs.png,
    test_confusion_matrices.png, test_roc_curves.png
  - roc_data.pkl      dict por dataset com fpr/tpr/auc/model/thresholds (para
                       comparação posterior entre trilhas)

Exemplos:
    # Trilha A: usa o vencedor de experiments/trilha_a/task5_top_models.csv,
    # retreinando com SelectKBest (mesmo K reportado no CSV)
    python scripts/train_final.py --feature-selection kbest \
        --top-models-csv experiments/trilha_a/task5_top_models.csv \
        --output-dir experiments/trilha_a_final

    # Trilha B: sem seleção de features
    python scripts/train_final.py --feature-selection none \
        --top-models-csv experiments/trilha_b/task5_top_models.csv \
        --output-dir experiments/trilha_b_final
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # execução via terminal: evita plt.show() bloqueando sem display interativo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split

sys.path.insert(0, str((Path(__file__).parent / ".." / "modules").resolve()))

from evaluation import (
    DATASETS,
    DS_COLORS,
    META_COLS,
    evaluate_at_threshold,
    get_scores,
    youden_threshold,
)
from models import build_plain_pipeline, build_selectkbest_pipeline, get_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino final + avaliação no test set, a partir do vencedor do k-fold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature-selection", choices=["kbest", "none"], default="kbest",
        help="'kbest' aplica SelectKBest com o K reportado no CSV (Trilha A); 'none' usa todas as features (Trilha B).",
    )
    parser.add_argument(
        "--top-models-csv", type=Path, required=True,
        help="CSV gerado por train_kfold.py (task5_top_models*.csv) com o vencedor (rank==1) por dataset.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/images/datasets"), help="Pasta com os CSVs radiômicos.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pasta onde os resultados/plots são salvos.")
    parser.add_argument("--val-size", type=float, default=0.20, help="Fração do trainval separada para calibrar o threshold de Youden.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
    return parser.parse_args()


def build_pipeline_for_row(args: argparse.Namespace, row: pd.Series, n_features: int):
    clf = get_classifier(row["model"])
    if args.feature_selection == "none":
        return build_plain_pipeline(clf)
    k = row.get("k_features")
    k_eff = n_features if pd.isna(k) else min(int(k), n_features)
    return build_selectkbest_pipeline(clf, k_eff)


def run_dataset(args: argparse.Namespace, ds_name: str, row: pd.Series) -> tuple[dict, dict]:
    csv_path = args.data_dir / DATASETS[ds_name]
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    trainval = df[df["split"] == "trainval"]
    test = df[df["split"] == "test"]
    X_tv, y_tv = trainval[feat_cols].to_numpy(), trainval["target"].to_numpy()
    X_test, y_test = test[feat_cols].to_numpy(), test["target"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=args.val_size, stratify=y_tv, random_state=args.seed,
    )

    pipeline = build_pipeline_for_row(args, row, n_features=X_tv.shape[1])
    pipe = clone(pipeline)
    pipe.fit(X_tv, y_tv)

    prob_val, _ = get_scores(pipe, X_val)
    prob_test, _ = get_scores(pipe, X_test)
    threshold_youden = youden_threshold(y_val, prob_val)

    result = {"dataset": ds_name, "model": row["model"], "threshold_youden": threshold_youden}
    if "k_features" in row and not pd.isna(row["k_features"]):
        result["k_features"] = int(row["k_features"])

    for split_name, y_true, y_prob in (("val", y_val, prob_val), ("test", y_test, prob_test)):
        for thr_name, thr in (("05", 0.5), ("youden", threshold_youden)):
            metrics = evaluate_at_threshold(y_true, y_prob, thr)
            for k, v in metrics.items():
                result[f"{split_name}_{k}_{thr_name}"] = v

    roc_entry = {
        "model": row["model"],
        "fpr": None, "tpr": None, "auc": result["test_auc_05"],
        "thr_youden": threshold_youden,
        "y_test": y_test, "prob_test": prob_test,
    }
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    roc_entry["fpr"], roc_entry["tpr"] = fpr, tpr

    return result, roc_entry


def plot_auc_barh(df_test: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [DS_COLORS[ds] for ds in df_test["dataset"]]
    ax.barh(df_test["dataset"] + " — " + df_test["model"], df_test["test_auc_05"], color=colors, alpha=0.85)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC no teste (threshold=0.5)")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvo: %s", output_path)


def plot_sens_spec(df_test: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_test))
    width = 0.35
    ax.bar(x - width / 2, df_test["test_sensitivity_05"], width, label="Sensibilidade")
    ax.bar(x + width / 2, df_test["test_specificity_05"], width, label="Especificidade")
    ax.set_xticks(x)
    ax.set_xticklabels(df_test["dataset"] + "\n" + df_test["model"], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvo: %s", output_path)


def plot_confusion_matrices(df_test: pd.DataFrame, output_path: Path) -> None:
    n = len(df_test)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes = [axes] if n == 1 else axes
    labels = ["Benigno", "Maligno"]
    for ax, (_, row) in zip(axes, df_test.iterrows()):
        cm = np.array([[row["test_tn_05"], row["test_fp_05"]], [row["test_fn_05"], row["test_tp_05"]]], dtype=float)
        cm = cm / cm.sum(axis=1, keepdims=True)
        cm_df = pd.DataFrame(cm, index=[f"Real: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])
        sns.heatmap(cm_df, annot=True, fmt=".2f", cmap="Blues", cbar=False, vmin=0, vmax=1, ax=ax)
        ax.set_title(f'{row["dataset"]} — {row["model"]}', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvo: %s", output_path)


def plot_predicted_probs(roc_data: dict, output_path: Path) -> None:
    n = len(roc_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = [axes] if n == 1 else axes
    for ax, (ds_name, entry) in zip(axes, roc_data.items()):
        for cls, label in ((0, "Benigno"), (1, "Maligno")):
            mask = entry["y_test"] == cls
            ax.hist(entry["prob_test"][mask], bins=10, alpha=0.6, label=label, range=(0, 1))
        ax.axvline(entry["thr_youden"], color="black", linestyle="--", linewidth=1, label="Youden")
        ax.set_title(f'{ds_name} — {entry["model"]}', fontsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvo: %s", output_path)


def plot_roc_curves(roc_data: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for ds_name, entry in roc_data.items():
        color = DS_COLORS.get(ds_name)
        ax.plot(entry["fpr"], entry["tpr"], label=f'{ds_name} (AUC={entry["auc"]:.3f})', color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("1 - Especificidade")
    ax.set_ylabel("Sensibilidade")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvo: %s", output_path)


def main() -> None:
    args = parse_args()

    if not args.top_models_csv.is_file():
        raise FileNotFoundError(f"--top-models-csv não encontrado: {args.top_models_csv}")
    df_top = pd.read_csv(args.top_models_csv)

    results, roc_data = [], {}
    for ds_name in df_top["dataset"].unique():
        row = df_top[df_top["dataset"] == ds_name].iloc[0]
        logger.info("Treino final: dataset=%s modelo=%s", ds_name, row["model"])
        result, roc_entry = run_dataset(args, ds_name, row)
        results.append(result)
        roc_data[ds_name] = roc_entry
        logger.info("  test AUC (0.5)=%.3f | Youden thr=%.3f", result["test_auc_05"], result["threshold_youden"])

    df_test = pd.DataFrame(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(args.output_dir / "test_results.csv", index=False)
    logger.info("Salvo: %s", args.output_dir / "test_results.csv")

    with open(args.output_dir / "roc_data.pkl", "wb") as f:
        pickle.dump(roc_data, f)
    logger.info("Salvo: %s", args.output_dir / "roc_data.pkl")

    plot_auc_barh(df_test, args.output_dir / "test_auc_barh.png")
    plot_sens_spec(df_test, args.output_dir / "test_sens_spec_comparison.png")
    plot_confusion_matrices(df_test, args.output_dir / "test_confusion_matrices.png")
    plot_predicted_probs(roc_data, args.output_dir / "test_predicted_probs.png")
    plot_roc_curves(roc_data, args.output_dir / "test_roc_curves.png")


if __name__ == "__main__":
    main()
