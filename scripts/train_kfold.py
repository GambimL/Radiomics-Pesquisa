#!/usr/bin/env python
"""
Treino + seleção de modelo via k-fold estratificado (une Trilha A e Trilha B).

Para cada combinação (K de features × modelo × dataset), roda k-fold
estratificado, agrega as métricas por fold e seleciona os top-N modelos por
AUC médio em cada dataset. Suporta uma varredura de múltiplos valores de K
(equivalente ao antigo k-sweep): cada K elege seu próprio vencedor.

Saídas em --output-dir, uma vez por valor de K (sufixo `_k{K}` quando mais
de um K é passado):
  - task3_fold_metrics.csv       métricas por fold, modelo e dataset
  - task4_aggregated_metrics.csv métricas agregadas (mean/std) por modelo e dataset
  - task5_top_models.csv         top-N modelos por dataset (coluna 'rank')
  - task5_auc_barh.png, task5_metrics_heatmap.png, task5_auc_boxplot.png

Exemplos:
    # Trilha A (equivalente a trilha_a.ipynb): SelectKBest com K=20
    python scripts/train_kfold.py --feature-selection kbest --k-values 20 \
        --output-dir experiments/trilha_a

    # Trilha B (equivalente a trilha_b.ipynb): sem seleção de features
    python scripts/train_kfold.py --feature-selection none \
        --output-dir experiments/trilha_b

    # K-sweep (equivalente a trilha_a_k_sweep.ipynb)
    python scripts/train_kfold.py --feature-selection kbest \
        --k-values 5 10 15 20 30 50 100 all \
        --output-dir experiments/trilha_a_k_sweep

    # Restringir a alguns modelos e datasets
    python scripts/train_kfold.py --feature-selection kbest --k-values 20 \
        --models "Linear SVM" "Weighted KNN" --datasets BW Doppler \
        --output-dir experiments/trilha_a_subset
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # execução via terminal: evaluation.py chama plt.show(), que bloqueia sem display interativo
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str((Path(__file__).parent / ".." / "modules").resolve()))

from evaluation import (
    DATASETS,
    META_COLS,
    aggregate_folds,
    plot_auc_boxplot,
    plot_metrics_barh,
    plot_metrics_heatmap,
    run_kfold,
    select_top_models,
)
from models import build_plain_pipeline, build_selectkbest_pipeline, get_classifier, list_classifier_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino + seleção de modelo via k-fold (Trilha A / Trilha B / k-sweep).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature-selection", choices=["kbest", "none"], default="kbest",
        help="'kbest' aplica SelectKBest (Trilha A); 'none' usa todas as features (Trilha B).",
    )
    parser.add_argument(
        "--k-values", nargs="+", default=["20"],
        help="Valores de K para o SelectKBest (ignorado se --feature-selection=none). "
             "Aceita 'all' para usar todas as features. Mais de um valor roda uma varredura (k-sweep).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Nomes dos classificadores a treinar (default: todos os de modules/models.py).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, choices=list(DATASETS.keys()),
        help="Datasets a processar (default: todos — BW, Doppler, BW+Doppler).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/images/datasets"), help="Pasta com os CSVs radiômicos.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pasta onde os CSVs/plots são salvos.")
    parser.add_argument("--n-folds", type=int, default=5, help="Número de folds do k-fold estratificado.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade.")
    parser.add_argument("--top-n", type=int, default=1, help="Top N modelos selecionados por dataset (Tarefa 5).")
    return parser.parse_args()


def parse_k_value(raw: str):
    return raw if raw == "all" else int(raw)


def build_pipeline(feature_selection: str, clf, k, n_features: int):
    if feature_selection == "none":
        return build_plain_pipeline(clf)
    k_eff = n_features if k == "all" else min(k, n_features)
    return build_selectkbest_pipeline(clf, k_eff), k_eff


def load_datasets(data_dir: Path, dataset_names: list[str]) -> dict[str, pd.DataFrame]:
    dfs = {}
    for ds_name in dataset_names:
        csv_path = data_dir / DATASETS[ds_name]
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV não encontrado para dataset '{ds_name}': {csv_path}")
        df = dfs[ds_name] = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns if c not in META_COLS]
        n_train = int((df["split"] == "trainval").sum())
        n_test = int((df["split"] == "test").sum())
        logger.info("%-10s: trainval=%d test=%d features=%d", ds_name, n_train, n_test, len(feat_cols))
    return dfs


def run_for_k(args: argparse.Namespace, dfs: dict[str, pd.DataFrame], model_names: list[str], k) -> None:
    fold_rows, agg_rows = [], []

    for ds_name, df in dfs.items():
        trainval = df[df["split"] == "trainval"]
        feat_cols = [c for c in df.columns if c not in META_COLS]
        X = trainval[feat_cols].to_numpy()
        y = trainval["target"].to_numpy()

        for clf_name in model_names:
            try:
                clf = get_classifier(clf_name)
                if args.feature_selection == "none":
                    pipeline = build_plain_pipeline(clf)
                    k_eff = None
                else:
                    k_eff = X.shape[1] if k == "all" else min(k, X.shape[1])
                    pipeline = build_selectkbest_pipeline(clf, k_eff)

                fold_results = run_kfold(X, y, pipeline, n_folds=args.n_folds, seed=args.seed)
                for r in fold_results:
                    row = {"dataset": ds_name, "model": clf_name, **r}
                    if k_eff is not None:
                        row["k_features"] = k_eff
                    fold_rows.append(row)

                agg = aggregate_folds(fold_results)
                agg_row = {"dataset": ds_name, "model": clf_name, **agg}
                if k_eff is not None:
                    agg_row["k_features"] = k_eff
                agg_rows.append(agg_row)
                logger.info("[K=%s] %-10s | %-32s | AUC=%.3f ± %.3f", k, ds_name, clf_name, agg["auc_mean"], agg["auc_std"])
            except Exception:
                logger.exception("Falhou: dataset=%s modelo=%s K=%s", ds_name, clf_name, k)

    df_folds = pd.DataFrame(fold_rows)
    df_agg = pd.DataFrame(agg_rows)
    df_top = select_top_models(df_agg, top_n=args.top_n)

    suffix = "" if k is None else f"_k{k}"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_folds.to_csv(out_dir / f"task3_fold_metrics{suffix}.csv", index=False)
    df_agg.to_csv(out_dir / f"task4_aggregated_metrics{suffix}.csv", index=False)
    df_top.to_csv(out_dir / f"task5_top_models{suffix}.csv", index=False)
    logger.info("CSVs salvos em %s (sufixo '%s')", out_dir, suffix)

    fig, axes = plt.subplots(1, len(dfs), figsize=(7 * len(dfs), 6))
    axes = [axes] if len(dfs) == 1 else axes
    for ax, ds_name in zip(axes, dfs.keys()):
        plot_metrics_barh(df_agg, ds_name, ax)
    plt.tight_layout()
    plt.savefig(out_dir / f"task5_auc_barh{suffix}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    plot_metrics_heatmap(df_top, args.top_n, str(out_dir / f"task5_metrics_heatmap{suffix}.png"),
                          title=f"Top {args.top_n} modelos por dataset (K={k})")
    plot_auc_boxplot(df_folds, df_top, args.top_n, str(out_dir / f"task5_auc_boxplot{suffix}.png"),
                      title=f"Distribuição de AUC por fold — Top {args.top_n} (K={k})")


def main() -> None:
    args = parse_args()

    dataset_names = args.datasets or list(DATASETS.keys())
    model_names = args.models or list_classifier_names()
    logger.info("Feature selection: %s", args.feature_selection)
    logger.info("Modelos: %d | Datasets: %s", len(model_names), dataset_names)

    dfs = load_datasets(args.data_dir, dataset_names)

    if args.feature_selection == "none":
        run_for_k(args, dfs, model_names, k=None)
    else:
        k_values = [parse_k_value(v) for v in args.k_values]
        for k in k_values:
            run_for_k(args, dfs, model_names, k=k)


if __name__ == "__main__":
    main()
