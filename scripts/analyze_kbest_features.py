#!/usr/bin/env python
"""
Análise de seleção de features via SelectKBest (migra kbest_analysis.ipynb).

Para cada dataset radiômico (BW, Doppler, BW+Doppler), calcula os scores de
todas as features via SelectKBest (k='all') sob dois critérios — F-Score
(ANOVA) e Informação Mútua — usando apenas o split `trainval`. Gera:
  - gráficos de barra com o ranking completo por dataset × critério
  - análise de consenso (features eleitas por ambos os critérios no top-K)
  - mapa de sobreposição entre critérios
  - curva de score (score vs. posição no ranking)
  - CSV com o ranking completo (dataset, criterion, rank, feature, score)

Saídas em --output-dir:
  - task7a_kbest_rankings.csv
  - task7a_kbest_all_{dataset}.png       (um por dataset)
  - task7a_kbest_consensus.png
  - task7a_kbest_overlap.png
  - task7a_kbest_score_curves.png

Exemplo:
    python scripts/analyze_kbest_features.py --output-dir experiments/kbest_analysis
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # execução via terminal: evita plt.show() bloqueando sem display interativo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str((Path(__file__).parent / ".." / "modules").resolve()))

from evaluation import DATASETS, META_COLS, plot_kbest_overlap, plot_kbest_score_curves
from kbest_utils import build_criteria, shorten_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DS_COLORS = {
    "BW": "#A3C4F3",
    "Doppler": "#80CED7",
    "BW+Doppler": "#C3AED6",
}
ACCENT_COLORS = {
    "BW": "#5B8DB8",
    "Doppler": "#4A9EA6",
    "BW+Doppler": "#8B6FB0",
}
CRIT_COLORS = {
    "F-Score (ANOVA)": "#5B8DB8",
    "Mutual Information": "#8B6FB0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analisa scores SelectKBest (ANOVA/MI) por dataset radiômico.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/images/datasets"), help="Pasta com os CSVs radiômicos.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pasta onde os CSVs/plots são salvos.")
    parser.add_argument("--datasets", nargs="+", default=None, choices=list(DATASETS.keys()), help="Datasets a processar (default: todos).")
    parser.add_argument("--top-consensus", type=int, default=15, help="Quantas features exibir no gráfico de consenso.")
    parser.add_argument("--top-k-consensus", type=int, default=20, help="Top-K de cada critério usado para definir consenso/sumário.")
    parser.add_argument("--top-overlap", type=int, default=20, help="Top-K de cada critério usado no mapa de sobreposição.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para mutual_info_classif.")
    return parser.parse_args()


def load_datasets(data_dir: Path, dataset_names: list[str]) -> dict[str, pd.DataFrame]:
    dfs = {}
    for ds_name in dataset_names:
        csv_path = data_dir / DATASETS[ds_name]
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV não encontrado para dataset '{ds_name}': {csv_path}")
        df = dfs[ds_name] = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns if c not in META_COLS]
        trainval = df[df["split"] == "trainval"]
        n_benign = int((trainval["target"] == 0).sum())
        n_malignant = int((trainval["target"] == 1).sum())
        logger.info(
            "%-10s: trainval=%d test=%d features=%d benigno=%d maligno=%d",
            ds_name, len(trainval), len(df) - len(trainval), len(feat_cols), n_benign, n_malignant,
        )
    return dfs


def compute_scores(dfs: dict[str, pd.DataFrame], criteria: dict) -> dict:
    all_results = {}
    for ds_name, df in dfs.items():
        feat_cols = [c for c in df.columns if c not in META_COLS]
        trainval = df[df["split"] == "trainval"]
        X = trainval[feat_cols].to_numpy(dtype=float)
        y = trainval["target"].to_numpy()

        all_results[ds_name] = {}
        for crit_name, fn in criteria.items():
            scores = fn(X, y)
            idx_sorted = np.argsort(scores)[::-1]
            all_results[ds_name][crit_name] = {
                "full_names": [feat_cols[i] for i in idx_sorted],
                "short_names": [shorten_name(feat_cols[i]) for i in idx_sorted],
                "scores": scores[idx_sorted],
            }
            logger.info(
                "%-10s / %-20s: max=%.4f @ %s",
                ds_name, crit_name, scores[idx_sorted[0]], shorten_name(feat_cols[idx_sorted[0]]),
            )
    return all_results


def plot_kbest_bars(all_results: dict, criteria: dict, output_dir: Path) -> None:
    for ds_name, results_by_crit in all_results.items():
        n_feats = len(next(iter(results_by_crit.values()))["scores"])
        bar_height = 0.18
        fig_h = max(8, n_feats * bar_height)

        fig, axes = plt.subplots(1, len(criteria), figsize=(18, fig_h))
        axes = [axes] if len(criteria) == 1 else axes
        fig.suptitle(f"SelectKBest — todas as {n_feats} features | {ds_name}", fontsize=13, fontweight="bold")

        for ax, crit_name in zip(axes, criteria.keys()):
            res = results_by_crit[crit_name]
            names = res["short_names"][::-1]
            values = res["scores"][::-1]
            color = CRIT_COLORS.get(crit_name, "#5B8DB8")

            ax.barh(range(n_feats), values, color=color, alpha=0.75, edgecolor="none", height=0.85)
            ax.set_yticks(range(n_feats))
            ax.set_yticklabels(names, fontsize=4.5)
            ax.set_xlabel("Score", fontsize=9)
            ax.set_title(crit_name, fontsize=10, fontweight="bold", color=color)
            ax.tick_params(axis="x", labelsize=8)

        plt.tight_layout()
        safe_name = ds_name.replace("+", "_plus_")
        out_path = output_dir / f"task7a_kbest_all_{safe_name}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Salvo: %s", out_path)


def plot_consensus(all_results: dict, criteria: dict, output_path: Path, top_k: int, top_consensus: int) -> None:
    ds_names = list(all_results.keys())
    fig, axes = plt.subplots(1, len(ds_names), figsize=(20, 7))
    axes = [axes] if len(ds_names) == 1 else axes
    fig.suptitle(f"Consenso entre critérios — top-{top_consensus} features mais votadas", fontsize=13, fontweight="bold")

    for ax, ds_name in zip(axes, ds_names):
        results_by_crit = all_results[ds_name]
        n_feats = len(next(iter(results_by_crit.values()))["scores"])
        k_eff = min(top_k, n_feats)

        counter = Counter()
        for crit_name in criteria.keys():
            counter.update(results_by_crit[crit_name]["full_names"][:k_eff])

        top = counter.most_common(top_consensus)
        names = [shorten_name(n) for n, _ in top][::-1]
        counts = [c for _, c in top][::-1]

        accent = ACCENT_COLORS.get(ds_name, "#5B8DB8")
        pastel = DS_COLORS.get(ds_name, "#A3C4F3")
        colors_bar = [accent if c == len(criteria) else pastel for c in counts]

        bars = ax.barh(range(len(names)), counts, color=colors_bar, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(f"Nº de critérios que elegeram esta feature (top-{k_eff})", fontsize=8)
        ax.set_title(ds_name, fontsize=11, fontweight="bold", color=accent)
        ax.set_xlim(0, len(criteria) + 0.5)
        ax.set_xticks(range(1, len(criteria) + 1))
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f"{cnt}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Salvo: %s", output_path)


def print_consensus_summary(all_results: dict, criteria: dict, top_k: int) -> None:
    logger.info("=== FEATURES COM CONSENSO TOTAL (%d/%d CRITÉRIOS) — top-%d de cada ===", len(criteria), len(criteria), top_k)
    for ds_name, results_by_crit in all_results.items():
        counter = Counter()
        for crit_name in criteria.keys():
            counter.update(results_by_crit[crit_name]["full_names"][:top_k])
        top_full = [n for n, c in counter.most_common() if c == len(criteria)]
        logger.info("--- %s: %d features com consenso total ---", ds_name, len(top_full))
        for i, feat in enumerate(top_full, 1):
            logger.info("  %2d. %s", i, feat)


def build_rankings_df(all_results: dict) -> pd.DataFrame:
    rows = []
    for ds_name, results_by_crit in all_results.items():
        for crit_name, res in results_by_crit.items():
            for rank, (full_name, score) in enumerate(zip(res["full_names"], res["scores"]), 1):
                rows.append({
                    "dataset": ds_name,
                    "criterion": crit_name,
                    "rank": rank,
                    "feature": full_name,
                    "feature_short": shorten_name(full_name),
                    "score": score,
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    dataset_names = args.datasets or list(DATASETS.keys())
    dfs = load_datasets(args.data_dir, dataset_names)
    criteria = build_criteria(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = compute_scores(dfs, criteria)

    plot_kbest_bars(all_results, criteria, args.output_dir)
    plot_consensus(all_results, criteria, args.output_dir / "task7a_kbest_consensus.png",
                   top_k=args.top_k_consensus, top_consensus=args.top_consensus)

    df_rankings = build_rankings_df(all_results)

    plot_kbest_overlap(df_rankings, str(args.output_dir / "task7a_kbest_overlap.png"), top_n=args.top_overlap)

    df_plot = df_rankings[df_rankings["dataset"] != "BW+Doppler"] if "BW+Doppler" in dataset_names and len(dataset_names) > 1 else df_rankings
    plot_kbest_score_curves(
        df_plot, str(args.output_dir / "task7a_kbest_score_curves.png"),
        n_graphic_lgd="Mutual Information",
        plot_title="Curva de Score SelectKBest",
    )

    csv_out = args.output_dir / "task7a_kbest_rankings.csv"
    df_rankings.to_csv(csv_out, index=False)
    logger.info("Rankings salvos em: %s (total=%d linhas)", csv_out, len(df_rankings))

    print_consensus_summary(all_results, criteria, top_k=args.top_k_consensus)


if __name__ == "__main__":
    main()
