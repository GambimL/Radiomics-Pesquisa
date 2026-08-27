"""
Funções compartilhadas entre Trilha A e Trilha B para avaliação de modelos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, f1_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

DS_COLORS = {'BW': '#5B8DB8', 'Doppler': '#4A9EA6', 'BW+Doppler': '#8B6FB0'}

META_COLS = ['image_id', 'label', 'target', 'dx', 'split', 'fold']

DATASETS = {
    'BW':         'radiomics_bw.csv',
    'Doppler':    'radiomics_doppler.csv',
    'BW+Doppler': 'radiomics_bw_doppler.csv',
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcula métricas de qualidade para um fold ou conjunto de avaliação."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'auc':         roc_auc_score(y_true, y_prob),
        'accuracy':    accuracy_score(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision':   precision_score(y_true, y_pred, zero_division=0),
        'f1':          f1_score(y_true, y_pred, zero_division=0),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calcula o threshold ótimo pelo índice J de Youden (argmax TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Binariza y_prob no threshold informado e calcula métricas via compute_metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return compute_metrics(y_true, y_pred, y_prob)


def get_scores(pipe, X):
    """Scores do pipeline para a classe positiva.

    Usa predict_proba quando disponivel; caso contrario (modelos sem
    probabilidade calibrada), recorre a decision_function. Retorna
    (scores, is_proba) para que o plot saiba qual escala esta sendo usada.
    """
    if hasattr(pipe, 'predict_proba'):
        return pipe.predict_proba(X)[:, 1], True
    return pipe.decision_function(X), False


def aggregate_folds(fold_results: list[dict]) -> dict:
    """Calcula média e desvio-padrão das métricas dos folds (Tarefa 4)."""
    metric_keys = ['auc', 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1']
    agg = {}
    for m in metric_keys:
        vals = [r[m] for r in fold_results]
        agg[f'{m}_mean'] = float(np.mean(vals))
        agg[f'{m}_std']  = float(np.std(vals))
    return agg


def run_kfold(X: np.ndarray, y: np.ndarray, pipeline: Pipeline,
              n_folds: int = 5, seed: int = 42) -> list[dict]:
    """
    Executa k-fold estratificado sobre o pipeline fornecido.

    O pipeline é reconstruído (clone) a cada fold para garantir isolamento.
    Retorna lista de dicts com métricas por fold.
    """
    from sklearn.base import clone

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        pipe = clone(pipeline)
        pipe.fit(X[train_idx], y[train_idx])
        y_prob = pipe.predict_proba(X[val_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y[val_idx], y_pred, y_prob)
        metrics['fold'] = fold_idx
        fold_results.append(metrics)
    return fold_results


def select_top_models(df_agg: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Seleciona os top_n modelos por AUC médio em cada dataset (Tarefa 5).
    Retorna DataFrame com coluna 'rank' adicionada.
    """
    top_rows = []
    for ds_name in df_agg['dataset'].unique():
        sub = df_agg[df_agg['dataset'] == ds_name].dropna(subset=['auc_mean'])
        top = sub.nlargest(top_n, 'auc_mean').copy()
        top['rank'] = range(1, len(top) + 1)
        top_rows.append(top)
    return pd.concat(top_rows, ignore_index=True)


def plot_metrics_barh(df_agg: pd.DataFrame, ds_name: str, ax,
                      metric: str = 'auc_mean') -> None:
    """Gráfico horizontal de barras das métricas médias, ordenado pela métrica alvo."""
    std_col = metric.replace('_mean', '_std')
    df_sorted = df_agg[df_agg['dataset'] == ds_name].sort_values(metric)
    xerr = df_sorted[std_col] if std_col in df_sorted.columns else None
    ax.barh(df_sorted['model'], df_sorted[metric],
            xerr=xerr, color=DS_COLORS[ds_name],
            alpha=0.80, capsize=3, edgecolor='white')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric.replace('_mean', '').upper(), fontsize=8)
    ax.set_title(f'{ds_name} — {metric.replace("_mean", "").upper()}',
                 fontsize=9, fontweight='bold', color=DS_COLORS[ds_name])
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=6)
    ax.tick_params(axis='x', labelsize=7)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)


def plot_metrics_heatmap(df_top: pd.DataFrame, top_n: int, output_path: str,
                         title: str) -> None:
    """Heatmap das principais métricas para os top_n modelos de cada dataset."""
    metric_cols   = ['auc_mean', 'accuracy_mean', 'sensitivity_mean', 'specificity_mean', 'f1_mean']
    metric_labels = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1']

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for ax, ds_name in zip(axes, DATASETS.keys()):
        sub = df_top[df_top['dataset'] == ds_name][['model'] + metric_cols].copy()
        sub = sub.sort_values('auc_mean', ascending=False).set_index('model')
        sub.columns = metric_labels

        im = ax.imshow(sub.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, fontsize=8, rotation=30, ha='right')
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.index, fontsize=8)
        ax.set_title(ds_name, fontsize=11, fontweight='bold', color=DS_COLORS[ds_name])

        for i in range(len(sub)):
            for j in range(len(metric_labels)):
                ax.text(j, i, f'{sub.values[i, j]:.2f}',
                        ha='center', va='center', fontsize=8,
                        color='black' if 0.3 < sub.values[i, j] < 0.8 else 'white')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Salvo: {output_path}')


def plot_auc_boxplot(df_folds: pd.DataFrame, df_top: pd.DataFrame,
                     top_n: int, output_path: str, title: str) -> None:
    """Boxplot da distribuição de AUC por fold para os top_n modelos de cada dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for ax, ds_name in zip(axes, DATASETS.keys()):
        top_models = df_top[df_top['dataset'] == ds_name]['model'].tolist()
        sub_folds  = df_folds[
            (df_folds['dataset'] == ds_name) &
            (df_folds['model'].isin(top_models))
        ]
        data_list = [sub_folds[sub_folds['model'] == m]['auc'].values for m in top_models]
        labels    = [m[:25] + '...' if len(m) > 25 else m for m in top_models]

        bp = ax.boxplot(data_list, vert=True, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2})
        for patch in bp['boxes']:
            patch.set_facecolor(DS_COLORS[ds_name])
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=7)
        ax.set_ylabel('AUC')
        ax.set_ylim(0, 1.05)
        ax.set_title(ds_name, fontsize=11, fontweight='bold', color=DS_COLORS[ds_name])
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Salvo: {output_path}')


# -----
# Tabelas e figuras da Seção de Resultados (comparativo Trilha A vs B)
# -----
                      
def build_top5_comparison_table(df_agg_a: pd.DataFrame, df_agg_b: pd.DataFrame,
                                top_n: int = 5) -> pd.DataFrame:
    """
    Monta a tabela 'Top 5 modelos por dataset' (Trilha A e B) com AUC mean ± std no k-fold.

    Recebe os DataFrames de task4_aggregated_metrics.csv de cada trilha e
    recalcula o top_n por dataset com select_top_models (os CSVs
    task5_top_models.csv salvos trazem apenas o rank 1). Retorna uma tabela
    única com colunas: trilha, dataset, rank, model, auc_mean, auc_std, auc_fmt.
    """
    rows = []
    for trilha_name, df_agg in (('Trilha A', df_agg_a), ('Trilha B', df_agg_b)):
        top = select_top_models(df_agg, top_n=top_n)
        sub = top[['dataset', 'rank', 'model', 'auc_mean', 'auc_std']].copy()
        sub.insert(0, 'trilha', trilha_name)
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    df['auc_fmt'] = df.apply(lambda r: f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}", axis=1)
    return df.sort_values(['dataset', 'trilha', 'rank']).reset_index(drop=True)


def build_test_results_comparison_table(df_test_a: pd.DataFrame, df_test_b: pd.DataFrame) -> pd.DataFrame:
    """
    Monta a tabela única comparativa Trilha A vs B com os resultados finais no
    teste (threshold=0.5 e Youden), a partir dos test_results.csv de cada trilha.

    Colunas retornadas: trilha, dataset, model, k_features, threshold_youden,
    e as métricas de teste (auc, accuracy, sensitivity, specificity, f1) para
    thr=0.5 e Youden.
    """
    metric_cols = ['auc', 'acc', 'sens', 'spec', 'f1']
    base_cols = ['dataset', 'model', 'threshold_youden']
    keep_05     = [f'test_{m}_05' for m in metric_cols]
    keep_youden = [f'test_{m}_youden' for m in metric_cols]

    rows = []
    for trilha_name, df_test in (('Trilha A', df_test_a), ('Trilha B', df_test_b)):
        sub = df_test.copy()
        if 'k_features' not in sub.columns:
            sub['k_features'] = np.nan
        sub = sub[base_cols + ['k_features'] + keep_05 + keep_youden].copy()
        sub.insert(0, 'trilha', trilha_name)
        rows.append(sub)

    df = pd.concat(rows, ignore_index=True)
    return df[['trilha', 'dataset', 'model', 'k_features', 'threshold_youden'] + keep_05 + keep_youden]


def build_kbest_top_features_table(df_rankings: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Seleciona, para cada combinação dataset × critério (ANOVA / MI), as top_n
    features por score — tabela 'Top features por ANOVA e MI'.

    Espera o CSV gerado em kbest_analysis (colunas: dataset, criterion, rank,
    feature, feature_short, score) e retorna apenas as linhas com rank <= top_n.
    """
    df = df_rankings[df_rankings['rank'] <= top_n].copy()
    return df[['dataset', 'criterion', 'rank', 'feature_short', 'score']].sort_values(
        ['dataset', 'criterion', 'rank']
    ).reset_index(drop=True)


def _confusion_from_counts(tn: int, fp: int, fn: int, tp: int) -> np.ndarray:
    """Reconstrói a matriz de confusão normalizada por linha (classe real) a partir das contagens."""
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return cm / row_sums


def plot_test_confusion_matrices(df_test_a: pd.DataFrame, df_test_b: pd.DataFrame,
                                 output_path: str) -> None:
    """
    Matrizes de confusão (normalizadas por classe real) dos modelos finais no
    teste — uma linha por dataset, comparando Trilha A vs Trilha B (threshold=0.5).

    Reconstrói as matrizes a partir das contagens já salvas em test_results.csv
    (test_tn_05, test_fp_05, test_fn_05, test_tp_05), sem re-treinar modelos.
    """
    labels = ['Benigno', 'Maligno']
    fig, axes = plt.subplots(3, 2, figsize=(11, 14))
    fig.suptitle('Matrizes de Confusão no Teste — Trilha A vs Trilha B (threshold=0.5)',
                 fontsize=13, fontweight='bold')

    for row_idx, ds_name in enumerate(DATASETS.keys()):
        for col_idx, (trilha_name, df_test) in enumerate((('Trilha A', df_test_a), ('Trilha B', df_test_b))):
            row = df_test[df_test['dataset'] == ds_name].iloc[0]
            cm = _confusion_from_counts(row['test_tn_05'], row['test_fp_05'],
                                        row['test_fn_05'], row['test_tp_05'])
            cm_df = pd.DataFrame(cm, index=[f'Real: {l}' for l in labels],
                                 columns=[f'Pred: {l}' for l in labels])

            ax = axes[row_idx][col_idx]
            sns.heatmap(cm_df, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                        annot_kws={'size': 12}, vmin=0, vmax=1, ax=ax)
            ax.set_title(f'{trilha_name} — {ds_name} — {row["model"]}',
                         fontsize=9, color=DS_COLORS[ds_name], fontweight='bold')
            ax.set_xlabel('Classe Predita', fontsize=8)
            ax.set_ylabel('Classe Real', fontsize=8)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Salvo: {output_path}')


def plot_test_sens_spec_comparison(df_test_a: pd.DataFrame, df_test_b: pd.DataFrame,
                                   output_path: str) -> None:
    """
    Compara sensibilidade e especificidade no teste (threshold=0.5) entre
    Trilha A e Trilha B, lado a lado para cada dataset.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Sensibilidade e Especificidade no Teste — Trilha A vs Trilha B (threshold=0.5)',
                 fontsize=13, fontweight='bold')

    width = 0.35
    x = np.arange(2)  # Sensibilidade, Especificidade

    for ax, ds_name in zip(axes, DATASETS.keys()):
        row_a = df_test_a[df_test_a['dataset'] == ds_name].iloc[0]
        row_b = df_test_b[df_test_b['dataset'] == ds_name].iloc[0]
        color = DS_COLORS[ds_name]

        vals_a = [row_a['test_sens_05'], row_a['test_spec_05']]
        vals_b = [row_b['test_sens_05'], row_b['test_spec_05']]

        ax.bar(x - width / 2, vals_a, width, label=f"Trilha A ({row_a['model']})",
               color=color, alpha=0.85)
        ax.bar(x + width / 2, vals_b, width, label=f"Trilha B ({row_b['model']})",
               color=color, alpha=0.45, hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels(['Sensibilidade', 'Especificidade'], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(ds_name, fontsize=11, fontweight='bold', color=color)
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Salvo: {output_path}')


def plot_k_sweep_auc_line(df_sweep: pd.DataFrame, output_path: str) -> None:
    """
    Gráfico de linha do AUC no teste por valor de K no k-sweep (Trilha A),
    uma linha por dataset, anotando o melhor modelo em cada ponto.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle('Trilha A — AUC no Teste por Número de Features (K)',
                 fontsize=13, fontweight='bold')

    for ds_name, color in DS_COLORS.items():
        sub = df_sweep[df_sweep['dataset'] == ds_name].sort_values('k_numeric')
        x_labels = [str(k) for k in sub['k'].tolist()]
        x_pos = range(len(x_labels))

        ax.plot(x_pos, sub['test_auc'], marker='o', label=ds_name,
                color=color, linewidth=2, markersize=6)
        for xi, yi, model in zip(x_pos, sub['test_auc'], sub['best_model']):
            ax.annotate(model[:14], (xi, yi), textcoords='offset points',
                        xytext=(0, 6), fontsize=6, ha='center', color=color, alpha=0.85)

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel('Número de features (K)', fontsize=9)
    ax.set_ylabel('AUC no teste', fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Salvo: {output_path}')


def plot_kbest_score_curves(
        df_rankings: pd.DataFrame, output_path: str, 
        n_graphic_lgd: str,
        top_n: int = None, plot_title: str = '',
    ) -> None:
    """
    Curva de dispersão dos scores do SelectKBest (score vs. posição no
    ranking), uma linha por dataset e um painel por critério (ANOVA/MI).

    É o equivalente "em curva" do gráfico de barras de `kbest_analysis`:
    mostra a queda do score conforme a posição no ranking aumenta,
    permitindo comparar visualmente a concentração de informação entre
    BW, Doppler e BW+Doppler. Se `top_n` for informado, cada curva é
    cortada nas `top_n` primeiras posições.
    """
    criteria = sorted(df_rankings['criterion'].unique())

    fig, axes = plt.subplots(1, len(criteria), figsize=(12, 5))
    fig.suptitle(plot_title,
                  fontsize=13, fontweight='bold')

    ds_names = [ds for ds in DATASETS if ds in df_rankings['dataset'].unique()]

    for ax, crit_name in zip(axes, criteria):
        for ds_name in ds_names:
            sub = df_rankings[(df_rankings['dataset'] == ds_name) &
                               (df_rankings['criterion'] == crit_name)]
            sub = sub.sort_values('rank')
            if top_n is not None:
                sub = sub[sub['rank'] <= top_n]

            color = DS_COLORS[ds_name]
            ax.plot(sub['rank'], sub['score'], color=color, lw=1.8, label=ds_name)
            ax.fill_between(sub['rank'], sub['score'], color=color, alpha=0.12)

        ax.set_xlabel('Posição no ranking', fontsize=9)
        ax.set_ylabel('Score', fontsize=9)
        ax.set_title(crit_name, fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        if crit_name == n_graphic_lgd:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Salvo: {output_path}')


def plot_kbest_overlap(df_rankings: pd.DataFrame, output_path: str, top_n: int = 20) -> None:
    """
    Heatmap de sobreposição entre os critérios ANOVA (F-Score) e Mutual
    Information — quantas das top_n features de cada critério coincidem,
    um painel por dataset.
    """
    criteria = sorted(df_rankings['criterion'].unique())
    crit_short = [c.split(' ')[0] for c in criteria]
    n_c = len(criteria)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(f'Sobreposição das top-{top_n} features entre critérios (ANOVA vs MI)',
                 fontsize=13, fontweight='bold')

    for ax, ds_name in zip(axes, DATASETS.keys()):
        sub = df_rankings[df_rankings['dataset'] == ds_name]
        top_sets = {
            crit: set(sub[(sub['criterion'] == crit) & (sub['rank'] <= top_n)]['feature'])
            for crit in criteria
        }

        overlap = np.zeros((n_c, n_c))
        for i, ca in enumerate(criteria):
            for j, cb in enumerate(criteria):
                overlap[i, j] = len(top_sets[ca] & top_sets[cb])

        im = ax.imshow(overlap, cmap='Blues', vmin=0, vmax=top_n, aspect='auto')
        ax.set_xticks(range(n_c))
        ax.set_yticks(range(n_c))
        ax.set_xticklabels(crit_short, fontsize=9)
        ax.set_yticklabels(crit_short, fontsize=9)
        ax.set_title(ds_name, fontsize=11, fontweight='bold', color=DS_COLORS[ds_name])
        for i in range(n_c):
            for j in range(n_c):
                ax.text(j, i, f'{int(overlap[i, j])}/{top_n}', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if overlap[i, j] > top_n * 0.6 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Features em comum')

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Salvo: {output_path}')
