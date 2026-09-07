"""
Funções auxiliares para análise de seleção de features via SelectKBest.
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


def run_kbest(score_func, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    """Ajusta SelectKBest(k='all') e retorna scores_ para todas as features.

    Usa k='all' para não descartar features — a análise é feita sobre o ranking completo.
    NaN scores (features com variância zero) são substituídos por 0.
    """
    selector = SelectKBest(score_func=score_func, k='all')
    selector.fit(X, y)
    scores = np.nan_to_num(selector.scores_, nan=0.0)
    return scores


def shorten_name(name: str) -> str:
    """Encurta nome de feature para exibição nos gráficos."""
    name = name.replace('bw__', '[BW] ').replace('doppler__', '[D] ')
    parts = name.replace('[BW] ', '').replace('[D] ', '').split('_')
    prefix = '[BW] ' if name.startswith('[BW]') else '[D] ' if name.startswith('[D]') else ''
    short = '_'.join(parts[-3:]) if len(parts) > 3 else '_'.join(parts)
    return prefix + short


def build_criteria(seed: int = 42) -> dict:
    """Retorna o dict de critérios de scoring (ANOVA F-Score e Mutual Information)."""
    return {
        'F-Score (ANOVA)':    lambda X, y: run_kbest(f_classif, X, y),
        'Mutual Information': lambda X, y: run_kbest(
            lambda X, y: mutual_info_classif(X, y, random_state=seed), X, y
        ),
    }
