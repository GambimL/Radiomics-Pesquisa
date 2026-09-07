"""
Funções para mesclar datasets radiômicos de diferentes modalidades (BW/Doppler).
"""

import pandas as pd


def add_modality_prefix(df: pd.DataFrame, prefix: str, meta_cols: list) -> pd.DataFrame:
    """Renomeia colunas de features com um prefixo de modalidade, preservando meta_cols."""
    rename_map = {
        col: f"{prefix}__{col}"
        for col in df.columns
        if col not in meta_cols
    }
    return df.rename(columns=rename_map)


def merge_modalities(bw: pd.DataFrame, doppler: pd.DataFrame, meta_cols: list) -> pd.DataFrame:
    """Mescla radiomics de BW e Doppler por image_id + split, prefixando as features de cada modalidade."""
    bw_prefixed = add_modality_prefix(bw, "bw", meta_cols)
    doppler_prefixed = add_modality_prefix(doppler, "doppler", meta_cols)
    doppler_features_only = doppler_prefixed.drop(
        columns=[c for c in meta_cols if c != "image_id" and c != "split"]
    )
    merged = bw_prefixed.merge(
        doppler_features_only,
        on=["image_id", "split"],
        how="inner",
    )
    return merged
