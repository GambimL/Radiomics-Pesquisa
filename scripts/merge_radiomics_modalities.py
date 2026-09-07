#!/usr/bin/env python
"""
Merge de features radiômicas BW + Doppler.

Combina os CSVs `radiomics_bw.csv` e `radiomics_doppler.csv` em um único
CSV, por `image_id` + `split`. Features de BW recebem prefixo `bw__` e
features de Doppler recebem prefixo `doppler__`, preservando as colunas de
metadados (--meta-cols) sem duplicação.

Exemplo:
    python scripts/merge_radiomics_modalities.py \
        --input-bw data/images/datasets/radiomics_bw.csv \
        --input-doppler data/images/datasets/radiomics_doppler.csv \
        --output data/images/datasets/radiomics_bw_doppler.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).parent / ".." / "modules").resolve()))

from data_merge import merge_modalities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mescla os CSVs de features radiômicas BW e Doppler por image_id + split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-bw", type=Path, required=True, help="CSV de features BW (ex.: radiomics_bw.csv).")
    parser.add_argument("--input-doppler", type=Path, required=True, help="CSV de features Doppler (ex.: radiomics_doppler.csv).")
    parser.add_argument("--output", type=Path, required=True, help="Caminho do CSV final mesclado.")
    parser.add_argument(
        "--meta-cols", nargs="+", default=["image_id", "label", "target", "split"],
        help="Colunas de metadados preservadas sem prefixo/duplicação.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_bw.is_file():
        raise FileNotFoundError(f"--input-bw não encontrado: {args.input_bw}")
    if not args.input_doppler.is_file():
        raise FileNotFoundError(f"--input-doppler não encontrado: {args.input_doppler}")

    bw = pd.read_csv(args.input_bw)
    doppler = pd.read_csv(args.input_doppler)
    logger.info("BW:      %s", bw.shape)
    logger.info("Doppler: %s", doppler.shape)

    merged = merge_modalities(bw, doppler, args.meta_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    logger.info("Salvo em: %s (shape=%s)", args.output, merged.shape)

    dist_cols = [c for c in ["split", "label", "target"] if c in merged.columns]
    logger.info("Distribuição por split e label:\n%s", merged[dist_cols].value_counts().sort_index().to_string())
    logger.info("Colunas BW (amostra):      %s", [c for c in merged.columns if c.startswith("bw__")][:4])
    logger.info("Colunas Doppler (amostra): %s", [c for c in merged.columns if c.startswith("doppler__")][:4])


if __name__ == "__main__":
    main()
