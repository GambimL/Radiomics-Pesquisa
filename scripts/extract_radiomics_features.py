#!/usr/bin/env python
"""
Extração de features radiômicas (PyRadiomics) para os splits train/test.

Consome diretamente a saída de `split_data.py --input-type folder
--copy-files`: uma pasta `--images-dir` contendo `train_ids.csv`/
`test_ids.csv` + os arquivos em `train/<classe>/` e `test/<classe>/`.
Extrai features com PyRadiomics para cada imagem e salva um CSV final com
colunas `image_id`, `label`, `target`, `split` + todas as features
radiômicas. `--modality` é usado apenas para nomear o CSV de saída — a
extração em si não depende da modalidade.

Exemplos:
    # 1. Split (uma vez por modalidade, apontando para as imagens já organizadas em classe_b/classe_m)
    python scripts/split_data.py --input-type folder --copy-files \
        --input data/raw/bw/CLASSES --output data/splits/bw

    # 2. Extração (consome a saída do split acima)
    python scripts/extract_radiomics_features.py \
        --images-dir data/splits/bw --modality bw \
        --output-dir data/images/datasets

    python scripts/extract_radiomics_features.py \
        --images-dir data/splits/doppler --modality doppler --no-wavelet \
        --output-dir data/images/datasets
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str((Path(__file__).parent / ".." / "modules").resolve()))

from radiomics_utils import build_extractor, extract_features_for_catalog, load_split_catalog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai features radiômicas a partir da saída de split_data.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-dir", type=Path, required=True,
        help="Pasta gerada por split_data.py --copy-files (contém train_ids.csv, test_ids.csv, train/, test/).",
    )
    parser.add_argument(
        "--modality", required=True,
        help="Rótulo usado para nomear o CSV de saída (radiomics_{modality}.csv). Não afeta a extração.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Pasta onde o CSV final é salvo.",
    )
    parser.add_argument("--bin-width", type=int, default=25, help="binWidth do PyRadiomics.")
    parser.add_argument("--normalize-scale", type=int, default=100, help="normalizeScale do PyRadiomics.")
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Desativa a normalização de intensidade do PyRadiomics.",
    )
    parser.add_argument(
        "--no-wavelet", action="store_true",
        help="Desativa a extração de features wavelet (mantém apenas 'Original').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_catalog = load_split_catalog(args.images_dir, "train")
    train_catalog["split"] = "trainval"  # split_data.py grava 'train'; scripts de treino esperam 'trainval'
    test_catalog = load_split_catalog(args.images_dir, "test")

    catalog = pd.concat([train_catalog, test_catalog], ignore_index=True)
    logger.info("Catálogo: %d imagens", len(catalog))
    logger.info("%s", catalog[["split", "label"]].value_counts().sort_index().to_string())

    extractor = build_extractor(
        bin_width=args.bin_width,
        normalize=not args.no_normalize,
        normalize_scale=args.normalize_scale,
        enable_wavelet=not args.no_wavelet,
    )
    features = extract_features_for_catalog(catalog, extractor)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"radiomics_{args.modality}.csv"
    features.to_csv(output_path, index=False)
    logger.info("Salvo em: %s (shape=%s)", output_path, features.shape)


if __name__ == "__main__":
    main()
