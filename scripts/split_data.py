#!/usr/bin/env python
"""
Split treino/teste agnóstico a dataset.

Suporta dois tipos de entrada (--input-type):

  folder  Pasta organizada como `input/<classe>/<arquivo>`. Faz split
          estratificado por classe e grava `train_ids.csv` / `test_ids.csv`
          em --output (colunas: filename, class_name, target). Com
          --copy-files, também copia os arquivos para
          output/train/<classe>/ e output/test/<classe>/.

  csv     Arquivo CSV (ex.: features de radiômica). Faz split estratificado
          pela coluna --label-col e grava em --output uma cópia do CSV
          original com uma coluna `split` ('train'/'test') adicionada,
          preservando todas as demais colunas.

Exemplos:
    python scripts/split_data.py --input-type folder \
        --input data/raw/DATASET-ISA/CLASSES \
        --output data/splits/dataset_isa \
        --test-size 0.10 --seed 42 --copy-files

    python scripts/split_data.py --input-type csv \
        --input data/CSVs/radiomics_bw.csv \
        --output data/CSVs/radiomics_bw_split.csv \
        --id-col image_id --label-col target \
        --test-size 0.10 --seed 42
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split treino/teste estratificado, agnóstico a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-type", choices=["folder", "csv"], required=True,
        help="Tipo da entrada: 'folder' (input/<classe>/<arquivo>) ou 'csv' (tabela de features).",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Pasta de classes (--input-type folder) ou caminho do CSV (--input-type csv).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Pasta de saída (modo folder) ou caminho do CSV de saída (modo csv).",
    )
    parser.add_argument("--test-size", type=float, default=0.10, help="Fração para o conjunto de teste.")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade do split.")
    parser.add_argument(
        "--copy-files", action="store_true",
        help="[modo folder] Além dos CSVs, copia fisicamente os arquivos para output/train|test/<classe>/.",
    )
    parser.add_argument(
        "--id-col", default="image_id",
        help="[modo csv] Nome da coluna identificadora (usada apenas para logging).",
    )
    parser.add_argument(
        "--label-col", default="target",
        help="[modo csv] Nome da coluna de rótulo/classe usada para o split estratificado.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Modo folder
# --------------------------------------------------------------------------

def load_paths_and_classes(input_dir: Path) -> pd.DataFrame:
    """Percorre as subpastas de classe e retorna um DataFrame (path, filename, class_name)."""
    rows = []
    for class_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file():
                rows.append({
                    "path": file_path,
                    "filename": file_path.name,
                    "class_name": class_dir.name,
                })
    if not rows:
        raise ValueError(f"Nenhum arquivo encontrado em subpastas de classe dentro de {input_dir}")
    return pd.DataFrame(rows)


def split_folder_dataframe(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Faz o split estratificado por class_name e retorna (df_train, df_test)."""
    classes = sorted(df["class_name"].unique())
    class_to_target = {name: i for i, name in enumerate(classes)}
    targets = df["class_name"].map(class_to_target).to_numpy()

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=targets,
    )

    df = df.assign(target=targets)
    df_train = df.iloc[idx_train].sort_values("filename").reset_index(drop=True)
    df_test = df.iloc[idx_test].sort_values("filename").reset_index(drop=True)
    return df_train, df_test


def copy_split_files(df: pd.DataFrame, split_name: str, output_dir: Path) -> None:
    """Copia os arquivos do split para output_dir/<split_name>/<classe>/."""
    for class_name, group in df.groupby("class_name"):
        dest_dir = output_dir / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in group["path"]:
            shutil.copy2(path, dest_dir / path.name)
    logger.info("Arquivos de %s copiados para %s", split_name, output_dir / split_name)


def log_class_balance(name: str, df: pd.DataFrame, label_col: str) -> None:
    counts = df[label_col].value_counts().to_dict()
    logger.info("%-8s: %d amostras — %s", name, len(df), counts)


def run_folder_split(args: argparse.Namespace) -> None:
    input_dir: Path = args.input
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input não encontrado ou não é uma pasta: {input_dir}")

    df = load_paths_and_classes(input_dir)
    log_class_balance("Total", df, "class_name")

    df_train, df_test = split_folder_dataframe(df, test_size=args.test_size, seed=args.seed)
    log_class_balance("Train", df_train, "class_name")
    log_class_balance("Test", df_test, "class_name")

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["filename", "class_name", "target"]
    df_train[cols].to_csv(output_dir / "train_ids.csv", index=False)
    df_test[cols].to_csv(output_dir / "test_ids.csv", index=False)
    logger.info("CSVs salvos em %s (train_ids.csv, test_ids.csv)", output_dir)

    if args.copy_files:
        copy_split_files(df_train, "train", output_dir)
        copy_split_files(df_test, "test", output_dir)


# --------------------------------------------------------------------------
# Modo csv
# --------------------------------------------------------------------------

def run_csv_split(args: argparse.Namespace) -> None:
    input_path: Path = args.input
    if not input_path.is_file():
        raise FileNotFoundError(f"--input não encontrado: {input_path}")

    df = pd.read_csv(input_path)
    for col in (args.id_col, args.label_col):
        if col not in df.columns:
            raise KeyError(
                f"Coluna '{col}' não encontrada em {input_path}. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

    log_class_balance("Total", df, args.label_col)

    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=df[args.label_col].to_numpy(),
    )

    df = df.copy()
    df["split"] = ""
    df.iloc[idx_train, df.columns.get_loc("split")] = "train"
    df.iloc[idx_test, df.columns.get_loc("split")] = "test"

    log_class_balance("Train", df[df["split"] == "train"], args.label_col)
    log_class_balance("Test", df[df["split"] == "test"], args.label_col)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("CSV com coluna 'split' salvo em %s", output_path)


def main() -> None:
    args = parse_args()
    if args.input_type == "folder":
        run_folder_split(args)
    else:
        run_csv_split(args)


if __name__ == "__main__":
    main()
