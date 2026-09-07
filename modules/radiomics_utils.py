import logging
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from radiomics import featureextractor
from tqdm.auto import tqdm


def load_split_catalog(images_dir: Path, split_name: str) -> pd.DataFrame:
    """Carrega o catálogo de um split (train/test) gerado por split_data.py.

    Espera a estrutura produzida por `split_data.py --input-type folder
    --copy-files`: `images_dir/{split_name}_ids.csv` (colunas filename,
    class_name, target) + arquivos em `images_dir/{split_name}/<class_name>/`.
    """
    ids_csv_path = images_dir / f"{split_name}_ids.csv"
    if not ids_csv_path.is_file():
        raise FileNotFoundError(
            f"{ids_csv_path} não encontrado. Rode split_data.py --input-type folder "
            f"--copy-files apontando --output para {images_dir} antes de extrair features."
        )
    ids_csv = pd.read_csv(ids_csv_path)

    split_dir = images_dir / split_name
    if not split_dir.is_dir():
        raise NotADirectoryError(
            f"{split_dir} não encontrado. Rode split_data.py com --copy-files para "
            f"gerar os arquivos de imagem em {split_dir}/<classe>/."
        )

    rows = []
    for _, row in ids_csv.iterrows():
        image_path = split_dir / row["class_name"] / row["filename"]
        if not image_path.exists():
            continue
        rows.append({
            "image_id": Path(row["filename"]).stem,
            "label": row["class_name"],
            "target": int(row["target"]),
            "split": split_name,
            "image_path": image_path.resolve(),
        })
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def build_sitk_image_and_mask(image_array: np.ndarray):
    image_3d = image_array[np.newaxis, :, :]
    mask_3d = np.zeros_like(image_3d, dtype=np.uint8)
    if image_3d.shape[1] > 2 and image_3d.shape[2] > 2:
        mask_3d[:, 1:-1, 1:-1] = 1
    else:
        mask_3d[:, :, :] = 1
        mask_3d[:, 0, :] = 0
        mask_3d[:, -1, :] = 0
        mask_3d[:, :, 0] = 0
        mask_3d[:, :, -1] = 0
    sitk_image = sitk.GetImageFromArray(image_3d)
    sitk_mask = sitk.GetImageFromArray(mask_3d)
    sitk_image.SetSpacing((1.0, 1.0, 1.0))
    sitk_mask.SetSpacing((1.0, 1.0, 1.0))
    return sitk_image, sitk_mask


def build_extractor(
    bin_width: int = 25,
    normalize: bool = True,
    normalize_scale: int = 100,
    enable_wavelet: bool = True,
) -> featureextractor.RadiomicsFeatureExtractor:
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=bin_width,
        normalize=normalize,
        normalizeScale=normalize_scale,
        force2D=True,
        force2Ddimension=0,
        minimumROIDimensions=2,
        label=1,
    )
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    if enable_wavelet:
        extractor.enableImageTypeByName("Wavelet")
    extractor.disableAllFeatures()
    for feature_class in ["firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]:
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def extract_features_for_catalog(
    catalog: pd.DataFrame,
    extractor: featureextractor.RadiomicsFeatureExtractor,
) -> pd.DataFrame:
    """Extrai features radiômicas para cada imagem no catálogo."""
    rows = []
    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Extraindo"):
        image_array = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.float32)
        sitk_image, sitk_mask = build_sitk_image_and_mask(image_array)
        raw = extractor.execute(sitk_image, sitk_mask)
        features = {k: v for k, v in raw.items() if not k.startswith("diagnostics_")}
        rows.append({
            "image_id": row["image_id"],
            "label": row["label"],
            "target": row["target"],
            "split": row["split"],
            **features,
        })
    return pd.DataFrame(rows)
