# Radiomics-Pesquisa

Pesquisa em radiômica para classificação de lesões (benigno vs. maligno) a partir de features de ultrassom (modo B e Doppler), com benchmark de classificadores de machine learning e seleção de atributos (k-best).

O pipeline é executado inteiramente via scripts de terminal em `scripts/` (não há mais notebooks no fluxo principal — os notebooks originais ficam em `notebooks/` apenas como referência histórica).

## Setup

```bash
python -m venv .venv
./.venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

A extração de features (`extract_radiomics_features.py`) depende do `pyradiomics`, que não tem wheel para Python 3.11 e precisa compilar uma extensão C — requer Microsoft C++ Build Tools instalado. Ver instruções em `requirements.txt`. As demais etapas do pipeline não dependem dele.

## Pipeline

Dentro da pasta scripts/ estão todos os scripts do pipeline do projeto. Cada script tem --help com todas as flags e exemplos.

Como usar cada script:

### Split Data

Split treino/teste estratificado, agnóstico a dataset. Suporta dois tipos de entrada via --input-type:

- folder: pasta com as imagens organizadas como "input/classe/arquivo"
- csv: arquivo CSV com as informações do dataset de imagens (o arquivo precisa ter a coluna das classes e dos paths das imagens)



Use --input para marcar o caminho para os dados de entrada e --output para os dados de saida. Para escolher o tipo de imput --imput_type
```bash
# pasta organizada por classe
python scripts/split_data.py --input-type folder \
    --input data/raw/DATASET/CLASSES \
    --output data/splits/dataset \
    --test-size 0.10 --seed 42 --copy-files

# CSV de features
python scripts/split_data.py --input-type csv \
    --input data/CSVs/radiomics.csv \
    --output data/CSVs/radiomics_split.csv \
    --id-col image_id --label-col target \
    --test-size 0.10 --seed 42
```

### Extract Radiomics Features

Extrai features radiômicas (PyRadiomics). Pasta adicionada à --images_dir precisa ter o formato: - 

- train_ids.csv/test_ids.csv + os arquivos train/classe/ e test/classe/
 
 Gera radiomics_{modality}.csv com colunas image_id, label, target, split (trainval/test) + todas as features. --modality só nomeia o CSV de saída 


--images-dir (obrigatório, saída do split),
--modality (obrigatório, rótulo do arquivo de saída), --output-dir (obrigatório), --bin-width (default 25), --normalize-scale (default 100), --no-normalize, --no-wavelet.

```bash
# Extração
python scripts/extract_radiomics_features.py \
    --images-dir data/splits/bw --modality bw \
    --output-dir data/images/datasets

python scripts/extract_radiomics_features.py \
    --images-dir data/splits/doppler --modality doppler --no-wavelet \
    --output-dir data/images/datasets
```

### Merge Radiomics Modalities

Combina o csv de features radiomicas das imagens bw e doppler, por image_id + split. Features de BW recebem prefixo bw__ e de Doppler doppler__, preservando as colunas de metadados sem duplicação.

Principais flags: --input-bw, --input-doppler, --output, --meta-cols (default image_id label target split).

```bash
python scripts/merge_radiomics_modalities.py \
    --input-bw data/images/datasets/radiomics_bw.csv \
    --input-doppler data/images/datasets/radiomics_doppler.csv \
    --output data/images/datasets/radiomics_bw_doppler.csv
```

### Train Kfold

Treino + seleção de modelo via k-fold estratificado. Para cada combinação (K de features × modelo × dataset), roda k-fold, agrega métricas por fold e seleciona o(s) top-N modelo(s) por AUC médio em cada dataset. Passar múltiplos valores em --k-values roda uma varredura (k-sweep), elegendo um vencedor por K.

Principais flags: --feature-selection {kbest,none} (default kbest), --k-values (default 20, aceita all e múltiplos valores), --models, --datasets (default: todos), --output-dir (obrigatório), --n-folds (default 5), --top-n (default 1).

Saídas em --output-dir (sufixo _k{K} quando mais de um K é passado): task3_fold_metrics.csv, task4_aggregated_metrics.csv, task5_top_models.csv, task5_auc_barh.png, task5_metrics_heatmap.png, task5_auc_boxplot.png.

```bash
# SelectKBest com K=20
python scripts/train_kfold.py --feature-selection kbest --k-values 20 \
    --output-dir experiments/teste_2

# sem seleção de features (todas as features)
python scripts/train_kfold.py --feature-selection none \
    --output-dir experiments/teste_1

# k-sweep
python scripts/train_kfold.py --feature-selection kbest \
    --k-values 5 10 15 20 30 50 100 all \
    --output-dir experiments/trilha_a_k_sweep

# restringir modelos e datasets
python scripts/train_kfold.py --feature-selection kbest --k-values 20 \
    --models "Linear SVM" "Weighted KNN" --datasets BW Doppler \
    --output-dir experiments/subset
```

### Train Final

Lê o task5_top_models.csv (ou variante _k{K}) gerado por train_kfold.py para saber, por dataset, o modelo vencedor (assume rank==1). Para cada dataset: separa --val-size do trainval para calibrar o threshold de Youden, retreina o pipeline vencedor no trainval completo e avalia no val e no test set, nos thresholds 0.5 e Youden.

Principais flags: --feature-selection {kbest,none} (deve bater com o usado no train_kfold.py), --top-models-csv (obrigatório), --output-dir (obrigatório), --val-size (default 0.20).

Saídas em --output-dir: test_results.csv, test_auc_barh.png, test_sens_spec_comparison.png, test_predicted_probs.png, test_confusion_matrices.png, test_roc_curves.png, roc_data.pkl (dict por dataset com fpr/tpr/auc/thresholds, para comparações futuras).

```bash
python scripts/train_final.py --feature-selection kbest \
    --top-models-csv experiments/trilha_a/top_models.csv \
    --output-dir experiments/trilha_a_final

python scripts/train_final.py --feature-selection none \
    --top-models-csv experiments/trilha_b/top_models.csv \
    --output-dir experiments/trilha_b_final
```

### Analyze Kbest Features

Análise exploratória de seleção de features, independente do pipeline de treino. Para cada dataset radiômico (BW, Doppler, BW+Doppler), calcula os scores de todas as features via SelectKBest(k='all') sob dois critérios (F-Score/ANOVA e Mutual Information), usando apenas o split trainval.

Principais flags: --data-dir (default data/images/datasets), --output-dir (obrigatório), --datasets (default: todos), --top-consensus, --top-k-consensus, --top-overlap.

Saídas em --output-dir: task7a_kbest_rankings.csv, task7a_kbest_all_{dataset}.png, task7a_kbest_consensus.png, task7a_kbest_overlap.png, task7a_kbest_score_curves.png.

```bash
python scripts/analyze_kbest_features.py --output-dir experiments/kbest_analysis
```



