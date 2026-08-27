# Radiomics-Pesquisa

Pesquisa em radiômica para classificação de lesões a partir de features de ultrassom (modo B e Doppler), com benchmark de 23 classificadores de machine learning e seleção de atributos (k-best).

## Estrutura

- `data/` — datasets de radiômica extraídos (B-mode, Doppler, combinado)
- `modules/` — registro de classificadores (`models.py`) e funções de avaliação (`evaluation.py`)
- `experiments/` — trilhas experimentais (A e B), sweep de k-best, análise por tipo de lesão e resultados agregados
- `notebooks/` — extração de features, treino/validação, análise de resultados e geração de gráficos
- `TCC_MANUSCRITO/` — manuscrito original do TCC (LaTeX)
- `ERAMIA/` — template e reescrita do artigo para submissão à ERAMIA 2026
- `papers/` — referências bibliográficas
