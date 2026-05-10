# Classificació de malignitat de tumors de mama mitjançant imatges histopatològiques en R

**Autora:** Anna Nofuentes Creus  
**Màster en Ciència de Dades — Universitat Oberta de Catalunya (UOC)**  
**Àrea:** Machine Learning and Computer Vision in Healthcare and Medical Application  
**Data:** Maig 2026

---

## Descripció

Aquest repositori conté el codi R del Treball Final de Màster (TFM), que desenvolupa un model de **classificació binària (benigne vs maligne)** de tumors de mama a partir d'imatges histopatològiques digitals, basat en **Transfer Learning amb MobileNetV2**.

El projecte s'estructura en 4 fases seqüencials, cadascuna implementada en un script R independent.

---

## Datasets utilitzats

- **BreakHis** (dataset principal d'entrenament): [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)  
- **BACH** (dataset de validació externa): [Zenodo](https://zenodo.org/record/3632035)

> Els datasets **no s'inclouen** en aquest repositori. Cal descarregar-los manualment i actualitzar les rutes `BASE_PATH` i `BACH_PATH` a cada script.

---

## Estructura del projecte

```
├── FASE_1.R   # Càrrega i exploració del dataset (EDA)
├── FASE_2.R   # Preprocessament, divisió i augmentació d'imatges
├── FASE_3.R   # Entrenament del model CNN (MobileNetV2) + avaluació
├── FASE_4.R   # Validació externa cross-dataset (BreakHis → BACH)
└── resultats/ # Carpeta generada automàticament amb outputs
```

---

## Requisits

### R i paquets necessaris

- R >= 4.2
- `tidyverse`, `magick`, `fs`, `patchwork`, `scales`
- `keras3`, `tensorflow`, `tfdatasets`
- `reticulate`

Els paquets s'instal·len automàticament si no estan disponibles a l'inici de cada script.

### Python (backend de TensorFlow)

```r
reticulate::install_miniconda()
keras3::install_keras(backend = "tensorflow")
```

---

## Com executar el codi

Els scripts s'han d'executar **en ordre**, ja que cada fase depèn dels outputs de l'anterior:

```
FASE_1.R → FASE_2.R → FASE_3.R → FASE_4.R
```

Abans d'executar, actualitza les rutes dels datasets als scripts:

- **FASE_1.R i FASE_2.R:** modifica `BASE_PATH` amb la ruta local del dataset BreakHis
- **FASE_4.R:** modifica `BACH_PATH` amb la ruta local del dataset BACH

---

## Resultats principals

| Magnificació | Accuracy | AUC    | F1-Score | Recall |
|:------------:|:--------:|:------:|:--------:|:------:|
| 40X          | 83.56%   | 0.9140 | 0.8972   | 0.9366 |
| 100X         | 76.53%   | 0.9186 | 0.8864   | 0.9070 |
| 200X         | 80.86%   | 0.9347 | 0.8995   | 0.8995 |
| 400X         | 83.88%   | 0.9202 | 0.9129   | 0.9351 |

**Validació externa sobre BACH (model 200X):** AUC = 0.713, F1 = 0.652

---

## Llicència

Aquest projecte és de caràcter acadèmic. El codi es distribueix lliurement per a fins educatius i de recerca.
