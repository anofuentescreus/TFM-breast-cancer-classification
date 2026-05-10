# Fase 2: Preprocessat i Augmentació d'Imatges
# Autora: Anna Nofuentes Creus


# PREREQUISIT: Executa primer FASE 1.R

# 0. CONFIGURACIÓ INICAL
BASE_PATH <- "/Users/annanofuentescreus/Documents/Màster en Ciència de Dades UOC/5 Semestre/TFM /DATASETS/BreakHis/BreakHis - Breast Cancer Histopathological Database/dataset_cancer_v1/dataset_cancer_v1"
PATH_BINARI <- file.path(BASE_PATH, "classificacao_binaria")
OUTPUT_DIR  <- "resultats/"
IMATGES_DIR <- "resultats/imatges_preprocessades/"

# Paràmetres de normalització d'imatges
# Definim 224x224 perquè és l'estàndard per a Transfer Learning
IMG_AMPLADA <- 224
IMG_ALCADA  <- 224

# Iterem sobre les 4 magnificacions disponibles al dataset BreakHis
MAGNIFICACIONS <- c("40X", "100X", "200X", "400X")

# Proporcions de divisió del dataset
# 70% TRAIN | 15% VAL | 15% TEST
PROPORCIO_TRAIN <- 0.70
PROPORCIO_VAL   <- 0.15
PROPORCIO_TEST  <- 0.15

# Fixem la llavor aleatòria per garantir la reproductibilitat de la divisió del dataset
SEED <- 42
set.seed(SEED)

# Creem les carpetes si no existeixen
if (!dir.exists(OUTPUT_DIR))  dir.create(OUTPUT_DIR,  recursive = TRUE)
if (!dir.exists(IMATGES_DIR)) dir.create(IMATGES_DIR, recursive = TRUE)

# 1. INSTAL·LACIÓ I CÀRREGA DE LLIBRERIES
# keras3 i tensorflow són les eines principals per al modelatge de xarxes neuronals
paquets <- c("tidyverse", "magick", "keras3", "tensorflow", "tfdatasets")

for (paquet in paquets) {
  if (!requireNamespace(paquet, quietly = TRUE)) {
    install.packages(paquet)
  }
}

library(tidyverse)    # Manipulació de dades i visualització
library(magick)       # Lectura i manipulació d'imatges
library(keras3)       # Definició i entrenament de xarxes neuronals
library(tensorflow)   # Backend de Deep Learning
library(tfdatasets)   # Necessari per gestionar pipelines de dades massives (ETL)

# 2. CARREGAR EL CATÀLEG BINARI
# Recuperem el catàleg creat en la FASE 1
cataleg_binari <- read_csv(file.path(OUTPUT_DIR, "cataleg_binari.csv"),
                           show_col_types = FALSE)

cat(sprintf("Catàleg carregat: %d imatges totals\n", nrow(cataleg_binari)))

# 3. DIVISIÓ TRAIN / VALIDATION / TEST
# Divisió estratificada per malignitat per assegurar proporcions similars a cada split
set.seed(SEED)

dividir_dataset <- function(df, prop_train, prop_val) {
  df %>%
    group_by(malignitat) %>%
    mutate(
      split = sample(
        c(rep("train", round(prop_train * n())),
          rep("val",   round(prop_val   * n())),
          rep("test",  n() - round(prop_train * n()) - round(prop_val * n())))
      )
    ) %>%
    ungroup()
}

# 4. FUNCIÓ DE PREPROCESSAT D'UNA IMATGE
# Llegim una imatge, la redimensionem a 224x224 i la normalitzem a [0, 1]
preprocessar_imatge <- function(path_imatge,
                                amplada = IMG_AMPLADA,
                                alcada  = IMG_ALCADA) {
  img <- image_read(path_imatge)
  img <- image_resize(img, paste0(amplada, "x", alcada, "!"))
  arr <- as.integer(image_data(img, channels = "rgb"))
  arr <- array(arr, dim = c(alcada, amplada, 3))
  arr <- arr / 255.0
  return(arr)
}

# 5. CREAR ESTRUCTURA DE DIRECTORIS PER A KERAS
# Keras requereix una jerarquia de carpetes (split/classe/imatge) per carregar dades automàticament
crear_estructura_keras <- function(df, split_name, base_dir, col_classe) {
  
  classes <- unique(df[[col_classe]])
  
  for (classe in classes) {
    dir.create(file.path(base_dir, split_name, classe),
               recursive = TRUE, showWarnings = FALSE)
  }
  
  df %>%
    filter(split == split_name) %>%
    rowwise() %>%
    mutate(dest = file.path(base_dir, split_name,
                            .data[[col_classe]], nom_fitxer)) %>%
    filter(!file.exists(dest)) %>%
    rowwise() %>%
    group_walk(~ file.copy(.x$path, .x$dest))
  
  n <- df %>% filter(split == split_name) %>% nrow()
  cat(sprintf("    %s: %d imatges copiades\n",
              split_name, n))
}

# Llistes per emmagatzemar els datasets de cada magnificació
datasets_per_magnificacio <- list()

# INICI DEL BUCLE PER MAGNIFICACIONS
for (MAGNIFICACIO_OBJECTIU in MAGNIFICACIONS) {
  
  cat(sprintf("\n=============================================\n"))
  cat(sprintf(" PROCESSANT MAGNIFICACIÓ: %s\n", MAGNIFICACIO_OBJECTIU))
  cat(sprintf("=============================================\n"))
  
  # Filtrem el catàleg per la magnificació actual
  cataleg_filtrat <- cataleg_binari %>% filter(magnificacio == MAGNIFICACIO_OBJECTIU)
  
  # Apliquem la divisió
  cataleg_binari_split <- dividir_dataset(cataleg_filtrat, PROPORCIO_TRAIN, PROPORCIO_VAL)
  
  # Estructura per classificació BINÀRIA per a la magnificació actual
  IMATGES_DIR_BINARI <- file.path(IMATGES_DIR, "binari", MAGNIFICACIO_OBJECTIU)
  
  crear_estructura_keras(cataleg_binari_split, "train", IMATGES_DIR_BINARI, "malignitat")
  crear_estructura_keras(cataleg_binari_split, "val", IMATGES_DIR_BINARI, "malignitat")
  crear_estructura_keras(cataleg_binari_split, "test", IMATGES_DIR_BINARI, "malignitat")

# 6. CREACIÓ DE DATASETS DE TENSORFLOW (PIPELINE)
# L'objectiu és carregar les imatges de forma eficient des del disc durant l'entrenament
# TRAIN: amb augmentació | VAL/TEST: sense augmentació (només normalitza)
BATCH_SIZE <- 16  # Reduït per CPU

train_ds_binari <- image_dataset_from_directory(
  directory  = file.path(IMATGES_DIR_BINARI, "train"),
  labels     = "inferred",
  label_mode = "binary",
  image_size = c(IMG_ALCADA, IMG_AMPLADA),
  batch_size = BATCH_SIZE,
  shuffle    = TRUE,
  seed       = SEED
)

val_ds_binari <- image_dataset_from_directory(
  directory  = file.path(IMATGES_DIR_BINARI, "val"),
  labels     = "inferred",
  label_mode = "binary",
  image_size = c(IMG_ALCADA, IMG_AMPLADA),
  batch_size = BATCH_SIZE,
  shuffle    = FALSE,
  seed       = SEED
)

test_ds_binari <- image_dataset_from_directory(
  directory  = file.path(IMATGES_DIR_BINARI, "test"),
  labels     = "inferred",
  label_mode = "binary",
  image_size = c(IMG_ALCADA, IMG_AMPLADA),
  batch_size = BATCH_SIZE,
  shuffle    = FALSE,
  seed       = SEED
)

# 7. AUGMENTACIÓ I NORMALITZACIÓ
# L'augmentació només s'aplica al set d'entrenament per simular variabilitat i evitar l'overfitting
augmentacio <- keras_model_sequential() %>%
  layer_rescaling(scale = 1/255) %>% # Normalització
  layer_random_flip("horizontal_and_vertical") %>% # Inversions aleatòries
  layer_random_rotation(factor = 0.08) %>% # Rotacions lleugeres
  layer_random_zoom(height_factor = 0.08) %>% # Zooms
  layer_random_contrast(factor = 0.1) # Variacions de contrast

# El set de validació/test només es normalitza (no s'augmenta)
normalitzacio <- keras_model_sequential() %>%
  layer_rescaling(scale = 1/255)

# Aplicació de les capes de preprocessat als datasets
train_ds_binari <- train_ds_binari %>%
  dataset_map(function(x, y) list(augmentacio(x, training = TRUE), y))

val_ds_binari <- val_ds_binari %>%
  dataset_map(function(x, y) list(normalitzacio(x), y))

test_ds_binari <- test_ds_binari %>%
  dataset_map(function(x, y) list(normalitzacio(x), y))

# Optimizació de la càrrega mitjançant pre-fetching a la memòria
train_ds_binari <- train_ds_binari %>% dataset_prefetch(buffer_size = -1)
val_ds_binari   <- val_ds_binari   %>% dataset_prefetch(buffer_size = -1)
test_ds_binari  <- test_ds_binari  %>% dataset_prefetch(buffer_size = -1)

# Emmagatzemem els datasets en una llista
datasets_per_magnificacio[[MAGNIFICACIO_OBJECTIU]] <- list(
  train = train_ds_binari,
  val = val_ds_binari,
  test = test_ds_binari,
  info = list(
    n_train = nrow(cataleg_binari_split %>% filter(split == "train")),
    n_val   = nrow(cataleg_binari_split %>% filter(split == "val")),
    n_test  = nrow(cataleg_binari_split %>% filter(split == "test"))
  )
)

# 8. GUARDAR INFORMACIÓ (Un fitxer per Magnificació)
info_datasets <- list(
  magnificacio   = MAGNIFICACIO_OBJECTIU,
  img_amplada    = IMG_AMPLADA,
  img_alcada     = IMG_ALCADA,
  batch_size     = BATCH_SIZE,
  classes_binari = c("benigne", "maligne"),
  n_train_binari = datasets_per_magnificacio[[MAGNIFICACIO_OBJECTIU]]$info$n_train,
  n_val_binari   = datasets_per_magnificacio[[MAGNIFICACIO_OBJECTIU]]$info$n_val,
  n_test_binari  = datasets_per_magnificacio[[MAGNIFICACIO_OBJECTIU]]$info$n_test
)

saveRDS(info_datasets, file.path(OUTPUT_DIR, paste0("info_datasets_", MAGNIFICACIO_OBJECTIU, ".rds")))
write_csv(cataleg_binari_split, file.path(OUTPUT_DIR, paste0("cataleg_binari_split_", MAGNIFICACIO_OBJECTIU, ".csv")))

# 9. NETEJA DE MEMÒRIA PER A LA SEGÜENT MAGNIFICACIÓ
# Eliminem els objectes pesats del workspace de R
rm(train_ds_binari, val_ds_binari, test_ds_binari, cataleg_binari_split, cataleg_filtrat)

# Netegem la sessió interna de Keras/TensorFlow per alliberar la memòria de vídeo (si n'hi ha) o RAM de backend
clear_session()

# Executem el recol·lector de brossa de R per netejar la RAM del sistema
gc()

}

cat("\n=============================================\n")
cat(" FASE 2 FINALITZADA PER A TOTES LES MAGNIFICACIONS \n")
cat("=============================================\n")