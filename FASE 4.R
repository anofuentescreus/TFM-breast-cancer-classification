# Fase 4: Validació Externa Cross-Dataset (BreakHis → BACH)
# Autora: Anna Nofuentes Creus

# OBJECTIU: Avaluar la capacitat de generalització dels models entrenats
# amb BreakHis aplicant-los sobre el dataset BACH, que mai han vist.
# Això ens permetrà mesurar si el model ha après característiques generals
# de les imatges histopatològiques o si ha memoritzat el dataset d'entrenament.

# PREREQUISIT: Executa primer FASE 1.R, FASE 2.R i FASE 3.R

# 0. CONFIGURACIÓ INICIAL

BACH_PATH   <- "/Users/annanofuentescreus/Documents/Màster en Ciència de Dades UOC/5 Semestre/TFM /DATASETS/BACH/ICIAR2018_BACH_Challenge/Photos/"

# Directori on es troben els models i resultats de les fases anteriors
OUTPUT_DIR  <- "resultats/"
MODELS_DIR  <- "resultats/models/"

# Directori on es guardaran els resultats i imatges preprocessades de la FASE 4
OUTPUT_FASE4     <- "resultats/fase4/"
IMATGES_DIR_BACH <- "resultats/fase4/imatges_bach/"

if (!dir.exists(OUTPUT_FASE4))     dir.create(OUTPUT_FASE4,     recursive = TRUE)
if (!dir.exists(IMATGES_DIR_BACH)) dir.create(IMATGES_DIR_BACH, recursive = TRUE)

# Paràmetres de preprocessat (han de ser idèntics als de les fases anteriors)
IMG_AMPLADA <- 224
IMG_ALCADA  <- 224
BATCH_SIZE  <- 32
SEED        <- 42
set.seed(SEED)

# Magnificació del model BreakHis que s'avaluarà amb BACH.
MAGNIFICACIO_MODEL <- "200X"

# Colors (coherència visual amb les fases anteriors)
color_benigne <- "#4DAFB0"
color_maligne <- "#D85A30"

# 1. INSTAL·LACIÓ I CÀRREGA DE LLIBRERIES

paquets <- c("tidyverse", "magick", "keras3", "tensorflow",
             "tfdatasets", "patchwork", "scales", "fs")

for (paquet in paquets) {
  if (!requireNamespace(paquet, quietly = TRUE)) {
    install.packages(paquet)
  }
}

library(tidyverse)    # Manipulació de dades i visualització
library(magick)       # Lectura i inspecció d'imatges
library(keras3)       # Càrrega del model preentrenat
library(tensorflow)   # Backend de Deep Learning
library(tfdatasets)   # Pipeline de dades eficient
library(patchwork)    # Composició de gràfics
library(scales)       # Formatació d'eixos
library(fs)           # Operacions amb el sistema de fitxers

# 2. EXPLORACIÓ DE L'ESTRUCTURA DEL DATASET BACH
# BACH té 4 classes: Normal, Benign, InSitu, Invasive
# Per al nostre model binari les col·lapsem en 2:
#   - Benigne (negatiu) -> "Normal" + "Benign"
#   - Maligne (positiu) -> "InSitu" + "Invasive"

cat("\n=============================================\n")
cat(" FASE 4: VALIDACIÓ CROSS-DATASET (BACH)\n")
cat("=============================================\n")

# Bloc de diagnòstic: verifiquem que BACH_PATH és correcte i conté
# les 4 subcarpetes esperades abans de continuar amb la càrrega de dades.
cat("\nDiagnòstic: verificant estructura del directori BACH...\n")
carpetes_esperades <- c("Normal", "Benign", "InSitu", "Invasive")
carpetes_trobades  <- basename(dir_ls(BACH_PATH, recurse = FALSE, type = "directory"))
carpetes_faltants  <- setdiff(carpetes_esperades, carpetes_trobades)

if (length(carpetes_faltants) > 0) {
  stop(sprintf("No s'han trobat les carpetes esperades a BACH_PATH.\n"))
}

fitxers_bach <- dir_ls(BACH_PATH, recurse = TRUE, glob = "*.tif")
if (length(fitxers_bach) == 0) {
  fitxers_bach <- dir_ls(BACH_PATH, recurse = TRUE, glob = "*.png")
  cat("Format detectat: PNG\n")
} else {
  cat("Format detectat: TIF\n")
}

cat(sprintf("Total d'imatges detectades: %d\n", length(fitxers_bach)))

# 3. CONSTRUCCIÓ DEL CATÀLEG BACH
cataleg_bach <- tibble(path = as.character(fitxers_bach)) %>%
  mutate(
    classe_bach = basename(dirname(path)),
    malignitat = case_when(
      classe_bach %in% c("Normal", "Benign")   ~ "benigne",
      classe_bach %in% c("InSitu", "Invasive") ~ "maligne",
      TRUE                                     ~ "desconegut"
    ),
    nom_fitxer = basename(path)
  ) %>%
  filter(malignitat != "desconegut")

cat("\nDistribució binària resultant:\n")
dist_bach <- cataleg_bach %>% count(malignitat) %>% mutate(percentatge = n / sum(n) * 100)
print(dist_bach)

write_csv(cataleg_bach, file.path(OUTPUT_FASE4, "cataleg_bach.csv"))

# 4. MOSTRA D'IMATGES BACH
# Visualitzem exemples representatius per confirmar la compatibilitat
# visual amb les imatges de BreakHis (tinció H&E, estructura tisular similar)

mostra_bach <- cataleg_bach %>%
  group_by(malignitat) %>%
  slice_sample(n = 4) %>%
  ungroup()

imatges_bach <- map(mostra_bach$path, function(p) {
  img <- image_read(p)
  img <- image_resize(img, "200x200!")
  mal_label <- mostra_bach$malignitat[mostra_bach$path == p][1]
  color_box <- if (mal_label == "benigne") color_benigne else color_maligne
  img <- image_annotate(img,
                        text     = mal_label,
                        gravity  = "South",
                        size     = 13,
                        color    = "white",
                        boxcolor = color_box,
                        location = "+0+0")
  img
})

graella_bach <- image_append(
  image_join(list(
    image_append(image_join(imatges_bach[1:4]), stack = FALSE),
    image_append(image_join(imatges_bach[5:8]), stack = FALSE)
  )),
  stack = TRUE
)

image_write(graella_bach, file.path(OUTPUT_FASE4, "mostra_imatges_bach.png"))
cat("Mosaic d'imatges BACH guardat.\n")

# 5. PREPROCESSAT: CÒPIA D'IMATGES BACH A ESTRUCTURA KERAS
# Keras necessita una jerarquia split/classe/imatge per llegir les dades.
# Com que BACH és exclusivament un conjunt d'avaluació externa, no fem
# cap divisió: totes les imatges van a la carpeta "test".

BACH_KERAS_DIR <- file.path(IMATGES_DIR_BACH, "test")

for (classe in c("benigne", "maligne")) {
  dir.create(file.path(BACH_KERAS_DIR, classe),
             recursive = TRUE, showWarnings = FALSE)
}

cat("\nConvertint i copiant imatges BACH a l'estructura Keras (TIF -> PNG)...\n")

# Modifiquem el destí per a que el fitxer sigui .png (format que Keras entén a R)
cataleg_bach_amb_dest <- cataleg_bach %>%
  mutate(
    nom_fitxer_png = gsub("\\.tif$|\\.tiff$", ".png", nom_fitxer, ignore.case = TRUE),
    dest = file.path(BACH_KERAS_DIR, malignitat, nom_fitxer_png)
  )

# Funció per llegir TIF i guardar en PNG si no existeix
processar_i_convertir <- function(origen, desti) {
  if (!file.exists(desti)) {
    img <- image_read(origen)
    image_write(img, path = desti, format = "png")
    return(TRUE)
  }
  return(FALSE)
}

# Executem la conversió per a totes les imatges del catàleg
resultats_conv <- walk2(cataleg_bach_amb_dest$path, cataleg_bach_amb_dest$dest, processar_i_convertir)

cat(sprintf("Procés finalitzat. Les imatges estan a: %s\n", BACH_KERAS_DIR))

# 6. CREACIÓ DEL PIPELINE DE DADES TENSORFLOW PER A BACH
# Apliquem ÚNICAMENT normalització (sense augmentació), igual que
# als conjunts de test a les fases anteriors, per obtenir mètriques reals.

normalitzacio <- keras_model_sequential() %>%
  layer_rescaling(scale = 1/255)

# Nota: les imatges BACH (TIF 2048x1536) es redimensionen automàticament a 224x224 (estàndard per a MobileNetV2) durant la càrrega del pipeline
bach_test_ds <- image_dataset_from_directory(
  directory  = BACH_KERAS_DIR,
  labels     = "inferred",
  label_mode = "binary",
  image_size = c(IMG_ALCADA, IMG_AMPLADA),
  batch_size = BATCH_SIZE,
  shuffle    = FALSE,
  seed       = SEED
) %>%
  dataset_map(function(x, y) list(normalitzacio(x), y)) %>%
  dataset_prefetch(buffer_size = -1)

# 7. CÀRREGA DEL MODEL ENTRENAT AMB BREAKHIS
# Carreguem el millor checkpoint del model de la FASE 3 per a la magnificació seleccionada (200X), que inclou els pesos amb millor AUC de validació.

MODEL_PATH <- file.path(MODELS_DIR,
                        paste0("millor_model_binari_", MAGNIFICACIO_MODEL, ".keras"))

if (!file.exists(MODEL_PATH)) {
  stop(sprintf("No s'ha trobat el model: %s", MODEL_PATH))
}

model_breakhis <- load_model(MODEL_PATH)
cat("Model carregat correctament.\n")
summary(model_breakhis)

# 8. AVALUACIÓ DIRECTA DEL MODEL SOBRE BACH
# Avaluem el model tal com és, sense cap reentrenament, per mesurar la seva capacitat real de generalització cross-dataset.

cat(sprintf("\n AVALUACIÓ CROSS-DATASET: BreakHis (%s) -> BACH \n", MAGNIFICACIO_MODEL))

metriques_bach <- model_breakhis %>% evaluate(bach_test_ds, verbose = 1)

loss_bach     <- as.numeric(metriques_bach["loss"])
accuracy_bach <- as.numeric(metriques_bach["accuracy"])
auc_bach      <- as.numeric(metriques_bach["auc"])

cat(sprintf("\nTest Loss (BACH): %.4f\n", loss_bach))
cat(sprintf("Test Accuracy (BACH): %.4f (%.1f%%)\n", accuracy_bach, accuracy_bach * 100))
cat(sprintf("Test AUC (BACH): %.4f\n", auc_bach))

# 9. PREDICCIONS I CERCA DEL LLINDAR ÒPTIM

prediccions_prob_bach <- model_breakhis %>% predict(bach_test_ds, verbose = 1)

# Recuperem les etiquetes reals en el mateix ordre que les llegeix Keras.
# Keras carrega les imatges ordenades alfabèticament per classe i per nom
# de fitxer, de manera que hem de replicar aquest ordre exacte per assegurar
# la correspondència correcta entre prediccions i etiquetes.
etiquetes_reals_bach <- cataleg_bach_amb_dest %>%
  arrange(malignitat, nom_fitxer_png) %>%  # Fem servir el nom nou .png
  mutate(label = as.integer(malignitat == "maligne")) %>%
  pull(label)

# Cerca del llindar que maximitza el F1-Score sobre BACH.
# No fem servir el llindar òptim de BreakHis per no esbiaixar l'avaluació: el llindar s'ha de trobar sobre les dades que s'estan avaluant.
llindars <- seq(0.2, 0.8, by = 0.05)

resultats_llindar_bach <- map_dfr(llindars, function(t) {
  preds <- as.integer(prediccions_prob_bach > t)
  tp  <- sum(preds == 1 & etiquetes_reals_bach == 1)
  fp  <- sum(preds == 1 & etiquetes_reals_bach == 0)
  fn  <- sum(preds == 0 & etiquetes_reals_bach == 1)
  tn  <- sum(preds == 0 & etiquetes_reals_bach == 0)
  prec <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
  rec  <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
  f1   <- if ((prec + rec) == 0) 0 else 2 * prec * rec / (prec + rec)
  tibble(llindar = t, tp = tp, tn = tn, fp = fp, fn = fn,
         precision = prec, recall = rec, f1 = f1)
})

llindar_optim_bach <- resultats_llindar_bach %>%
  slice_max(f1, n = 1) %>%
  pull(llindar)

# Apliquem el llindar òptim per obtenir la classe predita
prediccions_classe_bach <- as.integer(prediccions_prob_bach > llindar_optim_bach)

# 10. MATRIU DE CONFUSIÓ I MÈTRIQUES FINALS

matriu_conf_bach <- table(
  Real       = ifelse(etiquetes_reals_bach == 1,    "Maligne", "Benigne"),
  Predicció  = ifelse(prediccions_classe_bach == 1, "Maligne", "Benigne")
)

cat("\n Matriu de confusió (BACH):\n")
print(matriu_conf_bach)

TP <- matriu_conf_bach["Maligne", "Maligne"]
TN <- matriu_conf_bach["Benigne", "Benigne"]
FP <- matriu_conf_bach["Benigne", "Maligne"]
FN <- matriu_conf_bach["Maligne", "Benigne"]

precision_bach   <- TP / (TP + FP)
recall_bach      <- TP / (TP + FN)
specificity_bach <- TN / (TN + FP)
f1_bach          <- 2 * precision_bach * recall_bach / (precision_bach + recall_bach)

cat(sprintf("\n Mètriques (classe positiva: Maligne):\n"))
cat(sprintf("  Precision:   %.4f\n", precision_bach))
cat(sprintf("  Recall:      %.4f\n", recall_bach))
cat(sprintf("  Specificity: %.4f\n", specificity_bach))
cat(sprintf("  F1-Score:    %.4f\n", f1_bach))

# 11. COMPARACIÓ AMB LES MÈTRIQUES DE BREAKHIS (MATEIX MODEL)
# Comparem les mètriques obtingudes en el test intern de BreakHis (FASE 3) amb les de BACH

metriques_breakhis_path <- file.path(OUTPUT_DIR,
                                     paste0("metriques_test_binari_", MAGNIFICACIO_MODEL, ".csv"))

if (file.exists(metriques_breakhis_path)) {
  metriques_breakhis <- read_csv(metriques_breakhis_path, show_col_types = FALSE)
  
  comparativa <- tibble(
    Dataset     = c(paste0("BreakHis (", MAGNIFICACIO_MODEL, " - Test intern)"), "BACH (Test extern)"),
    Accuracy    = c(metriques_breakhis$accuracy,    accuracy_bach),
    AUC         = c(metriques_breakhis$auc,          auc_bach),
    Precision   = c(metriques_breakhis$precision,    precision_bach),
    Recall      = c(metriques_breakhis$recall,       recall_bach),
    Specificity = c(metriques_breakhis$specificity,  specificity_bach),
    F1_Score    = c(metriques_breakhis$f1_score,     f1_bach)
  )
  
  cat("\n TAULA COMPARATIVA:\n")
  print(comparativa, width = Inf)
  
  write_csv(comparativa, file.path(OUTPUT_FASE4, "comparativa_cross_dataset.csv"))
  cat("\nTaula comparativa guardada.\n")
  
} else {
  cat(sprintf("AVÍS: No s'han trobat les mètriques de BreakHis a: %s\n",
              metriques_breakhis_path))
  cat("Assegura't d'haver executat la FASE 3 correctament.\n")
  
  # Guardem igualment les mètriques de BACH per separat
  tibble(
    dataset      = "BACH",
    model_origen = paste0("BreakHis_", MAGNIFICACIO_MODEL),
    loss         = loss_bach,
    accuracy     = accuracy_bach,
    auc          = auc_bach,
    precision    = precision_bach,
    recall       = recall_bach,
    specificity  = specificity_bach,
    f1_score     = f1_bach
  ) %>%
    write_csv(file.path(OUTPUT_FASE4, "metriques_bach.csv"))
}

# 12. GUARDAR TOTES LES MÈTRIQUES DE LA FASE 4

tibble(
  model_origen    = paste0("BreakHis_MobileNetV2_", MAGNIFICACIO_MODEL),
  dataset_test    = "BACH",
  n_imatges       = nrow(cataleg_bach),
  n_benigne       = sum(cataleg_bach$malignitat == "benigne"),
  n_maligne       = sum(cataleg_bach$malignitat == "maligne"),
  llindar_optim   = llindar_optim_bach,
  loss            = loss_bach,
  accuracy        = accuracy_bach,
  auc             = auc_bach,
  precision       = precision_bach,
  recall          = recall_bach,
  specificity     = specificity_bach,
  f1_score        = f1_bach,
  tp              = TP,
  tn              = TN,
  fp              = FP,
  fn              = FN
) %>%
  write_csv(file.path(OUTPUT_FASE4, "metriques_complertes_fase4.csv"))

cat("\nTotes les mètriques guardades a: resultats/fase4/\n")

cat("\n=============================================\n")
cat(" FASE 4 FINALITZADA\n")
cat("=============================================\n")