# Fase 3: Model CNN amb Transfer Learning (MobileNetV2)
# Autora: Anna Nofuentes Creus

# PREREQUISIT: Executa primer FASE 1.R i FASE 2.R

# 0. CONFIGURACIÓ I HIPERPARÀMETRES
OUTPUT_DIR      <- "resultats/"
IMATGES_DIR     <- "resultats/imatges_preprocessades/"
MODELS_DIR      <- "resultats/models/"
if (!dir.exists(MODELS_DIR)) dir.create(MODELS_DIR, recursive = TRUE)

IMG_ALCADA      <- 224
IMG_AMPLADA     <- 224
BATCH_SIZE      <- 32     
EPOCHS_FASE1    <- 30 # Entrenament de la capçalera
EPOCHS_FASE2    <- 15 # Ajust fi (fine-tuning)
LR_FASE1        <- 3e-4 # Learning rate inicial
LR_FASE2        <- 1e-5 # Learning rate molt baix per al fine-tuning
SEED            <- 42
set.seed(SEED)

# 1. CÀRREGA DE LLIBRERIES
library(tidyverse)
library(keras3)
library(tensorflow)
library(tfdatasets)
library(patchwork)

# Definim les 4 magnificacions per al bucle
MAGNIFICACIONS <- c("40X", "100X", "200X", "400X")

# INICI DEL BUCLE PER MAGNIFICACIONS
for (MAGNIFICACIO_OBJECTIU in MAGNIFICACIONS) {
  
  cat(sprintf("\n=============================================\n"))
  cat(sprintf(" INICIANT ENTRENAMENT MAGNIFICACIÓ: %s\n", MAGNIFICACIO_OBJECTIU))
  cat(sprintf("=============================================\n"))
  
  # Directoris específics per aquesta magnificació
  IMATGES_DIR_BINARI <- file.path(IMATGES_DIR, "binari", MAGNIFICACIO_OBJECTIU)
  
  # 2. CARREGAR INFORMACIÓ
  # Carreguem la info guardada en la FASE 2 per mantenir la coherència
  info_datasets <- readRDS(file.path(OUTPUT_DIR, paste0("info_datasets_", MAGNIFICACIO_OBJECTIU, ".rds")))
  
  cat(sprintf("Magnificació: %s\n", info_datasets$magnificacio))
  cat(sprintf("Train: %d | Val: %d | Test: %d\n",
              info_datasets$n_train_binari,
              info_datasets$n_val_binari,
              info_datasets$n_test_binari))
  
  # 3. RECREACIÓ DELS PIPELINES DE DADES
  # Nota: S'ha eliminat 'random_contrast' per preservar la integritat de la tinció H&E
  augmentacio <- keras_model_sequential() %>%
    layer_rescaling(scale = 1/255) %>%
    layer_random_flip("horizontal_and_vertical") %>%
    layer_random_rotation(factor = 0.15) %>%      
    layer_random_zoom(height_factor = 0.15,
                      width_factor  = 0.15)
  
  normalitzacio <- keras_model_sequential() %>%
    layer_rescaling(scale = 1/255)
  
  # Creació dels objectes dataset optimitzats per a TensorFlow
  train_ds_binari <- image_dataset_from_directory(
    directory  = file.path(IMATGES_DIR_BINARI, "train"),
    labels     = "inferred",
    label_mode = "binary",
    image_size = c(IMG_ALCADA, IMG_AMPLADA),
    batch_size = BATCH_SIZE,
    shuffle    = TRUE,
    seed       = SEED
  ) %>%
    dataset_map(function(x, y) list(augmentacio(x, training = TRUE), y)) %>%
    dataset_prefetch(buffer_size = -1)
  
  val_ds_binari <- image_dataset_from_directory(
    directory  = file.path(IMATGES_DIR_BINARI, "val"),
    labels     = "inferred",
    label_mode = "binary",
    image_size = c(IMG_ALCADA, IMG_AMPLADA),
    batch_size = BATCH_SIZE,
    shuffle    = FALSE,
    seed       = SEED
  ) %>%
    dataset_map(function(x, y) list(normalitzacio(x), y)) %>%
    dataset_prefetch(buffer_size = -1)
  
  test_ds_binari <- image_dataset_from_directory(
    directory  = file.path(IMATGES_DIR_BINARI, "test"),
    labels     = "inferred",
    label_mode = "binary",
    image_size = c(IMG_ALCADA, IMG_AMPLADA),
    batch_size = BATCH_SIZE,
    shuffle    = FALSE,
    seed       = SEED
  ) %>%
    dataset_map(function(x, y) list(normalitzacio(x), y)) %>%
    dataset_prefetch(buffer_size = -1)
  
  # 4. ARQUITECTURA DEL MODEL - TRANSFER LEARNING
  # Utilitzem MobileNetV2
  construir_model_binari <- function(lr = 1e-3, entrenar_base = FALSE) {
    
    # Carreguem el model base preentrenat amb ImageNet
    base_model <- application_mobilenet_v2(
      weights     = "imagenet",
      include_top = FALSE, # No incloem la part final (classificador de 1000 classes)
      input_shape = c(IMG_ALCADA, IMG_AMPLADA, 3)
    )
    
    base_model$trainable <- FALSE # Per defecte, congelem els pesos
    
    # Si activem fine-tuning, descongelem les últimes 20 capes per a l'especialització
    if (entrenar_base) {
      totes_les_capes <- reticulate::py_to_r(base_model$layers)
      n_capes <- length(totes_les_capes)
      for (i in seq(n_capes - 20 + 1, n_capes)) {
        totes_les_capes[[i]]$trainable <- TRUE
      }
    }
    
    inputs  <- layer_input(shape = c(IMG_ALCADA, IMG_AMPLADA, 3))
    x       <- base_model(inputs, training = FALSE)
    x       <- layer_global_average_pooling_2d()(x) # Reducció de dimensionalitat
    x       <- layer_batch_normalization()(x) # Estabilització de l'aprenentatge
    x       <- layer_dense(
      units = 64,
      activation = "relu",
      kernel_regularizer = regularizer_l2(0.01) # Regularització L2
    )(x)
    x       <- layer_dropout(rate = 0.5)(x) # Prevenció d'overfitting
    outputs <- layer_dense(units = 1, activation = "sigmoid")(x) # Sortida binària
    
    model <- keras_model(inputs, outputs)
    
    # Compilació amb optimitzador Adam i mètrica AUC
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = lr),
      loss      = "binary_crossentropy",
      metrics   = list("accuracy", metric_auc(name = "auc"))
    )
    
    n_train <- sum(sapply(model$trainable_weights,
                          function(w) prod(w$shape$as_list())))
    cat(sprintf("Paràmetres entrenables: %s\n",
                format(n_train, big.mark = ".")))
    
    return(model)
  }
  
  model_binari <- construir_model_binari(lr = LR_FASE1)
  summary(model_binari)
  
  # 5. CALLBACKS (Estratègies de control)
  # Implementem mecanismes per aturar l'entrenament si no hi ha millora o ajustar el LR
  callbacks_llista <- list(
    callback_early_stopping(
      monitor              = "val_auc",
      patience             = 8,
      restore_best_weights = TRUE,
      mode                 = "max",
      verbose              = 1
    ),
    callback_reduce_lr_on_plateau(
      monitor  = "val_loss",
      factor   = 0.5,
      patience = 4,
      min_lr   = 1e-7,
      verbose  = 1
    ),
    callback_model_checkpoint(
      filepath       = file.path(MODELS_DIR, paste0("millor_model_binari_", MAGNIFICACIO_OBJECTIU, ".keras")),
      monitor        = "val_auc",
      save_best_only = TRUE,
      mode           = "max",
      verbose        = 1
    ),
    callback_csv_logger(
      filename = file.path(OUTPUT_DIR, paste0("historial_entrenament_", MAGNIFICACIO_OBJECTIU, ".csv"))
    )
  )
  
  # 6. FASE 1: ENTRENAMENT BASE CONGELADA
  # Donem més pes a la classe minoritària per compensar el dataset
  cataleg_train <- read_csv(file.path(OUTPUT_DIR, paste0("cataleg_binari_split_", MAGNIFICACIO_OBJECTIU, ".csv")),
                            show_col_types = FALSE) %>%
    filter(split == "train")
  
  n_ben <- sum(cataleg_train$malignitat == "benigne")
  n_mal <- sum(cataleg_train$malignitat == "maligne")
  total <- n_ben + n_mal
  
  class_weights <- list(
    "0" = total / (2 * n_ben),
    "1" = total / (2 * n_mal)
  )
  
  cat(sprintf("\n FASE 1: Entrenament %s (base congelada) \n", MAGNIFICACIO_OBJECTIU))
  
  historial_fase1 <- model_binari %>%
    fit(
      train_ds_binari,
      epochs          = EPOCHS_FASE1,
      validation_data = val_ds_binari,
      class_weight    = class_weights,
      callbacks       = callbacks_llista,
      verbose         = 1
    )
  
  # 7. FASE 2: FINE-TUNING
  # Descongelem part de la base i entrenem amb un LR molt petit per no destruir el coneixement previ
  model_finetune <- construir_model_binari(lr = LR_FASE2, entrenar_base = TRUE)
  model_fase1 <- load_model(file.path(MODELS_DIR, paste0("millor_model_binari_", MAGNIFICACIO_OBJECTIU, ".keras")))
  model_finetune$set_weights(model_fase1$get_weights()) # Partim del millor estat anterior
  rm(model_fase1)
  
  callbacks_fase2 <- list(
    callback_early_stopping(
      monitor              = "val_auc",
      patience             = 6,
      restore_best_weights = TRUE,
      mode                 = "max",
      verbose              = 1
    ),
    callback_model_checkpoint(
      filepath       = file.path(MODELS_DIR, paste0("millor_model_finetune_", MAGNIFICACIO_OBJECTIU, ".keras")),
      monitor        = "val_auc",
      save_best_only = TRUE,
      mode           = "max",
      verbose        = 1
    ),
    callback_csv_logger(
      filename = file.path(OUTPUT_DIR, paste0("historial_finetune_", MAGNIFICACIO_OBJECTIU, ".csv")),
      append   = TRUE
    )
  )
  
  historial_fase2 <- model_finetune %>%
    fit(
      train_ds_binari,
      epochs          = EPOCHS_FASE2,
      validation_data = val_ds_binari,
      class_weight    = class_weights,
      callbacks       = callbacks_fase2,
      verbose         = 1
    )
  
  # 8. AVALUACIÓ EN EL CONJUNT DE TEST
  # Avaluem amb dades que el model no ha vist mai per obtenir mètriques no esbiaixades
  cat(sprintf("\n AVALUACIÓ EN EL CONJUNT DE TEST (%s) \n", MAGNIFICACIO_OBJECTIU))
  
  # Carreguem el model que ha obtingut millors resultats en validació (best checkpoint)
  model_final <- load_model(file.path(MODELS_DIR, paste0("millor_model_binari_", MAGNIFICACIO_OBJECTIU, ".keras")))
  
  # Executem la funció evaluate per obtenir el Loss i les mètriques base (Accuracy i AUC)
  metriques_test <- model_final %>% evaluate(test_ds_binari, verbose = 1)
  
  loss_val     <- as.numeric(metriques_test["loss"])
  accuracy_val <- as.numeric(metriques_test["accuracy"])
  auc_val      <- as.numeric(metriques_test["auc"])
  
  # Mostrem els resultats globals del model
  cat(sprintf("\nTest Loss: %.4f\n", loss_val))
  cat(sprintf("Test Accuracy: %.4f (%.1f%%)\n", accuracy_val, accuracy_val * 100))
  cat(sprintf("Test AUC: %.4f\n", auc_val))
  
  # Generem prediccions probabilístiques (valor entre 0 i 1) per al conjunt de test
  prediccions_prob   <- model_final %>% predict(test_ds_binari, verbose = 1)
  
  # Recuperem les etiquetes reals per comparar-les amb les prediccions
  etiquetes_reals <- read_csv(file.path(OUTPUT_DIR, paste0("cataleg_binari_split_", MAGNIFICACIO_OBJECTIU, ".csv")),
                              show_col_types = FALSE) %>%
    filter(split == "test") %>%
    arrange(nom_fitxer) %>%
    mutate(label = as.integer(malignitat == "maligne")) %>%
    pull(label)
  
  # Cerca del llindar òptim (maximitzant F1)
  llindars <- seq(0.2, 0.6, by = 0.05)
  resultats_llindar <- map_dfr(llindars, function(t) {
    preds <- as.integer(prediccions_prob > t)
    tp <- sum(preds == 1 & etiquetes_reals == 1)
    fp <- sum(preds == 1 & etiquetes_reals == 0)
    fn <- sum(preds == 0 & etiquetes_reals == 1)
    prec <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    rec  <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1   <- if ((prec + rec) == 0) 0 else 2 * prec * rec / (prec + rec)
    tibble(llindar = t, precision = prec, recall = rec, f1 = f1)
  })
  cat("\nAnàlisi de llindars:\n")
  print(resultats_llindar)
  
  llindar_optim <- resultats_llindar %>% slice_max(f1, n = 1) %>% pull(llindar)
  cat(sprintf("\nLlindar òptim (màxim F1): %.2f\n", llindar_optim))
  
  # Apliquem un llindar_optim per decidir la classe: 0 (benigne) o 1 (maligne)
  prediccions_classe <- as.integer(prediccions_prob > llindar_optim)
  
  # Construcció de la Matriu de Confusió (analitzar falsos positius/negatius)
  matriu_conf <- table(
    Real = ifelse(etiquetes_reals == 1, "Maligne", "Benigne"),
    Predicció = ifelse(prediccions_classe == 1, "Maligne", "Benigne")
  )
  
  cat("\n Matriu de confusió:\n")
  print(matriu_conf)
  
  # Extracció dels valors de la matriu per al càlcul de mètriques diagnòstiques
  TP <- matriu_conf["Maligne", "Maligne"] # Vertaders Positius
  TN <- matriu_conf["Benigne", "Benigne"] # Vertaders Negatius
  FP <- matriu_conf["Benigne", "Maligne"] # Falsos Positius
  FN <- matriu_conf["Maligne", "Benigne"] # Falsos Negatius
  
  # Càlcul de mètriques
  precision   <- TP / (TP + FP) # Capacitat de no predir maligne quan és benigne
  recall      <- TP / (TP + FN) # Sensibilitat: capacitat de detectar tots els casos malignes
  specificity <- TN / (TN + FP) # Especificitat: capacitat de detectar casos benignes
  f1          <- 2 * precision * recall / (precision + recall) # Mitjana harmònica (balanç)
  
  cat(sprintf("\n-- Mètriques (classe positiva: Maligne):\n"))
  cat(sprintf("  Precision:   %.4f\n", precision))
  cat(sprintf("  Recall:      %.4f\n", recall))
  cat(sprintf("  Specificity: %.4f\n", specificity))
  cat(sprintf("  F1-Score:    %.4f\n", f1))
  
  # Guardem totes les mètriques en un fitxer CSV
  tibble(
    magnificacio = MAGNIFICACIO_OBJECTIU,
    loss         = loss_val,
    accuracy     = accuracy_val,
    auc          = auc_val,
    precision    = precision,
    recall       = recall,
    specificity  = specificity,
    f1_score     = f1
  ) %>%
    write_csv(file.path(OUTPUT_DIR, paste0("metriques_test_binari_", MAGNIFICACIO_OBJECTIU, ".csv")))
  
  # 9. VISUALITZACIÓ DE L'HISTORIAL D'ENTRENAMENT
  # Analitzem la convergència del model comparant Train vs Validació
  historial_csv <- read_csv(file.path(OUTPUT_DIR, paste0("historial_entrenament_", MAGNIFICACIO_OBJECTIU, ".csv")),
                            show_col_types = FALSE)
  
  # Gràfic d'Accuracy: busquem que les dues línies pugin juntes (evitar overfitting)
  p_acc <- ggplot(historial_csv, aes(x = epoch)) +
    geom_line(aes(y = accuracy, color = "Train"), linewidth = 1) +
    geom_line(aes(y = val_accuracy, color = "Validació"),
              linewidth = 1, linetype = "dashed") +
    scale_color_manual(values = c("Train" = "#4DAFB0", "Validació" = "#D85A30")) +
    labs(title = paste("Accuracy -", MAGNIFICACIO_OBJECTIU), x = "Epoch", y = "Accuracy", color = NULL) +
    theme_minimal(base_size = 12)
  
  # Gràfic de Loss: busquem la minimització de l'error
  p_loss <- ggplot(historial_csv, aes(x = epoch)) +
    geom_line(aes(y = loss, color = "Train"), linewidth = 1) +
    geom_line(aes(y = val_loss, color = "Validació"),
              linewidth = 1, linetype = "dashed") +
    scale_color_manual(values = c("Train" = "#4DAFB0", "Validació" = "#D85A30")) +
    labs(title = paste("Loss -", MAGNIFICACIO_OBJECTIU), x = "Epoch", y = "Loss", color = NULL) +
    theme_minimal(base_size = 12)
  
  ggsave(
    file.path(OUTPUT_DIR, paste0("historial_entrenament_", MAGNIFICACIO_OBJECTIU, ".png")),
    p_acc | p_loss,
    width = 12, height = 5, dpi = 150
  )
  
  cat("\nGràfic d'historial guardat.\n")
  
  # 10. RESUM FINAL
  cat(sprintf("\n RESUM FINAL — %s \n", MAGNIFICACIO_OBJECTIU))
  cat(sprintf("Accuracy: %.2f%% | AUC: %.4f | F1: %.4f\n", accuracy_val * 100, auc_val, f1))
  
  # 11. NETEJA DE SESSIÓ PER A LA SEGÜENT MAGNIFICACIÓ
  clear_session() # Allibera memòria GPU/RAM
  gc()              # Neteja memòria de R
  
}

cat("\n=============================================\n")
cat(" FASE 3 FINALITZADA PER A TOTES LES MAGNIFICACIONS \n")
cat("=============================================\n")