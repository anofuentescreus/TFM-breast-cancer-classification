# FASE 1: Càrrega i Exploració del dataset BreakHis
# Autora: Anna Nofuentes Creus

# 0. CONFIGURACIÓ INICIAL
BASE_PATH <- "/Users/annanofuentescreus/Documents/Màster en Ciència de Dades UOC/5 Semestre/TFM /DATASETS/BreakHis/BreakHis - Breast Cancer Histopathological Database/dataset_cancer_v1/dataset_cancer_v1"
PATH_BINARI <- file.path(BASE_PATH, "classificacao_binaria")

OUTPUT_DIR <- "resultats/"
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)

# 1. INSTAL·LACIÓ I CÀRREGA DE LLIBRERIES
paquets <- c("tidyverse", "magick", "fs", "patchwork", "scales")

for (paquet in paquets) {
  if (!requireNamespace(paquet, quietly = TRUE)) {
    install.packages(paquet)
  }
}

library(tidyverse)
library(magick)
library(fs)
library(patchwork)
library(scales)

# 2. CONSTRUCCIÓ DEL CATÀLEG D'IMATGES
# L'estructura del dataset és la següent:
#   classificacao_binaria/
#     {40X, 100X, 200X, 400X}/
#       benign/      -> imatges .png
#       malignant/   -> imatges .png

fitxers_binari <- dir_ls(PATH_BINARI, recurse = TRUE, glob = "*.png")

# Creem un dataframe (tibble)
cataleg_binari <- tibble(path = as.character(fitxers_binari)) %>%
  mutate(
    # Extracció del nivell de magnificació (40x, 100x, 200x o 400x) des de la ruta del fitxer
    magnificacio = path %>% str_extract("/(40X|100X|200X|400X)/") %>% str_remove_all("/"),
    # Classificació segons l'etiqueta de la carpeta (benigne vs maligne)
    malignitat   = case_when(
      str_detect(path, "/benign/")    ~ "benigne",
      str_detect(path, "/malignant/") ~ "maligne",
      TRUE ~ "desconegut"
    ),
    nom_fitxer = basename(path) # Conservació del nom original de la mostra
  ) %>%
  # Filtrem possibles fitxers que no segueixin l'estructura esperada
  filter(malignitat != "desconegut")

# Validació de la càrrega: nombre total de mostres detectades
cat(sprintf("Total imatges (binari): %d\n", nrow(cataleg_binari)))

# 3. EXPLORACIÓ BÀSICA DEL DATASET
# Realitzem un resum estadístic per entendre el balanceig de les classes
cat("\nRESUM DEL DATASET\n")

# 3.1 Distribució per malignitat (Variable objectiu)
cat("\nDistribució per malignitat:\n")
dist_malignitat <- cataleg_binari %>%
  count(malignitat) %>%
  mutate(percentatge = n / sum(n) * 100) # Càlcul de la proporció relativa
print(dist_malignitat)

# 3.2 Distribució creuada per magnificació i malignitat
cat("\nImatges per magnificació i malignitat:\n")
dist_magnif <- cataleg_binari %>%
  count(magnificacio, malignitat) %>%
  pivot_wider(names_from = magnificacio, values_from = n, values_fill = 0) %>%
  mutate(total = `40X` + `100X` + `200X` + `400X`)
print(dist_magnif)

# 4. VISUALITZACIONS (EDA)
color_benigne <- "#4DAFB0"
color_maligne <- "#D85A30"

tema_comú <- theme_minimal(base_size = 10) +
  theme(
    plot.title      = element_text(face = "bold", size = 11, margin = margin(b = 8)),
    axis.text       = element_text(size = 8, color = "grey40"),
    axis.text.x     = element_text(angle = 0, hjust = 0.5),
    axis.title.y    = element_text(size = 8, color = "grey40", margin = margin(r = 8)),
    axis.ticks      = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.major= element_line(color = "grey93", linewidth = 0.3),
    panel.grid.minor= element_blank(),
    legend.title    = element_text(size = 7, color = "grey30", face = "bold"),
    legend.text     = element_text(size = 7, color = "grey40"),
    legend.key.size = unit(0.35, "cm"),
    legend.position = "bottom",
    plot.margin     = margin(10, 15, 10, 10)
  )

# 4.1 Distribució per malignitat (Gràfic de barres)
p1 <- dist_malignitat %>%
  ggplot(aes(x = malignitat, y = n, fill = malignitat)) +
  geom_col(width = 0.45, alpha = 0.88) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, percentatge)),
            vjust = -0.3, size = 2.3, color = "grey35", fontface = "plain",
            lineheight = 1.2) +
  scale_fill_manual(values = c("benigne" = color_benigne, "maligne" = color_maligne)) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.18))) +
  labs(title = "Distribució per malignitat", x = NULL,
       y = "Nombre d'imatges", fill = NULL) +
  tema_comú +
  guides(fill = guide_legend(override.aes = list(alpha = 0.88)))

# 4.2 Distribució per magnificació (Gràfic de barres agrupat)
p2 <- cataleg_binari %>%
  count(magnificacio, malignitat) %>%
  mutate(magnificacio = factor(magnificacio, levels = c("40X", "100X", "200X", "400X"))) %>%
  ggplot(aes(x = magnificacio, y = n, fill = malignitat)) +
  geom_col(position = position_dodge(width = 0.55), width = 0.5, alpha = 0.88) +
  geom_text(aes(label = n),
            position = position_dodge(width = 0.55),
            vjust = -0.35, size = 2.2, color = "grey35") +
  scale_fill_manual(values = c("benigne" = color_benigne, "maligne" = color_maligne)) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.18))) +
  labs(title = "Distribució per magnificació", x = NULL,
       y = "Nombre d'imatges", fill = NULL) +
  tema_comú +
  guides(fill = guide_legend(override.aes = list(alpha = 0.88)))

# Combinació d'ambdós gràfics en una sola imatge
grafic_eda <- p1 | p2 +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(file.path(OUTPUT_DIR, "eda_distribucio.png"),
       grafic_eda, width = 10, height = 5, dpi = 200)

cat("\nGràfic guardat a:", file.path(OUTPUT_DIR, "eda_distribucio.png"), "\n")

# 5. MOSTRA D'IMATGES REPRESENTATIVES
# Seleccionem aleatòriament 4 exemples de cada classe a 40X

mostra <- cataleg_binari %>%
  filter(magnificacio == "40X") %>%
  group_by(malignitat) %>%
  slice_sample(n = 4) %>%
  ungroup()

# Processament d'imatges per crear un mosaic
imatges <- map(mostra$path, function(p) {
  img <- image_read(p)
  img <- image_resize(img, "200x200!")
  mal_label <- mostra$malignitat[mostra$path == p][1]
  color_box <- if (mal_label == "benigne") color_benigne else color_maligne
  
  # Afegim una etiqueta visual directament sobre la imatge
  img <- image_annotate(img,
                        text = mal_label,
                        gravity = "South",
                        size = 13,
                        color = "white",
                        boxcolor = color_box,
                        location = "+0+0")
  img
})

# Unim les imatges en un únic fitxer de sortida
graella <- image_append(
  image_join(
    list(
      image_append(image_join(imatges[1:4]), stack = FALSE),
      image_append(image_join(imatges[5:8]), stack = FALSE)
    )
  ),
  stack = TRUE
)

image_write(graella, file.path(OUTPUT_DIR, "mostra_imatges.png"))
cat("Mosaic d'imatges guardat a:", file.path(OUTPUT_DIR, "mostra_imatges.png"), "\n")

# 6. GUARDAR EL CATÀLEG
write_csv(cataleg_binari, file.path(OUTPUT_DIR, "cataleg_binari.csv"))

cat("\n=============================================\n")
cat(sprintf("\ FASE 1 FINALITZADA. CATÀLEG I GRÀFICS DISPONIBLES A: %s\n", OUTPUT_DIR))
cat("=============================================\n")