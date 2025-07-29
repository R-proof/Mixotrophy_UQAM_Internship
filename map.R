# Chargement des packages nécessaires
library(sf)
library(ggplot2)
library(dplyr)
library(ggspatial)
library(rnaturalearth)
library(rnaturalearthdata)
library(viridis)

# === 1. Charger les écorégions EPA ===
ecoregions <- st_read("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/na_cec_eco_l1/NA_CEC_Eco_Level1.shp")

# Reprojeter en NAD83 (EPSG:4269)
ecoregions_nad83 <- st_transform(ecoregions, crs = 4269)

# === 2. Charger les données des lacs ===
df_lakes = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/df_final/df_LP_NLA.csv")

# Conversion en objet spatial
lakes_sf <- st_as_sf(df_lakes, coords = c("long", "lat"), crs = 4326)  # WGS84
lakes_nad83 <- st_transform(lakes_sf, crs = 4269)

# Ajouter une colonne "Survey" (ex : selon une colonne déjà présente ou créer manuellement)
# Ici, on suppose qu'une colonne "Survey" existe, sinon adapte :
# lakes_nad83$Survey <- ifelse(lakes_nad83$Survey == "LP", "Lake Pulse", "NLA 2017")

# === 3. Affichage avec ggplot ===
# ggplot() +
#   geom_sf(data = ecoregions_nad83, aes(fill = NA_L1NAME), color = NA, alpha = 0.4) +
#   scale_fill_viridis_d(name = "Ecoregions", option = "turbo") +
#   theme_minimal()
# Exemple avec tes 5 coordonnées (latitude, longitude)
df_ela <- data.frame(
  long = c(-93.7550, -93.7172, -93.7226, -93.7994, -93.8175),
  lat  = c(49.6712, 49.6905, 49.6637, 49.7450, 49.7749),
  Survey = "ELA"
)

# Convertir en sf (en NAD83)
df_ela_sf <- st_as_sf(df_ela, coords = c("long", "lat"), crs = 4326) %>%
  st_transform(crs = 4269)


# ggplot() +
#   geom_sf(data = ecoregions_nad83, aes(fill = NA_L1NAME), color = NA, alpha = 0.4) +
#   geom_sf(data = lakes_nad83, aes(color = Survey), size = 1, alpha = 0.8) +
#   scale_fill_viridis_d(name = "Ecoregions", option = "turbo") +
#   scale_color_manual(values = c("Lake Pulse" = "black", "NLA 2017" = "red")) +
#   theme_minimal()
# 
# ggplot(df_final_LP_NLA, aes(x = TP_ug.L, y = prev_Mixo, fill = shannon)) +
#   geom_tile() +
#   scale_fill_viridis_c() +
#   labs(x = "TP (µg/L)", y = "Prévalence mixo", fill = "Shannon")

gg = ggplot() +
  geom_sf(data = ecoregions_nad83, aes(fill = NA_L1NAME), color = NA, alpha = 0.4) +
  geom_sf(data = lakes_nad83, aes(color = Survey), size = 1, alpha = 0.8) +
  
  # 🌟 Étoiles pour ELA
  geom_sf(data = df_ela_sf, aes(color = Survey), shape = 8, size = 3) +  # shape = 8 → étoile
  
  scale_fill_viridis_d(name = "Ecoregions", option = "turbo") +
  scale_color_manual(
    values = c("Lake Pulse" = "black", "NLA 2017" = "red", "ELA" = "blue"),
    breaks = c("Lake Pulse", "NLA 2017", "ELA")
  ) +
  theme_minimal() +
  labs(title = "", color = "Relevés")
ggsave("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/figures/map_lakes.png", plot = gg, width = 10, height = 7, dpi = 300)

# Libraries
library(sf)
library(ggplot2)
library(viridis)

# === 1. Correction des géométries invalides ===
ecoregions_nad83 <- st_make_valid(ecoregions_nad83)

# === 2. Vérification et transformation CRS pour les points ELA ===
st_crs(df_ela_sf) <- 4326  # WGS84 si manquant
df_ela_sf <- st_transform(df_ela_sf, 4269)  # NAD83

# === 3. Création d'une bbox manuelle autour des points ELA ===
ela_coords <- st_coordinates(df_ela_sf)
xrange <- range(ela_coords[, 1])
yrange <- range(ela_coords[, 2])
expand <- 0.5  # marge en degrés

ela_bbox <- st_bbox(c(
  xmin = xrange[1] - expand,
  xmax = xrange[2] + expand,
  ymin = yrange[1] - expand,
  ymax = yrange[2] + expand
), crs = st_crs(df_ela_sf))

ela_zoom <- st_as_sfc(ela_bbox)

# === 4. Filtrage des couches à la zone zoomée ===
ecoregions_zoom <- st_crop(ecoregions_nad83, ela_zoom)
lakes_zoom <- st_crop(lakes_nad83, ela_zoom)

unique(ecoregions_zoom$NA_L1NAME)
# === 5. Affichage de la carte zoomée ===
gg2 = ggplot() +
  geom_sf(data = ecoregions_zoom, aes(fill = NA_L1NAME), color = NA) +  # enlever alpha ici
  
  geom_sf(data = lakes_zoom, aes(color = Survey), size = 1, alpha = 0.8) +
  geom_sf(data = df_ela_sf, aes(color = Survey), shape = 8, size = 3, stroke = 1.2) +
  
  scale_fill_manual(
    name = "Ecoregions",
    values = c("NORTHERN FORESTS" = "#D2FFBE")  # 🌿 couleur verte personnalisée
  ) +
  
  scale_color_manual(
    values = c("Lake Pulse" = "black", "NLA 2017" = "red", "ELA" = "blue"),
    breaks = c("Lake Pulse", "NLA 2017", "ELA")
  ) +
  
  coord_sf(xlim = c(ela_bbox["xmin"], ela_bbox["xmax"]),
           ylim = c(ela_bbox["ymin"], ela_bbox["ymax"]),
           crs = 4269) +
  theme_minimal() +
  labs(
    title = "Zoom sur les lacs ELA",
    subtitle = "Ecoregions Level I (NAD83) – étoiles dorées : sites ELA",
    color = "Survey"
  )
ggsave("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/figures/zoom_map_ELA.png", plot = gg2, width = 10, height = 7, dpi = 300)




col="#D2FFBE"



























library(lubridate)
df_final_IISD_ELA_2 = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/df_final/df_IISD_ELA.csv")
df_final_IISD_ELA_2$lake_id = as.factor(df_final_IISD_ELA_2$lake_id) 
# Plot des dates d’échantillonnage
time_ELA = ggplot(df_final_IISD_ELA_2, aes(x = year(date), y = lake_id)) +
  geom_violin(color = "black") +
  labs(x = "Date", y = "Lac") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.text.x = element_text(size = 17),  
    axis.text.y = element_text(size = 17)   
  )

ggsave("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/figures/time_ELA.png", plot = time_ELA, width = 10, height = 7, dpi = 300)
