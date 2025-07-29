# =============================================================================
# ANALYSE PACF POUR LES DONNÉES ELA - VERSION SIMPLIFIÉE SANS LAGS
# =============================================================================

# Charger les packages nécessaires
library(ggplot2)
library(dplyr)
library(gridExtra)
library(reshape2)
library(forecast)
library(tseries)

# =============================================================================
# Configuration
# =============================================================================

# Chemin vers les données ELA
data_path <- "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/DATA_RAW/df_final/ELA_py.csv"

# Répertoire de sortie pour les figures
output_dir <- "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/ELA_model"

# Variables de diversité à analyser
diversity_vars <- c("rich_genus_no_cyano", "shannon_no_cyano", "eveness_piel_no_cyano")
diversity_labels <- c("Richesse (S)", "Shannon (H')", "Équitabilité (J')")

# IDs des lacs à analyser
lake_ids <- c(114, 224, 239, 373, 442)
lake_names <- c("114", "224", "239", "373", "442")

# =============================================================================
# Fonctions principales
# =============================================================================

calculate_pacf_for_lake <- function(data, lake_id, variable, max_lag = 4) {
  """
  Calculer PACF pour un lac et une variable spécifiques
  """
  
  # Filtrer pour un lac spécifique
  lake_data <- data %>%
    filter(lake_id == !!lake_id) %>%
    arrange(year, doy)
  
  if(nrow(lake_data) < 10) {
    return(NULL)
  }
  
  # Extraire la série temporelle
  ts_data <- lake_data[[variable]]
  ts_data <- ts_data[!is.na(ts_data)]
  
  if(length(ts_data) < 10) {
    return(NULL)
  }
  
  # Calculer PACF
  tryCatch({
    pacf_result <- pacf(ts_data, lag.max = max_lag, plot = FALSE)
    
    # Tests de stationnarité
    adf_test <- adf.test(ts_data, alternative = "stationary")
    kpss_test <- kpss.test(ts_data)
    
    # Déterminer la significativité
    adf_signif <- adf_test$p.value < 0.05
    kpss_signif <- kpss_test$p.value > 0.05  # Pour KPSS, p > 0.05 indique stationnarité
    
    return(list(
      pacf_values = as.numeric(pacf_result$acf),
      lags = 1:max_lag,
      adf_significant = adf_signif,
      kpss_significant = kpss_signif,
      n_obs = length(ts_data)
    ))
  }, error = function(e) {
    return(NULL)
  })
}

create_pacf_combined_plot <- function(data) {
  """
  Créer le graphique PACF combiné pour toutes les métriques et tous les lacs
  """
  
  # Créer une grille de graphiques
  plots_list <- list()
  
  for (i in 1:length(diversity_vars)) {
    for (j in 1:length(lake_ids)) {
      
      variable <- diversity_vars[i]
      lake_id <- lake_ids[j]
      
      # Calculer PACF
      pacf_result <- calculate_pacf_for_lake(data, lake_id, variable, max_lag = 4)
      
      if (!is.null(pacf_result)) {
        # Créer le dataframe pour le graphique
        plot_data <- data.frame(
          lag = pacf_result$lags,
          pacf = pacf_result$pacf_values
        )
        
        # Déterminer le titre avec les résultats des tests
        adf_status <- ifelse(pacf_result$adf_significant, "S", "NS")
        kpss_status <- ifelse(pacf_result$kpss_significant, "S", "NS")
        
        title_text <- paste0(lake_names[j], " - ", diversity_labels[i], " - ADF:", adf_status, ", KPSS:", kpss_status)
        
        # Créer le graphique
        p <- ggplot(plot_data, aes(x = lag, y = pacf)) +
          geom_col(fill = "steelblue", alpha = 0.7, width = 0.6) +
          geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
          geom_hline(yintercept = c(-0.2, 0.2), color = "red", linetype = "dashed", alpha = 0.5) +
          scale_x_continuous(breaks = 1:4, limits = c(0.5, 4.5)) +
          scale_y_continuous(limits = c(-1, 1)) +
          labs(
            title = if(i == 1) title_text else "",
            x = if(i == 3) "Lag" else "",
            y = if(j == 1) "PACF" else ""
          ) +
          theme_minimal() +
          theme(
            plot.title = element_text(size = 8, hjust = 0.5),
            axis.title = element_text(size = 8),
            axis.text = element_text(size = 7),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank()
          )
        
      } else {
        # Pas de données suffisantes
        p <- ggplot() +
          annotate("text", x = 0.5, y = 0.5, label = "Insufficient\\ndata", 
                   hjust = 0.5, vjust = 0.5, size = 3) +
          scale_x_continuous(limits = c(0.5, 4.5)) +
          scale_y_continuous(limits = c(-1, 1)) +
          labs(
            title = if(i == 1) paste0(lake_names[j], " - ", diversity_labels[i], " - ADF:NS, KPSS:NS") else "",
            x = if(i == 3) "Lag" else "",
            y = if(j == 1) "PACF" else ""
          ) +
          theme_minimal() +
          theme(
            plot.title = element_text(size = 8, hjust = 0.5),
            axis.title = element_text(size = 8),
            axis.text = element_text(size = 7),
            panel.grid = element_blank()
          )
      }
      
      # Ajouter à la liste
      plot_index <- (i - 1) * length(lake_ids) + j
      plots_list[[plot_index]] <- p
    }
  }
  
  # Combiner tous les graphiques
  combined_plot <- do.call(grid.arrange, c(plots_list, ncol = 5, nrow = 3))
  
  # Sauvegarder
  ggsave(
    filename = file.path(output_dir, "PACF_combined_all_metrics.png"),
    plot = combined_plot,
    width = 20, height = 12, dpi = 300, units = "in"
  )
  
  cat("Graphique PACF combiné créé avec succès\\n")
  return(combined_plot)
}

# =============================================================================
# Fonction principale
# =============================================================================

main <- function() {
  cat("=== ANALYSE PACF POUR LES DONNÉES ELA ===\\n")
  
  # Charger les données
  if (!file.exists(data_path)) {
    cat("Erreur: Fichier de données non trouvé:", data_path, "\\n")
    return()
  }
  
  data <- read.csv(data_path)
  cat("Données chargées:", nrow(data), "observations\\n")
  
  # Vérifier les colonnes nécessaires
  required_cols <- c("lake_id", "year", "doy", diversity_vars)
  missing_cols <- setdiff(required_cols, names(data))
  
  if (length(missing_cols) > 0) {
    cat("Erreur: Colonnes manquantes:", paste(missing_cols, collapse = ", "), "\\n")
    return()
  }
  
  # Créer le répertoire de sortie
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Créer l'analyse PACF combinée
  create_pacf_combined_plot(data)
  
  cat("=== ANALYSE TERMINÉE ===\\n")
  cat("Figure sauvegardée dans:", output_dir, "\\n")
}

# Exécuter l'analyse
main()
