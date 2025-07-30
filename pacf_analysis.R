# =============================================================================
# ANALYSE PACF ROBUSTE POUR LES DONNÉES ELA - TRAITEMENT DE LA STATIONNARITÉ
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

# Chemin vers les résidus XGBoost
residuals_dir <- "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/DATA_MODEL"

# Répertoire de sortie pour les figures
output_dir <- "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/ELA_model"

# Variables de diversité à analyser
diversity_vars <- c("rich_genus_no_cyano", "shannon_no_cyano", "eveness_piel_no_cyano")
diversity_labels <- c("Richesse (S)", "Shannon (H')", "Équitabilité (J')")

# IDs des lacs à analyser
lake_ids <- c(114, 224, 239, 373, 442)
lake_names <- c("114", "224", "239", "373", "442")

# Couleurs pour les métriques
metric_colors <- c("#E69F00", "#56B4E9", "#009E73")
names(metric_colors) <- diversity_vars

# =============================================================================
# Fonctions de traitement de la stationnarité
# =============================================================================

check_stationarity <- function(ts_data) {
  # Tester la stationnarité avec ADF et KPSS
  
  # Test ADF (H0: série non-stationnaire)
  adf_test <- tryCatch({
    adf.test(ts_data)
  }, error = function(e) {
    list(p.value = 1)  # Si erreur, considérer comme non-stationnaire
  })
  
  # Test KPSS (H0: série stationnaire)
  kpss_test <- tryCatch({
    kpss.test(ts_data, null = "Trend")
  }, error = function(e) {
    list(p.value = 0)  # Si erreur, considérer comme non-stationnaire
  })
  
  adf_stationary <- adf_test$p.value < 0.05
  kpss_stationary <- kpss_test$p.value > 0.05
  
  return(list(
    adf_stationary = adf_stationary,
    kpss_stationary = kpss_stationary,
    both_stationary = adf_stationary && kpss_stationary,
    adf_pvalue = adf_test$p.value,
    kpss_pvalue = kpss_test$p.value
  ))
}

make_stationary <- function(ts_data) {
  # Rendre une série stationnaire si nécessaire
  
  original_data <- ts_data
  stationarity_info <- check_stationarity(ts_data)
  transformation <- "none"
  
  # Si non-stationnaire, essayer différentes transformations
  if (!stationarity_info$both_stationary) {
    
    # 1. Essayer la différenciation première
    if (length(ts_data) > 2) {
      diff_data <- diff(ts_data, differences = 1)
      diff_stationarity <- check_stationarity(diff_data)
      
      if (diff_stationarity$both_stationary) {
        ts_data <- diff_data
        transformation <- "diff1"
      } else {
        # 2. Essayer la différenciation seconde
        if (length(diff_data) > 2) {
          diff2_data <- diff(diff_data, differences = 1)
          diff2_stationarity <- check_stationarity(diff2_data)
          
          if (diff2_stationarity$both_stationary) {
            ts_data <- diff2_data
            transformation <- "diff2"
          } else {
            # 3. Utiliser la détrend (centrer et réduire)
            ts_data <- scale(ts_data)[,1]
            transformation <- "detrend"
          }
        }
      }
    }
  }
  
  final_stationarity <- check_stationarity(ts_data)
  
  return(list(
    data = ts_data,
    original_data = original_data,
    transformation = transformation,
    stationarity = final_stationarity
  ))
}

calculate_pacf_for_lake <- function(residuals_data, lake_id, variable, max_lag = 5) {
  # Calculer PACF pour un lac et une variable spécifiques sur les résidus XGBoost
  
  # Nom de la colonne des résidus
  residuals_col <- paste0(variable, "_residuals")
  
  # Filtrer pour un lac spécifique
  lake_data <- residuals_data %>%
    filter(lake_id == !!lake_id) %>%
    arrange(year, doy)
  
  if(nrow(lake_data) < 10) {
    return(NULL)
  }
  
  # Extraire la série temporelle des résidus
  ts_raw <- lake_data[[residuals_col]]
  
  # Supprimer les valeurs manquantes
  ts_raw <- ts_raw[!is.na(ts_raw)]
  
  if(length(ts_raw) < 10) {
    return(NULL)
  }
  
  # Rendre la série stationnaire si nécessaire (les résidus devraient déjà être stationnaires)
  stationary_result <- make_stationary(ts_raw)
  ts_data <- stationary_result$data
  
  if(length(ts_data) < max_lag + 1) {
    return(NULL)
  }
  
  # Calculer PACF avec intervalles de confiance
  pacf_result <- tryCatch({
    pacf(ts_data, lag.max = max_lag, plot = FALSE)
  }, error = function(e) {
    return(NULL)
  })
  
  if(is.null(pacf_result)) {
    return(NULL)
  }
  
  # Calculer les intervalles de confiance (approximation asymptotique)
  n <- length(ts_data)
  conf_int <- 1.96 / sqrt(n)  # IC à 95%
  
  return(list(
    pacf_values = as.numeric(pacf_result$acf),
    conf_int = conf_int,
    n_obs = n,
    stationarity = stationary_result$stationarity,
    transformation = stationary_result$transformation,
    original_n = length(ts_raw)
  ))
}

create_pacf_combined_plot <- function(residuals_data) {
  # Créer le graphique PACF combiné pour toutes les métriques et tous les lacs (sur résidus)
  
  # Créer une grille 3x5 (3 métriques x 5 lacs)
  plots_list <- list()
  plot_counter <- 1
  
  for(i in 1:length(diversity_vars)) {
    variable <- diversity_vars[i]
    var_label <- diversity_labels[i]
    var_color <- metric_colors[variable]
    
    for(j in 1:length(lake_ids)) {
      lake_id <- lake_ids[j]
      lake_name <- lake_names[j]
      
      # Calculer PACF pour ce lac et cette variable (sur résidus)
      pacf_data <- calculate_pacf_for_lake(residuals_data, lake_id, variable, max_lag = 5)
      
      if(!is.null(pacf_data)) {
        # Créer le data frame pour ggplot (inclure lag 0 et ajouter lag 5)
        plot_df <- data.frame(
          lag = 0:5,
          pacf = c(1, pacf_data$pacf_values),  # lag 0 = 1, puis lags 1-5
          conf_lower = c(0, rep(-pacf_data$conf_int, 5)),
          conf_upper = c(0, rep(pacf_data$conf_int, 5))
        )
        
        # Créer le graphique
        p <- ggplot(plot_df, aes(x = lag, y = pacf)) +
          geom_bar(stat = "identity", fill = var_color, alpha = 0.7, width = 0.6) +
          geom_hline(yintercept = 0, color = "black", linewidth = 0.8) +
          geom_hline(yintercept = c(-0.3, 0.3), color = "red", 
                    linetype = "dashed", alpha = 0.7, linewidth = 1) +
          scale_x_continuous(breaks = 0:5) +
          ylim(-1, 1) +
          labs(x = ifelse(i == 3, "Lag", ""),
               y = ifelse(j == 1, paste0(var_label, "\nPACF"), "")) +
          theme_minimal() +
          theme(
            panel.grid.minor = element_blank(),
            panel.grid.major = element_line(linetype = "dotted"),
            axis.text = element_text(size = 10),
            axis.title = element_text(size = 12)
          )
        
        # Ajouter le titre seulement pour la première ligne
        if(i == 1) {
          adf_status <- ifelse(pacf_data$stationarity$adf_stationary, "S", "NS")
          kpss_status <- ifelse(pacf_data$stationarity$kpss_stationary, "S", "NS")
          # Enlever "(diff1)" et supprimer le gras
          title <- paste0("Lake ", lake_name, "\n", 
                         "ADF:", adf_status, ", KPSS:", kpss_status)
          p <- p + ggtitle(title) +
            theme(plot.title = element_text(size = 10, face = "plain"))
        }
        
        # Mettre en évidence les valeurs significatives
        significant_lags <- which(abs(plot_df$pacf) > 0.3)
        if(length(significant_lags) > 0) {
          p <- p + geom_bar(data = plot_df[significant_lags, ], 
                           aes(x = lag, y = pacf), 
                           stat = "identity", fill = var_color, 
                           alpha = 1.0, width = 0.6, color = "black", linewidth = 1)
        }
        
      } else {
        # Graphique vide avec message
        p <- ggplot() +
          annotate("text", x = 0.5, y = 0.5, 
                  label = "Données\ninsuffisantes", 
                  hjust = 0.5, vjust = 0.5, size = 4) +
          xlim(0.5, 4.5) +
          ylim(-1, 1) +
          scale_x_continuous(breaks = 1:4) +
          labs(x = ifelse(i == 3, "Lag", ""),
               y = ifelse(j == 1, paste0(var_label, "\nPACF"), "")) +
          theme_minimal() +
          theme(
            panel.grid.minor = element_blank(),
            panel.grid.major = element_line(linetype = "dotted"),
            axis.text = element_text(size = 10),
            axis.title = element_text(size = 12)
          )
        
        if(i == 1) {
          p <- p + ggtitle(paste0("Lake ", lake_name, "\nADF:NS, KPSS:NS"))
        }
      }
      
      plots_list[[plot_counter]] <- p
      plot_counter <- plot_counter + 1
    }
  }
  
  # Arranger tous les graphiques en grille 3x5
  combined_plot <- do.call(grid.arrange, c(plots_list, ncol = 5, nrow = 3))
  
  return(combined_plot)
}

# =============================================================================
# Script principal
# =============================================================================

main <- function() {
  cat("=== ANALYSE PACF SUR RÉSIDUS XGBOOST POUR ELA ===\n")
  
  # Charger et combiner les résidus de tous les modèles
  cat("Chargement des résidus XGBoost...\n")
  
  residuals_combined <- NULL
  
  for(variable in diversity_vars) {
    residuals_file <- file.path(residuals_dir, paste0("residuals_", variable, "_ELA.csv"))
    
    if(file.exists(residuals_file)) {
      residuals_data <- read.csv(residuals_file)
      cat(paste("  ", variable, ":", nrow(residuals_data), "résidus chargés\n"))
      
      if(is.null(residuals_combined)) {
        residuals_combined <- residuals_data
      } else {
        # Fusionner par lake_id, year, doy
        residuals_combined <- merge(residuals_combined, residuals_data, 
                                  by = c("lake_id", "year", "doy"), all = TRUE)
      }
    } else {
      cat(paste("  ATTENTION: Fichier de résidus manquant pour", variable, "\n"))
      cat(paste("  Veuillez d'abord exécuter model_final.py pour générer les résidus\n"))
      return(NULL)
    }
  }
  
  if(is.null(residuals_combined)) {
    stop("Aucun fichier de résidus trouvé. Exécutez d'abord model_final.py")
  }
  
  cat(paste("Résidus combinés:", nrow(residuals_combined), "observations\n"))
  
  # S'assurer que le répertoire de sortie existe
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Créer le graphique PACF combiné (sur résidus)
  cat("Création du graphique PACF combiné sur résidus...\n")
  
  # Ouvrir le device graphique
  png(filename = file.path(output_dir, "PACF_combined_all_metrics.png"),
      width = 20, height = 12, units = "in", res = 300)
  
  # Créer et afficher le graphique
  combined_plot <- create_pacf_combined_plot(residuals_combined)
  
  # Fermer le device
  dev.off()
  
  cat("Graphique PACF sauvegardé:", file.path(output_dir, "PACF_combined_all_metrics.png"), "\n")
  
  # Statistiques de résumé
  cat("\n=== RÉSUMÉ DE L'ANALYSE (RÉSIDUS) ===\n")
  
  for(variable in diversity_vars) {
    cat(paste("\nVariable:", variable, "(résidus)\n"))
    
    for(lake_id in lake_ids) {
      pacf_result <- calculate_pacf_for_lake(residuals_combined, lake_id, variable)
      
      if(!is.null(pacf_result)) {
        adf_status <- ifelse(pacf_result$stationarity$adf_stationary, "Stationnaire", "Non-stationnaire")
        kpss_status <- ifelse(pacf_result$stationarity$kpss_stationary, "Stationnaire", "Non-stationnaire")
        
        cat(paste("  Lac", lake_id, ":", 
                 "n=", pacf_result$original_n, 
                 "-> n=", pacf_result$n_obs,
                 "| Transformation:", pacf_result$transformation,
                 "| ADF:", adf_status,
                 "| KPSS:", kpss_status, "\n"))
        
        # Afficher les valeurs PACF significatives
        significant_lags <- which(abs(pacf_result$pacf_values) > 0.2)
        if(length(significant_lags) > 0) {
          cat(paste("    PACF significatives (>0.2):", 
                   paste(paste("Lag", significant_lags, "=", 
                              round(pacf_result$pacf_values[significant_lags], 3)), 
                        collapse = ", "), "\n"))
        } else {
          cat("    Aucune PACF significative > 0.2 (bon modèle!)\n")
        }
      } else {
        cat(paste("  Lac", lake_id, ": Données insuffisantes\n"))
      }
    }
  }
  
  cat("\n=== ANALYSE TERMINÉE ===\n")
  cat("Note: L'absence de PACF significatives indique que le modèle XGBoost\n")
  cat("      capture bien la structure temporelle des données.\n")
}

# Exécuter le script principal
if(!interactive()) {
  main()
}
