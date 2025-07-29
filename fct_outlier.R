# Fonction pour détecter les outliers (avec seuil ajustable)
detect_outliers <- function(df, id_col = "lake_id", dataset_name = "Dataset", iqr_threshold = 5) {
  numeric_vars <- names(df)[sapply(df, is.numeric) & !names(df) %in% vars_to_remove]
  outlier_list <- list()
  
  for (var in numeric_vars) {
    Q1 <- quantile(df[[var]], 0.25, na.rm = TRUE)
    Q3 <- quantile(df[[var]], 0.75, na.rm = TRUE)
    IQR_val <- Q3 - Q1
    lower_bound <- Q1 - iqr_threshold * IQR_val
    upper_bound <- Q3 + iqr_threshold * IQR_val
    
    is_outlier <- df[[var]] < lower_bound | df[[var]] > upper_bound
    if (any(is_outlier, na.rm = TRUE)) {
      rows_with_outliers <- which(is_outlier)
      for (i in rows_with_outliers) {
        key <- if (id_col %in% names(df)) as.character(df[[id_col]][i]) else as.character(i)
        if (key %in% names(outlier_list)) {
          outlier_list[[key]] <- c(outlier_list[[key]], var)
        } else {
          outlier_list[[key]] <- var
        }
      }
    }
  }
  
  cat("\n------ Résumé des outliers pour", dataset_name, "------\n")
  for (id in names(outlier_list)) {
    cat("Ligne:", id, "| Variables:", paste(outlier_list[[id]], collapse = ", "), "\n")
  }
  
  return(outlier_list)
}

# Fonction pour supprimer les lignes contenant des outliers
remove_outlier_rows <- function(df, outlier_list, df_name = "Dataset") {
  if ("lake_id" %in% colnames(df)) {
    rows_to_remove <- which(df$lake_id %in% names(outlier_list))
  } else {
    rows_to_remove <- as.integer(names(outlier_list))
  }
  
  n_before <- nrow(df)
  df_clean <- df[-rows_to_remove, ]
  n_after <- nrow(df_clean)
  n_removed <- n_before - n_after
  percent_removed <- round(100 * n_removed / n_before, 2)
  
  cat(paste0(">>> ", df_name, ": ", n_removed, " lignes supprimées (", percent_removed, "% du total).\n"))
  
  return(df_clean)
}


























#####################################################################################################
#####################################################################################################
# Blessing ---------------------------------------------------------------------------------------
#####################################################################################################
#####################################################################################################


library(stats)

# Fonction de calcul du seuil critique de Chauvenet
z_crit_chauvenet <- function(n) {
  if (n <= 1) return(Inf)
  qnorm(1 - 1 / (2 * n))  # seuil de probabilité à 1/(2n)
}

detect_outliers_blessing <- function(df, id_col = "lake_id", dataset_name = "Dataset", vars_to_remove = c()) {
  numeric_vars <- names(df)[sapply(df, is.numeric) & !names(df) %in% vars_to_remove]
  outlier_list <- list()
  
  for (var in numeric_vars) {
    x <- df[[var]]
    if (all(is.na(x))) next  # sauter les colonnes entièrement NA
    
    x_clean <- x[!is.na(x)]
    n <- length(x_clean)
    
    if (is.na(n) || n <= 3) next  # protection supplémentaire
    
    med <- median(x_clean)
    mad_val <- median(abs(x_clean - med)) * sqrt(n / (n - 1))
    if (mad_val == 0 || is.na(mad_val)) next  # éviter division par zéro
    
    z_crit <- qnorm(1 - 1 / (2 * n))
    z_score <- abs(x - med) / mad_val
    is_outlier <- z_score > z_crit
    
    if (any(is_outlier, na.rm = TRUE)) {
      rows_with_outliers <- which(is_outlier)
      for (i in rows_with_outliers) {
        key <- if (id_col %in% names(df)) as.character(df[[id_col]][i]) else as.character(i)
        if (key %in% names(outlier_list)) {
          outlier_list[[key]] <- c(outlier_list[[key]], var)
        } else {
          outlier_list[[key]] <- var
        }
      }
    }
  }
  
  cat("\n------ Résumé des outliers pour", dataset_name, "(méthode de Blessing) ------\n")
  for (id in names(outlier_list)) {
    cat("Ligne:", id, "| Variables:", paste(outlier_list[[id]], collapse = ", "), "\n")
  }
  
  return(outlier_list)
}



