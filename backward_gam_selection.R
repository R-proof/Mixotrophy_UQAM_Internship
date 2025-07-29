backward_gam_selection <- function(df, resp, Xs_lin, Xs_nolin, family = gaussian()) {
  
  current_Xs_lin <- Xs_lin
  current_Xs_nolin <- Xs_nolin
  
  # Construction de la formule complète
  build_formula <- function(lin, nolin) {
    lin_part <- lin
    nolin_part <- paste0("s(", nolin, ")")
    all_terms <- c(lin_part, nolin_part)
    as.formula(paste(resp, "~", paste(all_terms, collapse = " + ")))
  }
  
  current_formula <- build_formula(current_Xs_lin, current_Xs_nolin)
  best_model <- gam(current_formula, data = df, family = family)
  
  improving <- TRUE
  iteration <- 1
  
  while (improving) {
    cat("\nÉtape", iteration, "- AIC :", AIC(best_model), "\n")
    
    sm <- summary(best_model)
    
    # Récupérer les p-values des effets lissés uniquement
    if (!is.null(sm$s.table)) {
      pvals_smooth <- sm$s.table[, 4]
      names(pvals_smooth) <- rownames(sm$s.table)
      pvals_smooth <- pvals_smooth[!is.na(pvals_smooth)]
    } else {
      pvals_smooth <- numeric(0)
    }
    
    # Récupérer les p-values des effets linéaires
    pvals_linear <- sm$p.table[-1, 4]  # exclude intercept
    names(pvals_linear) <- rownames(sm$p.table)[-1]
    pvals_linear <- pvals_linear[!is.na(pvals_linear)]
    
    # Fusion des p-values
    all_pvals <- c(pvals_linear, pvals_smooth)
    
    if (length(all_pvals) == 0) break
    
    worst_term <- names(all_pvals)[which.max(all_pvals)]
    
    if (all_pvals[worst_term] > 0.05) {
      cat("Suppression de :", worst_term, "- p =", round(all_pvals[worst_term], 4), "\n")
      
      # Enlever correctement selon linéaire ou smooth
      if (grepl("^s\\((.*)\\)", worst_term)) {
        var_to_remove <- sub("^s\\((.*)\\)", "\\1", worst_term)
        current_Xs_nolin <- setdiff(current_Xs_nolin, var_to_remove)
      } else {
        current_Xs_lin <- setdiff(current_Xs_lin, worst_term)
      }
      
      current_formula <- build_formula(current_Xs_lin, current_Xs_nolin)
      best_model <- gam(current_formula, data = df, family = family)
      iteration <- iteration + 1
    } else {
      cat("Aucune variable non significative restante // => Fin de la sélection\n")
      improving <- FALSE
    }
  }
  
  return(best_model)
}