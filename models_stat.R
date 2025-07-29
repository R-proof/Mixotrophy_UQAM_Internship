  # Script by SERRE Renaud
# TITLE  : RECAP TOT 2

rm(list = ls())
graphics.off()
cat("\014") 

# ==== libray manipulation data ====
library(openxlsx)     # Lire/écrire fichiers Excel
library(tidyverse)    # Manipulation, visualisation, data wrangling
library(GGally)       # Matrices de paires ggplot + ggpairs()
library(corrplot)     # Visualisation de matrices de corrélation
library(plot3D)       # Graphiques 3D interactifs
library(lattice)      # Graphiques multipanels conditionnels
library(ggpubr)       # Graphiques prêts pour publication avec stats intégrées
library(gridExtra)    # Combiner plusieurs ggplots dans une grille
library(rsample)      # Pour separation data train/test
  
# ==== libray statistiques ====
library(vegan)        # Écologie : diversité, ordinations, PERMANOVA
library(mgcv)         # Modèles additifs généralisés (GAM)
library(gratia)       # Visualisation et diagnostics de GAM (mgcv)
library(ggeffects)    # Effets marginaux et prédictions conditionnelles (ggplot-friendly)
library(DHARMa)       # Diagnostics de résidus pour modèles GLM/GLMM
library(rcompanion)   # Tests stats + pseudo-R²
library(lme4)         # Modèles linéaires et mixtes (GLM/GLMM)
library(predictmeans) # Moyennes ajustées et diagnostics modèles
library(rsq)          # R² pour GLM/GLMM
library(ade4)         # Analyse multivariée : PCA, AFC, etc.
library(FactoMineR)   # PCA, CA, MFA orientés utilisateurs
library(factoextra)   # Visualisation facile des résultats de FactoMineR
library(RVAideMemoire) # Aide analyse canonique
library(PerformanceAnalytics) # Aide analyse canonique
library(adespatial)   # Analyse spatiale (dbMEM, etc.)
library(MASS)         # Fonctions stats classiques (GLM, LDA, etc.)
library(car)          # ANOVA type II/III, VIF, tests linéaires
library(emmeans)      # Moyennes marginales estimées et contrastes post-hoc
library(MuMIn)        # Pour selection de model exhaustive
library(effects)
library(car)
library(DALEX)
library(sensitivity)
library(lhs)              # Latin Hypercube Sampling
# library(pse)              # Propagation d'incertitudes

conflicts()

# ==== resolution library conflicts ====
library(conflicted)
conflict_prefer("filter", "dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("first", "dplyr")
conflict_prefer("step", "stats")
conflict_prefer("explain","DALEX")

#####################################################################################################
# Import data ---------------------------------------------------------------------------------------
#####################################################################################################

df1 = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/df_final/df_LP_NLA.csv",row.names = 1)
df1 = df1 %>% select(-c(lake_id,sech_m,COND_uS.cm,Temperature_bottom,DO_bottom,pH_bottom,
                     rich_sp_zoo,shannon_zoo,op_simpson_zoo,eveness_zoo,
                     ab_tot_bact,rich_order_bact,shannon_bact,op_simpson_bact,eveness_bact))
dim(df1) ; nrow_df_raw = nrow(df1)
# Traitement NA
colSums(is.na(df1))
df1 = na.omit(df1)
print(paste0(abs(nrow(df1) - nrow_df_raw)," row remove on a total of ",nrow_df_raw, 
             " // ",round((nrow_df_raw - nrow(df1)) * 100 / nrow_df_raw, 2), " % remove, ", nrow(df1)," remaining"))

colnames(df1)

#####################################################################################################
# Stats prélim ---------------------------------------------------------------------------------------
#####################################################################################################

# --- 1. Distribution de la variable réponse Y ---------------------------------------------------------------------------
Y = "shannon" # <<<<<<<<<<<<<< changer ici
resp = df1[[Y]]
par(mfrow=c(2,2)) ; boxplot(x = resp, main = paste0("Y : ", Y)) ; dotchart(resp)  ; hist(resp) ; qqnorm(resp) ; qqline(resp)

# --- 2. Distribution des variables explicatives ---------------------------------------------------------------------------
# 2.1 Xs cov
Xs_cov = names(df1[, sapply(df1, is.numeric)]) ; Xs_cov = Xs_cov[Xs_cov != Y]
for (names_cov in Xs_cov){
  par(mfrow=c(2,2)) ; boxplot(x = resp,main=paste0("X : ", names_cov))
  dotchart(df1[[names_cov]]) 
  hist(df1[[names_cov]]) 
  qqnorm(df1[[names_cov]])
  qqline(df1[[names_cov]])
}

# 2.2 Xs factors  
Xs_char = names(df1)[sapply(df1, is.character)]
# transfo character en factor pour stat par la suite
for (char in Xs_char){
  df1[[char]] = as.factor(df1[[char]])
}
Xs_fact = names(df1[, sapply(df1, is.factor), drop = FALSE])
for (names_fact in Xs_fact){
  print(names_fact) 
  print(summary(df1[[names_fact]]))
  cat("\n")
}

# --- 3. Relation Y ~ Xs ---------------------------------------------------------------------------
# 3.1 Xs cov
par(mfrow = c(2,2))
for (names_cov in Xs_cov){
  plot(x = df1[[names_cov]],y = resp,ylab = Y,xlab=paste0(names_cov))
}

# 3.2 Xs factors
par(mfrow = c(1,1))
for (names_fact in Xs_fact){
  boxplot(resp~df1[[names_fact]],ylab = Y,xlab=paste0(names_fact),las=2)
}

 # --- 4. Interactions Xs ---------------------------------------------------------------------------
# 4.1 interaction(cov,cov)
# quartz()
# pairs(df1[, colnames(df1) %in% Xs_cov],cex = 0.2)

# 4.2 interaction(cov,fact)
# for (names_fact in Xs_fact) {
#   print(ggpairs(df1, columns = Xs_cov, 
#                 aes(color = .data[[names_fact]])) +
#           ggtitle(paste("Color by", names_fact)))
# }

# --- 5. Correlation Xs ---------------------------------------------------------------------------
# 5.1 corr(cov,cov)
par(mfrow=c(1,1))
quartz()
cor_mat = cor(df1[, Xs_cov])
corrplot(cor_mat, method = "color", type = "upper",addCoef.col = "black",tl.col = "black",tl.cex = 0.7,number.cex = 0.3)

# 5.2 corr(fact,fact)
# Vérification du plan factoriel croisé complet
if (length(Xs_fact) >= 2) {
  for (i in 1:(length(Xs_fact)-1)) {
    for (j in (i+1):length(Xs_fact)) {
      if (i != j) {
        cat("\n======================================\n")
        cat("Plan factoriel :", Xs_fact[i], "vs", Xs_fact[j], "\n")
        
        # Vérification correcte des classes
        if (class(df1[[Xs_fact[i]]]) == "factor" & class(df1[[Xs_fact[j]]]) == "factor") {
          print(table(df1[[Xs_fact[i]]], df1[[Xs_fact[j]]]))
        }
      }
    }
  }
}
# 5.3 corr(cov,fact)
# quartz()
# if (length(Xs_fact>=1)){
#   par(mfrow = c(length(Xs_cov),length(Xs_fact)))
#   for (i in Xs_cov){
#     for (j in Xs_fact){
#       boxplot(df1[[i]]~df1[[j]], ylab = i, xlab = j)
#     }
#   }
# }

#####################################################################################################
#####################################################################################################
# MODELS ---------------------------------------------------------------------------------------
#####################################################################################################
#####################################################################################################

# lakes avec cook distance trop élevée
df1 = df1[-679,]
df1 = df1[-769,]

# -----------------------------------------------------------------------------------------------
# Rich_genus                                                                             
# -----------------------------------------------------------------------------------------------

# --- 1. Rich_genus  : Model naïf ---------------------------------------------------------------------------

Y = "rich_genus"
resp=df1[[Y]]
Xs = c("lat", "long", "color", "temp_mean","pH_mean","Chla_ug.L","TP_ug.L","TN_ug.L","MG_mg.L","K_mg.L","prev_Mixo")

print(paste0("var = ", round(var(resp),3)," // mean = ",round(mean(resp),3)))

formula = as.formula(paste(Y, "~", paste(Xs, collapse = " + ")))
print(formula)

modNB = glm.nb(formula,
               data = df1)

full_mod = glm.nb(formula, data = df1)
null_mod = glm.nb(as.formula(paste(Y, "~ 1")), data = df1)

# Pas obligé d'éxécuter toutes les types de selection mais utile pour diagnostique pb model
trace_mod = 1 # Verbose (0 : muet, 1 : default)
mod_bwd  = step(full_mod, direction = "backward", trace = trace_mod)
mod_fwd  = step(null_mod, scope = list(lower = null_mod, upper = full_mod), direction = "forward", trace = trace_mod)
mod_both = step(null_mod, scope = list(lower = null_mod, upper = full_mod), direction = "both", trace = trace_mod)

# Comparaison models obtenus
AIC(mod_fwd, mod_bwd, mod_both)
formula(mod_fwd)
formula(mod_bwd)
formula(mod_both)

summary(mod_fwd)
summary(mod_bwd)
summary(mod_both)

Xs = c("lat", "long","temp_mean","Chla_ug.L", # model reduce
       "TP_ug.L","prev_Mixo","MG_mg.L")
formula_red = as.formula(paste(Y, "~", paste(Xs, collapse = " + ")))
print(formula_red)
modNB_reduced = glm.nb(formula_red,
                       data = df1)
vif(modNB_reduced) # <5 pas de problème de colinéarité

# Scale parameter calculation
E1 = resid(modNB_reduced, type = "pearson") # (Y - mu) / sqrt(mu)
N  = nrow(df1)
p  = length(coef(modNB_reduced))
disp_stat = sum(E1^2) / (N - p) # valeur proche de 1 = dispersion correcte

# Use simulations for parameter estimation
par(mfrow=c(1,1))
testDispersion(modNB_reduced) # p> 0.05 : pas de surdispersion détectée
cat("Surdispersion (résidus de Pearson) :", round(disp_stat, 3),
    "\n(n = ", N, ", p = ", p, ")\n")

summary(modNB_reduced)

# Estimate of deviance explained
r2_dev = round((modNB_reduced$null.deviance - modNB_reduced$deviance) / modNB_reduced$null.deviance, 3)
r2s = nagelkerke(modNB_reduced)

cat("===== Résumé du pouvoir explicatif du modèle =====\n",
    "Déviance expliquée :", r2_dev * 100, "%\n",
    "Pseudo-R² de McFadden :", round(r2s$Pseudo.R.squared.for.model.vs.null["McFadden", ], 3), "\n",
    "Pseudo-R² de Cox & Snell :", round(r2s$Pseudo.R.squared.for.model.vs.null["Cox and Snell (ML)", ], 3), "\n",
    "Pseudo-R² de Nagelkerke :", round(r2s$Pseudo.R.squared.for.model.vs.null["Nagelkerke (Cragg and Uhler)", ], 3), "\n")

resid = residuals(modNB_reduced, type="pearson")

par(mfrow=c(2,2)) ; hist(resid,main="") ; qqnorm(resid,xlab='') ; qqline(resid) # Analyse res
plot(resid~fitted(modNB_reduced)) ; abline(h = 0) # residuals vs fitted => tendance ?
plot(resid~ df1$prev_Mixo, main = "") ; abline(h = 0)   # residuals vs cov => tendance ?

simulationOutput = simulateResiduals(fittedModel = modNB_reduced, plot = F)
residuals(simulationOutput)
plot(simulationOutput,main="rich genus naif")
residuals(simulationOutput, quantileFunction = qnorm, outlierValues = c(-7,7))
## QQplot
# KS test : compare la distribution des résidus simulés à une distribution uniforme.
# Dispersion test : Vérifie si la dispersion des résidus est conforme à ce qu’attend le modèle.
# Outlier test (p = 0.01598) : Test de la proportion d’observations extrêmes dans les résidus simulés.

## Res vs predict
# test n.s : Pas de déviation systématique détectée dans la forme de la relation prédite
# astérisque  = quantiles extrême sur/sous rpédits

par(mfrow = c(1, 1))
plot(cooks.distance(modNB_reduced), type = "h",ylim=c(0,1),main="rich genus naif")
abline(h = 1, col = "red", lty = 2)  # seuil classique

## Analyse de sensibilité
# 1. effets marginaux
get_percentiles = function(varname, data, probs = c(0.1, 0.5, 0.9)) {
  q = quantile(data[[varname]], probs = probs, na.rm = TRUE)
  sprintf("%s [%.2f, %.2f, %.2f]", varname, q[1], q[2], q[3])
}

effects = ggpredict(modNB_reduced,
                    terms = c("prev_Mixo",
                              get_percentiles("temp_mean", df1),
                              get_percentiles("long", df1)))

plot(effects)   

# 2. Analyse coefficients
plot(allEffects(modNB_reduced)) 

# 3. Explainer
used_vars = all.vars(formula(modNB_reduced))[-1]  ; df_used = df1[, used_vars]
explainer = explain(modNB_reduced, data = df_used, y = df1$rich_genus)
importance = model_parts(explainer) ; plot(importance) 

# 4. Analyse sensibility avec  Sobol
mod_predict = function(X) {
  X = as.data.frame(X)
  colnames(X) = Xs
  as.numeric(predict(modNB_reduced, newdata = X, type = "response"))
}
X_bounds = apply(df1[, Xs], 2, range)

n = 10000 # nb simulations

X1 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
X2 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
colnames(X1) = colnames(X2) = Xs

for (i in seq_along(Xs)) {
  X1[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
  X2[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
}
sobol_result = sobol2007(model = mod_predict, X1 = X1, X2 = X2, nboot = 100)
print(sobol_result)
plot(sobol_result)

# --- 2. Rich_genus : model corrigé ---------------------------------------------------------------------------

Y = "rich_genus_no_cyano"
resp=df1[[Y]]
Xs = c("lat", "long", "color", "temp_mean","pH_mean","Chla_ug.L","TP_ug.L","TN_ug.L","MG_mg.L","K_mg.L","prev_Mixo")

print(paste0("var = ", round(var(resp),3)," // mean = ",round(mean(resp),3)))

formula = as.formula(paste(Y, "~", paste(Xs, collapse = " + ")))
print(formula)

modNB = glm.nb(formula,
               data = df1)

full_mod = glm.nb(formula, data = df1)
null_mod = glm.nb(as.formula(paste(Y, "~ 1")), data = df1)

# Pas obligé d'éxécuter toutes les types de selection mais utile pour diagnostique pb model
trace_mod = 1 # Verbose (0 : muet, 1 : default)
mod_bwd  = step(full_mod, direction = "backward", trace = trace_mod)
mod_fwd  = step(null_mod, scope = list(lower = null_mod, upper = full_mod), direction = "forward", trace = trace_mod)
mod_both = step(null_mod, scope = list(lower = null_mod, upper = full_mod), direction = "both", trace = trace_mod)

# Comparaison models obtenus
AIC(mod_fwd, mod_bwd, mod_both)
formula(mod_fwd)
formula(mod_bwd)
formula(mod_both)

summary(mod_fwd)
summary(mod_bwd)
summary(mod_both)

Xs = c("lat", "long","temp_mean","Chla_ug.L", # model reduce
       "TP_ug.L","TN_ug.L","prev_Mixo","MG_mg.L","pH_mean")
formula_red = as.formula(paste(Y, "~", paste(Xs, collapse = " + ")))
print(formula_red)
modNB_reduced = glm.nb(formula_red,
                       data = df1)
vif(modNB_reduced) # <5 pas de problème de colinéarité

# Scale parameter calculation
E1 = resid(modNB_reduced, type = "pearson") # (Y - mu) / sqrt(mu)
N  = nrow(df1)
p  = length(coef(modNB_reduced))
disp_stat = sum(E1^2) / (N - p) # valeur proche de 1 = dispersion correcte

# Use simulations for parameter estimation
par(mfrow=c(1,1))
testDispersion(modNB_reduced) # p> 0.05 : pas de surdispersion détectée
cat("Surdispersion (résidus de Pearson) :", round(disp_stat, 3),
    "\n(n = ", N, ", p = ", p, ")\n")

summary(modNB_reduced)

# Estimate of deviance explained
r2_dev = round((modNB_reduced$null.deviance - modNB_reduced$deviance) / modNB_reduced$null.deviance, 3)
r2s = nagelkerke(modNB_reduced)

cat("===== Résumé du pouvoir explicatif du modèle =====\n",
    "Déviance expliquée :", r2_dev * 100, "%\n",
    "Pseudo-R² de McFadden :", round(r2s$Pseudo.R.squared.for.model.vs.null["McFadden", ], 3), "\n",
    "Pseudo-R² de Cox & Snell :", round(r2s$Pseudo.R.squared.for.model.vs.null["Cox and Snell (ML)", ], 3), "\n",
    "Pseudo-R² de Nagelkerke :", round(r2s$Pseudo.R.squared.for.model.vs.null["Nagelkerke (Cragg and Uhler)", ], 3), "\n")

resid = residuals(modNB_reduced, type="pearson")

par(mfrow=c(2,2)) ; hist(resid,main="") ; qqnorm(resid,xlab='') ; qqline(resid) # Analyse res
plot(resid~fitted(modNB_reduced)) ; abline(h = 0) # residuals vs fitted => tendance ?
plot(resid~ df1$prev_Mixo, main = "") ; abline(h = 0)   # residuals vs cov => tendance ?

simulationOutput = simulateResiduals(fittedModel = modNB_reduced, plot = F)
residuals(simulationOutput)
plot(simulationOutput,main="rich genus no cyano")
residuals(simulationOutput, quantileFunction = qnorm, outlierValues = c(-7,7))
## QQplot
# KS test : compare la distribution des résidus simulés à une distribution uniforme.
# Dispersion test : Vérifie si la dispersion des résidus est conforme à ce qu’attend le modèle.
# Outlier test (p = 0.01598) : Test de la proportion d’observations extrêmes dans les résidus simulés.

## Res vs predict
# test n.s : Pas de déviation systématique détectée dans la forme de la relation prédite
# astérisque  = quantiles extrême sur/sous rpédits

par(mfrow = c(1, 1))
plot(cooks.distance(modNB_reduced), type = "h",ylim=c(0,1))
abline(h = 1, col = "red", lty = 2,main="rich genus no cyano")  # seuil classique
which(cooks.distance(modNB_reduced)>0.7)

## Analyse de sensibilité
# 1. effets marginaux
effects = ggpredict(modNB_reduced, terms = c("prev_Mixo", "TP_ug.L","Chla_ug.L ")) ; plot(effects)

# 2. Analyse coefficients
plot(allEffects(modNB_reduced)) 

# 3. Explainer
used_vars = all.vars(formula(modNB_reduced))[-1]  ; df_used = df1[, used_vars]
explainer = explain(modNB_reduced, data = df_used, y = df1$rich_genus_no_cyano)
importance = model_parts(explainer) ; plot(importance) 

# 4. Analyse sensibility avec  Sobol
mod_predict = function(X) {
  X = as.data.frame(X)
  colnames(X) = Xs
  as.numeric(predict(modNB_reduced, newdata = X, type = "response"))
}
X_bounds = apply(df1[, Xs], 2, range)

n = 10000 # nb simulations

X1 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
X2 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
colnames(X1) = colnames(X2) = Xs

for (i in seq_along(Xs)) {
  X1[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
  X2[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
}
sobol_result = sobol2007(model = mod_predict, X1 = X1, X2 = X2, nboot = 100)
print(sobol_result)
plot(sobol_result)

# --- 3. Rich_genus : Model ajusté ---------------------------------------------------------------------------

Y = "rich_genus"
resp=df1[[Y]]
Xs = c("lat", "long", "color", "temp_mean","pH_mean","Chla_ug.L","TP_ug.L","TN_ug.L","MG_mg.L","K_mg.L","prev_Mixo","prev_Cyano")

print(paste0("var = ", round(var(resp),3)," // mean = ",round(mean(resp),3)))

formula = as.formula(paste(Y, "~", paste(Xs, collapse = " + ")))
print(formula)

modNB = glm.nb(formula,
               data = df1)

full_mod = glm.nb(formula, data = df1)
null_mod = glm.nb(as.formula(paste(Y, "~ 1")), data = df1)

# Pas obligé d'éxécuter toutes les types de selection mais utile pour diagnostique pb model
trace_mod = 1 # Verbose (0 : muet, 1 : default)
mod_bwd  = step(full_mod, direction = "backward", trace = trace_mod)
mod_fwd  = step(null_mod, scope = list(lower = null_mod, upper = full_mod), direction = "forward", trace = trace_mod)
mod_both = step(null_mod, scope = list(lower = null_mod, upper = full_mod), direction = "both", trace = trace_mod)

# Comparaison models obtenus
AIC(mod_fwd, mod_bwd, mod_both)
formula(mod_fwd)
formula(mod_bwd)
formula(mod_both)

summary(mod_fwd)
summary(mod_bwd)
summary(mod_both)

Xs = c("lat", "long","temp_mean","Chla_ug.L", # model reduce
       "TP_ug.L","prev_Mixo","prev_Cyano","MG_mg.L")
formula_red = as.formula(paste(Y, "~", paste(Xs, collapse = " + ")))
print(formula_red)
modNB_reduced = glm.nb(formula_red,
                       data = df1)
vif(modNB_reduced) # <5 pas de problème de colinéarité

# Scale parameter calculation
E1 = resid(modNB_reduced, type = "pearson") # (Y - mu) / sqrt(mu)
N  = nrow(df1)
p  = length(coef(modNB_reduced))
disp_stat = sum(E1^2) / (N - p) # valeur proche de 1 = dispersion correcte

# Use simulations for parameter estimation
par(mfrow=c(1,1))
testDispersion(modNB_reduced) # p> 0.05 : pas de surdispersion détectée
cat("Surdispersion (résidus de Pearson) :", round(disp_stat, 3),
    "\n(n = ", N, ", p = ", p, ")\n")

summary(modNB_reduced)

# Estimate of deviance explained
r2_dev = round((modNB_reduced$null.deviance - modNB_reduced$deviance) / modNB_reduced$null.deviance, 3)
r2s = nagelkerke(modNB_reduced)

cat("===== Résumé du pouvoir explicatif du modèle =====\n",
    "Déviance expliquée :", r2_dev * 100, "%\n",
    "Pseudo-R² de McFadden :", round(r2s$Pseudo.R.squared.for.model.vs.null["McFadden", ], 3), "\n",
    "Pseudo-R² de Cox & Snell :", round(r2s$Pseudo.R.squared.for.model.vs.null["Cox and Snell (ML)", ], 3), "\n",
    "Pseudo-R² de Nagelkerke :", round(r2s$Pseudo.R.squared.for.model.vs.null["Nagelkerke (Cragg and Uhler)", ], 3), "\n")

resid = residuals(modNB_reduced, type="pearson")

par(mfrow=c(2,2)) ; hist(resid,main="") ; qqnorm(resid,xlab='') ; qqline(resid) # Analyse res
plot(resid~fitted(modNB_reduced)) ; abline(h = 0) # residuals vs fitted => tendance ?
plot(resid~ df1$prev_Mixo, main = "") ; abline(h = 0)   # residuals vs cov => tendance ?

simulationOutput = simulateResiduals(fittedModel = modNB_reduced, plot = F)
residuals(simulationOutput)
plot(simulationOutput)
residuals(simulationOutput, quantileFunction = qnorm, outlierValues = c(-7,7))
## QQplot
# KS test : compare la distribution des résidus simulés à une distribution uniforme.
# Dispersion test : Vérifie si la dispersion des résidus est conforme à ce qu’attend le modèle.
# Outlier test (p = 0.01598) : Test de la proportion d’observations extrêmes dans les résidus simulés.

## Res vs predict
# test n.s : Pas de déviation systématique détectée dans la forme de la relation prédite
# astérisque  = quantiles extrême sur/sous rpédits

par(mfrow = c(1, 1))
plot(cooks.distance(modNB_reduced), type = "h",ylim=c(0,1))
abline(h = 1, col = "red", lty = 2)  # seuil classique

## Analyse de sensibilité
# 1. effets marginaux
effects = ggpredict(modNB_reduced, terms = c("prev_Mixo", "TP_ug.L","Chla_ug.L ")) ; plot(effects)

# 2. Analyse coefficients
plot(allEffects(modNB_reduced)) 

# 3. Explainer
used_vars = all.vars(formula(modNB_reduced))[-1]  ; df_used = df1[, used_vars]
explainer = explain(modNB_reduced, data = df_used, y = df1$rich_genus)
importance = model_parts(explainer) ; plot(importance) 

# 4. Analyse sensibility avec  Sobol
mod_predict = function(X) {
  X = as.data.frame(X)
  colnames(X) = Xs
  as.numeric(predict(modNB_reduced, newdata = X, type = "response"))
}
X_bounds = apply(df1[, Xs], 2, range)

n = 10000 # nb simulations

X1 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
X2 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
colnames(X1) = colnames(X2) = Xs

for (i in seq_along(Xs)) {
  X1[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
  X2[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
}
sobol_result = sobol2007(model = mod_predict, X1 = X1, X2 = X2, nboot = 100)
print(sobol_result)
plot(sobol_result)

# -----------------------------------------------------------------------------------------------
# Shannon                                                                             
# -----------------------------------------------------------------------------------------------

# --- 1. Shannon : Model naïf ---------------------------------------------------------------------------

Y = "shannon"
colnames(df1)
Xs_lin = c("long","lat", "color", "temp_mean","pH_mean","Chla_ug.L","TP_ug.L","TN_ug.L","MG_mg.L","K_mg.L")
Xs_nolin = c("prev_Mixo")

formula_gam = as.formula(
  paste(Y, "~",paste(c(Xs_lin, paste0("s(", Xs_nolin, ")")), collapse = " + ")))
print(formula_gam)

modGAM = gam(formula_gam, data = df1, family = gaussian())

modGAM = gam(formula_gam, data = df1, family = gaussian(), select = TRUE) # /!\ select = T != step !! il n'enleve que les effets qui n'apporte aucune info
summary(modGAM) # repérer VA significatives

source("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/CODE/OK_Backward_gam_selection.R")

modGAM_reduced = backward_gam_selection(df1, Y, Xs_lin, Xs_nolin)
summary(modGAM_reduced)

AIC(modGAM, modGAM_reduced)

#### /!\ Attention si famille != gaussian (en fonction du type de données), adaptez la suite en conséquence
#### => plus de r2 (déviance à calculer (rf glimNB ou glimPoiss)) et hypothèse  à vérifier différentes

summary(modGAM_reduced)
summary(modGAM_reduced)$r.sq
modGAM_reduced$deviance

plot(modGAM_reduced,page=1,residuals = T)

# Validation model
par(mfrow=c(2,2))
hist(residuals(modGAM_reduced), main="")
qqnorm(residuals(modGAM_reduced))
qqline(residuals(modGAM_reduced))

plot(residuals(modGAM_reduced) ~ fitted(modGAM_reduced)) ; abline(h = 0) # res vs fit
plot(residuals(modGAM_reduced) ~ df1$prev_Mixo) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl

par(mfrow = c(1, 1))
plot(cooks.distance(modGAM_reduced), type = "h", ylim = c(0, 1))
abline(h = 1, col = 2, lwd = 3)

sm = summary(modGAM_reduced)
cat("\n==============================================\n",
    "Formula : ", as.character(sm$formula),"\n",
    "r2 Adj : ", as.character(round(sm$r.sq,2))," (part de variance exliquée)","\n",
    "Deviance :", as.character(round(sm$dev.expl,2)),"(part de variabilité exliquée)","\n"
    )

## Analyse de sensibilité
Xs = all.vars(formula(modGAM_reduced))[-1]
# 1. effets marginaux
effects = ggpredict(modGAM_reduced, terms = c("prev_Mixo", "TP_ug.L","lat")) ; plot(effects)

# 2. Analyse coefficients
plots_list = list()
for (X in Xs){
  pred_var = ggpredict(modGAM_reduced, terms = X)
  p = plot(pred_var) + ggtitle(X)
  plots_list[[X]] = p
}
do.call(grid.arrange, c(plots_list, ncol = 2))

# 3. Explainer
used_vars = all.vars(formula(modGAM_reduced))[-1]  ; df_used = df1[, used_vars]
explainer = explain(modGAM_reduced, data = df_used, y = df1$shannon)
importance = model_parts(explainer) ; plot(importance) 

# 4. Analyse sensibility avec  Sobol
mod_predict = function(X) {
  X = as.data.frame(X)
  colnames(X) = Xs
  as.numeric(predict(modGAM_reduced, newdata = X, type = "response"))
}
X_bounds = apply(df1[, Xs], 2, range)

n = 10000 # nb simulations

X1 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
X2 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
colnames(X1) = colnames(X2) = Xs

for (i in seq_along(Xs)) {
  X1[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
  X2[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
}
sobol_result = sobol2007(model = mod_predict, X1 = X1, X2 = X2, nboot = 100)
print(sobol_result)
plot(sobol_result)

# --- 2. Shannon : model corrigé ---------------------------------------------------------------------------

Y = "shannon_no_cyano"
colnames(df1)
Xs_lin = c("long","lat", "color", "temp_mean","pH_mean","Chla_ug.L","TP_ug.L","TN_ug.L","MG_mg.L","K_mg.L")
Xs_nolin = c("prev_Mixo")

formula_gam = as.formula(
  paste(Y, "~",paste(c(Xs_lin, paste0("s(", Xs_nolin, ")")), collapse = " + ")))
print(formula_gam)

modGAM = gam(formula_gam, data = df1, family = gaussian())

modGAM = gam(formula_gam, data = df1, family = gaussian(), select = TRUE) # /!\ select = T != step !! il n'enleve que les effets qui n'apporte aucune info
summary(modGAM) # repérer VA significatives

source("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/CODE/OK_Backward_gam_selection.R")

modGAM_reduced = backward_gam_selection(df1, Y, Xs_lin, Xs_nolin)
summary(modGAM_reduced)

AIC(modGAM, modGAM_reduced)

#### /!\ Attention si famille != gaussian (en fonction du type de données), adaptez la suite en conséquence
#### => plus de r2 (déviance à calculer (rf glimNB ou glimPoiss)) et hypothèse  à vérifier différentes

summary(modGAM_reduced)
summary(modGAM_reduced)$r.sq
modGAM_reduced$deviance

plot(modGAM_reduced,page=1,residuals = T)

# Validation model
par(mfrow=c(2,2))
hist(residuals(modGAM_reduced), main="")
qqnorm(residuals(modGAM_reduced))
qqline(residuals(modGAM_reduced))

plot(residuals(modGAM_reduced) ~ fitted(modGAM_reduced)) ; abline(h = 0) # res vs fit
plot(residuals(modGAM_reduced) ~ df1$prev_Mixo) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl

par(mfrow = c(1, 1))
plot(cooks.distance(modGAM_reduced), type = "h", ylim = c(0, 1))
abline(h = 1, col = 2, lwd = 3)

sm = summary(modGAM_reduced)
cat("\n==============================================\n",
    "Formula : ", as.character(sm$formula),"\n",
    "r2 Adj : ", as.character(round(sm$r.sq,2))," (part de variance exliquée)","\n",
    "Deviance :", as.character(round(sm$dev.expl,2)),"(part de variabilité exliquée)","\n"
)

## Analyse de sensibilité
Xs = all.vars(formula(modGAM_reduced))[-1]
# 1. effets marginaux
effects = ggpredict(modGAM_reduced, terms = c("prev_Mixo", "TP_ug.L","lat")) ; plot(effects)

# 2. Analyse coefficients
plots_list = list()
for (X in Xs){
  pred_var = ggpredict(modGAM_reduced, terms = X)
  p = plot(pred_var) + ggtitle(X)
  plots_list[[X]] = p
}
do.call(grid.arrange, c(plots_list, ncol = 2))

# 3. Explainer
used_vars = all.vars(formula(modGAM_reduced))[-1]  ; df_used = df1[, used_vars]
explainer = explain(modGAM_reduced, data = df_used, y = df1$shannon_no_cyano)
importance = model_parts(explainer) ; plot(importance) 

# 4. Analyse sensibility avec  Sobol
mod_predict = function(X) {
  X = as.data.frame(X)
  colnames(X) = Xs
  as.numeric(predict(modGAM_reduced, newdata = X, type = "response"))
}
X_bounds = apply(df1[, Xs], 2, range)

n = 10000 # nb simulations

X1 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
X2 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
colnames(X1) = colnames(X2) = Xs

for (i in seq_along(Xs)) {
  X1[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
  X2[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
}
sobol_result = sobol2007(model = mod_predict, X1 = X1, X2 = X2, nboot = 100)
print(sobol_result)
plot(sobol_result)


# --- 3. Shannon : model ajusté ---------------------------------------------------------------------------
Y = "shannon"
colnames(df1)
Xs_lin = c("long","lat", "color", "temp_mean","pH_mean","Chla_ug.L","TP_ug.L","TN_ug.L","MG_mg.L","K_mg.L")
Xs_nolin = c("prev_Mixo","prev_Cyano")

Xs = c(Xs_lin,Xs_nolin)

# quelques verifications
cor(df1$prev_Mixo, df1$prev_Cyano)
cor(df1$prev_Mixo, df1$prev_Cyano, method = "spearman") # test correlation non lineaire

formula_check = as.formula(paste(Y, "~",paste(c(Xs_lin, paste0(Xs_nolin)), collapse = " + ")))
mod_check = lm(formula_check,data=df1)
vif(mod_check)

formula_gam = as.formula(
  paste(Y, "~",paste(c(Xs_lin, paste0("s(", Xs_nolin, ")")), collapse = " + ")))
print(formula_gam)

modGAM = gam(formula_gam, data = df1, family = gaussian())

modGAM = gam(formula_gam, data = df1, family = gaussian(), select = TRUE) # /!\ select = T != step !! il n'enleve que les effets qui n'apporte aucune info
summary(modGAM) # repérer VA significatives

source("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/CODE/OK_Backward_gam_selection.R")

modGAM_reduced = backward_gam_selection(df1, Y, Xs_lin, Xs_nolin)
summary(modGAM_reduced)

AIC(modGAM, modGAM_reduced)

#### /!\ Attention si famille != gaussian (en fonction du type de données), adaptez la suite en conséquence
#### => plus de r2 (déviance à calculer (rf glimNB ou glimPoiss)) et hypothèse  à vérifier différentes

summary(modGAM_reduced)
summary(modGAM_reduced)$r.sq
modGAM_reduced$deviance

plot(modGAM_reduced,page=1,residuals = T)

# Validation model
par(mfrow=c(2,2))
hist(residuals(modGAM_reduced), main="")
qqnorm(residuals(modGAM_reduced))
qqline(residuals(modGAM_reduced))

plot(residuals(modGAM_reduced) ~ fitted(modGAM_reduced)) ; abline(h = 0) # res vs fit
plot(residuals(modGAM_reduced) ~ df1$prev_Mixo) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl

par(mfrow = c(1, 1))
plot(cooks.distance(modGAM_reduced), type = "h", ylim = c(0, 1))
abline(h = 1, col = 2, lwd = 3)

sm = summary(modGAM_reduced)
cat("\n==============================================\n",
    "Formula : ", as.character(sm$formula),"\n",
    "r2 Adj : ", as.character(round(sm$r.sq,2))," (part de variance exliquée)","\n",
    "Deviance :", as.character(round(sm$dev.expl,2)),"(part de variabilité exliquée)","\n"
)

## Analyse de sensibilité
Xs = all.vars(formula(modGAM_reduced))[-1]
# 1. effets marginaux
effects = ggpredict(modGAM_reduced, terms = c("prev_Mixo", "TP_ug.L","lat")) ; plot(effects)

# 2. Analyse coefficients
plots_list = list()
for (X in Xs){
  pred_var = ggpredict(modGAM_reduced, terms = X)
  p = plot(pred_var) + ggtitle(X)
  plots_list[[X]] = p
}
do.call(grid.arrange, c(plots_list, ncol = 2))

# 3. Explainer
used_vars = all.vars(formula(modGAM_reduced))[-1]  ; df_used = df1[, used_vars]
explainer = explain(modGAM_reduced, data = df_used, y = df1$shannon)
importance = model_parts(explainer) ; plot(importance) 

# 4. Analyse sensibility avec  Sobol
mod_predict = function(X) {
  X = as.data.frame(X)
  colnames(X) = Xs
  as.numeric(predict(modGAM_reduced, newdata = X, type = "response"))
}
X_bounds = apply(df1[, Xs], 2, range)

n = 10000 # nb simulations

X1 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
X2 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
colnames(X1) = colnames(X2) = Xs

for (i in seq_along(Xs)) {
  X1[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
  X2[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
}
sobol_result = sobol2007(model = mod_predict, X1 = X1, X2 = X2, nboot = 100)
print(sobol_result)
plot(sobol_result)


# -----------------------------------------------------------------------------------------------
# Eveness                                                                             
# -----------------------------------------------------------------------------------------------

# --- 1. Eveness : Model naïf ---------------------------------------------------------------------------
Y = "eveness_piel"
colnames(df1)
Xs_lin = c("long","lat", "color", "temp_mean","pH_mean","Chla_ug.L","TP_ug.L","TN_ug.L","MG_mg.L","K_mg.L")
Xs_nolin = c("prev_Mixo")

formula_gam = as.formula(
  paste(Y, "~",paste(c(Xs_lin, paste0("s(", Xs_nolin, ")")), collapse = " + ")))
print(formula_gam)

df1$eveness_piel[df1$eveness_piel == 0] = 0.001
df1$eveness_piel[df1$eveness_piel == 1] = 0.999

modGAM = gam(formula_gam, data = df1, family = betar(link = "logit"), select = TRUE)
summary(modGAM) # repérer VA significatives

source("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/CODE/OK_Backward_gam_selection.R")

modGAM_reduced = backward_gam_selection(df1, Y, Xs_lin, Xs_nolin, family = betar(link = "logit"))
summary(modGAM_reduced)

AIC(modGAM, modGAM_reduced)

#### /!\ Attention si famille != gaussian (en fonction du type de données), adaptez la suite en conséquence
#### => plus de r2 (déviance à calculer (rf glimNB ou glimPoiss)) et hypothèse  à vérifier différentes

summary(modGAM_reduced)

modGAM_reduced$deviance

plot(modGAM_reduced,page=1,residuals = F)

# Validation model
par(mfrow=c(2,2))
hist(residuals(modGAM_reduced), main="")
qqnorm(residuals(modGAM_reduced))
qqline(residuals(modGAM_reduced))

plot(residuals(modGAM_reduced) ~ fitted(modGAM_reduced)) ; abline(h = 0) # res vs fit
plot(residuals(modGAM_reduced) ~ df1$prev_Mixo) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl

par(mfrow = c(1, 1))
plot(cooks.distance(modGAM_reduced), type = "h", ylim = c(0, 1))
abline(h = 1, col = 2, lwd = 3)

sm = summary(modGAM_reduced)
cat("\n==============================================\n",
    "Formula : ", as.character(sm$formula),"\n",
    "r2 Adj : ", as.character(round(sm$r.sq,2))," (part de variance exliquée)","\n",
    "Deviance :", as.character(round(sm$dev.expl,2)),"(part de variabilité exliquée)","\n"
)

## Analyse de sensibilité
Xs = all.vars(formula(modGAM_reduced))[-1]
# 1. effets marginaux
effects = ggpredict(modGAM_reduced, terms = c("long", "color","prev_Mixo")) ; plot(effects)

# 2. Analyse coefficients
plots_list = list()
for (X in Xs){
  pred_var = ggpredict(modGAM_reduced, terms = X)
  p = plot(pred_var) + ggtitle(X)
  plots_list[[X]] = p
}
do.call(grid.arrange, c(plots_list, ncol = 2))

# 3. Explainer
used_vars = all.vars(formula(modGAM_reduced))[-1]  ; df_used = df1[, used_vars]
explainer = explain(modGAM_reduced, data = df_used, y = df1$eveness_piel)
importance = model_parts(explainer) ; plot(importance) 

# 4. Analyse sensibility avec  Sobol
mod_predict = function(X) {
  X = as.data.frame(X)
  colnames(X) = Xs
  as.numeric(predict(modGAM_reduced, newdata = X, type = "response"))
}
X_bounds = apply(df1[, Xs], 2, range)

n = 90000 # nb simulations

X1 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
X2 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
colnames(X1) = colnames(X2) = Xs

for (i in seq_along(Xs)) {
  X1[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
  X2[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
}
sobol_result = sobol2007(model = mod_predict, X1 = X1, X2 = X2, nboot = 100)
print(sobol_result)
plot(sobol_result)


# --- 2. Eveness : model corrigé ---------------------------------------------------------------------------

Y = "eveness_piel_no_cyano"
colnames(df1)
Xs_lin = c("long","lat", "color", "temp_mean","pH_mean","Chla_ug.L","TP_ug.L","TN_ug.L","MG_mg.L","K_mg.L")
Xs_nolin = c("prev_Mixo")

formula_gam = as.formula(
  paste(Y, "~",paste(c(Xs_lin, paste0("s(", Xs_nolin, ")")), collapse = " + ")))
print(formula_gam)

df1$eveness_piel[df1$eveness_piel == 0] = 0.001
df1$eveness_piel[df1$eveness_piel == 1] = 0.999

modGAM = gam(formula_gam, data = df1, family = betar(link = "logit"), select = TRUE)
summary(modGAM) # repérer VA significatives

source("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/CODE/OK_Backward_gam_selection.R")

modGAM_reduced = backward_gam_selection(df1, Y, Xs_lin, Xs_nolin, family = betar(link = "logit"))
summary(modGAM_reduced)

AIC(modGAM, modGAM_reduced)

#### /!\ Attention si famille != gaussian (en fonction du type de données), adaptez la suite en conséquence
#### => plus de r2 (déviance à calculer (rf glimNB ou glimPoiss)) et hypothèse  à vérifier différentes

summary(modGAM_reduced)

modGAM_reduced$deviance

plot(modGAM_reduced,page=1,residuals = F)

# Validation model
par(mfrow=c(2,2))
hist(residuals(modGAM_reduced), main="")
qqnorm(residuals(modGAM_reduced))
qqline(residuals(modGAM_reduced))

plot(residuals(modGAM_reduced) ~ fitted(modGAM_reduced)) ; abline(h = 0) # res vs fit
plot(residuals(modGAM_reduced) ~ df1$prev_Mixo) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl

par(mfrow = c(1, 1))
plot(cooks.distance(modGAM_reduced), type = "h", ylim = c(0, 1))
abline(h = 1, col = 2, lwd = 3)

sm = summary(modGAM_reduced)
cat("\n==============================================\n",
    "Formula : ", as.character(sm$formula),"\n",
    "r2 Adj : ", as.character(round(sm$r.sq,2))," (part de variance exliquée)","\n",
    "Deviance :", as.character(round(sm$dev.expl,2)),"(part de variabilité exliquée)","\n"
)

## Analyse de sensibilité
Xs = all.vars(formula(modGAM_reduced))[-1]
# 1. effets marginaux
effects = ggpredict(modGAM_reduced, terms = c("prev_Mixo", "TP_ug.L","lat")) ; plot(effects)

# 2. Analyse coefficients
plots_list = list()
for (X in Xs){
  pred_var = ggpredict(modGAM_reduced, terms = X)
  p = plot(pred_var) + ggtitle(X)
  plots_list[[X]] = p
}
do.call(grid.arrange, c(plots_list, ncol = 2))

# 3. Explainer
used_vars = all.vars(formula(modGAM_reduced))[-1]  ; df_used = df1[, used_vars]
explainer = explain(modGAM_reduced, data = df_used, y = df1$eveness_piel_no_cyano)
importance = model_parts(explainer) ; plot(importance) 

# 4. Analyse sensibility avec  Sobol
mod_predict = function(X) {
  X = as.data.frame(X)
  colnames(X) = Xs
  as.numeric(predict(modGAM_reduced, newdata = X, type = "response"))
}
X_bounds = apply(df1[, Xs], 2, range)

n = 90000 # nb simulations

X1 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
X2 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
colnames(X1) = colnames(X2) = Xs

for (i in seq_along(Xs)) {
  X1[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
  X2[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
}
sobol_result = sobol2007(model = mod_predict, X1 = X1, X2 = X2, nboot = 100)
print(sobol_result)
plot(sobol_result)

# --- 3. Eveness : model ajusté ---------------------------------------------------------------------------

Y = "eveness_piel"
colnames(df1)
Xs_lin = c("long","lat", "color", "temp_mean","pH_mean","Chla_ug.L","TP_ug.L","TN_ug.L","MG_mg.L","K_mg.L")
Xs_nolin = c("prev_Mixo","prev_Cyano")

formula_gam = as.formula(
  paste(Y, "~",paste(c(Xs_lin, paste0("s(", Xs_nolin, ")")), collapse = " + ")))
print(formula_gam)

df1$eveness_piel[df1$eveness_piel == 0] = 0.001
df1$eveness_piel[df1$eveness_piel == 1] = 0.999

modGAM = gam(formula_gam, data = df1, family = betar(link = "logit"), select = TRUE)
summary(modGAM) # repérer VA significatives

source("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/CODE/OK_Backward_gam_selection.R")

modGAM_reduced = backward_gam_selection(df1, Y, Xs_lin, Xs_nolin, family = betar(link = "logit"))
summary(modGAM_reduced)

AIC(modGAM, modGAM_reduced)

#### /!\ Attention si famille != gaussian (en fonction du type de données), adaptez la suite en conséquence
#### => plus de r2 (déviance à calculer (rf glimNB ou glimPoiss)) et hypothèse  à vérifier différentes

summary(modGAM_reduced)

modGAM_reduced$deviance

plot(modGAM_reduced,page=1,residuals = F)

# Validation model
par(mfrow=c(2,2))
hist(residuals(modGAM_reduced), main="")
qqnorm(residuals(modGAM_reduced))
qqline(residuals(modGAM_reduced))

plot(residuals(modGAM_reduced) ~ fitted(modGAM_reduced)) ; abline(h = 0) # res vs fit
plot(residuals(modGAM_reduced) ~ df1$prev_Mixo) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl
# plot(residuals(modGAM_reduced) ~ df1$) ; abline(h = 0) # res vs expl

par(mfrow = c(1, 1))
plot(cooks.distance(modGAM_reduced), type = "h", ylim = c(0, 1))
abline(h = 1, col = 2, lwd = 3)

sm = summary(modGAM_reduced)
cat("\n==============================================\n",
    "Formula : ", as.character(sm$formula),"\n",
    "r2 Adj : ", as.character(round(sm$r.sq,2))," (part de variance exliquée)","\n",
    "Deviance :", as.character(round(sm$dev.expl,2)),"(part de variabilité exliquée)","\n"
)


## Analyse de sensibilité
Xs = all.vars(formula(modGAM_reduced))[-1]
# 1. effets marginaux
effects = ggpredict(modGAM_reduced, terms = c("prev_Mixo", "prev_Cyano","temp_mean")) ; plot(effects)

# 2. Analyse coefficients
plots_list = list()
for (X in Xs){
  pred_var = ggpredict(modGAM_reduced, terms = X)
  p = plot(pred_var) + ggtitle(X)
  plots_list[[X]] = p
}
do.call(grid.arrange, c(plots_list, ncol = 2))

# 3. Explainer
used_vars = all.vars(formula(modGAM_reduced))[-1]  ; df_used = df1[, used_vars]
explainer = explain(modGAM_reduced, data = df_used, y = df1$eveness_piel)
importance = model_parts(explainer) ; plot(importance) 

# 4. Analyse sensibility avec  Sobol
mod_predict = function(X) {
  X = as.data.frame(X)
  colnames(X) = Xs
  as.numeric(predict(modGAM_reduced, newdata = X, type = "response"))
}
X_bounds = apply(df1[, Xs], 2, range)

n = 100000 # nb simulations

X1 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
X2 = data.frame(matrix(NA, n, length(Xs))) # Matrices d'entrée
colnames(X1) = colnames(X2) = Xs

for (i in seq_along(Xs)) {
  X1[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
  X2[, i] = runif(n, min = X_bounds[1, i], max = X_bounds[2, i])
}
sobol_result = sobol2007(model = mod_predict, X1 = X1, X2 = X2, nboot = 100)
print(sobol_result)
plot(sobol_result)

