# Structure des données - NEW_ALL

## Répertoires

### DATA_MODEL/
Contient tous les fichiers CSV de données traités pour la modélisation :
- ELA_py.csv : Données ELA preprocessées pour Python
- LP_NLA_py.csv : Données LP-NLA preprocessées pour Python
- df_IISD_ELA.csv : Données brutes ELA
- df_LP_NLA.csv : Données brutes LP-NLA
- strat.csv : Données de stratification
TOUT LES AUTRES CSV CREES DEVRONT ETRE DANS CE DOSSIER

### FIGURES/
Contient toutes les figures générées :
- ELA_model/ : Figures spécifiques aux modèles ELA
  - SHAP_rich_genus_no_cyano_ELA.png 1
  - SHAP_shannon_no_cyano_ELA.png 2
  - SHAP_eveness_no_cyano_ELA.png 3
  - SHAP_all_ELA.png 4
  - res_ELA.png 5
  - PACF_combined_all_metrics.png 6 

- LPNLA_model/ : Figures spécifiques aux modèles LP-NLA
  - SHAP_rich_genus_no_cyano_LPNLA.png 7
  - SHAP_shannon_no_cyano_LPNLA.png 8
  - SHAP_eveness_no_cyano_LPNLA.png 9 
  - SHAP_all_LPNLA.png 10
  - res_LPNLA.png 11

- Figures combinées :
  - SHAP_rank_combined.png 12
  - SHAP_interactions_prev_mixo_heatmap.png 13
  - shap_table_LateX.tex 14

### CODE/
Contient tous les scripts de traitement et modélisation :
- model_final_complete.py : Script principal de modélisation XGBoost + SHAP => Création des figures 1,2,3,4,5,7,8,9,10,11,12,13,14
- pacf_analysis.R : Analyse PACF pour données ELA => Création de la figure 6

## Variables analysées

### Métriques de diversité (variables de réponse) :
- rich_genus_no_cyano : Richesse générique (sans cyanobactéries)
- shannon_no_cyano : Indice de Shannon (sans cyanobactéries)
- eveness_piel_no_cyano : Équitabilité de Pielou (sans cyanobactéries)

### Variables explicatives :

#### ELA (données temporelles) :
- year, doy : Variables temporelles
- lake_id : Identifiant du lac (variable catégorielle)
- COND_uS.cm : Conductivité
- Chla_ug.L : Chlorophylle a
- TNTP_mg.L : Ratio azote total/phosphore total
- pH_mean : pH moyen
- DO_up, DO_bottom : Oxygène dissous (surface et fond)
- prev_Cyano, prev_Mixo : Proportions de cyanobactéries et mixotrophes

#### LP-NLA (données spatiales) :
- lat, long : Coordonnées géographiques
- area_m2 : Surface du lac
- Variables limnologiques : COND_uS.cm, Chla_ug.L, TNTP_mg.L, pH_mean, DO_up, DO_bottom
- Biomasses zooplanctoniques : Biom_Cladocera_ugL, Biom_Copepoda_ugL
- Variables environnementales : color, temp_up, temp_bottom, wind_30d, tp_30d, degree_day_thr0
- prev_Cyano, prev_Mixo : Proportions de cyanobactéries et mixotrophes

## Notes importantes

1. **Suppression des variables lag** : Conformément aux demandes, aucune variable lag n'est utilisée dans la modélisation
2. **Validation croisée** : 
   - ELA : Leave-one-year-out (validation temporelle)
   - LP-NLA : K-fold standard
3. **Modèles XGBoost** :
   - Richesse : Régression Tweedie
   - Shannon et Équitabilité : Régression standard
4. **Analyse PACF** : Uniquement pour les données ELA (données temporelles)
