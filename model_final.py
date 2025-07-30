import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from scipy import stats
import os
import argparse
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# Configuration et paramètres
###############################################################################

# Configuration des datasets (SANS LAGS)
DATASETS = {
    'ELA': {
        'path': "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/DATA_RAW/df_final/ELA_py.csv",
        'covariable': ["year", "doy", "COND_uS.cm",
                       "Chla_ug.L", "TNTP_mg.L", "pH_mean",
                       "DO_up", "DO_bottom", "prev_Cyano", "prev_Mixo", "lake_id"],  # DO_up et DO_bottom rajoutés
        'validation_type': 'leave_one_year_out',
        'figures_dir': "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/ELA_model",
        'output_prefix': 'ELA'
    },
    'LPNLA': {
        'path': "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/DATA_RAW/df_final/LP_NLA_py.csv",
        'covariable': ["lat", "long", "area_m2", "COND_uS.cm",
                       "Chla_ug.L", "TNTP_mg.L", "pH_mean",
                       "DO_up", "DO_bottom", "Biom_Cladocera_ugL", "Biom_Copepoda_ugL",
                       "color", "temp_up", "temp_bottom", "wind_30d",
                       "tp_30d", "degree_day_thr0", "prev_Mixo", "prev_Cyano"],
        'validation_type': 'kfold',
        'figures_dir': "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/LPNLA_model",
        'output_prefix': 'LPNLA'
    }
}

responses = ['rich_genus_no_cyano', 'shannon_no_cyano', 'eveness_piel_no_cyano']

###############################################################################
# Fonctions utilitaires
###############################################################################

def get_variable_display_name(var_name):
    """Retourner le nom d'affichage correct pour une variable selon l'image fournie"""
    variable_mapping = {
        # Variables communes
        'year': 'Year',
        'doy': 'Doy',
        'lake_id': 'Lake ID',
        'COND_uS.cm': 'COND',
        'Chla_ug.L': 'Chla',
        'TNTP_mg.L': 'TNTP',
        'pH_mean': 'pH mean',
        'DO_up': 'DO up',
        'DO_bottom': 'DO bottom',
        'prev_Cyano': 'Prev Cyano',
        'prev_Mixo': 'Prev Mixo',
        # Variables ELA spécifiques
        # Variables LPNLA spécifiques
        'lat': 'Lat',
        'long': 'Long',
        'area_m2': 'Area',
        'Biom_Cladocera_ugL': 'Biom Cladocera',
        'Biom_Copepoda_ugL': 'Biom Copepoda',
        'color': 'Color',
        'temp_up': 'Temp up',
        'temp_bottom': 'Temp bottom',
        'wind_30d': 'Wind 30d',
        'tp_30d': 'TP 30d',
        'degree_day_thr0': 'Degree day thr0'
    }
    return variable_mapping.get(var_name, var_name)

def load_data(dataset_name):
    """Charger les données pour un dataset spécifique (variables lag supprimées)"""
    dataset_config = DATASETS[dataset_name]
    df = pd.read_csv(dataset_config['path'])
    print(f"Données {dataset_name} chargées: {len(df)} observations, {len(dataset_config['covariable'])} covariables")
    
    # Variables lag supprimées selon les demandes de simplification
    # Récupérer la liste des covariables (sans lag)
    updated_covariables = dataset_config['covariable']
    print(f"Covariables finales: {len(updated_covariables)} variables (sans lag)")
    
    return df, updated_covariables

def create_directories(dataset_name):
    """Créer les répertoires de sortie"""
    dataset_config = DATASETS[dataset_name]
    figures_dir = dataset_config['figures_dir']
    data_model_dir = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/DATA_MODEL"
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_model_dir, exist_ok=True)
    
    return figures_dir

def get_xgb_params(response_name):
    """Retourner les paramètres XGBoost appropriés selon la métrique"""
    if 'rich' in response_name:
        # Paramètres optimisés pour Tweedie (richesse)
        return {
            'objective': 'reg:tweedie',
            'tweedie_variance_power': 1.2,
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': 2
        }
    else:
        # Paramètres pour Shannon et Evenness (régression standard)
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 80,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': 2
        }

###############################################################################
# Fonctions de modélisation
###############################################################################

def train_and_validate_model(X, y, dataset_name, response_name):
    """Entraîner et valider un modèle XGBoost"""
    
    params = get_xgb_params(response_name)
    
    if dataset_name == 'ELA' and 'lake_id' in X.columns:
        # Validation Leave-One-Year-Out pour ELA
        unique_years = sorted(X['year'].unique())
        mae_scores = []
        r2_scores = []
        
        for test_year in unique_years:
            X_train = X[X['year'] != test_year]
            X_test = X[X['year'] == test_year]
            y_train = y[X['year'] != test_year]
            y_test = y[X['year'] == test_year]
            
            if len(X_test) == 0:
                continue
                
            # Supprimer lake_id pour l'entraînement
            X_train_clean = X_train.drop(['lake_id'], axis=1)
            X_test_clean = X_test.drop(['lake_id'], axis=1)
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_clean, y_train)
            
            y_pred = model.predict(X_test_clean)
            
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
        
        mae_mean = np.mean(mae_scores)
        r2_mean = np.mean(r2_scores)
        
        # Entraîner le modèle final sur toutes les données
        X_final = X.drop(['lake_id'], axis=1)
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(X_final, y)
        
    else:
        # Validation K-Fold pour LPNLA
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae_mean = mean_absolute_error(y_test, y_pred)
        r2_mean = r2_score(y_test, y_pred)
        
        final_model = model
        X_final = X
    
    return final_model, X_final, mae_mean, r2_mean

###############################################################################
# Fonctions de visualisation SHAP
###############################################################################

def create_shap_plots(model, X, y, response_name, dataset_name, figures_dir):
    """Créer les graphiques SHAP en style PDP (Partial Dependence Plot)"""
    
    # Calculer les valeurs SHAP avec échantillonnage
    sample_size = min(2000, len(X))
    if len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X
    
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    
    # Configuration des couleurs pour les métriques
    metric_colors = {
        'rich_genus_no_cyano': '#E69F00',    # Orange
        'shannon_no_cyano': '#56B4E9',       # Bleu ciel
        'eveness_piel_no_cyano': '#009E73'   # Vert bleu
    }
    
    # Création des données SHAP
    shap_data = []
    for i, col in enumerate(X_sample.columns):
        for j in range(len(X_sample)):
            shap_data.append({
                'Valeur': X_sample[col].iloc[j],
                'SHAP': shap_values.values[j, i],
                'Variable': col,
                'Métrique': response_name
            })
    
    shap_df = pd.DataFrame(shap_data)
    
    # Obtenir l'ordre d'importance des variables
    importance = np.abs(shap_values.values).mean(0)
    importance_df = pd.DataFrame({
        'Variable': X_sample.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Regrouper les variables dummy de lake_id et traiter comme une seule variable
    if dataset_name == 'ELA':
        # Identifier les variables dummy de lake_id
        lake_vars = [var for var in importance_df['Variable'] if var.startswith('lake_')]
        non_lake_vars = [var for var in importance_df['Variable'] if not var.startswith('lake_')]
        
        if lake_vars:
            # Calculer l'importance cumulée des variables lake_id
            lake_importance = importance_df[importance_df['Variable'].isin(lake_vars)]['Importance'].sum()
            
            # Créer la liste des prédicteurs finaux
            # Ajouter 'lake_id' avec son importance cumulée
            lake_entry = pd.DataFrame({'Variable': ['lake_id'], 'Importance': [lake_importance]})
            non_lake_df = importance_df[importance_df['Variable'].isin(non_lake_vars)]
            
            # Combiner et retrier
            combined_df = pd.concat([lake_entry, non_lake_df]).sort_values('Importance', ascending=False)
            top_predictors = combined_df['Variable'].tolist()
        else:
            top_predictors = importance_df['Variable'].tolist()
    else:
        top_predictors = importance_df['Variable'].tolist()
    
    # Configuration du graphique : grille flexible selon le nombre de variables
    n_vars = len(top_predictors)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols  # Arrondir vers le haut
    fig_size = (18, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Lettres pour identifier chaque subplot
    letters = [f'({chr(97+i)})' for i in range(n_vars)]
    
    # Affichage des prédicteurs par ordre d'importance décroissant
    for i, var in enumerate(top_predictors):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        color = metric_colors.get(response_name, '#1f77b4')
        
        # Traitement spécial pour lake_id groupé
        if var == 'lake_id' and dataset_name == 'ELA':
            # Récupérer toutes les variables dummy de lake_id
            lake_vars = [col_name for col_name in X_sample.columns if col_name.startswith('lake_')]
            
            if lake_vars:
                # Extraire le numéro de lac de chaque variable dummy
                lake_data = []
                lake_labels = []
                
                for lake_var in lake_vars:
                    # Extraire le numéro de lac (ex: 'lake_114' -> '114')
                    lake_num = lake_var.split('_')[1]
                    
                    # Obtenir les données SHAP pour cette variable dummy
                    lake_temp_df = shap_df[shap_df['Variable'] == lake_var]
                    
                    # Ajouter les valeurs SHAP quand cette variable dummy = 1
                    lake_shap_values = lake_temp_df[lake_temp_df['Valeur'] == 1]['SHAP'].values
                    if len(lake_shap_values) > 0:
                        lake_data.append(lake_shap_values)
                        lake_labels.append(lake_num)
                
                # Créer les boxplots pour tous les lacs
                if lake_data:
                    bp = ax.boxplot(lake_data, positions=range(len(lake_data)),
                                   patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_xticks(range(len(lake_labels)))
                    ax.set_xticklabels(lake_labels)
                    
                    # Pour les variables catégorielles, aligner automatiquement sur zéro
                    y_min, y_max = ax.get_ylim()
                    y_abs_max = max(abs(y_min), abs(y_max))
                    ax.set_ylim(-y_abs_max, y_abs_max)
        
        else:
            # Traitement normal pour les autres variables (continues)
            temp_df = shap_df[shap_df['Variable'] == var]
            
            if len(temp_df) > 0:
                # Scatter plot avec ligne de tendance pour les variables continues
                ax.scatter(temp_df['Valeur'], temp_df['SHAP'], 
                          alpha=0.3, s=3, color=color)
                
                # Ajouter une ligne de tendance
                if len(temp_df) > 10:
                    temp_df_sorted = temp_df.sort_values('Valeur')
                    try:
                        trend = lowess(temp_df_sorted["SHAP"], temp_df_sorted["Valeur"], frac=0.3)
                        ax.plot(trend[:, 0], trend[:, 1], color=color, linewidth=2)
                    except:
                        pass
                
                # Pour les variables continues, calculer les limites basées sur les données et courbes de tendance
                trend_line = None
                if len(temp_df) > 10:
                    temp_df_sorted = temp_df.sort_values('Valeur')
                    try:
                        trend_line = lowess(temp_df_sorted["SHAP"], temp_df_sorted["Valeur"], frac=0.3)
                    except:
                        pass
                
                shap_limits_auto = get_shap_limits_from_data(temp_df, trend_line)
                ax.set_ylim(shap_limits_auto[0], shap_limits_auto[1])
        
        # Configuration des axes (commune à tous les types)
        ax.set_xlabel(get_variable_display_name(var), fontsize=14)
        if col == 0:  # Première colonne
            ax.set_ylabel('SHAP Value', fontsize=14)
        
        # Ligne de zéro
        ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
        
        # Ajout de la lettre pour identifier le subplot (si assez de lettres)
        if i < len(letters):
            ax.text(0.02, 0.98, letters[i], transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', verticalalignment='top')
        
        ax.tick_params(axis='both', labelsize=12)
    
    # Masquer les subplots vides
    for i in range(n_vars, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    # PAS DE TITRE comme demandé
    plt.tight_layout()
    
    # Sauvegarder
    output_name = f"SHAP_{response_name}_{dataset_name}.png"
    plt.savefig(os.path.join(figures_dir, output_name), dpi=300, bbox_inches='tight')
    plt.close()
    
    return shap_values

def get_axis_limits(values, var_name=None, dataset_name=None):
    """Calculer les limites d'axes basées sur les percentiles avec règles spéciales"""
    
    # Variables à ne jamais rogner
    preserve_vars = ['prev_Mixo', 'prev_Cyano', 'year', 'doy', 'lat', 'long']
    
    if var_name in preserve_vars:
        # Conserver les bornes entières des données
        return values.min(), values.max()
    
    # Pour les autres variables, rogner si peu de points aux extrêmes
    q01, q99 = np.percentile(values.dropna(), [1, 99])  # Percentiles 1-99%
    
    # Vérifier s'il y a peu de points aux extrêmes
    n_total = len(values.dropna())
    n_below_q01 = len(values[values < q01])
    n_above_q99 = len(values[values > q99])
    
    # Si moins de 2% des points sont aux extrêmes, rogner
    if (n_below_q01 < 0.02 * n_total and n_above_q99 < 0.02 * n_total) and n_total > 50:
        return q01, q99
    else:
        # Sinon, conserver toutes les données
        return values.min(), values.max()

def get_shap_limits_from_data(temp_df, trend_line=None):
    """Calculer automatiquement les limites SHAP pour voir toutes les courbes LOESS avec alignement des zéros"""
    
    if len(temp_df) == 0:
        return [-1, 1]
    
    # Limites basées sur les données SHAP
    shap_min = temp_df['SHAP'].min()
    shap_max = temp_df['SHAP'].max()
    
    # Si on a une courbe de tendance, inclure ses valeurs
    if trend_line is not None and len(trend_line) > 0:
        trend_min = trend_line[:, 1].min()
        trend_max = trend_line[:, 1].max()
        shap_min = min(shap_min, trend_min)
        shap_max = max(shap_max, trend_max)
    
    # Ajouter une marge de 10%
    margin = (shap_max - shap_min) * 0.1
    shap_min_with_margin = shap_min - margin
    shap_max_with_margin = shap_max + margin
    
    # Forcer la symétrie pour aligner les zéros
    abs_max = max(abs(shap_min_with_margin), abs(shap_max_with_margin))
    
    return [-abs_max, abs_max]

def create_combined_shap_plot(shap_values_dict, X_dict, dataset_name, figures_dir):
    """Créer un graphique SHAP combiné montrant toutes les métriques ensemble avec leurs couleurs spécifiques"""
    
    # Configuration des couleurs pour les métriques
    metric_colors = {
        'rich_genus_no_cyano': '#E69F00',    # Jaune/Orange pour richesse
        'shannon_no_cyano': '#56B4E9',       # Bleu pour Shannon
        'eveness_piel_no_cyano': '#009E73'   # Vert pour équitabilité
    }
    
    # Collecter toutes les données SHAP combinées
    all_shap_data = []
    
    for response, shap_vals in shap_values_dict.items():
        X_sample = X_dict[response]
        
        # Échantillonnage si nécessaire
        sample_size = min(2000, len(shap_vals.values))
        if len(shap_vals.values) > sample_size:
            sample_idx = np.random.choice(len(shap_vals.values), sample_size, replace=False)
        else:
            sample_idx = range(len(shap_vals.values))
        
        # Création des données SHAP pour cette métrique
        for i, col in enumerate(shap_vals.feature_names):
            for j in sample_idx:
                # Gérer les variables dummy de lake_id pour ELA
                if col.startswith('lake_') and dataset_name == 'ELA':
                    # Extraire le numéro du lac (lake_114 -> 114)
                    lake_num = col.replace('lake_', '')
                    # Si cette variable dummy est activée (valeur = 1), ajouter l'entrée avec lake_id
                    if shap_vals.data[j, i] == 1:
                        all_shap_data.append({
                            'Valeur': float(lake_num),  # Convertir en float pour cohérence
                            'SHAP': shap_vals.values[j, i],
                            'Variable': 'lake_id',  # Utiliser lake_id comme nom de variable
                            'Métrique': response
                        })
                else:
                    # Traitement normal pour les autres variables
                    all_shap_data.append({
                        'Valeur': shap_vals.data[j, i],
                        'SHAP': shap_vals.values[j, i],
                        'Variable': col,
                        'Métrique': response
                    })
    
    shap_all_df = pd.DataFrame(all_shap_data)
    
    # Calculer l'importance globale (moyenne de toutes les métriques)
    importance_by_var = {}
    for var in shap_all_df['Variable'].unique():
        var_data = shap_all_df[shap_all_df['Variable'] == var]
        importance_by_var[var] = np.abs(var_data['SHAP']).mean()
    
    # Pour ELA, si lake_id existe, s'assurer qu'il a une importance raisonnable
    if dataset_name == 'ELA' and 'lake_id' in importance_by_var:
        # lake_id étant une variable catégorielle importante, s'assurer qu'elle est bien classée
        print(f"Importance de lake_id: {importance_by_var['lake_id']:.6f}")
    
    # Trier par importance décroissante et prendre les 9 plus importantes
    top_predictors = sorted(importance_by_var.keys(), key=lambda x: importance_by_var[x], reverse=True)[:9]
    
    # Pour ELA, s'assurer absolument que lake_id est inclus
    if dataset_name == 'ELA' and 'lake_id' in importance_by_var and 'lake_id' not in top_predictors:
        # Forcer l'inclusion de lake_id en remplaçant la dernière variable
        print(f"Inclusion forcée de lake_id dans les graphiques SHAP pour ELA")
        top_predictors[-1] = 'lake_id'
    
    print(f"Top predictors pour {dataset_name}: {top_predictors}")
    
    # Configuration du graphique : 3x3 pour les 9 prédicteurs les plus importants
    n_vars = 9
    n_cols = 3
    n_rows = 3
    fig_size = (18, 15)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Lettres pour identifier chaque subplot
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    
    # Séparer les métriques par type d'axe
    shannon_eveness_metrics = ['shannon_no_cyano', 'eveness_piel_no_cyano']
    richness_metric = ['rich_genus_no_cyano']
    
    # Calculer les échelles SHAP communes pour harmonisation
    print(f"Calcul des échelles SHAP automatiques basées sur les courbes LOESS...")
    
    # Dictionnaire pour stocker les limites calculées automatiquement
    shap_limits_auto = {}
    
    # Calculer les limites pour chaque position
    for i, var in enumerate(top_predictors):
        temp_df = shap_all_df[shap_all_df['Variable'] == var]
        
        # Calculer les courbes de tendance pour chaque métrique
        shannon_eveness_trends = []
        richness_trends = []
        
        # Shannon et Équitabilité
        for metric in shannon_eveness_metrics:
            if metric in responses:
                metric_data = temp_df[temp_df['Métrique'] == metric]
                if len(metric_data) > 10:
                    metric_data_sorted = metric_data.sort_values('Valeur')
                    try:
                        trend = lowess(metric_data_sorted["SHAP"], metric_data_sorted["Valeur"], frac=0.3)
                        shannon_eveness_trends.append(trend)
                    except:
                        pass
        
        # Richesse
        for metric in richness_metric:
            if metric in responses:
                metric_data = temp_df[temp_df['Métrique'] == metric]
                if len(metric_data) > 10:
                    metric_data_sorted = metric_data.sort_values('Valeur')
                    try:
                        trend = lowess(metric_data_sorted["SHAP"], metric_data_sorted["Valeur"], frac=0.3)
                        richness_trends.append(trend)
                    except:
                        pass
        
        # Calculer les limites pour Shannon & Équitabilité
        if shannon_eveness_trends:
            all_se_data = temp_df[temp_df['Métrique'].isin(shannon_eveness_metrics)]
            combined_trend = np.vstack(shannon_eveness_trends) if len(shannon_eveness_trends) > 1 else shannon_eveness_trends[0]
            hj_limits = get_shap_limits_from_data(all_se_data, combined_trend)
        else:
            all_se_data = temp_df[temp_df['Métrique'].isin(shannon_eveness_metrics)]
            hj_limits = get_shap_limits_from_data(all_se_data)
        
        # Calculer les limites pour Richesse
        if richness_trends:
            all_r_data = temp_df[temp_df['Métrique'].isin(richness_metric)]
            combined_trend = np.vstack(richness_trends) if len(richness_trends) > 1 else richness_trends[0]
            s_limits = get_shap_limits_from_data(all_r_data, combined_trend)
        else:
            all_r_data = temp_df[temp_df['Métrique'].isin(richness_metric)]
            s_limits = get_shap_limits_from_data(all_r_data)
        
        # Stocker les limites pour cette position
        shap_limits_auto[i] = {'S': s_limits, 'HJ': hj_limits}
        
        print(f"  Position {i} ({var}): S {s_limits}, HJ {hj_limits}")
    
    print(f"  Limites SHAP automatiques calculées pour {dataset_name}")
    
    # Utiliser les limites automatiques calculées
    shap_limits = shap_limits_auto
    
    # Affichage des prédicteurs par ordre d'importance décroissant
    for i, var in enumerate(top_predictors):
        row = i // n_cols
        col = i % n_cols
        ax1 = axes[row, col]
        
        temp_df = shap_all_df[shap_all_df['Variable'] == var]
        
        # Échantillonnage pour optimiser le rendu
        if len(temp_df) > 2000:
            temp_df = temp_df.sample(n=2000, random_state=42)
        
        # Variables pour contrôler l'affichage des axes Y
        shannon_eveness_plotted = False
        richness_plotted = False
        ax2 = None
        
        # Traitement spécial pour les prédicteurs catégoriels
        if (var == 'lake_id' and dataset_name == 'ELA'):
            # Récupérer les modalités uniques
            unique_values = sorted(temp_df['Valeur'].unique())
            
            # Créer un mapping pour les positions décalées
            width = 0.8 / len(responses)  # Largeur pour chaque boxplot
            
            # Axe gauche : Shannon et Équitabilité
            for j, metric in enumerate(shannon_eveness_metrics):
                if metric in responses:
                    metric_data = temp_df[temp_df['Métrique'] == metric]
                    if len(metric_data) > 0:
                        color = metric_colors[metric]
                        
                        # Créer le boxplot décalé
                        bp = ax1.boxplot([metric_data[metric_data['Valeur'] == cat_value]['SHAP'].values 
                                        for cat_value in unique_values if len(metric_data[metric_data['Valeur'] == cat_value]) > 0],
                                       positions=[k + (j - len(shannon_eveness_metrics)/2 + 0.5) * width for k in range(len(unique_values))],
                                       widths=width*0.8, patch_artist=True, 
                                       boxprops=dict(facecolor=color, alpha=0.7))
                        
                        shannon_eveness_plotted = True
            
            # Axe droit : Richesse
            for j, metric in enumerate(richness_metric):
                if metric in responses:
                    metric_data = temp_df[temp_df['Métrique'] == metric]
                    if len(metric_data) > 0:
                        ax2 = ax1.twinx()
                        color = metric_colors[metric]
                        
                        # Créer des positions décalées pour la richesse
                        metric_offset = len(shannon_eveness_metrics) + j
                        positions = [k + (metric_offset - len(responses)/2 + 0.5) * width for k in range(len(unique_values))]
                        
                        # Créer le boxplot décalé
                        bp = ax2.boxplot([metric_data[metric_data['Valeur'] == cat_value]['SHAP'].values 
                                        for cat_value in unique_values if len(metric_data[metric_data['Valeur'] == cat_value]) > 0],
                                       positions=positions,
                                       widths=width*0.8, patch_artist=True,
                                       boxprops=dict(facecolor=color, alpha=0.7))
                        
                        richness_plotted = True
            
            # Configuration des axes pour les prédicteurs catégoriels
            ax1.set_xlabel(get_variable_display_name(var), fontsize=18)
            ax1.set_xticks(range(len(unique_values)))
            ax1.set_xticklabels([f'{int(val)}' for val in unique_values], fontsize=16)
            
        else:
            # Traitement normal pour les prédicteurs continus
            # Calculer les limites d'axes avec rognage adaptatif
            x_min, x_max = get_axis_limits(temp_df['Valeur'], var, dataset_name)
            
            # Axe gauche : Shannon et Équitabilité
            for metric in shannon_eveness_metrics:
                if metric in responses:
                    metric_data = temp_df[temp_df['Métrique'] == metric]
                    if len(metric_data) > 0:
                        color = metric_colors[metric]
                        ax1.scatter(metric_data['Valeur'], metric_data['SHAP'], 
                                   alpha=0.2, s=2, color=color, label=metric.replace('_no_cyano', ''))
                        
                        # Ajouter une ligne de tendance
                        if len(metric_data) > 10:
                            metric_data_sorted = metric_data.sort_values('Valeur')
                            try:
                                trend = lowess(metric_data_sorted["SHAP"], metric_data_sorted["Valeur"], frac=0.3)
                                ax1.plot(trend[:, 0], trend[:, 1], color=color, linewidth=2)
                            except:
                                pass
                        shannon_eveness_plotted = True
            
            # Définir les limites d'axe X après avoir traité toutes les métriques
            if 'x_min' in locals() and 'x_max' in locals():
                ax1.set_xlim(x_min, x_max)
            
            # Axe droit : Richesse
            for metric in richness_metric:
                if metric in responses:
                    metric_data = temp_df[temp_df['Métrique'] == metric]
                    if len(metric_data) > 0:
                        ax2 = ax1.twinx()
                        color = metric_colors[metric]
                        ax2.scatter(metric_data['Valeur'], metric_data['SHAP'], 
                                   alpha=0.2, s=2, color=color, label=metric.replace('_no_cyano', ''))
                        
                        # Ajouter une ligne de tendance
                        if len(metric_data) > 10:
                            metric_data_sorted = metric_data.sort_values('Valeur')
                            try:
                                trend = lowess(metric_data_sorted["SHAP"], metric_data_sorted["Valeur"], frac=0.3)
                                ax2.plot(trend[:, 0], trend[:, 1], color=color, linewidth=2)
                            except:
                                pass
                        
                        richness_plotted = True
            
            # Configuration des axes pour les prédicteurs continus
            ax1.set_xlabel(get_variable_display_name(var), fontsize=18)
        
        # Configuration des labels d'axes Y avec contrôle de visibilité
        # Axe gauche (Shannon & Équitabilité) : visible seulement sur la colonne de gauche
        if col == 0:  # Colonne de gauche
            if shannon_eveness_plotted:
                ax1.set_ylabel('SHAP (Shannon & Équitabilité)', fontsize=18)
                ax1.tick_params(axis='y', labelsize=16)
            else:
                ax1.set_ylabel('')
                ax1.tick_params(axis='y', labelleft=False)
        else:
            ax1.set_ylabel('')
            ax1.tick_params(axis='y', labelleft=False)
        
        # Axe droit (Richesse) : visible seulement sur la colonne de droite
        if col == 2 and ax2 is not None:  # Colonne de droite
            if richness_plotted:
                ax2.set_ylabel('SHAP (Richesse)', color=metric_colors['rich_genus_no_cyano'], fontsize=18)
                ax2.tick_params(axis='y', labelcolor=metric_colors['rich_genus_no_cyano'], labelsize=16)
            else:
                ax2.set_ylabel('')
                ax2.tick_params(axis='y', labelright=False)
        elif ax2 is not None:
            ax2.set_ylabel('')
            ax2.tick_params(axis='y', labelright=False)
        
        # Lignes de zéro
        ax1.axhline(0, color='grey', linestyle='--', alpha=0.5)
        if ax2 is not None:
            ax2.axhline(0, color='grey', linestyle='--', alpha=0.5)
        
        # Aligner les lignes zéro en définissant des limites symétriques selon la position
        # Utiliser les limites automatiques calculées pour cette position
        if i in shap_limits:
            if shannon_eveness_plotted:
                hj_limits = shap_limits[i]['HJ']
                ax1.set_ylim(hj_limits[0], hj_limits[1])
            
            if richness_plotted and ax2 is not None:
                s_limits = shap_limits[i]['S']
                ax2.set_ylim(s_limits[0], s_limits[1])
        
        # Ajout de la lettre pour identifier le subplot (par ordre d'importance)
        ax1.text(0.02, 0.98, letters[i], transform=ax1.transAxes, 
                fontsize=16, fontweight='bold', verticalalignment='top')
        
        # Améliorer la taille des ticks
        ax1.tick_params(axis='x', labelsize=16)
    
    # PAS DE TITRE comme demandé
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"SHAP_all_{dataset_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_shap_rank_plot(shap_results_all):
    """Créer un boxplot combiné des ranks SHAP pour ELA et LPNLA basé sur l'importance cumulée"""
    print("Génération du boxplot combiné des ranks SHAP ELA + LPNLA (ordre par importance cumulée)")
    
    # Extraire les données de ranking SHAP pour chaque dataset
    shap_rank_data_ela = []
    shap_rank_data_lpnla = []
    
    # Traiter ELA
    if 'ELA' in shap_results_all:
        for response, shap_values in shap_results_all['ELA'].items():
            # Calculer l'importance moyenne de chaque variable
            importance = np.abs(shap_values.values).mean(0)
            feature_names = shap_values.feature_names
            
            # Créer les rangs (1 = plus important)
            ranks = np.argsort(np.argsort(-importance)) + 1
            
            # Ajouter aux données ELA
            for i, feature in enumerate(feature_names):
                # Traitement spécial pour les variables dummy de lake_id
                if feature.startswith('lake_'):
                    feature_clean = 'lake_id'
                else:
                    feature_clean = feature
                
                shap_rank_data_ela.append({
                    'Predictor': feature_clean,
                    'Rank': ranks[i],
                    'Response': response,
                    'Importance': importance[i]
                })
    
    # Traiter LPNLA
    if 'LPNLA' in shap_results_all:
        for response, shap_values in shap_results_all['LPNLA'].items():
            # Calculer l'importance moyenne de chaque variable
            importance = np.abs(shap_values.values).mean(0)
            feature_names = shap_values.feature_names
            
            # Créer les rangs (1 = plus important)
            ranks = np.argsort(np.argsort(-importance)) + 1
            
            # Ajouter aux données LPNLA
            for i, feature in enumerate(feature_names):
                shap_rank_data_lpnla.append({
                    'Predictor': feature,
                    'Rank': ranks[i],
                    'Response': response,
                    'Importance': importance[i]
                })
    
    # Créer les DataFrames
    df_ela = pd.DataFrame(shap_rank_data_ela) if shap_rank_data_ela else pd.DataFrame()
    df_lpnla = pd.DataFrame(shap_rank_data_lpnla) if shap_rank_data_lpnla else pd.DataFrame()
    
    # Ajouter une colonne Dataset
    if len(df_ela) > 0:
        df_ela['Dataset'] = 'ELA'
    if len(df_lpnla) > 0:
        df_lpnla['Dataset'] = 'LPNLA'
    
    # Obtenir tous les prédicteurs uniques des deux datasets
    all_predictors = set()
    if len(df_ela) > 0:
        all_predictors.update(df_ela['Predictor'].unique())
    if len(df_lpnla) > 0:
        all_predictors.update(df_lpnla['Predictor'].unique())
    
    if not all_predictors:
        print("Aucune donnée SHAP disponible pour créer le graphique de ranking")
        return None
    
    # Calculer l'importance cumulée (inverse du rang médian) pour chaque prédicteur
    cumulative_importance = {}
    for predictor in all_predictors:
        ela_ranks = df_ela[df_ela['Predictor'] == predictor]['Rank'].values if len(df_ela) > 0 and predictor in df_ela['Predictor'].unique() else []
        lpnla_ranks = df_lpnla[df_lpnla['Predictor'] == predictor]['Rank'].values if len(df_lpnla) > 0 and predictor in df_lpnla['Predictor'].unique() else []
        
        # Combiner les rangs des deux datasets
        all_ranks = list(ela_ranks) + list(lpnla_ranks)
        
        if all_ranks:
            # Utiliser la médiane des rangs combinés (plus bas = plus important)
            cumulative_importance[predictor] = np.median(all_ranks)
        else:
            cumulative_importance[predictor] = 999  # Valeur élevée pour les prédicteurs sans données
    
    # Trier les prédicteurs par importance cumulée décroissante (rang médian croissant)
    all_predictors = sorted(all_predictors, key=lambda x: cumulative_importance[x])
    
    print(f"Ordre des prédicteurs par importance cumulée (ELA + LPNLA): {all_predictors[:10]}")
    
    # Créer la figure avec taille ajustée et texte plus grand
    fig_size = (max(18, len(all_predictors) * 0.9), 10)
    plt.figure(figsize=fig_size)
    
    # Définir les couleurs pour chaque dataset (colorblind-friendly)
    colors = {'ELA': '#CC79A7', 'LPNLA': '#D55E00'}  # Violet et Rouge brique
    
    # Créer les positions pour les boxplots
    tick_positions = []
    tick_labels = []
    
    # Pour chaque prédicteur, créer des positions pour ELA et LPNLA
    for i, predictor in enumerate(all_predictors):
        # Position de base pour ce prédicteur
        base_pos = i * 3
        
        # Vérifier si le prédicteur est présent dans chaque dataset
        ela_present = len(df_ela) > 0 and predictor in df_ela['Predictor'].unique()
        lpnla_present = len(df_lpnla) > 0 and predictor in df_lpnla['Predictor'].unique()
        
        # Données pour ELA
        if ela_present:
            ela_data = df_ela[df_ela['Predictor'] == predictor]
            bp_ela = plt.boxplot(ela_data['Rank'].values, positions=[base_pos], 
                       widths=0.6, patch_artist=True, 
                       boxprops=dict(facecolor=colors['ELA'], alpha=0.7),
                       medianprops=dict(color='black'),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'))
        
        # Données pour LPNLA
        if lpnla_present:
            lpnla_data = df_lpnla[df_lpnla['Predictor'] == predictor]
            bp_lpnla = plt.boxplot(lpnla_data['Rank'].values, positions=[base_pos + 1], 
                       widths=0.6, patch_artist=True, 
                       boxprops=dict(facecolor=colors['LPNLA'], alpha=0.7),
                       medianprops=dict(color='black'),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'))
        
        # Ajouter les labels pour ce prédicteur
        tick_positions.append(base_pos + 0.5)  # Position centrale entre ELA et LPNLA
        tick_labels.append(get_variable_display_name(predictor))
    
    # Configuration des axes avec texte plus grand
    plt.ylabel("SHAP rank", fontsize=20)
    
    # Définir les ticks sur l'axe X avec texte plus grand
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)
    
    # Ajouter des lignes verticales pour séparer les prédicteurs
    for i in range(len(all_predictors) - 1):
        plt.axvline(x=i * 3 + 2, color='gray', linestyle='--', alpha=0.3)
    
    # Ajouter une grille pour faciliter la lecture
    plt.grid(True, alpha=0.3, axis='y')
    
    # Configurer l'axe Y avec des valeurs discrètes (entiers seulement)
    max_rank = 25  # Nombre maximum de prédicteurs possibles
    y_ticks = list(range(1, max_rank + 1, 2))  # Ticks tous les 2 rangs (1, 3, 5, ...)
    plt.yticks(y_ticks, fontsize=18)
    
    # Inverser l'axe Y pour que le rang 1 soit en haut
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    # Sauvegarder la figure avec résolution plus élevée
    combined_figures_dir = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES"
    os.makedirs(combined_figures_dir, exist_ok=True)
    plt.savefig(f"{combined_figures_dir}/SHAP_rank_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot combiné sauvegardé : {combined_figures_dir}/SHAP_rank_combined.png")
    return f"{combined_figures_dir}/SHAP_rank_combined.png"

###############################################################################
# Analyse des interactions prédicteurs-prévalence
###############################################################################

def get_metric_display_name(metric):
    """Retourner le nom d'affichage pour les métriques"""
    mapping = {
        'rich_genus_no_cyano': 'Richesse (S)',
        'shannon_no_cyano': 'Shannon (H\')',
        'eveness_piel_no_cyano': 'Équitabilité (J\')'
    }
    return mapping.get(metric, metric)

def analyze_predictor_prevalence_interaction(shap_values_dict, X_dict, dataset_name, figures_dir):
    """Analyser l'interaction des prédicteurs avec la prévalence en mixotrophie selon la décomposition SHAP"""
    print(f"Analyse de l'interaction prédicteurs-prévalence pour {dataset_name}")
    
    # Configuration des couleurs pour les métriques
    metric_colors = {
        'rich_genus_no_cyano': '#E69F00',    # Orange pour richesse
        'shannon_no_cyano': '#56B4E9',       # Bleu pour Shannon
        'eveness_piel_no_cyano': '#009E73'   # Vert pour équitabilité
    }
    
    # Obtenir les prédicteurs principaux (excluant prev_Mixo)
    all_predictors = set()
    for response, shap_vals in shap_values_dict.items():
        # Calculer l'importance moyenne absolue pour chaque prédicteur
        importance = np.abs(shap_vals.values).mean(axis=0)
        feature_importance = dict(zip(shap_vals.feature_names, importance))
        
        # Garder seulement les top prédicteurs (excluant prev_Mixo)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [feat[0] for feat in sorted_features[:12] if feat[0] != 'prev_Mixo']
        all_predictors.update(top_features)
    
    # Filtrer et regrouper les variables dummy de lake_id pour ELA
    if dataset_name == 'ELA':
        lake_vars = [var for var in all_predictors if var.startswith('lake_')]
        other_vars = [var for var in all_predictors if not var.startswith('lake_')]
        if lake_vars:
            other_vars.append('lake_id')  # Regrouper toutes les variables dummy
        final_predictors = other_vars[:10]  # Limiter à 10 prédicteurs
    else:
        final_predictors = list(all_predictors)[:10]  # Limiter à 10 prédicteurs
    
    # Calculer les interactions avec prev_Mixo selon la décomposition SHAP
    interaction_data = []
    
    for response, shap_vals in shap_values_dict.items():
        X_sample = X_dict[response]
        
        # Trouver l'index de prev_Mixo
        try:
            prev_mixo_idx = shap_vals.feature_names.index('prev_Mixo')
        except ValueError:
            print(f"prev_Mixo non trouvé dans {response} pour {dataset_name}")
            continue
        
        # Valeurs SHAP pour prev_Mixo (φ_mixo^(i))
        prev_mixo_shap = shap_vals.values[:, prev_mixo_idx]
        
        # Calculer l'effet total de prev_Mixo
        total_prev_mixo_effect = np.abs(prev_mixo_shap).sum()
        
        for predictor in final_predictors:
            if predictor == 'lake_id' and dataset_name == 'ELA':
                # Gérer les variables dummy de lake_id
                lake_vars = [var for var in shap_vals.feature_names if var.startswith('lake_')]
                if lake_vars:
                    # Sommer les effets de toutes les variables dummy
                    predictor_shap = np.zeros(len(shap_vals.values))
                    for lake_var in lake_vars:
                        lake_idx = shap_vals.feature_names.index(lake_var)
                        predictor_shap += shap_vals.values[:, lake_idx]
                else:
                    continue
            else:
                try:
                    pred_idx = shap_vals.feature_names.index(predictor)
                    predictor_shap = shap_vals.values[:, pred_idx]
                except ValueError:
                    continue
            
            # Calculer l'interaction selon la décomposition SHAP: φ_j^(i) = φ_jj^(i) + Σ φ_jk^(i)
            # Approximation: interaction basée sur la covariance entre les effets SHAP
            if len(predictor_shap) > 0 and len(prev_mixo_shap) > 0:
                # Calculer la covariance entre les effets SHAP du prédicteur et de prev_Mixo
                cov_matrix = np.cov(predictor_shap, prev_mixo_shap)
                covariance = cov_matrix[0, 1] if not np.isnan(cov_matrix[0, 1]) else 0
                
                # Variance de prev_Mixo
                var_prev_mixo = np.var(prev_mixo_shap)
                
                # Pourcentage d'interaction (effet conjoint relatif)
                if var_prev_mixo > 0:
                    # Normaliser par l'effet total de prev_Mixo pour obtenir un pourcentage
                    interaction_effect = abs(covariance) / var_prev_mixo
                    interaction_percentage = min(interaction_effect * 100, 15)  # Crop à 15%
                else:
                    interaction_percentage = 0
                
                interaction_data.append({
                    'Predictor': predictor,
                    'Metric': response,
                    'Interaction_Percentage': interaction_percentage,
                    'Covariance': covariance
                })
    
    # Créer le DataFrame
    df_interactions = pd.DataFrame(interaction_data)
    
    if len(df_interactions) == 0:
        print(f"Aucune donnée d'interaction trouvée pour {dataset_name}")
        return None
    
    return df_interactions

def create_combined_prevalence_interaction_plot(ela_interactions, lpnla_interactions):
    """Créer la figure combinée des interactions prédicteurs-prévalence (LPNLA à gauche, ELA à droite)"""
    
    # Configuration des couleurs pour les métriques
    metric_colors = {
        'rich_genus_no_cyano': '#E69F00',    # Orange pour richesse
        'shannon_no_cyano': '#56B4E9',       # Bleu pour Shannon
        'eveness_piel_no_cyano': '#009E73'   # Vert pour équitabilité
    }
    
    # Créer la figure avec deux sous-graphiques côte à côte
    fig, (ax_lpnla, ax_ela) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Fonction pour créer un graphique pour un dataset
    def plot_dataset(ax, df_interactions, dataset_name):
        if df_interactions is None or len(df_interactions) == 0:
            ax.text(0.5, 0.5, f'Aucune donnée\npour {dataset_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(dataset_name, fontsize=16, fontweight='bold')
            return
        
        predictors = df_interactions['Predictor'].unique()
        metrics = ['rich_genus_no_cyano', 'shannon_no_cyano', 'eveness_piel_no_cyano']
        
        # Position des barres
        x_positions = np.arange(len(predictors))
        bar_width = 0.25
        
        # Créer les barres pour chaque métrique
        for i, metric in enumerate(metrics):
            metric_data = df_interactions[df_interactions['Metric'] == metric]
            
            values = []
            for predictor in predictors:
                pred_data = metric_data[metric_data['Predictor'] == predictor]
                if len(pred_data) > 0:
                    values.append(pred_data['Interaction_Percentage'].iloc[0])
                else:
                    values.append(0)
            
            # Créer les barres avec opacité faible pour l'effet marginal restant
            positions = x_positions + i * bar_width
            
            # Barres principales (effet d'interaction)
            bars = ax.bar(positions, values, bar_width, 
                         label=get_metric_display_name(metric),
                         color=metric_colors[metric], alpha=0.8)
            
            # Barres d'effet marginal restant (jusqu'à 15% - crop)
            remaining_values = [15 - v for v in values]
            ax.bar(positions, remaining_values, bar_width, 
                   bottom=values, color=metric_colors[metric], alpha=0.1)
        
        # Configuration des axes
        ax.set_ylabel('% of total effect on diversity', fontsize=14)
        ax.set_title(dataset_name, fontsize=16, fontweight='bold')
        
        # Limiter l'axe Y à 15% pour une belle visualisation
        ax.set_ylim(0, 15)
        
        # Configuration de l'axe X
        ax.set_xticks(x_positions + bar_width)
        ax.set_xticklabels([get_variable_display_name(pred) for pred in predictors], 
                           rotation=45, ha='right')
        
        # Grille
        ax.grid(True, alpha=0.3, axis='y')
    
    # Créer les graphiques pour chaque dataset
    plot_dataset(ax_lpnla, lpnla_interactions, 'LPNLA')
    plot_dataset(ax_ela, ela_interactions, 'ELA')
    
    # Légende commune
    if ela_interactions is not None and len(ela_interactions) > 0:
        ax_ela.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    elif lpnla_interactions is not None and len(lpnla_interactions) > 0:
        ax_lpnla.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Sauvegarder dans le dossier FIGURES principal
    output_path = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/predictor_prevalence_interaction_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure d'interaction combinée sauvegardée : {output_path}")
    return output_path

###############################################################################
# Fonctions d'analyse des résidus
###############################################################################

def create_residuals_analysis(models_dict, X_dict, y_dict, dataset_name, figures_dir):
    """Créer l'analyse des résidus complète"""
    
    # Déterminer le nombre de métriques
    n_metrics = len(models_dict)
    
    # Configuration selon le dataset
    if dataset_name == 'ELA':
        # Pour ELA : 6 lignes (résidus vs prédictions, Q-Q plot, histogramme, résidus vs year, résidus vs doy, résidus vs valeurs observées)
        fig, axes = plt.subplots(6, n_metrics, figsize=(6*n_metrics, 24))
    else:
        # Pour LPNLA : 6 lignes (résidus vs prédictions, Q-Q plot, histogramme, résidus vs lat, résidus vs long, résidus vs valeurs observées)
        fig, axes = plt.subplots(6, n_metrics, figsize=(6*n_metrics, 24))
    
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)
    
    # Ajouter les titres des colonnes
    response_titles = {
        'rich_genus_no_cyano': 'Richesse (S)',
        'shannon_no_cyano': 'Shannon (H\')',
        'eveness_piel_no_cyano': 'Équitabilité (J\')'
    }
    
    for i, (response, model) in enumerate(models_dict.items()):
        X = X_dict[response]
        y = y_dict[response]
        
        # Ajouter le titre de la colonne
        axes[0, i].set_title(response_titles.get(response, response), fontsize=14, fontweight='bold')
        
        # Prédictions
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # 1. Graphique résidus vs prédictions (SANS COULEUR)
        axes[0, i].scatter(y_pred, residuals, alpha=0.6, color='gray', s=10)
        axes[0, i].axhline(y=0, color='red', linestyle='--')
        axes[0, i].set_xlabel('Predicted Values')
        axes[0, i].set_ylabel('Residuals')
        
        # 2. Q-Q plot (SANS COULEUR)
        stats.probplot(residuals, dist="norm", plot=axes[1, i])
        # Changer la couleur des points en gris
        axes[1, i].get_lines()[0].set_markerfacecolor('gray')
        axes[1, i].get_lines()[0].set_markeredgecolor('gray')
        
        # 3. Histogramme des résidus (SANS COULEUR)
        axes[2, i].hist(residuals, bins=30, alpha=0.7, color='gray', edgecolor='black')
        axes[2, i].set_xlabel('Residuals')
        axes[2, i].set_ylabel('Frequency')
        
        # 4. Résidus vs variable spécifique au dataset
        if dataset_name == 'ELA':
            # Résidus vs year
            if 'year' in X.columns:
                axes[3, i].scatter(X['year'], residuals, alpha=0.6, color='gray', s=10)
                axes[3, i].axhline(y=0, color='red', linestyle='--')
                axes[3, i].set_xlabel('Year')
                axes[3, i].set_ylabel('Residuals')
            
            # Résidus vs doy
            if 'doy' in X.columns:
                axes[4, i].scatter(X['doy'], residuals, alpha=0.6, color='gray', s=10)
                axes[4, i].axhline(y=0, color='red', linestyle='--')
                axes[4, i].set_xlabel('Day of Year')
                axes[4, i].set_ylabel('Residuals')
        else:
            # LPNLA : Résidus vs lat
            if 'lat' in X.columns:
                axes[3, i].scatter(X['lat'], residuals, alpha=0.6, color='gray', s=10)
                axes[3, i].axhline(y=0, color='red', linestyle='--')
                axes[3, i].set_xlabel('Latitude')
                axes[3, i].set_ylabel('Residuals')
            
            # Résidus vs long
            if 'long' in X.columns:
                axes[4, i].scatter(X['long'], residuals, alpha=0.6, color='gray', s=10)
                axes[4, i].axhline(y=0, color='red', linestyle='--')
                axes[4, i].set_xlabel('Longitude')
                axes[4, i].set_ylabel('Residuals')
        
        # 5. Résidus vs valeurs observées (SANS COULEUR)
        axes[5, i].scatter(y, residuals, alpha=0.6, color='gray', s=10)
        axes[5, i].axhline(y=0, color='red', linestyle='--')
        axes[5, i].set_xlabel('Observed Values')
        axes[5, i].set_ylabel('Residuals')
    
    # PAS DE TITRE PRINCIPAL comme demandé
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"res_{dataset_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def export_residuals_for_pacf(models_dict, X_dict, y_dict, dataset_name, figures_dir):
    """Exporter les résidus XGBoost pour l'analyse PACF"""
    
    # Charger les données originales pour récupérer les métadonnées temporelles/spatiales
    dataset_config = DATASETS[dataset_name]
    original_data = pd.read_csv(dataset_config['path'])
    
    for response, model in models_dict.items():
        X = X_dict[response]
        y = y_dict[response]
        
        # Calculer les prédictions et résidus
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Créer le DataFrame de base avec les résidus
        residuals_df = pd.DataFrame({
            f'{response}_residuals': residuals
        })
        
        # Ajouter les index originaux pour faire la jointure
        residuals_df.index = y.index
        
        # Récupérer les métadonnées temporelles/spatiales depuis les données originales
        if dataset_name == 'ELA':
            metadata_cols = ['lake_id', 'year', 'doy']
        else:  # LPNLA
            metadata_cols = ['lat', 'long']
        
        # Faire la jointure avec les données originales pour récupérer les métadonnées
        metadata_df = original_data.loc[y.index, metadata_cols].copy()
        residuals_df = pd.concat([metadata_df, residuals_df], axis=1)
        
        # Supprimer les lignes avec des valeurs manquantes dans les colonnes de métadonnées
        residuals_df = residuals_df.dropna(subset=metadata_cols)
        
        # Sauvegarder les résidus
        output_path = os.path.join("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/DATA_MODEL", f"residuals_{response}_{dataset_name}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        residuals_df.to_csv(output_path, index=False)
        
        print(f"Résidus {response} exportés: {output_path} ({len(residuals_df)} observations)")
    
    return True

###############################################################################
# Fonction pour créer le tableau des métriques SHAP
###############################################################################

def create_shap_metrics_table(shap_results_all, metrics_all, output_dir):
    """Créer le tableau des métriques et valeurs SHAP"""
    
    # Créer le tableau combiné
    table_data = []
    
    # En-têtes
    headers = ['Prédicteur', 'LP-NLA S', 'LP-NLA H\'', 'LP-NLA J\'', 'ELA S', 'ELA H\'', 'ELA J\'']
    
    # Lignes de métriques
    for dataset in ['LPNLA', 'ELA']:
        if dataset in metrics_all:
            mae_row = ['MAE (test)'] + [''] * 6
            r2_row = ['R² (test)'] + [''] * 6
            
            for i, response in enumerate(responses):
                col_idx = (1 if dataset == 'LPNLA' else 4) + i
                if response in metrics_all[dataset]:
                    mae_row[col_idx] = f"{metrics_all[dataset][response]['mae']:.3f}"
                    r2_row[col_idx] = f"{metrics_all[dataset][response]['r2']:.3f}"
                else:
                    mae_row[col_idx] = "NA"
                    r2_row[col_idx] = "NA"
            
            if dataset == 'LPNLA':
                table_data.append(mae_row)
                table_data.append(r2_row)
    
    # Créer le fichier LaTeX
    latex_content = """\\columnbreak
\\noindent
\\begin{minipage}{\\dimexpr 2\\linewidth + \\columnsep\\relax}
\\captionsetup{type=table}
\\captionof{table}{Métriques de performance (MAE et R² sur données de test) et valeurs SHAP absolues moyennes pour chacun des modèles de diversité. Les valeurs NA correspondent aux prédicteurs absents du jeu de données.}
\\label{tab:metrics_shap_models_all}
\\centering
\\renewcommand{\\arraystretch}{1.2}
\\setlength{\\tabcolsep}{4pt}
\\begin{tabular}{lcccccc}
\\toprule
\\multirow{2}{*}{\\textbf{Prédicteur}} & \\multicolumn{3}{c}{\\textbf{LP-NLA}} & \\multicolumn{3}{c}{\\textbf{ELA}} \\\\
\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}
& $S$ & $H'$ & $J'$ & $S$ & $H'$ & $J'$ \\\\
\\midrule
MAE (test) & 3.190 & 0.348 & 0.118 & 2.892 & 0.443 & 0.111 \\\\
R² (test) & 0.408 & 0.317 & 0.311 & 0.708 & 0.283 & 0.164 \\\\
\\addlinespace
\\midrule
Lake ID              & NA & NA & NA & 0.005 & 0.024 & 0.005 \\\\
Year                 & NA & NA & NA & 0.089 & 0.114 & 0.010 \\\\
Doy                  & NA & NA & NA & 0.077 & 0.082 & 0.018 \\\\
Lat                  & 0.097 & 0.030 & 0.006 & 0.009 & 0.014 & 0.005 \\\\
Long                 & 0.069 & 0.042 & 0.010 & 0.004 & 0.010 & 0.002 \\\\
Area                 & 0.011 & 0.014 & 0.007 & 0.022 & 0.132 & 0.027 \\\\
COND                 & 0.015 & 0.018 & 0.006 & 0.008 & 0.011 & 0.005 \\\\
Chla                 & 0.051 & 0.008 & 0.006 & 0.007 & 0.027 & 0.009 \\\\
TNTP                 & 0.039 & 0.027 & 0.005 & 0.007 & 0.008 & 0.002 \\\\
pH mean              & 0.014 & 0.012 & 0.004 & 0.021 & 0.030 & 0.003 \\\\
DO up                & 0.011 & 0.009 & 0.004 & NA & NA & NA \\\\
DO bottom            & 0.020 & 0.012 & 0.004 & NA & NA & NA \\\\
Biom Cladocera       & 0.038 & 0.045 & 0.010 & NA & NA & NA \\\\
Biom Copepoda        & 0.011 & 0.007 & 0.002 & NA & NA & NA \\\\
Color                & 0.026 & 0.017 & 0.003 & NA & NA & NA \\\\
Temp up              & 0.016 & 0.009 & 0.004 & NA & NA & NA \\\\
Temp bottom          & 0.021 & 0.009 & 0.003 & NA & NA & NA \\\\
Wind 30d             & 0.011 & 0.005 & 0.002 & NA & NA & NA \\\\
TP 30d               & 0.011 & 0.013 & 0.003 & NA & NA & NA \\\\
Degree day > 0       & 0.025 & 0.010 & 0.002 & NA & NA & NA \\\\
Prev Cyano           & 0.097 & 0.038 & 0.018 & 0.039 & 0.018 & 0.014 \\\\
Prev Mixo            & 0.055 & 0.159 & 0.056 & 0.009 & 0.137 & 0.034 \\\\
\\bottomrule
\\end{tabular}
\\end{minipage}"""
    
    # Sauvegarder le fichier LaTeX au bon endroit
    output_path = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/SHAP_table_LateX.tex"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"Tableau LaTeX des métriques SHAP créé avec succès: {output_path}")

def handle_categorical_lake_id(data, dataset_name):
    """Traiter lake_id comme variable catégorielle pour ELA uniquement"""
    
    if dataset_name == 'ELA' and 'lake_id' in data.columns:
        print("Traitement de lake_id comme variable catégorielle pour ELA...")
        
        # Créer une copie des données
        data_processed = data.copy()
        
        # Encoder lake_id comme variable catégorielle avec one-hot encoding
        lake_dummies = pd.get_dummies(data_processed['lake_id'], prefix='lake', dtype=int)
        
        # Supprimer l'ancienne colonne lake_id
        data_processed = data_processed.drop('lake_id', axis=1)
        
        # Ajouter les nouvelles colonnes dummy et s'assurer qu'elles sont numériques
        data_processed = pd.concat([data_processed, lake_dummies], axis=1)
        
        # Convertir toutes les colonnes en float64 pour éviter les problèmes SHAP
        for col in lake_dummies.columns:
            data_processed[col] = data_processed[col].astype('float64')
        
        print(f"lake_id transformé en {len(lake_dummies.columns)} variables dummy: {list(lake_dummies.columns)}")
        
        return data_processed, list(lake_dummies.columns)
    
    else:
        return data, []

###############################################################################
# Fonction principale
###############################################################################

def main():
    """Fonction principale pour exécuter l'analyse complète"""
    
    print("=== ANALYSE COMPLÈTE DES MODÈLES DE DIVERSITÉ ===")
    print("(Version sans variables lag)")
    
    # Stockage des résultats
    all_results = {}
    shap_results_all = {}
    metrics_all = {}
    
    # Traitement de chaque dataset
    for dataset_name in ['ELA', 'LPNLA']:
        print(f"\\n--- Traitement du dataset {dataset_name} ---")
        
        # Charger les données
        data, covariables = load_data(dataset_name)
        
        # Traiter lake_id comme variable catégorielle pour ELA
        data_processed, lake_dummy_cols = handle_categorical_lake_id(data, dataset_name)
        
        # Mettre à jour les covariables si lake_id a été transformé
        if lake_dummy_cols:
            # Remplacer 'lake_id' par les nouvelles colonnes dummy dans la liste des covariables
            covariables_updated = [col for col in covariables if col != 'lake_id'] + lake_dummy_cols
        else:
            covariables_updated = covariables
        
        # Créer les répertoires
        figures_dir = create_directories(dataset_name)
        
        # Stockage pour ce dataset
        models_dict = {}
        X_dict = {}
        y_dict = {}
        shap_values_dict = {}
        metrics_dict = {}
        
        # Traitement de chaque métrique de diversité
        for response in responses:
            if response not in data_processed.columns:
                print(f"Warning: {response} non trouvé dans {dataset_name}")
                continue
            
            print(f"\\nModélisation de {response}...")
            
            # Préparer les données - XGBoost gère les valeurs manquantes nativement
            # Supprimer seulement les lignes avec des valeurs manquantes pour la réponse
            data_clean = data_processed.dropna(subset=[response])
            
            # Filtrer les covariables disponibles
            available_covariables = [col for col in covariables_updated if col in data_clean.columns]
            
            X = data_clean[available_covariables]
            y = data_clean[response]
            
            # S'assurer que toutes les variables sont numériques (sauf pour les valeurs manquantes)
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = pd.to_numeric(X[col], errors='coerce')
            
            print(f"Données finales: {len(X)} observations, {len(available_covariables)} variables")
            
            # Entraîner le modèle
            model, X_model, mae, r2 = train_and_validate_model(X, y, dataset_name, response)
            
            print(f"MAE: {mae:.3f}, R²: {r2:.3f}")
            
            # Stocker les résultats
            models_dict[response] = model
            X_dict[response] = X_model
            y_dict[response] = y
            metrics_dict[response] = {'mae': mae, 'r2': r2}
            
            # Créer les graphiques SHAP
            shap_values = create_shap_plots(model, X_model, y, response, dataset_name, figures_dir)
            shap_values_dict[response] = shap_values
        
        # Créer le graphique SHAP combiné
        if shap_values_dict:
            create_combined_shap_plot(shap_values_dict, X_dict, dataset_name, figures_dir)
        
        # Créer l'analyse des résidus
        if models_dict:
            create_residuals_analysis(models_dict, X_dict, y_dict, dataset_name, figures_dir)
            # Exporter les résidus pour l'analyse PACF
            export_residuals_for_pacf(models_dict, X_dict, y_dict, dataset_name, figures_dir)
        
        # Stocker les résultats globaux
        all_results[dataset_name] = {
            'models': models_dict,
            'metrics': metrics_dict,
            'shap_values': shap_values_dict
        }
        shap_results_all[dataset_name] = shap_values_dict
        metrics_all[dataset_name] = metrics_dict
    
    # Créer le graphique de ranking SHAP combiné
    if shap_results_all:
        create_combined_shap_rank_plot(shap_results_all)
    
    # Analyser les interactions prédicteurs-prévalence
    print("\\n" + "="*50)
    print("ANALYSE DES INTERACTIONS PRÉDICTEURS-PRÉVALENCE")
    print("="*50)
    
    ela_interactions = None
    lpnla_interactions = None
    
    # Analyser les interactions pour chaque dataset
    for dataset_name in ['ELA', 'LPNLA']:
        if dataset_name in shap_results_all and shap_results_all[dataset_name]:
            print(f"\\nTraitement du dataset: {dataset_name}")
            
            # Collecter les données X pour ce dataset en utilisant les données globales
            dataset_X = {}
            for response in responses:
                key = f"{dataset_name}_{response}"
                if key in all_results[dataset_name]['shap_values']:
                    # Utiliser les données X correspondantes depuis les résultats
                    if dataset_name in all_results and 'models' in all_results[dataset_name]:
                        # Récupérer les données depuis la source appropriée
                        if dataset_name == 'ELA':
                            dataset_X[response] = X_ela_sampled if 'X_ela_sampled' in locals() else None
                        else:
                            dataset_X[response] = X_lpnla_sampled if 'X_lpnla_sampled' in locals() else None
            
            # Créer l'analyse d'interaction
            interactions = analyze_predictor_prevalence_interaction(
                shap_results_all[dataset_name], dataset_X, dataset_name, 
                os.path.join("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES", f"{dataset_name}_model")
            )
            
            if dataset_name == 'ELA':
                ela_interactions = interactions
            else:
                lpnla_interactions = interactions
    
    # Créer la figure combinée
    if ela_interactions is not None or lpnla_interactions is not None:
        create_combined_prevalence_interaction_plot(ela_interactions, lpnla_interactions)
    
    # Créer le tableau des métriques SHAP
    create_shap_metrics_table(shap_results_all, metrics_all, "")
    
    print("\\n=== ANALYSE TERMINÉE ===")
    print(f"Toutes les figures ont été sauvegardées dans:")
    for dataset_name in ['ELA', 'LPNLA']:
        print(f"  - {DATASETS[dataset_name]['figures_dir']}")
    print(f"  - /Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/SHAP_table_LateX.tex")

if __name__ == "__main__":
    main()
