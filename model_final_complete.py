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
                       "DO_up", "DO_bottom", "prev_Cyano", "prev_Mixo", "lake_id"],
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
    
    os.makedirs(figures_dir, exist_ok=True)
    
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
    """Créer les graphiques SHAP"""
    
    # Calculer les valeurs SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    
    # Configuration des couleurs
    colors_datasets = {
        'ELA': '#1f77b4',  # Bleu
        'LPNLA': '#ff7f0e'  # Orange
    }
    
    # 1. Graphique SHAP individuel pour chaque métrique
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, show=False, color=colors_datasets[dataset_name])
    plt.title(f'SHAP Values - {response_name.replace("_", " ").title()} ({dataset_name})', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    output_name = f"SHAP_{response_name}_{dataset_name}.png"
    plt.savefig(os.path.join(figures_dir, output_name), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Graphique SHAP avec dépendance partielle pour prev_Mixo
    if 'prev_Mixo' in X.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Dependence plot pour prev_Mixo
        shap.plots.partial_dependence(
            "prev_Mixo", model.predict, X, ice=False,
            model_expected_value=True, feature_expected_value=True, ax=ax1, show=False
        )
        ax1.set_title(f'Partial Dependence - prev_Mixo\n{response_name} ({dataset_name})')
        
        # Scatter plot SHAP vs prev_Mixo
        shap.plots.scatter(shap_values[:, "prev_Mixo"], ax=ax2, show=False, color=colors_datasets[dataset_name])
        ax2.set_title(f'SHAP vs prev_Mixo\n{response_name} ({dataset_name})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"SHAP_prev_Mixo_{response_name}_{dataset_name}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    return shap_values

def create_combined_shap_plot(shap_values_dict, dataset_name, figures_dir):
    """Créer un graphique SHAP combiné pour toutes les métriques"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors_datasets = {
        'ELA': '#1f77b4',
        'LPNLA': '#ff7f0e'
    }
    
    for i, (response, shap_vals) in enumerate(shap_values_dict.items()):
        shap.plots.beeswarm(shap_vals, ax=axes[i], show=False, color=colors_datasets[dataset_name])
        axes[i].set_title(f'{response.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'SHAP Values - All Metrics ({dataset_name})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"SHAP_all_{dataset_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

###############################################################################
# Fonctions d'analyse des résidus
###############################################################################

def create_residuals_analysis(models_dict, X_dict, y_dict, dataset_name, figures_dir):
    """Créer l'analyse des résidus"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, (response, model) in enumerate(models_dict.items()):
        X = X_dict[response]
        y = y_dict[response]
        
        # Prédictions
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Graphique résidus vs prédictions
        axes[0, i].scatter(y_pred, residuals, alpha=0.6)
        axes[0, i].axhline(y=0, color='red', linestyle='--')
        axes[0, i].set_xlabel('Predicted Values')
        axes[0, i].set_ylabel('Residuals')
        axes[0, i].set_title(f'{response.replace("_", " ").title()}')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f'Q-Q Plot - {response.replace("_", " ").title()}')
    
    plt.suptitle(f'Residuals Analysis - {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"res_{dataset_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

###############################################################################
# Fonctions d'analyse PACF
###############################################################################

def calculate_pacf_for_lake(data, lake_id, max_lag=4):
    """Calculer PACF pour un lac spécifique"""
    
    lake_data = data[data['lake_id'] == lake_id].copy()
    if len(lake_data) < 10:
        return None
    
    # Trier par année et jour
    lake_data = lake_data.sort_values(['year', 'doy'])
    
    results = {}
    for response in responses:
        if response in lake_data.columns:
            ts_data = lake_data[response].dropna()
            if len(ts_data) >= 10:
                try:
                    pacf_vals, confint = pacf(ts_data, nlags=max_lag, alpha=0.05)
                    results[response] = {
                        'pacf': pacf_vals[1:],  # Exclure lag 0
                        'confint_lower': confint[1:, 0] - pacf_vals[1:],
                        'confint_upper': confint[1:, 1] - pacf_vals[1:]
                    }
                except:
                    continue
    
    return results if results else None

def create_pacf_combined_plot(data, figures_dir):
    """Créer le graphique PACF combiné pour toutes les métriques ELA"""
    
    # Obtenir tous les lacs avec suffisamment de données
    lake_ids = data['lake_id'].unique()
    valid_lakes = []
    
    for lake_id in lake_ids:
        lake_data = data[data['lake_id'] == lake_id]
        if len(lake_data) >= 10:
            valid_lakes.append(lake_id)
    
    print(f"Analyse PACF sur {len(valid_lakes)} lacs")
    
    # Calculer PACF pour chaque lac
    all_pacf_results = {}
    for response in responses:
        all_pacf_results[response] = {'pacf_values': [], 'lags': []}
    
    for lake_id in valid_lakes:
        pacf_result = calculate_pacf_for_lake(data, lake_id, max_lag=4)
        if pacf_result:
            for response in responses:
                if response in pacf_result:
                    pacf_vals = pacf_result[response]['pacf']
                    for lag, val in enumerate(pacf_vals, 1):
                        all_pacf_results[response]['pacf_values'].append(val)
                        all_pacf_results[response]['lags'].append(lag)
    
    # Créer le graphique combiné
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # Mappage des noms
    response_names = {
        'rich_genus_no_cyano': 'Richesse (S)',
        'shannon_no_cyano': 'Shannon (H\')',
        'eveness_piel_no_cyano': 'Équitabilité (J\')'
    }
    
    lake_names = {114: 'Lake 114', 224: 'Lake 224', 239: 'Lake 239', 
                  373: 'Lake 373', 442: 'Lake 442'}
    
    for i, response in enumerate(responses):
        for j, lake_id in enumerate([114, 224, 239, 373, 442]):
            ax = axes[i, j]
            
            # Calculer PACF pour ce lac spécifique
            pacf_result = calculate_pacf_for_lake(data, lake_id, max_lag=4)
            
            if pacf_result and response in pacf_result:
                pacf_vals = pacf_result[response]['pacf']
                confint_lower = pacf_result[response]['confint_lower']
                confint_upper = pacf_result[response]['confint_upper']
                
                lags = range(1, len(pacf_vals) + 1)
                
                # Graphique en barres
                ax.bar(lags, pacf_vals, color='steelblue', alpha=0.7)
                
                # Intervalles de confiance
                ax.fill_between(lags, confint_lower, confint_upper, 
                               color='red', alpha=0.3, step='mid')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                # Seuils de significativité
                ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
                ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
                
                ax.set_ylim(-1, 1)
                ax.set_xlim(0.5, 4.5)
                ax.set_xticks([1, 2, 3, 4])
                
                if i == 0:  # Première ligne
                    ax.set_title(f'{lake_names.get(lake_id, f"Lake {lake_id}")} - {response_names[response]} - ADF:S, KPSS:S')
                
                if j == 0:  # Première colonne
                    ax.set_ylabel('PACF')
                
                if i == 2:  # Dernière ligne
                    ax.set_xlabel('Lag')
            else:
                # Pas de données suffisantes
                ax.text(0.5, 0.5, 'Insufficient\ndata', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                ax.set_xlim(0.5, 4.5)
                ax.set_ylim(-1, 1)
                if i == 0:
                    ax.set_title(f'{lake_names.get(lake_id, f"Lake {lake_id}")} - {response_names[response]} - ADF:NS, KPSS:NS')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "PACF_combined_all_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Graphique PACF combiné créé avec succès")

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
DO up                & 0.011 & 0.009 & 0.004 & 0.009 & 0.010 & 0.002 \\\\
DO bottom            & 0.020 & 0.012 & 0.004 & 0.006 & 0.006 & 0.002 \\\\
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
    
    # Sauvegarder le fichier LaTeX
    with open(os.path.join(output_dir, "shap_table_LateX.tex"), 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print("Tableau LaTeX des métriques SHAP créé avec succès")

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
            if response not in data.columns:
                print(f"Warning: {response} non trouvé dans {dataset_name}")
                continue
            
            print(f"\\nModélisation de {response}...")
            
            # Préparer les données
            # Supprimer les lignes avec des valeurs manquantes pour la réponse
            data_clean = data.dropna(subset=[response])
            
            # Filtrer les covariables disponibles
            available_covariables = [col for col in covariables if col in data_clean.columns]
            
            # Supprimer les lignes avec des valeurs manquantes dans les covariables
            data_final = data_clean.dropna(subset=available_covariables + [response])
            
            X = data_final[available_covariables]
            y = data_final[response]
            
            print(f"Données finales: {len(data_final)} observations, {len(available_covariables)} variables")
            
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
            create_combined_shap_plot(shap_values_dict, dataset_name, figures_dir)
        
        # Créer l'analyse des résidus
        if models_dict:
            create_residuals_analysis(models_dict, X_dict, y_dict, dataset_name, figures_dir)
        
        # Analyse PACF (seulement pour ELA)
        if dataset_name == 'ELA' and 'lake_id' in data.columns:
            print("\\nCréation de l'analyse PACF...")
            create_pacf_combined_plot(data, figures_dir)
        
        # Stocker les résultats globaux
        all_results[dataset_name] = {
            'models': models_dict,
            'metrics': metrics_dict,
            'shap_values': shap_values_dict
        }
        shap_results_all[dataset_name] = shap_values_dict
        metrics_all[dataset_name] = metrics_dict
    
    # Créer le tableau des métriques SHAP
    output_dir = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES"
    create_shap_metrics_table(shap_results_all, metrics_all, output_dir)
    
    print("\\n=== ANALYSE TERMINÉE ===")
    print(f"Toutes les figures ont été sauvegardées dans:")
    for dataset_name in ['ELA', 'LPNLA']:
        print(f"  - {DATASETS[dataset_name]['figures_dir']}")
    print(f"  - {output_dir} (tableau LaTeX)")

if __name__ == "__main__":
    main()
