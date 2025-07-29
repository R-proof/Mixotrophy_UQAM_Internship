import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
from scipy import stats
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# Configuration et paramètres
###############################################################################

# Configuration des datasets
DATASETS = {
    'ELA': {
        'path': "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/DATA_MODEL/ELA_py.csv",
        'covariable': ["year", "doy", "COND_uS.cm",
                       "Chla_ug.L", "TNTP_mg.L", "pH_mean",
                       "DO_up", "DO_bottom", "prev_Cyano", "prev_Mixo", "lake_id"],
        'validation_type': 'leave_one_year_out',
        'figures_dir': "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/ELA_model",
        'output_prefix': 'ELA'
    },
    'LPNLA': {
        'path': "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/DATA_MODEL/LP_NLA_py.csv",
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
    """Charge les données pour un dataset donné."""
    config = DATASETS[dataset_name]
    data = pd.read_csv(config['path'])
    
    # Nettoyage des données
    X = data[config['covariable']].copy()
    y_dict = {}
    
    for response in responses:
        y_dict[response] = data[response].copy()
        
        # Élimination des valeurs aberrantes (au-delà de 3 écarts-types)
        mean_val = y_dict[response].mean()
        std_val = y_dict[response].std()
        outlier_mask = np.abs(y_dict[response] - mean_val) > 3 * std_val
        
        X = X[~outlier_mask]
        for resp in y_dict:
            y_dict[resp] = y_dict[resp][~outlier_mask]
    
    return X, y_dict

def train_xgboost_model(X, y, response_name, validation_type='kfold'):
    """Entraîne un modèle XGBoost avec validation croisée."""
    
    # Configuration des paramètres selon la variable de réponse
    if response_name == 'rich_genus_no_cyano':
        params = {
            'objective': 'reg:tweedie',
            'tweedie_variance_power': 1.5,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    else:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    model = xgb.XGBRegressor(**params)
    
    if validation_type == 'leave_one_year_out':
        # Pour ELA : validation leave-one-year-out
        years = X['year'].unique()
        scores = []
        
        for year in years:
            test_mask = X['year'] == year
            train_mask = ~test_mask
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            # Enlever la variable 'year' pour l'entraînement
            X_train_no_year = X_train.drop('year', axis=1)
            X_test_no_year = X_test.drop('year', axis=1)
            
            model.fit(X_train_no_year, y_train)
            y_pred = model.predict(X_test_no_year)
            score = r2_score(y_test, y_pred)
            scores.append(score)
        
        # Entraînement final sur toutes les données
        X_final = X.drop('year', axis=1)
        model.fit(X_final, y)
        
    else:
        # Pour LPNLA : validation croisée k-fold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            scores.append(score)
        
        # Entraînement final sur toutes les données
        model.fit(X, y)
    
    return model, np.mean(scores), np.std(scores)

def calculate_shap_values(model, X, dataset_name):
    """Calcule les valeurs SHAP pour un modèle donné."""
    
    # Préparation des données selon le dataset
    if dataset_name == 'ELA' and 'year' in X.columns:
        X_shap = X.drop('year', axis=1)
    else:
        X_shap = X.copy()
    
    # Calcul des valeurs SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    return shap_values, X_shap

def create_shap_plots(shap_values, X_shap, response_name, dataset_name, figures_dir):
    """Crée les graphiques SHAP pour une variable de réponse."""
    
    # Assurer que le répertoire existe
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{dataset_name}_{response_name}_shap_summary.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot (importance)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{dataset_name}_{response_name}_shap_importance.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dependence plots pour les variables les plus importantes
    feature_importance = np.abs(shap_values).mean(0)
    top_features = np.argsort(feature_importance)[-5:]  # Top 5 features
    
    for i, feature_idx in enumerate(top_features):
        feature_name = X_shap.columns[feature_idx]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature_idx, shap_values, X_shap, show=False)
        plt.title(f'SHAP Dependence Plot - {feature_name}')
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/{dataset_name}_{response_name}_dependence_{feature_name}.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_combined_shap_plots(datasets_results, figures_base_dir):
    """Crée les graphiques SHAP combinés pour tous les datasets et variables."""
    
    # Assurer que le répertoire existe
    os.makedirs(figures_base_dir, exist_ok=True)
    
    # 1. Graphique d'importance combiné
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SHAP Feature Importance - Tous Datasets et Variables', fontsize=16)
    
    for i, response in enumerate(responses):
        for j, dataset in enumerate(['ELA', 'LPNLA']):
            ax = axes[j, i]
            
            if dataset in datasets_results and response in datasets_results[dataset]:
                shap_values = datasets_results[dataset][response]['shap_values']
                X_shap = datasets_results[dataset][response]['X_shap']
                
                feature_importance = np.abs(shap_values).mean(0)
                sorted_idx = np.argsort(feature_importance)
                
                ax.barh(range(len(feature_importance)), feature_importance[sorted_idx])
                ax.set_yticks(range(len(feature_importance)))
                ax.set_yticklabels([X_shap.columns[i] for i in sorted_idx])
                ax.set_title(f'{dataset} - {response}')
                ax.set_xlabel('Mean |SHAP value|')
            else:
                ax.text(0.5, 0.5, 'Données non disponibles', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dataset} - {response}')
    
    plt.tight_layout()
    plt.savefig(f"{figures_base_dir}/combined_shap_importance.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap des interactions SHAP
    for dataset in ['ELA', 'LPNLA']:
        if dataset in datasets_results:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'SHAP Interaction Heatmaps - {dataset}', fontsize=16)
            
            for i, response in enumerate(responses):
                ax = axes[i]
                
                if response in datasets_results[dataset]:
                    shap_values = datasets_results[dataset][response]['shap_values']
                    X_shap = datasets_results[dataset][response]['X_shap']
                    
                    # Calculer la matrice d'interaction
                    feature_importance = np.abs(shap_values).mean(0)
                    top_features_idx = np.argsort(feature_importance)[-8:]  # Top 8 features
                    
                    interaction_matrix = np.zeros((len(top_features_idx), len(top_features_idx)))
                    
                    for j, feat1_idx in enumerate(top_features_idx):
                        for k, feat2_idx in enumerate(top_features_idx):
                            if j != k:
                                # Corrélation entre les valeurs SHAP des deux features
                                corr = np.corrcoef(shap_values[:, feat1_idx], 
                                                 shap_values[:, feat2_idx])[0, 1]
                                interaction_matrix[j, k] = corr
                    
                    top_feature_names = [X_shap.columns[idx] for idx in top_features_idx]
                    
                    im = ax.imshow(interaction_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                    ax.set_xticks(range(len(top_feature_names)))
                    ax.set_yticks(range(len(top_feature_names)))
                    ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
                    ax.set_yticklabels(top_feature_names)
                    ax.set_title(f'{response}')
                    
                    # Ajouter les valeurs dans les cellules
                    for j in range(len(top_feature_names)):
                        for k in range(len(top_feature_names)):
                            text = ax.text(k, j, f'{interaction_matrix[j, k]:.2f}',
                                         ha="center", va="center", color="black", fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'Données non disponibles', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{response}')
            
            plt.tight_layout()
            plt.savefig(f"{figures_base_dir}/{dataset}_shap_interactions.png", 
                        dpi=300, bbox_inches='tight')
            plt.close()

def generate_latex_tables(datasets_results, figures_base_dir):
    """Génère les tableaux LaTeX avec les métriques et valeurs SHAP."""
    
    # Tableau des métriques de performance
    metrics_table = "\\begin{table}[htbp]\n\\centering\n"
    metrics_table += "\\begin{tabular}{|l|l|c|c|}\n\\hline\n"
    metrics_table += "Dataset & Variable & R² moyen & Écart-type \\\\\n\\hline\n"
    
    for dataset in ['ELA', 'LPNLA']:
        if dataset in datasets_results:
            for response in responses:
                if response in datasets_results[dataset]:
                    r2_mean = datasets_results[dataset][response]['r2_mean']
                    r2_std = datasets_results[dataset][response]['r2_std']
                    metrics_table += f"{dataset} & {response} & {r2_mean:.3f} & {r2_std:.3f} \\\\\n"
    
    metrics_table += "\\hline\n\\end{tabular}\n"
    metrics_table += "\\caption{Métriques de performance des modèles XGBoost}\n"
    metrics_table += "\\label{tab:model_performance}\n\\end{table}\n"
    
    # Sauvegarder le tableau des métriques
    with open(f"{figures_base_dir}/model_performance_table.tex", 'w') as f:
        f.write(metrics_table)
    
    # Tableau des importances SHAP
    shap_table = "\\begin{table}[htbp]\n\\centering\n"
    shap_table += "\\begin{tabular}{|l|l|l|c|}\n\\hline\n"
    shap_table += "Dataset & Variable & Feature & Importance SHAP \\\\\n\\hline\n"
    
    for dataset in ['ELA', 'LPNLA']:
        if dataset in datasets_results:
            for response in responses:
                if response in datasets_results[dataset]:
                    shap_values = datasets_results[dataset][response]['shap_values']
                    X_shap = datasets_results[dataset][response]['X_shap']
                    
                    feature_importance = np.abs(shap_values).mean(0)
                    sorted_idx = np.argsort(feature_importance)[::-1]  # Ordre décroissant
                    
                    # Top 5 features pour chaque combinaison
                    for i in range(min(5, len(sorted_idx))):
                        feature_name = X_shap.columns[sorted_idx[i]]
                        importance = feature_importance[sorted_idx[i]]
                        shap_table += f"{dataset} & {response} & {feature_name} & {importance:.4f} \\\\\n"
    
    shap_table += "\\hline\n\\end{tabular}\n"
    shap_table += "\\caption{Top 5 des importances SHAP par dataset et variable}\n"
    shap_table += "\\label{tab:shap_importance}\n\\end{table}\n"
    
    # Sauvegarder le tableau SHAP
    with open(f"{figures_base_dir}/shap_importance_table.tex", 'w') as f:
        f.write(shap_table)
    
    print(f"Tableaux LaTeX sauvegardés dans {figures_base_dir}/")

def process_single_combination(args):
    """Traite une seule combinaison dataset/response (pour parallélisation)."""
    dataset_name, response_name = args
    
    print(f"Traitement: {dataset_name} - {response_name}")
    
    # Chargement des données
    X, y_dict = load_data(dataset_name)
    y = y_dict[response_name]
    
    # Entraînement du modèle
    config = DATASETS[dataset_name]
    model, r2_mean, r2_std = train_xgboost_model(X, y, response_name, config['validation_type'])
    
    # Calcul des valeurs SHAP
    shap_values, X_shap = calculate_shap_values(model, X, dataset_name)
    
    # Création des graphiques SHAP individuels
    create_shap_plots(shap_values, X_shap, response_name, dataset_name, config['figures_dir'])
    
    print(f"Terminé: {dataset_name} - {response_name} (R² = {r2_mean:.3f} ± {r2_std:.3f})")
    
    return {
        'dataset': dataset_name,
        'response': response_name,
        'model': model,
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'shap_values': shap_values,
        'X_shap': X_shap
    }

def run_analysis(dataset_filter=None):
    """Lance l'analyse complète avec parallélisation."""
    
    print("Début de l'analyse SHAP unifiée")
    start_time = time.time()
    
    # Préparation des combinaisons à traiter
    combinations = []
    datasets_to_process = [dataset_filter] if dataset_filter else ['ELA', 'LPNLA']
    
    for dataset_name in datasets_to_process:
        if dataset_name in DATASETS:
            for response_name in responses:
                combinations.append((dataset_name, response_name))
    
    print(f"Traitement de {len(combinations)} combinaisons dataset/variable")
    
    # Traitement parallèle
    if len(combinations) > 1:
        n_cores = min(mp.cpu_count() - 1, 8)
        print(f"Utilisation de {n_cores} cœurs pour le traitement parallèle")
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(process_single_combination, combinations))
    else:
        # Traitement séquentiel pour un seul élément
        results = [process_single_combination(combinations[0])]
    
    # Organisation des résultats
    datasets_results = {}
    for result in results:
        dataset = result['dataset']
        response = result['response']
        
        if dataset not in datasets_results:
            datasets_results[dataset] = {}
        
        datasets_results[dataset][response] = {
            'model': result['model'],
            'r2_mean': result['r2_mean'],
            'r2_std': result['r2_std'],
            'shap_values': result['shap_values'],
            'X_shap': result['X_shap']
        }
    
    print(f"Traitement parallèle terminé en {time.time() - start_time:.2f} secondes")
    
    # Génération des graphiques combinés et tableaux LaTeX
    figures_base_dir = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES"
    
    print("Génération des graphiques combinés...")
    create_combined_shap_plots(datasets_results, figures_base_dir)
    
    print("Génération des tableaux LaTeX...")
    generate_latex_tables(datasets_results, figures_base_dir)
    
    print(f"Analyse {dataset_filter or 'complète'} terminée en {time.time() - start_time:.2f} secondes au total")
    
    return datasets_results

###############################################################################
# Point d'entrée principal
###############################################################################

def main():
    """Point d'entrée principal avec arguments en ligne de commande."""
    
    parser = argparse.ArgumentParser(description='Analyse SHAP unifiée pour les datasets ELA et LP-NLA')
    parser.add_argument('--dataset', choices=['ELA', 'LPNLA'], 
                       help='Dataset spécifique à traiter (optionnel)')
    parser.add_argument('--figures-only', action='store_true',
                       help='Générer seulement les figures combinées (nécessite des données pré-calculées)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANALYSE SHAP UNIFIÉE - MODÈLES XGBOOST")
    print("="*80)
    
    global_start_time = time.time()
    
    if not args.figures_only:
        # Analyse complète
        results = run_analysis(args.dataset)
        print(f"\nAnalyse terminée avec succès en {time.time() - global_start_time:.2f} secondes")
        print(f"Figures sauvegardées dans les répertoires respectifs")
        print(f"Graphiques combinés et tableaux dans /Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/")
    else:
        print("Mode figures seulement - non implémenté pour cette version")

if __name__ == "__main__":
    main()

###############################################################################
# Fonctions utilitaires
###############################################################################

def load_data(dataset_name):
    """Charger les données pour un dataset spécifique (variables lag supprimées)"""
    dataset_config = DATASETS[dataset_name]
    df = pd.read_csv(dataset_config['path'])
    print(f"Données {dataset_name} chargées: {len(df)} observations, {len(dataset_config['covariable'])} covariables")
    
    # Variables lag supprimées selon les demandes de simplification
    # df_with_lags, lag_vars = add_lagged_response_variables(df, dataset_name)
    
    # Récupérer la liste des covariables (sans lag)
    updated_covariables = dataset_config['covariable']
    print(f"Covariables finales: {len(updated_covariables)} variables (sans lag)")
    
    return df, updated_covariables

def create_directories(dataset_name):
    """Créer les répertoires de sortie"""
    dataset_config = DATASETS[dataset_name]
    figures_dir = dataset_config['figures_dir']
    tables_dir = figures_dir.replace('/figures_model/', '/figures_model/tables/')
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    return figures_dir, tables_dir

def get_xgb_params(response_name):
    """Retourner les paramètres XGBoost appropriés selon la métrique"""
    if 'rich' in response_name:
        # Paramètres optimisés pour Tweedie (richesse)
        return {
            'objective': 'reg:tweedie',
            'tweedie_variance_power': 1.2,  # Optimal pour les données de richesse
            'n_estimators': 100,  # Augmenté pour Tweedie
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
        # Paramètres pour Shannon et Équitabilité (MSE)
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 50,
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

###############################################################################
# Fonctions de traitement parallèle
###############################################################################

def process_response_year_ela(args):
    """Fonction pour traiter une combinaison réponse-année en parallèle (ELA)"""
    resp, test_year, df_data, covariable = args
    
    # Séparer les données d'entraînement/validation (85%) du test final (15%)
    train_val_df = df_data[df_data['year'] != test_year]
    test_df = df_data[df_data['year'] == test_year]
    
    # Supprimer les lignes avec des valeurs manquantes dans les covariables
    train_val_df = train_val_df.dropna(subset=covariable)
    test_df = test_df.dropna(subset=covariable)
    
    # Vérifier qu'il reste assez de données
    if len(train_val_df) < 10 or len(test_df) < 1:
        print(f"Pas assez de données pour {resp}, année {test_year} après suppression des NaN")
        return [], [], {'mae': np.nan, 'r2': np.nan, 'test_year': test_year, 'n_test': 0}
    
    X_train_val = train_val_df[covariable].copy()
    y_train_val = train_val_df[resp]
    X_test = test_df[covariable].copy()
    y_test = test_df[resp]
    
    # Convertir la richesse en entiers pour la distribution de Poisson
    if resp == 'rich_genus_no_cyano':
        y_train_val = np.round(y_train_val).astype(int)
        y_test = np.round(y_test).astype(int)
    
    # Encoder les variables catégorielles
    if 'lake_id' in X_train_val.columns:
        # Encoder lake_id en numérique
        lake_mapping = {lake: i for i, lake in enumerate(sorted(df_data['lake_id'].unique()))}
        X_train_val['lake_id'] = X_train_val['lake_id'].map(lake_mapping)
        X_test['lake_id'] = X_test['lake_id'].map(lake_mapping)
    
    # Modèle optimisé avec paramètres appropriés selon la métrique
    model = xgb.XGBRegressor(**get_xgb_params(resp))
    model.fit(X_train_val, y_train_val)
    
    # Prédictions pour calculer les métriques
    y_pred_train_val = model.predict(X_train_val)
    y_pred_test = model.predict(X_test)
    
    # Optimisation SHAP : échantillonnage stratifié pour garantir tous les lacs
    sample_size = min(500, len(X_train_val))
    
    # Échantillonnage stratifié par lac pour ELA (garantir tous les lacs)
    if 'lake_id' in X_train_val.columns:
        unique_lakes = X_train_val['lake_id'].unique()
        samples_per_lake = max(1, sample_size // len(unique_lakes))
        
        sample_indices = []
        for lake in unique_lakes:
            lake_indices = X_train_val[X_train_val['lake_id'] == lake].index
            n_samples = min(samples_per_lake, len(lake_indices))
            lake_sample = np.random.choice(lake_indices, n_samples, replace=False)
            sample_indices.extend(lake_sample)
        
        # Compléter jusqu'à sample_size si nécessaire
        remaining_size = sample_size - len(sample_indices)
        if remaining_size > 0:
            remaining_indices = X_train_val.index.difference(sample_indices)
            if len(remaining_indices) > 0:
                additional_samples = np.random.choice(remaining_indices, 
                                                    min(remaining_size, len(remaining_indices)), 
                                                    replace=False)
                sample_indices.extend(additional_samples)
        
        sample_idx = sample_indices[:sample_size]
        X_train_val_sample = X_train_val.loc[sample_idx]
        
        # Récupérer les valeurs originales (non encodées) pour les variables catégorielles
        X_train_val_sample_original = train_val_df[covariable].loc[sample_idx]
    else:
        # Échantillonnage simple pour les autres cas
        sample_idx = np.random.choice(len(X_train_val), sample_size, replace=False)
        X_train_val_sample = X_train_val.iloc[sample_idx]
        
        # Récupérer les valeurs originales (non encodées) pour les variables catégorielles
        X_train_val_sample_original = train_val_df[covariable].iloc[sample_idx]
    
    explainer = shap.Explainer(model, X_train_val_sample)
    shap_values = explainer(X_train_val_sample)
    
    # Création des données SHAP avec les valeurs originales pour les variables catégorielles
    shap_data = []
    for i, col in enumerate(X_train_val_sample.columns):
        for j in range(len(X_train_val_sample)):
            # Utiliser les valeurs originales pour les variables catégorielles
            if col in ['lake_id']:
                value = X_train_val_sample_original[col].iloc[j]
            else:
                value = X_train_val_sample[col].iloc[j]
            
            shap_data.append({
                'Valeur': value,
                'SHAP': shap_values.values[j, i],
                'Variable': col,
                'Année': test_year,
                'Métrique': resp
            })
    
    # Calcul des ranks
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranks = mean_abs_shap.argsort()[::-1]
    rank_data = []
    for i, idx in enumerate(ranks):
        rank_data.append({
            'Predictor': X_train_val_sample.columns[idx],
            'Rank': i + 1,
            'Response': resp
        })
    
    # Calculer MAE et R² pour cette année de test
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    metrics = {
        'mae': mae_test,
        'r2': r2_test,
        'test_year': test_year,
        'n_test': len(y_test)
    }
    
    return shap_data, rank_data, metrics

def process_response_fold_lpnla(args):
    """Fonction pour traiter une combinaison réponse-fold en parallèle (LP-NLA)"""
    resp, fold_idx, train_idx, val_idx, df_data, covariable = args
    
    # Données d'entraînement et validation
    X_train = df_data[covariable].iloc[train_idx].copy()
    y_train = df_data[resp].iloc[train_idx]
    X_val = df_data[covariable].iloc[val_idx].copy()
    y_val = df_data[resp].iloc[val_idx]
    
    # Convertir la richesse en entiers pour la distribution de Poisson
    if resp == 'rich_genus_no_cyano':
        y_train = np.round(y_train).astype(int)
        y_val = np.round(y_val).astype(int)
    
    # Modèle optimisé avec paramètres appropriés selon la métrique
    model = xgb.XGBRegressor(**get_xgb_params(resp))
    model.fit(X_train, y_train)
    
    # Prédictions pour calculer les métriques
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Optimisation SHAP : échantillonnage pour réduire le calcul
    sample_size = min(500, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train.iloc[sample_idx]
    
    explainer = shap.Explainer(model, X_train_sample)
    shap_values = explainer(X_train_sample)
    
    # Création des données SHAP
    shap_data = []
    for i, col in enumerate(X_train_sample.columns):
        for j in range(len(X_train_sample)):
            value = X_train_sample[col].iloc[j]
            
            shap_data.append({
                'Valeur': value,
                'SHAP': shap_values.values[j, i],
                'Variable': col,
                'Fold': fold_idx,
                'Métrique': resp
            })
    
    # Calcul des ranks
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranks = mean_abs_shap.argsort()[::-1]
    rank_data = []
    for i, idx in enumerate(ranks):
        rank_data.append({
            'Predictor': X_train_sample.columns[idx],
            'Rank': i + 1,
            'Response': resp
        })
    
    # Calculer MAE et R² pour ce fold
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    
    metrics = {
        'mae': mae_val,
        'r2': r2_val,
        'fold': fold_idx,
        'n_val': len(y_val)
    }
    
    return shap_data, rank_data, metrics

###############################################################################
# Fonctions de visualisation
###############################################################################

def get_axis_limits(data, percentile_lower=5, percentile_upper=95):
    """Calculer les limites d'axes basées sur les percentiles pour éviter les outliers"""
    if len(data) == 0:
        return None, None
    
    # Nettoyer les données (enlever NaN et infinis)
    clean_data = data.dropna()
    clean_data = clean_data[np.isfinite(clean_data)]
    
    if len(clean_data) == 0:
        return None, None
    
    lower_bound = np.percentile(clean_data, percentile_lower)
    upper_bound = np.percentile(clean_data, percentile_upper)
    
    # Vérifier que les bornes sont finies
    if not (np.isfinite(lower_bound) and np.isfinite(upper_bound)):
        return None, None
    
    # Ajouter une marge plus importante de 15% pour éviter le crop
    range_data = upper_bound - lower_bound
    if range_data > 0:
        margin = range_data * 0.15  # Augmenté de 10% à 15%
        return lower_bound - margin, upper_bound + margin
    else:
        return lower_bound - 1.5, upper_bound + 1.5  # Augmenté de 1 à 1.5

def create_combined_shap_plot(shap_dfs, responses, covariable, figures_dir, dataset_name):
    """Créer la figure SHAP combinée pour toutes les métriques avec deux axes des ordonnées"""
    print(f"Génération de la figure SHAP combinée pour {dataset_name}")
    
    # Mapper les noms de prédicteurs pour l'affichage
    predictor_mapping = {
        'year': 'Year',
        'doy': 'Doy', 
        'lat': 'Lat',
        'long': 'Long',
        'area_m2': 'Area',
        'COND_uS.cm': 'COND',
        'Chla_ug.L': 'Chla',
        'TNTP_mg.L': 'TNTP',
        'pH_mean': 'pH mean',
        'DO_up': 'DO up',
        'DO_bottom': 'DO bottom',
        'Biom_Cladocera_ugL': 'Biom Cladocera',
        'Biom_Copepoda_ugL': 'Biom Copepoda',
        'color': 'Color',
        'temp_up': 'Temp up',
        'temp_bottom': 'Temp bottom',
        'wind_30d': 'Wind 30d',
        'tp_30d': 'TP 30d',
        'degree_day_thr0': 'Degree day > 0',
        'prev_Cyano': 'Prev Cyano',
        'prev_Mixo': 'Prev Mixo',
        'lake_id': 'Lake ID'
        # Variables lag supprimées selon les demandes de simplification
    }
    
    shap_all_df = pd.concat(shap_dfs, axis=0)
    
    # Calculer l'importance moyenne absolue pour chaque prédicteur
    importance_by_var = shap_all_df.groupby('Variable')['SHAP'].apply(lambda x: np.abs(x).mean()).sort_values(ascending=False)
    
    # Sélectionner les 9 prédicteurs les plus importants (ordre décroissant d'importance)
    top_predictors = importance_by_var.head(9).index.tolist()
    print(f"Top 9 prédicteurs pour {dataset_name} (ordre décroissant d'importance): {top_predictors}")
    
    # Filtrer les données pour ne garder que les top 9 prédicteurs
    shap_all_df = shap_all_df[shap_all_df['Variable'].isin(top_predictors)]
    shap_all_df['Variable'] = pd.Categorical(shap_all_df['Variable'], categories=top_predictors, ordered=True)
    shap_all_df = shap_all_df.sort_values('Variable')
    
    # Définir les couleurs fixes pour chaque métrique (colorblind-friendly)
    metric_colors = {
        'rich_genus_no_cyano': '#E69F00',    # Orange
        'shannon_no_cyano': '#56B4E9',       # Bleu ciel
        'eveness_piel_no_cyano': '#009E73'   # Vert bleu
    }
    
    # Configuration uniforme : 3x3 pour les 9 prédicteurs les plus importants
    n_vars = 9  # Fixé à 9 prédicteurs
    n_cols = 3  # 3 colonnes pour avoir une grille 3x3
    fig_size = (18, 15)  # Taille augmentée pour un meilleur rendu
    
    n_rows = 3  # 3 lignes pour avoir une grille 3x3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Séparer les métriques : Shannon et Équitabilité (axe gauche), Richesse (axe droit)
    shannon_eveness_metrics = ['shannon_no_cyano', 'eveness_piel_no_cyano']
    richness_metric = ['rich_genus_no_cyano']
    
    # Lettres pour identifier chaque subplot par ordre d'importance
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    
    # Calculer les échelles SHAP communes pour harmonisation intra-rangée
    print(f"Calcul des échelles SHAP communes pour harmonisation intra-rangée...")
    
    # Pour Shannon et Équitabilité (axe gauche)
    all_shannon_eveness_shap = []
    for metric in shannon_eveness_metrics:
        if metric in responses:
            metric_shap_values = shap_all_df[shap_all_df['Métrique'] == metric]['SHAP'].values
            all_shannon_eveness_shap.extend(metric_shap_values)

    if all_shannon_eveness_shap:
        shannon_eveness_min = np.percentile(all_shannon_eveness_shap, 2.5)
        shannon_eveness_max = np.percentile(all_shannon_eveness_shap, 97.5)
        shannon_eveness_abs_max = max(abs(shannon_eveness_min), abs(shannon_eveness_max))
        # Ajouter une marge adaptée selon le dataset pour éviter le crop des données
        if dataset_name == "LPNLA":
            # Pour LPNLA, augmenter davantage les limites de Shannon et Équitabilité
            shannon_eveness_abs_max *= 1.25  # 25% de marge au lieu de 15%
        else:
            shannon_eveness_abs_max *= 1.15  # 15% de marge pour ELA
        common_shannon_eveness_lim = [-shannon_eveness_abs_max, shannon_eveness_abs_max]
    else:
        common_shannon_eveness_lim = [-1, 1]

    # Pour Richesse (axe droit)
    all_richness_shap = []
    for metric in richness_metric:
        if metric in responses:
            metric_shap_values = shap_all_df[shap_all_df['Métrique'] == metric]['SHAP'].values
            all_richness_shap.extend(metric_shap_values)

    if all_richness_shap:
        richness_min = np.percentile(all_richness_shap, 2.5)
        richness_max = np.percentile(all_richness_shap, 97.5)
        richness_abs_max = max(abs(richness_min), abs(richness_max))
        # Ajouter une marge de 15% pour éviter le crop des données
        richness_abs_max *= 1.15
        common_richness_lim = [-richness_abs_max, richness_abs_max]
    else:
        common_richness_lim = [-1, 1]
    
    print(f"  Échelle commune Shannon/Équitabilité: {common_shannon_eveness_lim}")
    print(f"  Échelle commune Richesse: {common_richness_lim}")
    
    # Affichage des prédicteurs par ordre d'importance décroissant (position 0,0 = plus important)
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
            # Utiliser le nom traduit pour l'affichage
            display_name = predictor_mapping.get(var, var)
            ax1.set_xlabel(display_name, fontsize=18)
            ax1.set_xticks(range(len(unique_values)))
            ax1.set_xticklabels(unique_values, fontsize=16)
            
        else:
            # Traitement normal pour les prédicteurs continus
            # Calculer les limites d'axes basées sur les percentiles pour ce prédicteur
            x_min, x_max = get_axis_limits(temp_df['Valeur'])
            
            # Règles spéciales pour conserver les bornes entières de certains prédicteurs ELA
            if dataset_name == "ELA" and var in ['prev_Mixo', 'doy']:
                # Pour prev_Mixo et doy dans ELA, garder les bornes entières des données
                x_min = temp_df['Valeur'].min()
                x_max = temp_df['Valeur'].max()
                print(f"  Conservation des bornes entières pour {var}: [{x_min:.1f}, {x_max:.1f}]")
            
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
            # Utiliser le nom traduit pour l'affichage
            display_name = predictor_mapping.get(var, var)
            ax1.set_xlabel(display_name, fontsize=18)
            
            # Appliquer les limites d'axes calculées pour borner aux valeurs principales
            if x_min is not None and x_max is not None:
                ax1.set_xlim(x_min, x_max)
                if ax2 is not None:
                    ax2.set_xlim(x_min, x_max)
        
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
        
        # Appliquer les échelles communes pour harmonisation intra-rangée
        if shannon_eveness_plotted:
            ax1.set_ylim(common_shannon_eveness_lim[0], common_shannon_eveness_lim[1])
        
        if richness_plotted and ax2 is not None:
            ax2.set_ylim(common_richness_lim[0], common_richness_lim[1])
        
        # Ajout de la lettre pour identifier le subplot (par ordre d'importance)
        ax1.text(0.02, 0.98, letters[i], transform=ax1.transAxes, 
                fontsize=16, fontweight='bold', verticalalignment='top')
        
        # Améliorer la taille des ticks
        ax1.tick_params(axis='x', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/SHAP_all_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_shap_plots(shap_dfs, responses, covariable, figures_dir, dataset_name):
    """Créer des figures SHAP individuelles pour chaque métrique de diversité"""
    print(f"Génération des figures SHAP individuelles pour {dataset_name}")
    
    # Mapper les noms de prédicteurs pour l'affichage
    predictor_mapping = {
        'year': 'Year',
        'doy': 'Doy', 
        'lat': 'Lat',
        'long': 'Long',
        'area_m2': 'Area',
        'COND_uS.cm': 'COND',
        'Chla_ug.L': 'Chla',
        'TNTP_mg.L': 'TNTP',
        'pH_mean': 'pH mean',
        'DO_up': 'DO up',
        'DO_bottom': 'DO bottom',
        'Biom_Cladocera_ugL': 'Biom Cladocera',
        'Biom_Copepoda_ugL': 'Biom Copepoda',
        'color': 'Color',
        'temp_up': 'Temp up',
        'temp_bottom': 'Temp bottom',
        'wind_30d': 'Wind 30d',
        'tp_30d': 'TP 30d',
        'degree_day_thr0': 'Degree day > 0',
        'prev_Cyano': 'Prev Cyano',
        'prev_Mixo': 'Prev Mixo',
        'lake_id': 'Lake ID'
        # Variables lag supprimées selon les demandes de simplification
    }
    
    # Définir les couleurs fixes pour chaque métrique (colorblind-friendly)
    metric_colors = {
        'rich_genus_no_cyano': '#E69F00',    # Orange
        'shannon_no_cyano': '#56B4E9',       # Bleu ciel
        'eveness_piel_no_cyano': '#009E73'   # Vert bleu
    }
    
    metric_names = {
        'rich_genus_no_cyano': 'Richesse Générique',
        'shannon_no_cyano': 'Diversité Shannon',
        'eveness_piel_no_cyano': 'Équitabilité Pielou'
    }
    
    # Pour chaque métrique, créer une figure individuelle
    for i, response in enumerate(responses):
        print(f"  Création de la figure SHAP pour {metric_names[response]}")
        
        shap_df = shap_dfs[i]
        
        # Calculer l'importance moyenne absolue pour chaque prédicteur pour cette métrique
        importance_by_var = shap_df.groupby('Variable')['SHAP'].apply(lambda x: np.abs(x).mean()).sort_values(ascending=False)
        
        # Garder tous les prédicteurs pour les figures individuelles
        top_predictors = importance_by_var.index.tolist()
        print(f"    Tous les prédicteurs pour {metric_names[response]} ({len(top_predictors)} variables): {top_predictors[:5]}...")
        
        # Filtrer les données pour tous les prédicteurs
        metric_shap_df = shap_df[shap_df['Variable'].isin(top_predictors)].copy()
        metric_shap_df['Variable'] = pd.Categorical(metric_shap_df['Variable'], categories=top_predictors, ordered=True)
        metric_shap_df = metric_shap_df.sort_values('Variable')
        
        # Configuration dynamique basée sur le nombre de prédicteurs
        n_predictors = len(top_predictors)
        n_cols = 4  # 4 colonnes pour un meilleur rendu
        n_rows = (n_predictors + n_cols - 1) // n_cols  # Calcul du nombre de lignes nécessaires
        fig_size = (20, 5 * n_rows)  # Taille adaptée au nombre de lignes
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(n_rows, n_cols)
        
        # Lettres pour identifier chaque subplot
        letters = [f'({chr(97+i)})' for i in range(n_predictors)]  # (a), (b), (c), etc.
        
        # Couleur unique pour cette métrique
        color = metric_colors[response]
        
        # Affichage des prédicteurs par ordre d'importance décroissant
        for j, var in enumerate(top_predictors):
            row = j // n_cols
            col = j % n_cols
            
            # Vérifier que nous ne dépassons pas les axes disponibles
            if row >= n_rows:
                break
                
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            temp_df = metric_shap_df[metric_shap_df['Variable'] == var]
            
            # Échantillonnage pour optimiser le rendu
            if len(temp_df) > 2000:
                temp_df = temp_df.sample(n=2000, random_state=42)
            
            # Traitement spécial pour les prédicteurs catégoriels
            if (var == 'lake_id' and dataset_name == 'ELA'):
                # Récupérer les modalités uniques
                unique_values = sorted(temp_df['Valeur'].unique())
                
                # Créer le boxplot pour la variable catégorielle
                bp = ax.boxplot([temp_df[temp_df['Valeur'] == cat_value]['SHAP'].values 
                               for cat_value in unique_values if len(temp_df[temp_df['Valeur'] == cat_value]) > 0],
                               patch_artist=True, 
                               boxprops=dict(facecolor=color, alpha=0.7),
                               medianprops=dict(color='black'),
                               whiskerprops=dict(color='black'),
                               capprops=dict(color='black'))
                
                # Configuration des axes pour les prédicteurs catégoriels
                display_name = predictor_mapping.get(var, var)
                ax.set_xlabel(display_name, fontsize=18)
                ax.set_xticks(range(len(unique_values)))
                ax.set_xticklabels(unique_values, fontsize=16)
                
            else:
                # Traitement normal pour les prédicteurs continus
                # Calculer les limites d'axes basées sur les percentiles
                x_min, x_max = get_axis_limits(temp_df['Valeur'])
                
                # Scatter plot avec ligne de tendance
                ax.scatter(temp_df['Valeur'], temp_df['SHAP'], 
                          alpha=0.3, s=3, color=color)
                
                # Ajouter une ligne de tendance
                if len(temp_df) > 10:
                    temp_df_sorted = temp_df.sort_values('Valeur')
                    try:
                        trend = lowess(temp_df_sorted["SHAP"], temp_df_sorted["Valeur"], frac=0.3)
                        ax.plot(trend[:, 0], trend[:, 1], color=color, linewidth=3)
                    except:
                        pass
                
                # Configuration des axes pour les prédicteurs continus
                display_name = predictor_mapping.get(var, var)
                ax.set_xlabel(display_name, fontsize=18)
                
                # Appliquer les limites d'axes calculées
                if x_min is not None and x_max is not None:
                    ax.set_xlim(x_min, x_max)
            
            # Configuration des labels d'axes Y
            if col == 0:  # Colonne de gauche
                ax.set_ylabel(f'SHAP ({metric_names[response]})', fontsize=18)
                ax.tick_params(axis='y', labelsize=16)
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)
            
            # Ligne de zéro
            ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
            
            # Définir des limites symétriques pour l'axe Y
            y_min, y_max = ax.get_ylim()
            y_abs_max = max(abs(y_min), abs(y_max))
            ax.set_ylim(-y_abs_max, y_abs_max)
            
            # Ajout de la lettre pour identifier le subplot
            if j < len(letters):
                ax.text(0.02, 0.98, letters[j], transform=ax.transAxes, 
                       fontsize=16, fontweight='bold', verticalalignment='top')
            
            # Améliorer la taille des ticks
            ax.tick_params(axis='x', labelsize=16)
        
        # Masquer les axes vides s'il y en a
        total_subplots = n_rows * n_cols
        for empty_idx in range(n_predictors, total_subplots):
            empty_row = empty_idx // n_cols
            empty_col = empty_idx % n_cols
            if empty_row < n_rows:
                empty_ax = axes[empty_row, empty_col] if n_rows > 1 else axes[empty_col]
                empty_ax.set_visible(False)
        
        # Titre global pour la figure
        fig.suptitle(f'Analyse SHAP - {metric_names[response]} ({dataset_name})', 
                     fontsize=24, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)  # Ajuster pour le titre
        
        # Sauvegarder la figure individuelle
        response_clean = response.replace('_no_cyano', '').replace('_', '')
        plt.savefig(f"{figures_dir}/SHAP_{response_clean}_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_combined_shap_rank_plot(shap_rank_data_ela, shap_rank_data_lpnla):
    """Créer un boxplot combiné des ranks SHAP pour ELA et LPNLA basé sur l'importance cumulée"""
    print("Génération du boxplot combiné des ranks SHAP ELA + LPNLA (ordre par importance cumulée)")
    
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
        tick_labels.append(predictor)
    
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
    combined_figures_dir = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/figures/figures_model"
    os.makedirs(combined_figures_dir, exist_ok=True)
    plt.savefig(f"{combined_figures_dir}/SHAP_rank_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot combiné sauvegardé : {combined_figures_dir}/SHAP_rank_combined.png")
    return f"{combined_figures_dir}/SHAP_rank_combined.png"

###############################################################################
# Fonction d'export LaTeX
###############################################################################

def export_combined_latex_results(shap_data_ela, shap_data_lpnla, metrics_ela=None, metrics_lpnla=None):
    """Exporter un tableau LaTeX combiné avec les valeurs SHAP, MAE et R² des deux datasets"""
    print("Génération du tableau LaTeX combiné ELA + LPNLA avec MAE et R²")
    
    # Définir le dossier tables correct
    tables_dir = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/figures/figures_model/tables"
    os.makedirs(tables_dir, exist_ok=True)
    
    # Calculer les valeurs SHAP moyennes pour ELA
    shap_means_ela = {}
    if shap_data_ela:
        for i, resp in enumerate(responses):
            if i < len(shap_data_ela):
                shap_df = shap_data_ela[i]
                mean_abs_shap = shap_df.groupby('Variable')['SHAP'].apply(lambda x: np.abs(x).mean())
                shap_means_ela[resp] = mean_abs_shap
    
    # Calculer les valeurs SHAP moyennes pour LPNLA
    shap_means_lpnla = {}
    if shap_data_lpnla:
        for i, resp in enumerate(responses):
            if i < len(shap_data_lpnla):
                shap_df = shap_data_lpnla[i]
                mean_abs_shap = shap_df.groupby('Variable')['SHAP'].apply(lambda x: np.abs(x).mean())
                shap_means_lpnla[resp] = mean_abs_shap
    
    # Mapper les noms de prédicteurs aux noms LaTeX
    predictor_mapping = {
        'year': 'Year',
        'doy': 'Doy', 
        'lat': 'Lat',
        'long': 'Long',
        'area_m2': 'Area',
        'COND_uS.cm': 'COND',
        'Chla_ug.L': 'Chla',
        'TNTP_mg.L': 'TNTP',
        'pH_mean': 'pH mean',
        'DO_up': 'DO up',
        'DO_bottom': 'DO bottom',
        'Biom_Cladocera_ugL': 'Biom Cladocera',
        'Biom_Copepoda_ugL': 'Biom Copepoda',
        'color': 'Color',
        'temp_up': 'Temp up',
        'temp_bottom': 'Temp bottom',
        'wind_30d': 'Wind 30d',
        'tp_30d': 'TP 30d',
        'degree_day_thr0': 'Degree day > 0',
        'prev_Cyano': 'Prev Cyano',
        'prev_Mixo': 'Prev Mixo',
        'lake_id': 'Lake ID',
        # Variables lag
        'rich_genus_no_cyano_lag1m_o1': 'Richesse lag 1',
        'shannon_no_cyano_lag1m_o1': 'Shannon lag 1',
        'eveness_piel_no_cyano_lag1m_o1': 'Équitabilité lag 1'
    }
    
    # Liste des prédicteurs dans l'ordre voulu
    all_predictors = [
        'lake_id', 'year', 'doy', 'lat', 'long', 'area_m2', 'COND_uS.cm', 'Chla_ug.L', 
        'TNTP_mg.L', 'pH_mean', 'DO_up', 'DO_bottom', 'Biom_Cladocera_ugL',
        'Biom_Copepoda_ugL', 'color', 'temp_up', 'temp_bottom', 'wind_30d',
        'tp_30d', 'degree_day_thr0', 'prev_Cyano', 'prev_Mixo',
        # Variables lag
        'rich_genus_no_cyano_lag1m_o1', 'shannon_no_cyano_lag1m_o1', 'eveness_piel_no_cyano_lag1m_o1'
    ]
    
    # Définir quels prédicteurs sont disponibles pour chaque dataset
    lpnla_predictors = ['lat', 'long', 'area_m2', 'COND_uS.cm', 'Chla_ug.L', 
                        'TNTP_mg.L', 'pH_mean', 'DO_up', 'DO_bottom', 
                        'Biom_Cladocera_ugL', 'Biom_Copepoda_ugL', 'color', 
                        'temp_up', 'temp_bottom', 'wind_30d', 'tp_30d', 
                        'degree_day_thr0', 'prev_Mixo', 'prev_Cyano']
    
    ela_predictors = ['lake_id', 'year', 'doy', 'COND_uS.cm',
                      'Chla_ug.L', 'TNTP_mg.L', 'pH_mean', 'DO_up', 'DO_bottom',
                      'prev_Cyano', 'prev_Mixo',
                      # Variables lag pour ELA
                      'rich_genus_no_cyano_lag1m_o1', 'shannon_no_cyano_lag1m_o1', 'eveness_piel_no_cyano_lag1m_o1']
    
    # Générer le tableau LaTeX
    latex_lines = []
    
    latex_lines.append("\\begin{table}[H]")
    latex_lines.append("\\centering")
    latex_lines.append("\\renewcommand{\\arraystretch}{1.2}")
    latex_lines.append("\\setlength{\\tabcolsep}{4pt}")
    latex_lines.append("\\caption{Métriques de performance (MAE et R² moyens sur validation) et valeurs SHAP absolues moyennes pour chacun des modèles de diversité. Les valeurs NA correspondent aux prédicteurs absents du jeu de données.}")
    latex_lines.append("\\label{tab:metrics_shap_models_all}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\multirow{2}{*}{\\textbf{Prédicteur}} & \\multicolumn{3}{c}{\\textbf{LP-NLA}} & \\multicolumn{3}{c}{\\textbf{ELA}} \\\\")
    latex_lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    latex_lines.append("& $S$ & $H'$ & $J'$ & $S$ & $H'$ & $J'$ \\\\")
    latex_lines.append("\\midrule")
    
    # Calculer les moyennes des métriques
    def calculate_mean_metrics(metrics_list):
        if not metrics_list:
            return {'mae': {'rich_genus_no_cyano': 0, 'shannon_no_cyano': 0, 'eveness_piel_no_cyano': 0},
                    'r2': {'rich_genus_no_cyano': 0, 'shannon_no_cyano': 0, 'eveness_piel_no_cyano': 0}}
        
        mae_means = {}
        r2_means = {}
        
        for resp in responses:
            resp_metrics = [m for m in metrics_list if resp in [item for item in metrics_list[0].keys() if 'mae' in str(item)]]
            mae_values = [m['mae'] for m in metrics_list if 'mae' in m]
            r2_values = [m['r2'] for m in metrics_list if 'r2' in m]
            
            mae_means[resp] = np.mean(mae_values) if mae_values else 0
            r2_means[resp] = np.mean(r2_values) if r2_values else 0
        
        return {'mae': mae_means, 'r2': r2_means}
    
    # Obtenir les moyennes (simplifiées pour cette version)
    mae_s_lpnla = mae_h_lpnla = mae_j_lpnla = "--"
    mae_s_ela = mae_h_ela = mae_j_ela = "--"
    r2_s_lpnla = r2_h_lpnla = r2_j_lpnla = "--"
    r2_s_ela = r2_h_ela = r2_j_ela = "--"
    
    latex_lines.append(f"MAE          & {mae_s_lpnla} & {mae_h_lpnla} & {mae_j_lpnla} & {mae_s_ela} & {mae_h_ela} & {mae_j_ela} \\\\")
    latex_lines.append(f"R² validation & {r2_s_lpnla} & {r2_h_lpnla} & {r2_j_lpnla} & {r2_s_ela} & {r2_h_ela} & {r2_j_ela} \\\\")
    latex_lines.append("\\addlinespace")
    latex_lines.append("\\midrule")
    
    for predictor in all_predictors:
        latex_name = predictor_mapping.get(predictor, predictor)
        
        # Valeurs pour LP-NLA
        if predictor in lpnla_predictors:
            if shap_means_lpnla:
                s_val = f"{shap_means_lpnla.get('rich_genus_no_cyano', {}).get(predictor, 0):.3f}" if 'rich_genus_no_cyano' in shap_means_lpnla and predictor in shap_means_lpnla.get('rich_genus_no_cyano', {}) else "--"
                h_val = f"{shap_means_lpnla.get('shannon_no_cyano', {}).get(predictor, 0):.3f}" if 'shannon_no_cyano' in shap_means_lpnla and predictor in shap_means_lpnla.get('shannon_no_cyano', {}) else "--"
                j_val = f"{shap_means_lpnla.get('eveness_piel_no_cyano', {}).get(predictor, 0):.3f}" if 'eveness_piel_no_cyano' in shap_means_lpnla and predictor in shap_means_lpnla.get('eveness_piel_no_cyano', {}) else "--"
            else:
                s_val = h_val = j_val = "--"
        else:
            s_val = h_val = j_val = "NA"
        
        # Valeurs pour ELA
        if predictor in ela_predictors:
            if shap_means_ela:
                s_ela = f"{shap_means_ela.get('rich_genus_no_cyano', {}).get(predictor, 0):.3f}" if 'rich_genus_no_cyano' in shap_means_ela and predictor in shap_means_ela.get('rich_genus_no_cyano', {}) else "--"
                h_ela = f"{shap_means_ela.get('shannon_no_cyano', {}).get(predictor, 0):.3f}" if 'shannon_no_cyano' in shap_means_ela and predictor in shap_means_ela.get('shannon_no_cyano', {}) else "--"
                j_ela = f"{shap_means_ela.get('eveness_piel_no_cyano', {}).get(predictor, 0):.3f}" if 'eveness_piel_no_cyano' in shap_means_ela and predictor in shap_means_ela.get('eveness_piel_no_cyano', {}) else "--"
            else:
                s_ela = h_ela = j_ela = "--"
        else:
            s_ela = h_ela = j_ela = "NA"
        
        latex_lines.append(f"{latex_name:<20} & {s_val} & {h_val} & {j_val} & {s_ela} & {h_ela} & {j_ela} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Sauvegarder le fichier LaTeX combiné
    latex_file = f"{tables_dir}/shap_values_table_combined.tex"
    with open(latex_file, 'w', encoding='utf-8') as f:
        for line in latex_lines:
            f.write(line + '\n')
    
    print(f"Tableau LaTeX combiné généré : {latex_file}")
    return latex_file

###############################################################################
# Fonction principale d'analyse
###############################################################################

def analyze_dataset(dataset_name):
    """Analyser un dataset spécifique"""
    print(f"\\n=== Analyse du dataset {dataset_name} ===")
    start_time = time.time()
    
    # Charger les données et créer les répertoires
    df, covariable = load_data(dataset_name)
    figures_dir, tables_dir = create_directories(dataset_name)
    
    # Préparation des arguments pour la parallélisation
    all_args = []
    
    if dataset_name == 'ELA':
        # Leave-One-Year-Out validation
        for resp in responses:
            years = df['year'].unique()
            for test_year in years:
                all_args.append((resp, test_year, df, covariable))
        
        # Traitement parallèle
        n_cores = min(mp.cpu_count() - 1, 8)
        print(f"Utilisation de {n_cores} cœurs CPU")
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(process_response_year_ela, all_args))
        
        # Regroupement des résultats
        response_data = {resp: [] for resp in responses}
        shap_rank_data = []
        all_metrics = []
        
        for i, (resp, test_year, _, _) in enumerate(all_args):
            shap_data, rank_data, metrics = results[i]
            response_data[resp].extend(shap_data)
            shap_rank_data.extend(rank_data)
            all_metrics.append(metrics)
    
    else:  # LPNLA
        # K-fold validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for resp in responses:
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df)):
                all_args.append((resp, fold_idx, train_idx, val_idx, df, covariable))
        
        # Traitement parallèle
        n_cores = min(mp.cpu_count() - 1, 8)
        print(f"Utilisation de {n_cores} cœurs CPU")
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(process_response_fold_lpnla, all_args))
        
        # Regroupement des résultats
        response_data = {resp: [] for resp in responses}
        shap_rank_data = []
        all_metrics = []
        
        for i, (resp, fold_idx, train_idx, val_idx, _, _) in enumerate(all_args):
            shap_data, rank_data, metrics = results[i]
            response_data[resp].extend(shap_data)
            shap_rank_data.extend(rank_data)
            all_metrics.append(metrics)
    
    print(f"Traitement parallèle terminé en {time.time() - start_time:.2f} secondes")
    
    # Traitement des résultats par réponse
    shap_dfs = []
    for resp in responses:
        print(f"Traitement des données SHAP pour: {resp}")
        shap_df = pd.DataFrame(response_data[resp])
        shap_df.to_csv(f"{tables_dir}/{resp}_SHAP_values_{dataset_name}.csv", index=False)
        shap_dfs.append(shap_df)
    
    # Génération des visualisations
    create_combined_shap_plot(shap_dfs, responses, covariable, figures_dir, dataset_name)
    create_individual_shap_plots(shap_dfs, responses, covariable, figures_dir, dataset_name)
    
    print(f"Analyse {dataset_name} terminée en {time.time() - start_time:.2f} secondes au total")
    
    return {
        'dataset': dataset_name,
        'shap_dfs': shap_dfs,
        'shap_rank_data': shap_rank_data,
        'metrics': all_metrics
    }

###############################################################################
# Fonction principale
###############################################################################

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Analyse SHAP unifiée pour les datasets ELA et LP-NLA')
    parser.add_argument('--dataset', choices=['ELA', 'LPNLA', 'both'], default='both',
                        help='Dataset à analyser (default: both)')
    parser.add_argument('--latex', action='store_true',
                        help='Générer un export LaTeX (actif par défaut pour LPNLA)')
    
    args = parser.parse_args()
    
    print("=== ANALYSE SHAP UNIFIÉE ===")
    print(f"Dataset(s) sélectionné(s): {args.dataset}")
    print(f"Export LaTeX: {args.latex or args.dataset in ['LPNLA', 'both']}")
    
    global_start_time = time.time()
    results = []
    shap_data_ela = None
    shap_data_lpnla = None
    
    if args.dataset == 'both':
        # Analyser ELA
        print("\\n" + "="*50)
        print("ANALYSE ELA")
        print("="*50)
        result_ela = analyze_dataset('ELA')
        results.append(result_ela)
        shap_data_ela = result_ela['shap_dfs']
        
        # Analyser LPNLA
        print("\\n" + "="*50)
        print("ANALYSE LPNLA")
        print("="*50)
        result_lpnla = analyze_dataset('LPNLA')
        results.append(result_lpnla)
        shap_data_lpnla = result_lpnla['shap_dfs']
        
        # Créer le graphique de rang combiné
        create_combined_shap_rank_plot(result_ela['shap_rank_data'], result_lpnla['shap_rank_data'])
        
        # Export LaTeX combiné
        export_combined_latex_results(shap_data_ela, shap_data_lpnla)
        
    elif args.dataset == 'ELA':
        result = analyze_dataset('ELA')
        results.append(result)
        shap_data_ela = result['shap_dfs']
        
    else:  # LPNLA
        result = analyze_dataset('LPNLA')
        results.append(result)
        shap_data_lpnla = result['shap_dfs']
        
        if args.latex:
            export_combined_latex_results(None, shap_data_lpnla)
    
    print(f"\\n{'='*60}")
    print("RÉSUMÉ FINAL")
    print("="*60)
    
    for result in results:
        dataset_name = result['dataset']
        print(f"\\n{dataset_name}:")
        print(f"  - Figures SHAP générées dans: {DATASETS[dataset_name]['figures_dir']}")
        print(f"  - Données SHAP exportées dans: {DATASETS[dataset_name]['figures_dir'].replace('/figures_model/', '/figures_model/tables/')}")
    
    print(f"\\nTemps total d'exécution: {time.time() - global_start_time:.2f} secondes")
    print("Analyse terminée avec succès !")

if __name__ == "__main__":
    main()
