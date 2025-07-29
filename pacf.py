import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
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
    """Entraîne un modèle XGBoost."""
    
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
        X_final = X.drop('year', axis=1)
        model.fit(X_final, y)
    else:
        # Pour LPNLA : entraînement direct
        model.fit(X, y)
    
    return model

def calculate_residuals(model, X, y, dataset_name):
    """Calcule les résidus du modèle."""
    
    # Préparation des données selon le dataset
    if dataset_name == 'ELA' and 'year' in X.columns:
        X_pred = X.drop('year', axis=1)
    else:
        X_pred = X.copy()
    
    # Prédiction et calcul des résidus
    y_pred = model.predict(X_pred)
    residuals = y - y_pred
    
    return residuals, y_pred

def analyze_residuals(residuals, y_true, y_pred, response_name, dataset_name, figures_dir):
    """Analyse complète des résidus."""
    
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Graphique résidus vs prédictions
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Valeurs prédites')
    plt.ylabel('Résidus')
    plt.title(f'Résidus vs Prédictions - {dataset_name} {response_name}')
    
    # Ajouter une ligne de tendance
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    plt.plot(y_pred, p(y_pred), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{dataset_name}_{response_name}_residuals_vs_fitted.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Q-Q plot pour normalité
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot - {dataset_name} {response_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{dataset_name}_{response_name}_qq_plot.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Histogramme des résidus
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    # Superposer une courbe normale
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
             label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
    
    plt.xlabel('Résidus')
    plt.ylabel('Densité')
    plt.title(f'Distribution des Résidus - {dataset_name} {response_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{dataset_name}_{response_name}_residuals_hist.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Test de normalité
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
    
    # 5. Test d'autocorrélation (Ljung-Box)
    ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
    
    # Sauvegarder les statistiques
    stats_text = f"""STATISTIQUES DES RÉSIDUS - {dataset_name} {response_name}
    
Normalité:
- Test de Shapiro-Wilk: statistique = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}
- Test de Kolmogorov-Smirnov: statistique = {ks_stat:.4f}, p-value = {ks_p:.4f}

Autocorrélation (Ljung-Box):
{ljung_box.to_string()}

Métriques de performance:
- R² = {r2_score(y_true, y_pred):.4f}
- MAE = {mean_absolute_error(y_true, y_pred):.4f}
- RMSE = {np.sqrt(np.mean(residuals**2)):.4f}
"""
    
    with open(f"{figures_dir}/{dataset_name}_{response_name}_residuals_stats.txt", 'w') as f:
        f.write(stats_text)
    
    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'ljung_box': ljung_box
    }

def calculate_pacf_analysis(residuals, dataset_name, figures_dir, max_lags=20):
    """Calcule et visualise l'analyse PACF."""
    
    os.makedirs(figures_dir, exist_ok=True)
    
    # Calcul PACF
    pacf_values, confint = pacf(residuals, nlags=max_lags, alpha=0.05)
    
    # Graphique PACF
    plt.figure(figsize=(12, 6))
    
    # Plot des valeurs PACF
    lags = np.arange(len(pacf_values))
    plt.bar(lags, pacf_values, alpha=0.7)
    
    # Intervalles de confiance
    lower_ci = confint[:, 0] - pacf_values
    upper_ci = confint[:, 1] - pacf_values
    plt.errorbar(lags, pacf_values, yerr=[np.abs(lower_ci), upper_ci], 
                fmt='none', color='red', alpha=0.7)
    
    # Ligne de référence à 0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.xlabel('Lag')
    plt.ylabel('Autocorrélation partielle')
    plt.title(f'Fonction d\'autocorrélation partielle (PACF) - {dataset_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{dataset_name}_pacf_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return pacf_values, confint

def analyze_temporal_patterns_ela(X, residuals, response_name, figures_dir):
    """Analyse spécifique des patterns temporels pour ELA."""
    
    if 'year' not in X.columns or 'doy' not in X.columns:
        print("Variables temporelles non disponibles pour l'analyse ELA")
        return
    
    os.makedirs(figures_dir, exist_ok=True)
    
    # Résidus par année
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Résidus par année
    plt.subplot(2, 2, 1)
    years = sorted(X['year'].unique())
    annual_residuals = [residuals[X['year'] == year] for year in years]
    plt.boxplot(annual_residuals, labels=years)
    plt.xlabel('Année')
    plt.ylabel('Résidus')
    plt.title('Distribution des résidus par année')
    plt.xticks(rotation=45)
    
    # Subplot 2: Résidus par jour de l'année (binned)
    plt.subplot(2, 2, 2)
    doy_bins = np.arange(0, 366, 30)  # Bins de 30 jours
    doy_binned = pd.cut(X['doy'], bins=doy_bins)
    doy_residuals = [residuals[doy_binned == bin_val] for bin_val in doy_binned.cat.categories]
    
    valid_doy_residuals = [res for res in doy_residuals if len(res) > 0]
    valid_labels = [f"{int(cat.left)}-{int(cat.right)}" for cat, res in zip(doy_binned.cat.categories, doy_residuals) if len(res) > 0]
    
    if valid_doy_residuals:
        plt.boxplot(valid_doy_residuals, labels=valid_labels)
        plt.xlabel('Jour de l\'année (binned)')
        plt.ylabel('Résidus')
        plt.title('Distribution des résidus par période')
        plt.xticks(rotation=45)
    
    # Subplot 3: Série temporelle des résidus moyens par année
    plt.subplot(2, 2, 3)
    annual_mean_residuals = [np.mean(residuals[X['year'] == year]) for year in years]
    plt.plot(years, annual_mean_residuals, 'bo-')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Année')
    plt.ylabel('Résidus moyens')
    plt.title('Évolution temporelle des résidus moyens')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Autocorrélation des résidus moyens annuels
    plt.subplot(2, 2, 4)
    if len(annual_mean_residuals) > 3:
        pacf_annual, confint_annual = pacf(annual_mean_residuals, nlags=min(5, len(annual_mean_residuals)-2), alpha=0.05)
        lags = np.arange(len(pacf_annual))
        plt.bar(lags, pacf_annual, alpha=0.7)
        
        lower_ci = confint_annual[:, 0] - pacf_annual
        upper_ci = confint_annual[:, 1] - pacf_annual
        plt.errorbar(lags, pacf_annual, yerr=[np.abs(lower_ci), upper_ci], 
                    fmt='none', color='red', alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Lag (années)')
        plt.ylabel('PACF')
        plt.title('PACF des résidus annuels moyens')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/ELA_{response_name}_temporal_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset_pacf(dataset_name):
    """Traite un dataset complet pour l'analyse PACF et résidus."""
    
    print(f"Traitement PACF pour {dataset_name}")
    
    # Chargement des données
    X, y_dict = load_data(dataset_name)
    config = DATASETS[dataset_name]
    
    results = {}
    
    for response_name in responses:
        print(f"  - {response_name}")
        
        y = y_dict[response_name]
        
        # Entraînement du modèle
        model = train_xgboost_model(X, y, response_name, config['validation_type'])
        
        # Calcul des résidus
        residuals, y_pred = calculate_residuals(model, X, y, dataset_name)
        
        # Analyse des résidus
        residuals_stats = analyze_residuals(residuals, y, y_pred, response_name, 
                                          dataset_name, config['figures_dir'])
        
        # Analyse PACF
        pacf_values, confint = calculate_pacf_analysis(residuals, 
                                                      f"{dataset_name}_{response_name}", 
                                                      config['figures_dir'])
        
        # Analyse temporelle spécifique pour ELA
        if dataset_name == 'ELA':
            analyze_temporal_patterns_ela(X, residuals, response_name, config['figures_dir'])
        
        results[response_name] = {
            'residuals_stats': residuals_stats,
            'pacf_values': pacf_values,
            'confint': confint
        }
    
    return results

def create_combined_pacf_plots():
    """Crée les graphiques PACF combinés pour tous les datasets."""
    
    figures_base_dir = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES"
    os.makedirs(figures_base_dir, exist_ok=True)
    
    # Graphique combiné des analyses PACF
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analyse PACF des Résidus - Tous Datasets et Variables', fontsize=16)
    
    for i, response in enumerate(responses):
        for j, dataset in enumerate(['ELA', 'LPNLA']):
            ax = axes[j, i]
            
            # Rechargement des données pour chaque combinaison
            try:
                X, y_dict = load_data(dataset)
                y = y_dict[response]
                
                config = DATASETS[dataset]
                model = train_xgboost_model(X, y, response, config['validation_type'])
                residuals, _ = calculate_residuals(model, X, y, dataset)
                
                # Calcul PACF
                pacf_values, confint = pacf(residuals, nlags=10, alpha=0.05)
                
                # Plot
                lags = np.arange(len(pacf_values))
                ax.bar(lags, pacf_values, alpha=0.7)
                
                lower_ci = confint[:, 0] - pacf_values
                upper_ci = confint[:, 1] - pacf_values
                ax.errorbar(lags, pacf_values, yerr=[np.abs(lower_ci), upper_ci], 
                           fmt='none', color='red', alpha=0.7)
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax.set_title(f'{dataset} - {response}')
                ax.set_xlabel('Lag')
                ax.set_ylabel('PACF')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Erreur: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dataset} - {response}')
    
    plt.tight_layout()
    plt.savefig(f"{figures_base_dir}/combined_pacf_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graphique PACF combiné sauvegardé dans {figures_base_dir}/")

###############################################################################
# Point d'entrée principal
###############################################################################

def main():
    """Point d'entrée principal."""
    
    parser = argparse.ArgumentParser(description='Analyse PACF et résidus pour les modèles XGBoost')
    parser.add_argument('--dataset', choices=['ELA', 'LPNLA'], 
                       help='Dataset spécifique à traiter (optionnel)')
    parser.add_argument('--combined-only', action='store_true',
                       help='Générer seulement les graphiques combinés')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ANALYSE PACF ET RÉSIDUS - MODÈLES XGBOOST")
    print("="*80)
    
    if not args.combined_only:
        # Traitement des datasets individuels
        datasets_to_process = [args.dataset] if args.dataset else ['ELA', 'LPNLA']
        
        for dataset_name in datasets_to_process:
            if dataset_name in DATASETS:
                results = process_dataset_pacf(dataset_name)
                print(f"Analyse {dataset_name} terminée")
    
    # Génération des graphiques combinés
    print("Génération des graphiques PACF combinés...")
    create_combined_pacf_plots()
    
    print("Analyse PACF terminée avec succès!")

if __name__ == "__main__":
    main()
