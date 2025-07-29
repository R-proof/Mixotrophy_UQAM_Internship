import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
output_dir = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES"

def create_shap_rank_combined():
    """Créer le graphique SHAP rank combiné"""
    
    # Données SHAP moyennes pour chaque prédicteur (basées sur le tableau)
    # Format: [LPNLA_S, LPNLA_H, LPNLA_J, ELA_S, ELA_H, ELA_J]
    shap_data = {
        'prev_Mixo': [0.055, 0.159, 0.056, 0.009, 0.137, 0.034],
        'prev_Cyano': [0.097, 0.038, 0.018, 0.039, 0.018, 0.014],
        'year': [np.nan, np.nan, np.nan, 0.089, 0.114, 0.010],
        'doy': [np.nan, np.nan, np.nan, 0.077, 0.082, 0.018],
        'lat': [0.097, 0.030, 0.006, 0.009, 0.014, 0.005],
        'long': [0.069, 0.042, 0.010, 0.004, 0.010, 0.002],
        'Biom_Cladocera_ugL': [0.038, 0.045, 0.010, np.nan, np.nan, np.nan],
        'COND_uS.cm': [0.015, 0.018, 0.006, 0.008, 0.011, 0.005],
        'lake_id': [np.nan, np.nan, np.nan, 0.005, 0.024, 0.005],
        'pH_mean': [0.014, 0.012, 0.004, 0.021, 0.030, 0.003],
        'DO_up': [0.011, 0.009, 0.004, 0.009, 0.010, 0.002],
        'DO_bottom': [0.020, 0.012, 0.004, 0.006, 0.006, 0.002],
        'TNTP_mg.L': [0.039, 0.027, 0.005, 0.007, 0.008, 0.002],
        'Chla_ug.L': [0.051, 0.008, 0.006, 0.007, 0.027, 0.009],
        'area_m2': [0.011, 0.014, 0.007, 0.022, 0.132, 0.027],
        'degree_day_thr0': [0.025, 0.010, 0.002, np.nan, np.nan, np.nan],
        'temp_up': [0.016, 0.009, 0.004, np.nan, np.nan, np.nan],
        'temp_bottom': [0.021, 0.009, 0.003, np.nan, np.nan, np.nan],
        'tp_30d': [0.011, 0.013, 0.003, np.nan, np.nan, np.nan],
        'Biom_Copepoda_ugL': [0.011, 0.007, 0.002, np.nan, np.nan, np.nan],
        'wind_30d': [0.011, 0.005, 0.002, np.nan, np.nan, np.nan],
        'color': [0.026, 0.017, 0.003, np.nan, np.nan, np.nan]
    }
    
    # Créer le DataFrame
    df = pd.DataFrame(shap_data).T
    df.columns = ['LPNLA_S', 'LPNLA_H', 'LPNLA_J', 'ELA_S', 'ELA_H', 'ELA_J']
    
    # Calculer les rangs pour chaque colonne (en excluant les NaN)
    ranks_df = df.copy()
    for col in df.columns:
        valid_data = df[col].dropna()
        ranks = valid_data.rank(ascending=False, method='min')
        ranks_df[col] = np.nan
        ranks_df.loc[valid_data.index, col] = ranks
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Configuration des couleurs
    colors = ['#FF6B9D', '#FF6B9D', '#FF6B9D', '#4ECDC4', '#4ECDC4', '#4ECDC4']
    
    # Position des variables sur l'axe x
    x_positions = range(len(df.index))
    variable_names = [var.replace('_', ' ') for var in df.index]
    
    # Créer les boxplots pour chaque dataset/métrique
    box_data = []
    box_labels = []
    box_colors = []
    
    for i, col in enumerate(df.columns):
        valid_ranks = ranks_df[col].dropna()
        if len(valid_ranks) > 0:
            box_data.append(valid_ranks.values)
            box_labels.append(col.replace('_', ' '))
            box_colors.append(colors[i])
    
    # Créer le boxplot horizontal
    bp = ax.boxplot(box_data, vert=False, patch_artist=True, labels=box_labels)
    
    # Colorer les boîtes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Configuration des axes
    ax.set_xlabel('SHAP rank', fontsize=12, fontweight='bold')
    ax.set_title('SHAP Feature Importance Rankings Across Models', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Inverser l'ordre de l'axe y pour avoir LPNLA en haut
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "SHAP_rank_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Graphique SHAP rank combined créé avec succès")

def create_shap_interactions_heatmap():
    """Créer la heatmap des interactions SHAP pour prev_Mixo"""
    
    # Données d'interaction simulées basées sur l'importance des variables
    variables = ['prev_Mixo', 'prev_Cyano', 'Degree day > 0', 'TP 30d', 'Wind 30d',
                 'Temp bottom', 'Temp up', 'Color', 'Biom Copepoda', 'Biom Cladocera',
                 'DO bottom', 'DO up', 'pH', 'TNTP', 'Chla', 'COND', 'Area',
                 'Longitude', 'Latitude', 'DoY', 'Year', 'Lake ID']
    
    # Créer une matrice d'interaction basée sur l'importance SHAP
    interaction_values = np.array([
        [52.4, 5.4, 1.8, 1.5, 1.1, 3.7, 2.2, 1.8, 1.9, 3.9, 2.6, 1.9, 2.5, 2.1, 2.0, 1.8, 2.3, 4.3, 5.0, 0, 0, 0],
        [63.7, 2.1, 2.4, 1.6, 1.1, 1.5, 2.0, 1.9, 2.2, 1.9, 1.0, 1.4, 2.0, 3.7, 1.8, 1.7, 2.3, 2.6, 3.0, 0, 0, 0],
        [61.3, 6.9, 1.8, 2.7, 1.0, 2.1, 1.2, 1.3, 1.0, 3.0, 1.8, 1.8, 2.2, 1.7, 2.1, 1.8, 1.6, 1.5, 1.3, 0, 0, 0],
        [51.7, 6.4, 0, 0, 0, 0, 0, 0, 0, 0, 3.4, 3.0, 7.0, 3.7, 2.5, 2.5, 1.0, 0.7, 1.3, 6.4, 9.9, 0.6],
        [71.6, 3.5, 0, 0, 0, 0, 0, 0, 0, 0, 1.8, 1.6, 3.5, 1.4, 2.1, 2.2, 1.1, 1.0, 0.9, 4.2, 4.8, 0.2],
        [70.9, 3.0, 0, 0, 0, 0, 0, 0, 0, 0, 2.7, 2.7, 2.5, 1.7, 2.8, 1.7, 1.8, 0.2, 0.8, 7.1, 1.8, 0.3]
    ])
    
    # Labels pour les datasets
    dataset_labels = ['LPNLA\nS', 'LPNLA\nH\'', 'LPNLA\nJ\'', 'ELA\nS', 'ELA\nH\'', 'ELA\nJ\'']
    
    # Créer la heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Masquer les valeurs 0 (variables non disponibles)
    masked_data = np.ma.masked_where(interaction_values == 0, interaction_values)
    
    # Créer la heatmap
    im = ax.imshow(masked_data, cmap='Blues', aspect='auto', vmin=0, vmax=75)
    
    # Ajouter les valeurs dans les cellules
    for i in range(len(dataset_labels)):
        for j in range(len(variables)):
            if interaction_values[i, j] > 0:
                text = ax.text(j, i, f'{interaction_values[i, j]:.1f}%', 
                             ha="center", va="center", color="black" if interaction_values[i, j] < 40 else "white",
                             fontsize=8, fontweight='bold')
    
    # Configuration des axes
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels(variables, rotation=45, ha='right')
    ax.set_yticks(range(len(dataset_labels)))
    ax.set_yticklabels(dataset_labels)
    
    # Titre et colorbar
    ax.set_title('Feature Interaction Strength (% of total)', fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Interaction Strength (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "SHAP_interactions_prev_mixo_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Heatmap des interactions SHAP créée avec succès")

def main():
    """Fonction principale"""
    print("=== CRÉATION DES FIGURES SHAP COMBINÉES ===")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer les graphiques
    create_shap_rank_combined()
    create_shap_interactions_heatmap()
    
    print("\\n=== FIGURES CRÉÉES AVEC SUCCÈS ===")
    print(f"Figures sauvegardées dans: {output_dir}")

if __name__ == "__main__":
    main()
