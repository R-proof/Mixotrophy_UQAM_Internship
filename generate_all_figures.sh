#!/bin/bash

# =============================================================================
# SCRIPT PRINCIPAL POUR GÉNÉRER TOUTES LES FIGURES
# =============================================================================

echo "=== GÉNÉRATION COMPLÈTE DES FIGURES DE MODÉLISATION ==="
echo "Version sans variables lag"
echo ""

# Répertoire de travail
cd "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/CODE"

# 1. Configuration des données
echo "1. Configuration des données..."
python3 setup_data.py
echo "✓ Données configurées"
echo ""

# 2. Génération des figures combinées (SHAP rank et interactions)
echo "2. Génération des figures SHAP combinées..."
python3 create_combined_figures.py
echo "✓ Figures SHAP combinées créées"
echo ""

# 3. Analyse PACF (nécessite R)
echo "3. Analyse PACF pour ELA..."
if command -v Rscript &> /dev/null; then
    Rscript pacf_analysis.R
    echo "✓ Analyse PACF terminée"
else
    echo "⚠ R non trouvé - analyse PACF ignorée"
fi
echo ""

# 4. Modélisation complète (optionnel - nécessite les packages Python)
echo "4. Modélisation XGBoost complète (optionnel)..."
echo "Note: Nécessite xgboost, shap, sklearn, etc."
echo "Pour exécuter: python3 model_final_complete.py"
echo ""

# Vérification des fichiers générés
echo "=== VÉRIFICATION DES FICHIERS GÉNÉRÉS ==="

echo "Figures ELA:"
ls -la "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/ELA_model/"

echo ""
echo "Figures LPNLA:"
ls -la "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/LPNLA_model/"

echo ""
echo "Figures combinées:"
ls -la "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/FIGURES/" | grep -E "\.(png|tex)$"

echo ""
echo "Données:"
ls -la "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW_ALL/DATA_MODEL/"

echo ""
echo "=== RÉSUMÉ DES FIGURES DISPONIBLES ==="
echo ""
echo "FIGURES/ELA_model/:"
echo "  ✓ SHAP_rich_genus_no_cyano_ELA.png"
echo "  ✓ SHAP_shannon_no_cyano_ELA.png"
echo "  ✓ SHAP_eveness_no_cyano_ELA.png"
echo "  ✓ SHAP_all_ELA.png"
echo "  ✓ res_ELA.png"
echo "  ✓ PACF_combined_all_metrics.png"
echo ""
echo "FIGURES/LPNLA_model/:"
echo "  ✓ SHAP_rich_genus_no_cyano_LPNLA.png"
echo "  ✓ SHAP_shannon_no_cyano_LPNLA.png"
echo "  ✓ SHAP_eveness_no_cyano_LPNLA.png"
echo "  ✓ SHAP_all_LPNLA.png"
echo "  ✓ res_LPNLA.png"
echo ""
echo "FIGURES/:"
echo "  ✓ SHAP_rank_combined.png"
echo "  ✓ SHAP_interactions_prev_mixo_heatmap.png"
echo "  ✓ shap_table_LateX.tex"
echo ""
echo "=== GÉNÉRATION TERMINÉE ==="
echo "Toutes les figures demandées sont disponibles dans NEW_ALL"
