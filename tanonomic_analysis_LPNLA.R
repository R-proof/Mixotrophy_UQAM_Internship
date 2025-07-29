# Script by SERRE Renaud
# Internship UQAM-GRIL under supervision of BEISNER Beatrix

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
library(cowplot)  

#####################################################################################################
# Taxons majoritaire pour relation div~prevMixo ---------------------------------------------------------------------------------------
#####################################################################################################

source("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/CODE/OK_Process_data.R")

data_taxon = data_phyto %>%
  rename("Genus" = "Target_taxon",
         "biovol_tot" = "total.biov") %>%
  left_join(taxo_info,by=c("Genus")) %>%
  rename(biovol_um3.mL = biovol) %>%
  left_join(summary_data_phyto,by="lake_id") %>%
  select(-c(classic.group,Survey.x,Survey.y,biovol_tot,X)) %>%
  select(lake_id,biovol_um3.mL,biovol_tot_um3.mL,Empire,Kingdom,Phylum,Class,Order,Family,Genus,everything())

data_taxon = data_taxon %>% # <<<<<<<<<<<<<<<<<< a commenter si on veux diversité TOT
  filter(!Classic_group_name == "CYANOBACTERIA")

all_type_taxon = c("Empire","Kingdom","Phylum",
                   "Class","Order","Family","Genus","strat")

palette_taxon = c(
  "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
  "#e6ab02", "#a6761d", "#666666", "#8dd3c7", "#fb8072",  
  "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#bc80bd",  
  "#cab2d6", "#ffffb3", "#b15928",                        
  "black", "white" # Other et NA                                
)
nb_bins = 20  # nombre de colonnes/bins souhaitées
breaks = seq(0, 1, length.out = nb_bins + 1)

data_taxon = data_taxon %>%
  mutate(bins = cut(prev_Mixo, breaks = breaks, include.lowest = TRUE, labels = FALSE))

top_n_taxa = 15  # nombre de modalités à garder
i = 1

for (taxon in all_type_taxon) {
  
  data_taxon_grouped = data_taxon %>%
    mutate(taxon_grouped = fct_lump_n(.data[[taxon]], n = top_n_taxa, other_level = "Other"))
  
  table_taxon = data_taxon_grouped %>%
    group_by(bin = bins, taxon_grouped) %>%
    summarise(
      freq = n(),
      biovol_tot = sum(biovol_um3.mL, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      percent_freq = 100 * freq / ave(freq, bin, FUN = sum),
      percent_biovol = 100 * biovol_tot / ave(biovol_tot, bin, FUN = sum)
    )
  
  # recup legend
  gg_for_legend = ggplot(table_taxon, aes(x = bin, y = percent_biovol, fill = taxon_grouped)) +
    geom_bar(stat = "identity") +
    labs(fill = taxon) +
    scale_fill_manual(values = palette_taxon) +
    theme_minimal(base_size = 9) +  
    theme(legend.position = "right",
          legend.title = element_text(size = 15),
          legend.text = element_text(size = 12))
  legend_plot = cowplot::get_legend(gg_for_legend)
  
  # Freq apparition taxons
  gg_freq1 = ggplot(table_taxon, aes(x = bin, y = freq, fill = taxon_grouped)) +
    geom_bar(stat = "identity") +
    labs(y = "Fréquence apparition taxon") +
    scale_fill_manual(values = palette_taxon) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # % apparition taxons
  gg_percent1 = ggplot(table_taxon, aes(x = bin, y = percent_freq, fill = taxon_grouped)) +
    geom_bar(stat = "identity") +
    labs(y = "% apparition taxon") +
    scale_fill_manual(values = palette_taxon) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # Biomass cumulée taxons
  gg_freq2 = ggplot(table_taxon, aes(x = bin, y = biovol_tot, fill = taxon_grouped)) +
    geom_bar(stat = "identity") +
    labs(y = "Biovol cumulé") +
    scale_fill_manual(values = palette_taxon) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # % Biomass taxon
  gg_percent2 = ggplot(table_taxon, aes(x = bin, y = percent_biovol, fill = taxon_grouped)) +
    geom_bar(stat = "identity") +
    labs(y = "% biovol par taxon") +
    scale_fill_manual(values = palette_taxon) +
    theme_minimal() +
    theme(legend.position = "none")
  
  # 8 plots sans légende
  plots_no_legend = plot_grid(
    gg_freq2, gg_freq1, 
    gg_percent2, gg_percent1,
    ncol = 2
  )
  
  # Ajout legend
  final_plot = plot_grid(legend_plot, plots_no_legend, ncol = 2, rel_widths = c(0.25, 1))
  
  filename = paste0("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/figures/panels/panels_LP_NLA/",i,"_Panel_", taxon, ".png")
  ggsave(filename, plot = final_plot, width = 18, height = 9, dpi = 500, bg = "white")
  
  i = i+1
}
