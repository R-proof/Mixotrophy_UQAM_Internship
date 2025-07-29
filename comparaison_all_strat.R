# Script by SERRE Renaud
# TITLE  : Comparaison all Strat all dataset

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
  
# ==== resolution library conflicts ====
library(conflicted)
conflict_prefer("filter", "dplyr")
conflict_prefer("select", "dplyr")

#####################################################################################################
# DATA ---------------------------------------------------------------------------------------
#####################################################################################################

setwd("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/Mixotroph_strat")

s1_raw = read.csv("info_genus.csv") # OK
s2_raw = read.csv("MatMix_AnnexeL_NLA2017LP_05012023.csv")
s3_raw = read.csv("Names_phyto.csv",sep=";")
s4_raw = read.csv("nanoplanktonnutritionstrategies.csv",sep=";")
s5_raw = read.csv("Tabular Data - Phytoplankton Species Codes and Taxonomy.csv")
s6_raw = read.csv("Trophic_type_Spec_list_updated.csv")
s7_raw = read.xlsx("NLA2017LakePulse-zooplankton-taxa-list-data-06062022.xlsx")


# -----------------------------------------------------------------------------------------------
# Data Manips                                                                             
# -----------------------------------------------------------------------------------------------

s2 = s2_raw %>%
  select(Genus,Resource.acquisition.strategy) %>%
  rename(strat.2 = Resource.acquisition.strategy)

s5 = s5_raw %>%
  left_join(s6_raw,by="species_code") %>%
  rename(strat.3 = trophic_type,
         Genus = genus) %>%
  mutate(strat.3 = case_when(
    strat.3 == "a" ~ "Autotrophe",
    strat.3 == "m" ~ "Mixotrophe",
    strat.3 == "h" ~ "Heterotrophe",
  )) %>%
  select(Genus,strat.3)
  
s_final = s1_raw %>%
  select(Classic.group.name,Empire,Kingdom,Phylum,Class,Order,Family,Genus,Final_Nutrition_Strategy) %>%
  rename(Classic_group_name = Classic.group.name,
         strat.1 = Final_Nutrition_Strategy) %>%
  left_join(s2, by="Genus") %>%
  left_join(s5, by="Genus") 

s_final = s_final %>% 
  select(-c(strat.2,strat.3)) %>%
  rename(strat = strat.1)


#####################################################################################################
# Export ---------------------------------------------------------------------------------------
#####################################################################################################

write.csv(s_final,file = "/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/df_final/strat.csv")

