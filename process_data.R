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
library(car)
library(sensitivity)      # méthodes globales (Sobol, FAST, etc.)
library(lhs)              # Latin Hypercube Sampling
library(effects)
library(DALEX)
library(fuzzyjoin)
library(lubridate)
library(forcats)
library(zoo)   # pour na.approx
conflicts()

# ==== resolution library conflicts ====
library(conflicted)
conflict_prefer("filter", "dplyr")
conflict_prefer("select", "dplyr")
conflict_prefer("first", "dplyr")

#####################################################################################################
#####################################################################################################
# LP-NLA ---------------------------------------------------------------------------------------
#####################################################################################################
#####################################################################################################

#####################################################################################################
# IMPORT DATA ---------------------------------------------------------------------------------------
#####################################################################################################

data_phyto1 = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/LP_NLA/biovol_phyto.csv")

data_zoo1 = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/LP_NLA/biomass_zoo2.csv",sep=";",dec=",") %>%
  rename("lake_id"="Lake_ID")

env1 = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/LP_NLA/env.csv") %>%
  rename("lake_id"="Lake_ID") %>% select(Survey,everything())

data_bact = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/LP_NLA/ab_bacteria.csv",row.names = 1, check.names = FALSE) %>%
  select(-c(asv_code,sequence,kingdom)) %>%
  rename(Phylum  = phylum,
         Class  = class, 
         Order  = order)

taxo_info = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/df_final/strat.csv")
all_mixo = taxo_info$Genus[!is.na(taxo_info$Genus) & taxo_info$strat == "Mixotroph"]

# --- qq infos rapport ---------------------------------------------------------------------------
data_phyto1.2 = data_phyto1 %>%
  rename(Genus = Target_taxon) %>%
  left_join(taxo_info[,c("Genus","Class","strat")],by=c("Genus"))

a = sum(data_phyto1$biovol[data_phyto1.2$Class=="Cryptophyceae"],na.rm=T)
b = sum(data_phyto1$biovol[data_phyto1.2$Class=="Chrysophyceae"],na.rm=T)
c = sum(data_phyto1$biovol[data_phyto1.2$Class=="Dinophyceae"],na.rm=T)
d = sum(data_phyto1$biovol[data_phyto1.2$Class=="Coccolithophyceae"],na.rm=T)

print(paste(a , ": Cryptophyceae"))
print(paste(b , ": Chrysophyceae"))
print(paste(c , ": Dinophyceae"))
print(paste(d , ": Coccolithophyceae"))
sort(c(a,b,c,d),decreasing = T)

#####################################################################################################
# Summary DATA ---------------------------------------------------------------------------------------
#####################################################################################################

# --- Phyto ---------------------------------------------------------------------------
summary_data_phyto1 = data_phyto1 %>%
  rename("genus" = "Target_taxon",
         "biovol_tot" = "total.biov") %>%
  mutate(strat = ifelse(genus %in% all_mixo, "Mixotroph", "Autotroph")) %>%
  filter(biovol > 0) %>%
  group_by(lake_id)%>%
  summarize(
    Survey = first(Survey),
    
    biovol_tot_um3.mL = first(biovol_tot),
    biovol_mixo_um3.mL = sum(ifelse(strat == "Mixotroph", biovol, 0)),
    biovol_cyano_um3.mL = sum(ifelse(classic.group == "CYANOBACTERIA", biovol, 0)),
    
    rich_genus = n_distinct(genus),
    shannon = -sum((biovol/biovol_tot_um3.mL) * log(biovol/biovol_tot_um3.mL)),
    op_simpson = 1-sum((biovol/biovol_tot_um3.mL)**2),
  ) %>%
  ungroup() %>%
  mutate(eveness_piel = ifelse(rich_genus > 1, shannon / log(rich_genus), NA),
         prev_Mixo = biovol_mixo_um3.mL/biovol_tot_um3.mL,
         prev_Cyano = biovol_cyano_um3.mL/biovol_tot_um3.mL) %>%
  select(-c(biovol_mixo_um3.mL,biovol_cyano_um3.mL))

# --- Phyto whitout cyano ---------------------------------------------------------------------------

summary_data_phyto_no_cyano1 = data_phyto1 %>%
  rename("genus" = "Target_taxon",
         "biovol_tot" = "total.biov") %>%
  select(-c("biovol_tot"))  %>%
  
  mutate(strat = ifelse(genus %in% all_mixo, "Mixotroph", "Autotroph")) %>%
  filter(biovol > 0) %>%
  filter(classic.group!="CYANOBACTERIA") %>%
  
  group_by(lake_id) %>%
  summarize(
    biovol_tot_um3.mL = sum(biovol),
    
    rich_genus_no_cyano = n_distinct(genus),
    shannon_no_cyano = -sum((biovol/biovol_tot_um3.mL) * log(biovol/biovol_tot_um3.mL) ),
    op_simpson_no_cyano = 1-sum((biovol/biovol_tot_um3.mL)**2),
    
  ) %>%
  ungroup() %>%
  mutate(eveness_piel_no_cyano = shannon_no_cyano/log(rich_genus_no_cyano)) %>%
  left_join(summary_data_phyto1[,c("lake_id","prev_Mixo")],by="lake_id")

# --- Zoo ---------------------------------------------------------------------------

lake_id_zoo = data_zoo1$lake_id 

mat_zoo = data_zoo1 %>%
  select(-lake_id) %>%
  as.matrix()

summary_data_zoo1 = data.frame(
  lake_id = lake_id_zoo,
  biomass_tot_zoo_um3.mL = rowSums(mat_zoo),
  rich_sp_zoo = rowSums(mat_zoo > 0),
  shannon_zoo = diversity(mat_zoo, index = "shannon"),
  op_simpson_zoo = diversity(mat_zoo, index = "simpson")
) %>%
  mutate(eveness_zoo = ifelse(rich_sp_zoo > 1, shannon_zoo / log(rich_sp_zoo), NA)) %>%
  left_join(summary_data_phyto1[,c("lake_id","prev_Mixo")],by="lake_id")

dim(summary_data_zoo1) ; nrow_df_raw = nrow(summary_data_zoo1)
colSums(is.na(summary_data_zoo1))
summary_data_zoo1 = na.omit(summary_data_zoo1)
print(paste0(abs(nrow(summary_data_zoo1) - nrow_df_raw)," row remove on a total of ",nrow_df_raw, 
             " // ",round((nrow_df_raw - nrow(summary_data_zoo1)) * 100 / nrow_df_raw, 2), " % remove, ", nrow(summary_data_zoo1)," remaining"))

# --- Bacteria ---------------------------------------------------------------------------

bact_long = data_bact %>%
  pivot_longer(
    cols = -c(Phylum, Class, Order,clade,lineage,tribe),
    names_to = "lake_id",
    values_to = "abundance"
  ) %>% select(lake_id,Phylum,Class,Order,abundance)

# summary par ordre
summary_data_bact = bact_long %>%
  filter(abundance > 0) %>%
  group_by(lake_id, Order) %>%
  summarize(abundance = sum(abundance), .groups = "drop") %>%
  
  group_by(lake_id) %>%
  mutate(ab_tot_bact = sum(abundance)) %>%
  summarize(
    ab_tot_bact = first(ab_tot_bact),
    rich_order_bact = n_distinct(Order),
    shannon_bact = -sum((abundance / ab_tot_bact) * log(abundance / ab_tot_bact)),
    op_simpson_bact = 1 - sum((abundance / ab_tot_bact)^2)
  ) %>%
  ungroup() %>%
  mutate(
    eveness_bact = ifelse(rich_order_bact > 1, shannon_bact / log(rich_order_bact), NA)
  ) %>%
  left_join(summary_data_phyto1[, c("lake_id", "prev_Mixo")], by = "lake_id")

#####################################################################################################
#####################################################################################################
# IISD-ELA ---------------------------------------------------------------------------------------
#####################################################################################################
#####################################################################################################

#####################################################################################################
# Import data ---------------------------------------------------------------------------------------
#####################################################################################################

setwd("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/IISD-ELA/csv_traites")
dt1 = read.csv("dt1.csv") # env chimie
dt2 = read.csv("dt2.csv") # infos spatial
dt3 = read.csv("dt3.csv") # biomass plantons => pas utile ici
dt4 = read.csv("dt4.csv") # biomass / sp # 
dt7 = read.csv("dt7.csv") # env physics
setwd("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/IISD-ELA")
secchi = read.csv("RID_375_field_obs.csv")
temp_and_O2 = read.csv("RID_375_profiles.csv")
temp_surface = read.csv("RID_375_water_surf_temp.csv") # info redondante pas utile ici
sp_code = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/OLD/DATA/4_IISD-ELA/csv_traites/dt5.csv")[,-1]

# --- Phtyo ---------------------------------------------------------------------------
data_phyto2 = dt4[,c(2,3,7,8:10)] %>% 
  rename(lake_id = monitoring_location_name,
         date = date_collected) %>%
  mutate(date = as.Date(date), format = "%Y-%m-%d") %>%
  left_join(sp_code[,c(1,7)],by="species_code") %>%
  rename(Genus = genus) %>%
  left_join(taxo_info,by="Genus", relationship = "many-to-many") %>%
  mutate(biovol_um3.mL = (density*volume_cell)/10^6) %>% # calcul biovolume, /10^6 pour conversion m3 en mL
  select(lake_id,date,species_code,Classic_group_name,Empire,Kingdom,Phylum,Class,Order,Family,Genus,strat,biovol_um3.mL)

data_phyto2 = data_phyto2 %>% select(-c(Empire,Kingdom,Phylum,Class,Order,Family))
colSums(is.na(data_phyto2))
data_phyto2 = na.omit(data_phyto2)

# --- Env ---------------------------------------------------------------------------
# ENV CHIMIE
env_chim = dt1 %>%
  select(monitoring_location_name,activity_start_date,activity_start_time,
         characteristic_name,result_value,result_unit,
         activity_depth_height_measure,activity_depth_height_unit) %>%
  rename(lake_id = monitoring_location_name,
         date = activity_start_date,
         hour = activity_start_time,
         prof_ech = activity_depth_height_measure,
         type_ech = activity_depth_height_unit) %>%
  filter(type_ech %in% c("epi","m")) %>%
  filter(hour == "") %>%
  
  mutate(characteristic_name = replace_na(characteristic_name, "SO")) %>%
  mutate(characteristic_name = paste0(characteristic_name, "_", result_unit)) %>%
  mutate(characteristic_name = stringr::str_replace_all(characteristic_name, "/", ".")) %>%
  
  mutate(type_simplifie = case_when(
    type_ech == "epi" ~ "epi",
    type_ech == "m" & prof_ech <= 1 ~ "pseudo_epi",
    TRUE ~ "autre"
  )) %>%
  filter(type_simplifie %in% c("epi", "pseudo_epi")) %>%
  
  select(-c(prof_ech,type_ech,type_simplifie,hour,result_unit)) %>%
  group_by(lake_id, date, characteristic_name) %>%
  summarise(result_value = mean(as.numeric(result_value), na.rm = TRUE), .groups = "drop") %>%
  
  pivot_wider(
    names_from = characteristic_name,
    values_from = result_value
  ) %>%
  
  rename(Chla_ug.L = CHLA_ug.L,
         pH_mean = PH_,
         color = COLOUR_,
         DO_up = O2_mg.L) %>%
  select(lake_id,date,
         pH_mean, color, DO_up,
         CA_mg.L,MG_mg.L,CL_mg.L,FE_mg.L,K_mg.L,NH3_ug.L,NO2_ug.L,NO3_ug.L,SO4_mg.L,SO_mg.L,
         COND_uS.cm,Chla_ug.L,DOC_umol.L,TDN_ug.L,TDP_ug.L)

DO_bottom_envchim = dt1 %>%
  filter(characteristic_name == "O2", !is.na(activity_depth_height_measure)) %>%
  mutate(date = as.Date(activity_start_date)) %>%
  group_by(monitoring_location_name, date) %>%
  slice_max(order_by = activity_depth_height_measure, n = 1, with_ties = FALSE) %>%
  select(lake_id = monitoring_location_name, date, DO_bottom = result_value) %>%
  mutate(date = as.character(date))
env_chim = env_chim %>%
  left_join(DO_bottom_envchim, by = c("lake_id", "date"))

# ENV CHIMIE PART.2 
env_temp_and_O2 = temp_and_O2 %>%
  select(monitoring_location_name,date,depth_bin_m,temp_c,oxygen_mg_l1) %>%
  rename(lake_id = monitoring_location_name) %>%
  group_by(lake_id, date) %>%
  reframe(
    temp_up = temp_c[which.min(depth_bin_m)],
    temp_bottom = temp_c[which.max(depth_bin_m)],
    ox_mg.L_up = oxygen_mg_l1[which.min(depth_bin_m)],
    ox_mg.L_bottom = oxygen_mg_l1[which.max(depth_bin_m)],
    .groups = "drop"
  ) %>%
  select(lake_id, date, temp_up, temp_bottom, ox_mg.L_up, ox_mg.L_bottom)

env_secchi = secchi %>%
  select(monitoring_location_name, date, secchi_depth) %>%
  rename(lake_id = monitoring_location_name) %>%
  group_by(lake_id,date) %>%
  summarise(
    secchi_depth = mean(secchi_depth, na.rm = TRUE),
    .groups = "drop"
  )

# ENV PHYSIQUE
env_phy = dt7 %>%
  select(-c(X,order_lake)) %>%
  rename(lake_id = monitoring_location_name,
         long = longitude,
         lat = latitude,
         Stratification = mixing_status) %>%
  mutate(Stratification = case_when(
    Stratification == "polymictic" ~ "mixed",
    Stratification == "dimictic" ~ "stratified"
  ))


# JOIN ENV TOT
env_chim$date = as.Date(env_chim$date)
env_temp_and_O2$date = as.Date(env_temp_and_O2$date)
env_secchi$date = as.Date(env_secchi$date)

env2 = env_chim %>%
  left_join(env_temp_and_O2, by = c("lake_id", "date")) %>%
  left_join(env_secchi,       by = c("lake_id", "date")) %>%
  left_join(env_phy,          by = "lake_id")  

colSums(is.na(env2))
dim(env2)

#####################################################################################################
# Summary data ---------------------------------------------------------------------------------------
#####################################################################################################

# --- Phyto ---------------------------------------------------------------------------
summary_data_phyto2 = data_phyto2 %>%
  group_by(lake_id, date) %>%
  summarize(
    biovol_tot_um3.mL = sum(biovol_um3.mL, na.rm = TRUE),
    biovol_mixo_um3.mL = sum(ifelse(strat == "Mixotroph", biovol_um3.mL, 0), na.rm = TRUE),
    biovol_cyano_um3.mL = sum(ifelse(Classic_group_name == "CYANOBACTERIA", biovol_um3.mL, 0), na.rm = TRUE),
    rich_genus = n_distinct(Genus),
    shannon = -sum((biovol_um3.mL / sum(biovol_um3.mL)) * log(biovol_um3.mL / sum(biovol_um3.mL))),
    op_simpson = 1 - sum((biovol_um3.mL / sum(biovol_um3.mL))^2),
    .groups = "drop"
  ) %>%
  mutate(
    eveness_piel = ifelse(rich_genus > 1, shannon / log(rich_genus), NA),
    prev_Mixo = ifelse(biovol_tot_um3.mL > 0, biovol_mixo_um3.mL / biovol_tot_um3.mL, 0),
    prev_Cyano = ifelse(biovol_tot_um3.mL > 0, biovol_cyano_um3.mL / biovol_tot_um3.mL, 0)
  )


# --- Phyto no cyano ---------------------------------------------------------------------------
summary_data_phyto_no_cyano2 = data_phyto2 %>%
  filter(Genus != "Cyanobacteria") %>%
  group_by(lake_id, date) %>%
  summarize(
    biovol_tot_um3.mL = sum(biovol_um3.mL, na.rm = TRUE),
    biovol_mixo_um3.mL = sum(ifelse(strat == "Mixotroph", biovol_um3.mL, 0), na.rm = TRUE),
    rich_genus_no_cyano = n_distinct(Genus),
    shannon_no_cyano = -sum((biovol_um3.mL / sum(biovol_um3.mL)) * log(biovol_um3.mL / sum(biovol_um3.mL))),
    op_simpson_no_cyano = 1 - sum((biovol_um3.mL / sum(biovol_um3.mL))^2),
    .groups = "drop"
  ) %>%
  mutate(
    eveness_piel_no_cyano = ifelse(rich_genus_no_cyano > 1, shannon_no_cyano / log(rich_genus_no_cyano), NA),
    prev_Mixo = ifelse(biovol_tot_um3.mL > 0, biovol_mixo_um3.mL / biovol_tot_um3.mL, 0)
  )


# --- Zoo ---------------------------------------------------------------------------

# data_zoo2 = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/IISD-ELA/ELA_zoopl_num_l_to_2024_common_taxonomy_beisner.csv",
#                      sep = ',')
# zoo_infos = read.csv("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/IISD-ELA/zoo/taxon_codes_all_common-taxonomy.csv")

#####################################################################################################
#####################################################################################################
# TRANSFO DATA ---------------------------------------------------------------------------------------
#####################################################################################################
#####################################################################################################

# --- LP-NLA ---------------------------------------------------------------------------
df_final_LP_NLA_raw = env1 %>%
  left_join(summary_data_phyto1,by="lake_id") %>%
  left_join(summary_data_phyto_no_cyano1[,-7],by="lake_id") %>%
  left_join(summary_data_zoo1[,-7],by="lake_id") %>%
  left_join(summary_data_bact[,-7],by="lake_id")
  
colSums(is.na(df_final_LP_NLA_raw))

df_final_LP_NLA = df_final_LP_NLA_raw %>%
  select(-c(Survey.x,secchi_bottom)) %>%
  rename(Survey = Survey.y,
         DOC_mg.L = DOC, 
         color = Colour,
         Chla_ug.L = Chla, 
         sech_m = secchi_depth,
         alt = altitude,
         lat = latitude,
         long = longitude,
         area_m2 = area,
         depth = lake_depth,
         SO_mg.L = Sodium,
         MG_mg.L = Magnesium,
         CL_mg.L = Chloride,
         K_mg.L = Potassium,
         CA_mg.L = Calcium,
         SO4_mg.L = Sulfate,
         TP_ug.L = TP,
         TN_mg.L = TN,
         temp_up = Temperature_up,
         temp_bottom = Temperature_bottom
  ) %>%
  mutate(temp_mean = rowMeans(cbind(temp_up, temp_bottom),na.rm=T),
         pH_mean = rowMeans(cbind(pH_up, pH_bottom),na.rm = T),
         DO_mean_mg.L = rowMeans(cbind(DO_up, DO_bottom),na.rm = T),
         COND_uS.cm = Conductivity*1000, # car en mS/cm
         Stratification = case_when(
           Stratification == "Stratification status could not be assessed" ~ "unclassifiable",
           .default = as.character(Stratification)),
          Biom_ZooTotal_ugL = Biom_Cladocera_ugL + Biom_Copepoda_ugL + Biom_OTHER_ugL,
         TNTP_mg.L = TN_mg.L + TP_ug.L/1000,
         ) %>%
  select(
    lake_id, Ecoregion, Survey, alt, area_m2, perimeter, lat, long, depth, sech_m,
    SO_mg.L, CA_mg.L, MG_mg.L, CL_mg.L, K_mg.L, SO4_mg.L,COND_uS.cm,
    
    temp_up, temp_bottom, temp_mean,
    DO_up, DO_bottom, DO_mean_mg.L,
    pH_up, pH_bottom, pH_mean,
    
    Chla_ug.L, TP_ug.L, TN_mg.L, TNTP_mg.L, DOC_mg.L, color,
    Stratification,
    wind_30d, t2m_30d, ssr_30d, tp_30d, degree_day_thr0,
    
    Biom_ZooTotal_ugL,
    Biom_Cladocera_ugL, Biom_Copepoda_ugL, Biom_OTHER_ugL,
    Biom_CladOther_ugL, Biom_CopCal_ugL, Biom_CopCyc_ugL, Biom_CopOther_ugL, Biom_Daphnia_ugL,
    
    rich_genus, shannon, op_simpson, eveness_piel,
    rich_genus_no_cyano,shannon_no_cyano,op_simpson_no_cyano,eveness_piel_no_cyano,
    rich_sp_zoo, shannon_zoo, op_simpson_zoo, eveness_zoo, 
    ab_tot_bact, rich_order_bact, shannon_bact, op_simpson_bact, eveness_bact,

    prev_Mixo, prev_Cyano
  )


df_final_LP_NLA = df_final_LP_NLA[!is.na(df_final_LP_NLA$shannon), ]
df_final_LP_NLA = df_final_LP_NLA[!is.na(df_final_LP_NLA$shannon_no_cyano), ]
df_final_LP_NLA = df_final_LP_NLA[!is.na(df_final_LP_NLA$eveness_piel), ]
df_final_LP_NLA = df_final_LP_NLA[!is.na(df_final_LP_NLA$eveness_piel_no_cyano), ]
df_final_LP_NLA = df_final_LP_NLA[!is.na(df_final_LP_NLA$prev_Mixo), ]
df_final_LP_NLA = df_final_LP_NLA[!is.na(df_final_LP_NLA$prev_Cyano), ]

# --- IISD-ELA ---------------------------------------------------------------------------
df_final_IISD_ELA_raw = env2 %>%
  left_join(summary_data_phyto2[,c("lake_id","date","rich_genus","shannon","op_simpson","eveness_piel","prev_Mixo","prev_Cyano")], by = c("lake_id", "date")) %>%
  left_join(summary_data_phyto_no_cyano2[,c("lake_id","date","rich_genus_no_cyano","shannon_no_cyano","op_simpson_no_cyano","eveness_piel_no_cyano")], by = c("lake_id", "date")
            ) %>%
  rename(
    area_m2 = area_surface
  ) %>%
  
  mutate(
    TNTP_mg.L = (TDP_ug.L + TDN_ug.L)/1000,
    DOC_mg.L = DOC_umol.L,
    temp_mean = rowMeans(cbind(temp_up, temp_bottom), na.rm=T),
    ox_mg.L_mean = rowMeans(cbind(ox_mg.L_up, ox_mg.L_bottom), na.rm=T),
    lake_id = case_when(
      lake_id == "114 LA CB" ~ 114,
      lake_id == "224 LA CB" ~ 224,
      lake_id == "239 LA CB" ~ 239,
      lake_id == "373 LA CB" ~ 373,
      lake_id == "442 LA CB" ~ 442,
    ),
    lake_id = as.factor(lake_id),
    year = year(date),
    doy = yday(date)
  ) %>%
  
  mutate(
    doy_sin = sin(2 * pi * doy / 365),
    doy_cos = cos(2 * pi * doy / 365)
  ) %>%
  
  select(
    lake_id, date, year, doy, doy_sin, doy_cos,
    lat, long, secchi_depth, depth_mean, depth_max, area_m2,
    CA_mg.L, MG_mg.L, CL_mg.L, FE_mg.L, K_mg.L, NH3_ug.L, NO2_ug.L, NO3_ug.L, SO4_mg.L, SO_mg.L,
    COND_uS.cm, Chla_ug.L, DOC_mg.L, TNTP_mg.L, TDN_ug.L, TDP_ug.L,
    temp_mean, temp_up, temp_bottom, 
    ox_mg.L_mean, ox_mg.L_up, ox_mg.L_bottom, DO_up, DO_bottom,
    
    Stratification, pH_mean, color,
    TNTP_mg.L, TDP_ug.L, TDN_ug.L, 

    rich_genus, shannon, op_simpson, eveness_piel, 
    rich_genus_no_cyano, shannon_no_cyano, op_simpson_no_cyano, eveness_piel_no_cyano,
    prev_Mixo, prev_Cyano
  )

# Interpolation par semaine
df_final_IISD_ELA = df_final_IISD_ELA_raw %>%
  arrange(lake_id, date) %>%
  group_by(lake_id) %>%
  mutate(across(
    where(is.numeric),
    ~ na.approx(., x = date, maxgap = 7, na.rm = FALSE)
  )) %>%
  ungroup()

df_final_IISD_ELA = df_final_IISD_ELA[!is.na(df_final_IISD_ELA$shannon), ]
df_final_IISD_ELA = df_final_IISD_ELA[!is.na(df_final_IISD_ELA$shannon_no_cyano), ]
df_final_IISD_ELA = df_final_IISD_ELA[!is.na(df_final_IISD_ELA$eveness_piel), ]
df_final_IISD_ELA = df_final_IISD_ELA[!is.na(df_final_IISD_ELA$eveness_piel_no_cyano), ]
df_final_IISD_ELA = df_final_IISD_ELA[!is.na(df_final_IISD_ELA$prev_Mixo), ]
df_final_IISD_ELA = df_final_IISD_ELA[!is.na(df_final_IISD_ELA$prev_Cyano), ]

colSums(is.na(df_final_IISD_ELA))

#####################################################################################################
#####################################################################################################
# Export data ---------------------------------------------------------------------------------------
#####################################################################################################
#####################################################################################################

# -----------------------------------------------------------------------------------------------
# Enlever corrélation                                                                  
# -----------------------------------------------------------------------------------------------

df_final_LP_NLA_2 = df_final_LP_NLA %>%
  select(-c(perimeter,alt,SO_mg.L,SO4_mg.L,CL_mg.L,CA_mg.L, K_mg.L, MG_mg.L, temp_mean,DO_mean_mg.L,pH_bottom,pH_up,ssr_30d,t2m_30d,TN_mg.L,TP_ug.L,sech_m,depth,DOC_mg.L))

df_final_IISD_ELA_2 = df_final_IISD_ELA %>%
  select(-c(secchi_depth,FE_mg.L,NO2_ug.L,SO4_mg.L,temp_mean,temp_up,temp_bottom,ox_mg.L_mean,ox_mg.L_up,ox_mg.L_bottom,
            SO_mg.L,TDN_ug.L,TDP_ug.L,NH3_ug.L,NO3_ug.L,CL_mg.L,K_mg.L,CA_mg.L, MG_mg.L, DOC_mg.L,depth_mean,depth_max,color)) # enlever variables avec trop de NA et corrélées


# --- LP-NLA ---------------------------------------------------------------------------
# quartz()
# df_corr = na.omit(df_final_LP_NLA_2)
# Xs_cov = names(df_corr[, sapply(df_corr, is.numeric)])
# vars_to_remove = c(
#   "Biom_OTHER_ugL", "Biom_CladOther_ugL", "Biom_CopCal_ugL", "Biom_CopCyc_ugL", "Biom_CopOther_ugL", "Biom_Daphnia_ugL", "rich_genus",
#   "shannon", "op_simpson", "eveness_piel", "rich_genus_no_cyano", "shannon_no_cyano", "op_simpson_no_cyano", "eveness_piel_no_cyano",
#   "rich_sp_zoo", "shannon_zoo", "op_simpson_zoo", "eveness_zoo", "ab_tot_bact", "rich_order_bact", "shannon_bact", "op_simpson_bact","eveness_bact","Biom_ZooTotal_ugL"
# )
# 
# Xs_cov = Xs_cov[!Xs_cov %in% vars_to_remove]
# cor_mat = cor(df_corr[, Xs_cov], method = "spearman")
# corrplot(cor_mat, method = "color", type = "upper",addCoef.col = "black",tl.col = "black",tl.cex = 0.7,number.cex = 0.3,title="LP-NLA")
# 
# # # --- IISD-ELA ---------------------------------------------------------------------------
# quartz()
# df_corr = na.omit(df_final_IISD_ELA_2)
# Xs_cov = names(df_corr[, sapply(df_corr, is.numeric)])
# vars_to_remove = c(
#   "rich_genus","shannon", "op_simpson", "eveness_piel", "rich_genus_no_cyano", "shannon_no_cyano", "op_simpson_no_cyano", "eveness_piel_no_cyano"
# )
# 
# Xs_cov = Xs_cov[!Xs_cov %in% vars_to_remove]
# cor_mat = cor(df_corr[, Xs_cov], method = "spearman")
# corrplot(cor_mat, method = "color", type = "upper",addCoef.col = "black",tl.col = "black",tl.cex = 0.7,number.cex = 0.3,title = "ELA")


# -----------------------------------------------------------------------------------------------
# Nettoyage Outlier                                                                             
# -----------------------------------------------------------------------------------------------

# Variables à exclure de l'analyse
vars_to_remove <- c(
  "Biom_OTHER_ugL", "Biom_CladOther_ugL", "Biom_CopCal_ugL", "Biom_CopCyc_ugL", "Biom_CopOther_ugL", "Biom_Daphnia_ugL", "rich_genus",
  "shannon", "op_simpson", "eveness_piel", "rich_genus_no_cyano", "shannon_no_cyano", "op_simpson_no_cyano", "eveness_piel_no_cyano",
  "rich_sp_zoo", "shannon_zoo", "op_simpson_zoo", "eveness_zoo", "ab_tot_bact", "rich_order_bact", "shannon_bact", "op_simpson_bact",
  "eveness_bact", "Biom_ZooTotal_ugL"
)

source("/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/CODE/OK_fct_outlier.R")

outliers_LP_NLA = detect_outliers(df_final_LP_NLA_2, id_col = "lake_id", dataset_name = "df_final_LP_NLA_2", iqr_threshold = 3)
outliers_LP_NLA_bless = detect_outliers_blessing(df_final_LP_NLA_2, id_col = "lake_id", dataset_name = "df_final_LP_NLA_2", )
  
# ou ----> outliers_LP_NLA_bless <------
df_final_LP_NLA_3 = remove_outlier_rows(df_final_LP_NLA_2, outliers_LP_NLA, "df_final_LP_NLA_2")


# -----------------------------------------------------------------------------------------------
# CSV EXPORT                                                                              
# -----------------------------------------------------------------------------------------------

write.csv(df_final_LP_NLA_3,"/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/df_final/df_LP_NLA.csv")
write.csv(df_final_IISD_ELA_2,"/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/df_final/df_IISD_ELA.csv")

colnames(df_final_LP_NLA_3)
dim(df_final_LP_NLA_3)
dim(df_final_LP_NLA_3[df_final_LP_NLA_3$Survey=="Lake Pulse",])
dim(df_final_LP_NLA_3[df_final_LP_NLA_3$Survey=="NLA 2017",])
colnames(df_final_IISD_ELA_2)
dim(df_final_IISD_ELA_2)

# -----------------------------------------------------------------------------------------------
# Dataset python                                                                             
# -----------------------------------------------------------------------------------------------

LP_NLA_py = df_final_LP_NLA_3 %>%
  select(c(lat,long,area_m2,COND_uS.cm,Chla_ug.L,TNTP_mg.L,pH_mean,Stratification, DO_up,DO_bottom,Biom_Cladocera_ugL,Biom_Copepoda_ugL,
            color,temp_up,temp_bottom,wind_30d,tp_30d,degree_day_thr0,prev_Mixo,prev_Cyano,
            rich_genus_no_cyano,shannon_no_cyano,eveness_piel_no_cyano))

ELA_py = df_final_IISD_ELA_2 %>%
  select(c(lake_id,year,doy,lat,long,area_m2,COND_uS.cm,Chla_ug.L,Stratification,TNTP_mg.L,pH_mean,DO_up,DO_bottom,prev_Cyano,prev_Mixo,
           rich_genus_no_cyano,shannon_no_cyano,eveness_piel_no_cyano))

write.csv(LP_NLA_py,"/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/df_final/LP_NLA_py.csv")
write.csv(ELA_py,"/Users/renaudsrr/Desktop/STAGE_MTL/MODELISATION/NEW/DATA/df_final/ELA_py.csv")








































