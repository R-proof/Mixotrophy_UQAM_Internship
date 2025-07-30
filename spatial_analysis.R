# Script by SERRE Renaud
# Internship UQAM-GRIL under supervision of BEISNER Beatrix

library(dplyr)
library(tidyverse)
library(ape)
library(vegan)
library(nlme)      
library(DHARMa) 
library(tidyverse)

# libraries for spatial series analyses
library(geoR) # for spatial object to perform geostatistic
library(sp)
library(sf)
library(pgirmess) # for spatial correlogram
library(spdep) #to extracted the neighbors is a spatial data
library(ade4)  # used for plotting spatial data

# these two libraries (spdep and ade4 also provive function that make object compatibles between library)
library(spatialreg) # used for spatial data modelling
library(spgwr) # to run geographically weighted regression

rm(list=ls()) 
graphics.off() 
cat("\014")

setwd("/Users/renaudsrr/Desktop/STAGE_MTL/Scripts/Data/new_csv/data_python")

data_phyton_LP_NLA = read.csv("data_python_LP_NLA.csv")
data_phyton_LP_NLA_harmz = read.csv("data_phyton_LP_NLA_harmz.csv") 

# Data --------------------------------------------------------------------

coord = data.frame("lake" = data_phyton_LP_NLA_harmz$lake_id,
                   "x" = data_phyton_LP_NLA_harmz$long,
                   "y" = data_phyton_LP_NLA_harmz$lat,
                   row.names=1)

div = data.frame("lake" = data_phyton_LP_NLA_harmz$lake_id,
                 "prev_Mixo" = data_phyton_LP_NLA_harmz$prev_Mixo,
                 "rich_genus" = data_phyton_LP_NLA_harmz$rich_genus,
                 "H" = data_phyton_LP_NLA_harmz$H,
                 "1-D" = data_phyton_LP_NLA_harmz$one_minus_D,
                 "J" = data_phyton_LP_NLA_harmz$J,
                 row.names=1)

env = data_phyton_LP_NLA_harmz %>%
  select(-c("region","lake_id","lat","long","rich_genus","H","one_minus_D","J","Stratification"))

# Analyse prelim ----------------------------------------------------------

plot(coord,type="n",main="lake location",
     xlab="long",ylab="lat",asp=1)
lines(coord,col="black",type="p")
text(coord, row.names(coord),cex=0.4,col="red")


plot(coord,cex=div$prev_Mixo/100,main="Prev Mixo",col="black")
par(mfrow=c(2,2))
plot(coord,cex=div$rich_genus/50,main="rich_genus",col="black")
plot(coord,cex=div$H/5,main="H",col="black")
plot(coord,cex=div$one_minus_D/20,main="1-D",col="black")
plot(coord,cex=div$J,main="J",col="black")

# pairs(env,panel = panel.smooth)

ggplot(data = data_phyton_LP_NLA_harmz, aes(x = Stratification, y = prev_Mixo)) +
  geom_boxplot()

# SPATIAL -----------------------------------------------------------------

coord_div = data.frame("lake" = data_phyton_LP_NLA_harmz$lake_id,
                   "x" = data_phyton_LP_NLA_harmz$long,
                   "y" = data_phyton_LP_NLA_harmz$lat,
                   "prev_Mixo" = data_phyton_LP_NLA_harmz$prev_Mixo,
                   "rich_genus" = data_phyton_LP_NLA_harmz$rich_genus,
                   "H" = data_phyton_LP_NLA_harmz$H,
                   "1-D" = data_phyton_LP_NLA_harmz$one_minus_D,
                   "J" = data_phyton_LP_NLA_harmz$J,
                   row.names=1)

coord_div = coord_div[!duplicated(coord_div[, c("x", "y")]), ]
coord_sf = st_as_sf(coord_div[,1:2], coords = c("x", "y"), crs = 4326)
coord_sf_lambert_can = st_transform(coord_sf, crs = 3347)
coord_div[,1:2] = st_coordinates(coord_sf_lambert_can)

# test sur rich_genus et prev_Mixo
geo_rich_genus=as.geodata(SpatialPointsDataFrame(data.frame(coord_div[,1:2]),data.frame(coord_div$rich_genus)))
geo_prev_Mixo=as.geodata(SpatialPointsDataFrame(data.frame(coord_div[,1:2]),data.frame(coord_div$prev_Mixo)))
geo_H =as.geodata(SpatialPointsDataFrame(data.frame(coord_div[,1:2]),data.frame(coord_div$H)))
geo_J =as.geodata(SpatialPointsDataFrame(data.frame(coord_div[,1:2]),data.frame(coord_div$J)))

par(mfrow=c(2,2))
points.geodata(geo_rich_genus,pt.divide="quartile", main="rich_genus")
points.geodata(geo_prev_Mixo,pt.divide="quartile", main="prev_Mixo")
points.geodata(geo_H,pt.divide="quartile", main="H")
points.geodata(geo_J,pt.divide="quartile", main="J")

# Gabriel
coord_div.gab = gabrielneigh(as.matrix(coord_div[,1:2]))
s.label(coord_div[,1:2],clabel=.03, cpoint=.01,neig=nb2neig(graph2nb(coord_div.gab)))
graph2nb(coord_div.gab)
gab=graph2nb(coord_div.gab)

# Delanay
coord_div.tri = tri2nb(coord_div[,1:2])
s.label(coord_div[,1:2],clabel=0.03,cpoint=.01,neig=nb2neig(coord_div.tri))
coord_div.tri

# matrice de connexion, ici binaire
pond_coordiv_tri.bin = nb2listw(coord_div.tri, style="B",zero.policy=TRUE)
pond_coordiv_tri.bin$weights[1]
pond_coordiv_gab.bin = nb2listw(graph2nb(coord_div.gab), style="B", zero.policy=TRUE)
pond_coordiv_gab.bin$weights[1]

moran.test(coord_div$rich_genus, pond_coordiv_tri.bin, zero.policy=TRUE)
moran.test(coord_div$prev_Mixo, pond_coordiv_gab.bin, zero.policy=TRUE)

corcond=correlog(coord_div[,1:2],coord_div$rich_genus,method="Moran")
plot(corcond, main="autocorrelogram")

