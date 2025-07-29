# Package ID: edi.1818.1 Cataloging System:https://pasta.edirepository.org.
# Data set title: IISD Experimental Lakes Area: Chemistry of LTER Lakes, 1968-2022.
# Data set creator:    - IISD Experimental Lakes Area 
# Data set creator:  Sonya Havens - IISD Experimental Lakes Area 
# Contact:  Chris Hay - Scientific Data Officer IISD Experimental Lakes Area  - chay@iisd-ela.org
# Contact:  Sonya Havens - Research Chemist IISD Experimental Lakes Area  - shavens@iisd-ela.org
# Contact:  Ken Sandilands - Biologist - Field Coordinator IISD Experimental Lakes Area  - ksandilands@iisd-ela.org
# Stylesheet v2.14 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu      
# Uncomment the following lines to have R clear previous work, or set a working directory
# rm(list=ls())      

# setwd("C:/users/my_name/my_dir")       



options(HTTPUserAgent="EDI_CodeGen")


inUrl1  <- "https://pasta.lternet.edu/package/data/eml/edi/1818/1/f2f82d7374785fb6e3910982750c5bde" 
infile1 <- tempfile()
try(download.file(inUrl1,infile1,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
if (is.na(file.size(infile1))) download.file(inUrl1,infile1,method="auto")


dt1 <-read.csv(infile1,header=F 
               ,skip=1
               ,sep=","  
               ,quot='"' 
               , col.names=c(
                 "dataset_name",     
                 "monitoring_location_id",     
                 "monitoring_location_name",     
                 "activity_media_name",     
                 "activity_start_date",     
                 "activity_start_time",     
                 "activity_end_date",     
                 "activity_end_time",     
                 "activity_depth_height_measure",     
                 "activity_depth_height_unit",     
                 "layer_collection_end_depth",     
                 "layer_collection_start_depth",     
                 "characteristic_name",     
                 "characteristic_name_long",     
                 "method_speciation",     
                 "result_sample_fraction",     
                 "result_value",     
                 "result_unit",     
                 "result_detection_quantitation_limit_measure",     
                 "result_detection_quantitation_limit_unit",     
                 "result_detection_quantitation_limit_type",     
                 "result_status_id",     
                 "analysis_start_date",     
                 "result_analytical_method_name",     
                 "result_analytical_method_instrument",     
                 "result_analytical_reference_method",     
                 "laboratory_name",     
                 "laboratory_sample_id",     
                 "field_comment",     
                 "analysis_comment",     
                 "result_comment"    ), check.names=TRUE)

unlink(infile1)

# Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings

if (class(dt1$dataset_name)!="factor") dt1$dataset_name<- as.factor(dt1$dataset_name)
if (class(dt1$monitoring_location_id)!="factor") dt1$monitoring_location_id<- as.factor(dt1$monitoring_location_id)
if (class(dt1$monitoring_location_name)!="factor") dt1$monitoring_location_name<- as.factor(dt1$monitoring_location_name)
if (class(dt1$activity_media_name)!="factor") dt1$activity_media_name<- as.factor(dt1$activity_media_name)                                   
# attempting to convert dt1$activity_start_date dateTime string to R date structure (date or POSIXct)                                
tmpDateFormat<-"%Y-%m-%d"
tmp1activity_start_date<-as.Date(dt1$activity_start_date,format=tmpDateFormat)
# Keep the new dates only if they all converted correctly
if(nrow(dt1[dt1$activity_start_date != "",]) == length(tmp1activity_start_date[!is.na(tmp1activity_start_date)])){dt1$activity_start_date <- tmp1activity_start_date } else {print("Date conversion failed for dt1$activity_start_date. Please inspect the data and do the date conversion yourself.")}                                                                    

# attempting to convert dt1$activity_end_date dateTime string to R date structure (date or POSIXct)                                
tmpDateFormat<-"%Y-%m-%d"
tmp1activity_end_date<-as.Date(dt1$activity_end_date,format=tmpDateFormat)
# Keep the new dates only if they all converted correctly
if(nrow(dt1[dt1$activity_end_date != "",]) == length(tmp1activity_end_date[!is.na(tmp1activity_end_date)])){dt1$activity_end_date <- tmp1activity_end_date } else {print("Date conversion failed for dt1$activity_end_date. Please inspect the data and do the date conversion yourself.")}                                                                    

if (class(dt1$activity_depth_height_measure)=="factor") dt1$activity_depth_height_measure <-as.numeric(levels(dt1$activity_depth_height_measure))[as.integer(dt1$activity_depth_height_measure) ]               
if (class(dt1$activity_depth_height_measure)=="character") dt1$activity_depth_height_measure <-as.numeric(dt1$activity_depth_height_measure)
if (class(dt1$activity_depth_height_unit)!="factor") dt1$activity_depth_height_unit<- as.factor(dt1$activity_depth_height_unit)
if (class(dt1$layer_collection_end_depth)=="factor") dt1$layer_collection_end_depth <-as.numeric(levels(dt1$layer_collection_end_depth))[as.integer(dt1$layer_collection_end_depth) ]               
if (class(dt1$layer_collection_end_depth)=="character") dt1$layer_collection_end_depth <-as.numeric(dt1$layer_collection_end_depth)
if (class(dt1$layer_collection_start_depth)=="factor") dt1$layer_collection_start_depth <-as.numeric(levels(dt1$layer_collection_start_depth))[as.integer(dt1$layer_collection_start_depth) ]               
if (class(dt1$layer_collection_start_depth)=="character") dt1$layer_collection_start_depth <-as.numeric(dt1$layer_collection_start_depth)
if (class(dt1$characteristic_name)!="factor") dt1$characteristic_name<- as.factor(dt1$characteristic_name)
if (class(dt1$characteristic_name_long)!="factor") dt1$characteristic_name_long<- as.factor(dt1$characteristic_name_long)
if (class(dt1$method_speciation)!="factor") dt1$method_speciation<- as.factor(dt1$method_speciation)
if (class(dt1$result_sample_fraction)!="factor") dt1$result_sample_fraction<- as.factor(dt1$result_sample_fraction)
if (class(dt1$result_value)=="factor") dt1$result_value <-as.numeric(levels(dt1$result_value))[as.integer(dt1$result_value) ]               
if (class(dt1$result_value)=="character") dt1$result_value <-as.numeric(dt1$result_value)
if (class(dt1$result_unit)!="factor") dt1$result_unit<- as.factor(dt1$result_unit)
if (class(dt1$result_detection_quantitation_limit_measure)=="factor") dt1$result_detection_quantitation_limit_measure <-as.numeric(levels(dt1$result_detection_quantitation_limit_measure))[as.integer(dt1$result_detection_quantitation_limit_measure) ]               
if (class(dt1$result_detection_quantitation_limit_measure)=="character") dt1$result_detection_quantitation_limit_measure <-as.numeric(dt1$result_detection_quantitation_limit_measure)
if (class(dt1$result_detection_quantitation_limit_unit)!="factor") dt1$result_detection_quantitation_limit_unit<- as.factor(dt1$result_detection_quantitation_limit_unit)
if (class(dt1$result_detection_quantitation_limit_type)!="factor") dt1$result_detection_quantitation_limit_type<- as.factor(dt1$result_detection_quantitation_limit_type)
if (class(dt1$result_status_id)!="factor") dt1$result_status_id<- as.factor(dt1$result_status_id)                                   
# attempting to convert dt1$analysis_start_date dateTime string to R date structure (date or POSIXct)                                
tmpDateFormat<-"%Y-%m-%d"
tmp1analysis_start_date<-as.Date(dt1$analysis_start_date,format=tmpDateFormat)
# Keep the new dates only if they all converted correctly
if(nrow(dt1[dt1$analysis_start_date != "",]) == length(tmp1analysis_start_date[!is.na(tmp1analysis_start_date)])){dt1$analysis_start_date <- tmp1analysis_start_date } else {print("Date conversion failed for dt1$analysis_start_date. Please inspect the data and do the date conversion yourself.")}                                                                    

if (class(dt1$result_analytical_method_name)!="factor") dt1$result_analytical_method_name<- as.factor(dt1$result_analytical_method_name)
if (class(dt1$result_analytical_method_instrument)!="factor") dt1$result_analytical_method_instrument<- as.factor(dt1$result_analytical_method_instrument)
if (class(dt1$result_analytical_reference_method)!="factor") dt1$result_analytical_reference_method<- as.factor(dt1$result_analytical_reference_method)
if (class(dt1$laboratory_name)!="factor") dt1$laboratory_name<- as.factor(dt1$laboratory_name)
if (class(dt1$laboratory_sample_id)!="factor") dt1$laboratory_sample_id<- as.factor(dt1$laboratory_sample_id)
if (class(dt1$field_comment)!="factor") dt1$field_comment<- as.factor(dt1$field_comment)
if (class(dt1$analysis_comment)!="factor") dt1$analysis_comment<- as.factor(dt1$analysis_comment)
if (class(dt1$result_comment)!="factor") dt1$result_comment<- as.factor(dt1$result_comment)

# Convert Missing Values to NA for non-dates

dt1$activity_depth_height_measure <- ifelse((trimws(as.character(dt1$activity_depth_height_measure))==trimws("null")),NA,dt1$activity_depth_height_measure)               
suppressWarnings(dt1$activity_depth_height_measure <- ifelse(!is.na(as.numeric("null")) & (trimws(as.character(dt1$activity_depth_height_measure))==as.character(as.numeric("null"))),NA,dt1$activity_depth_height_measure))
dt1$layer_collection_end_depth <- ifelse((trimws(as.character(dt1$layer_collection_end_depth))==trimws("null")),NA,dt1$layer_collection_end_depth)               
suppressWarnings(dt1$layer_collection_end_depth <- ifelse(!is.na(as.numeric("null")) & (trimws(as.character(dt1$layer_collection_end_depth))==as.character(as.numeric("null"))),NA,dt1$layer_collection_end_depth))
dt1$layer_collection_start_depth <- ifelse((trimws(as.character(dt1$layer_collection_start_depth))==trimws("null")),NA,dt1$layer_collection_start_depth)               
suppressWarnings(dt1$layer_collection_start_depth <- ifelse(!is.na(as.numeric("null")) & (trimws(as.character(dt1$layer_collection_start_depth))==as.character(as.numeric("null"))),NA,dt1$layer_collection_start_depth))
dt1$method_speciation <- as.factor(ifelse((trimws(as.character(dt1$method_speciation))==trimws("null")),NA,as.character(dt1$method_speciation)))
dt1$result_sample_fraction <- as.factor(ifelse((trimws(as.character(dt1$result_sample_fraction))==trimws("null")),NA,as.character(dt1$result_sample_fraction)))
dt1$result_unit <- as.factor(ifelse((trimws(as.character(dt1$result_unit))==trimws("null")),NA,as.character(dt1$result_unit)))
dt1$result_detection_quantitation_limit_measure <- ifelse((trimws(as.character(dt1$result_detection_quantitation_limit_measure))==trimws("null")),NA,dt1$result_detection_quantitation_limit_measure)               
suppressWarnings(dt1$result_detection_quantitation_limit_measure <- ifelse(!is.na(as.numeric("null")) & (trimws(as.character(dt1$result_detection_quantitation_limit_measure))==as.character(as.numeric("null"))),NA,dt1$result_detection_quantitation_limit_measure))
dt1$result_detection_quantitation_limit_unit <- as.factor(ifelse((trimws(as.character(dt1$result_detection_quantitation_limit_unit))==trimws("null")),NA,as.character(dt1$result_detection_quantitation_limit_unit)))
dt1$result_detection_quantitation_limit_type <- as.factor(ifelse((trimws(as.character(dt1$result_detection_quantitation_limit_type))==trimws("null")),NA,as.character(dt1$result_detection_quantitation_limit_type)))
dt1$result_analytical_method_name <- as.factor(ifelse((trimws(as.character(dt1$result_analytical_method_name))==trimws("null")),NA,as.character(dt1$result_analytical_method_name)))
dt1$result_analytical_method_name <- as.factor(ifelse((trimws(as.character(dt1$result_analytical_method_name))==trimws("blank")),NA,as.character(dt1$result_analytical_method_name)))
dt1$result_analytical_method_instrument <- as.factor(ifelse((trimws(as.character(dt1$result_analytical_method_instrument))==trimws("null")),NA,as.character(dt1$result_analytical_method_instrument)))
dt1$result_analytical_method_instrument <- as.factor(ifelse((trimws(as.character(dt1$result_analytical_method_instrument))==trimws("blank")),NA,as.character(dt1$result_analytical_method_instrument)))
dt1$result_analytical_reference_method <- as.factor(ifelse((trimws(as.character(dt1$result_analytical_reference_method))==trimws("null")),NA,as.character(dt1$result_analytical_reference_method)))
dt1$laboratory_name <- as.factor(ifelse((trimws(as.character(dt1$laboratory_name))==trimws("null")),NA,as.character(dt1$laboratory_name)))
dt1$field_comment <- as.factor(ifelse((trimws(as.character(dt1$field_comment))==trimws("null")),NA,as.character(dt1$field_comment)))
dt1$analysis_comment <- as.factor(ifelse((trimws(as.character(dt1$analysis_comment))==trimws("null")),NA,as.character(dt1$analysis_comment)))
dt1$result_comment <- as.factor(ifelse((trimws(as.character(dt1$result_comment))==trimws("null")),NA,as.character(dt1$result_comment)))


# Here is the structure of the input data frame:
str(dt1)                            
attach(dt1)                            
# The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 

summary(dataset_name)
summary(monitoring_location_id)
summary(monitoring_location_name)
summary(activity_media_name)
summary(activity_start_date)
summary(activity_start_time)
summary(activity_end_date)
summary(activity_end_time)
summary(activity_depth_height_measure)
summary(activity_depth_height_unit)
summary(layer_collection_end_depth)
summary(layer_collection_start_depth)
summary(characteristic_name)
summary(characteristic_name_long)
summary(method_speciation)
summary(result_sample_fraction)
summary(result_value)
summary(result_unit)
summary(result_detection_quantitation_limit_measure)
summary(result_detection_quantitation_limit_unit)
summary(result_detection_quantitation_limit_type)
summary(result_status_id)
summary(analysis_start_date)
summary(result_analytical_method_name)
summary(result_analytical_method_instrument)
summary(result_analytical_reference_method)
summary(laboratory_name)
summary(laboratory_sample_id)
summary(field_comment)
summary(analysis_comment)
summary(result_comment) 
# Get more details on character variables

summary(as.factor(dt1$dataset_name)) 
summary(as.factor(dt1$monitoring_location_id)) 
summary(as.factor(dt1$monitoring_location_name)) 
summary(as.factor(dt1$activity_media_name)) 
summary(as.factor(dt1$activity_depth_height_unit)) 
summary(as.factor(dt1$characteristic_name)) 
summary(as.factor(dt1$characteristic_name_long)) 
summary(as.factor(dt1$method_speciation)) 
summary(as.factor(dt1$result_sample_fraction)) 
summary(as.factor(dt1$result_unit)) 
summary(as.factor(dt1$result_detection_quantitation_limit_unit)) 
summary(as.factor(dt1$result_detection_quantitation_limit_type)) 
summary(as.factor(dt1$result_status_id)) 
summary(as.factor(dt1$result_analytical_method_name)) 
summary(as.factor(dt1$result_analytical_method_instrument)) 
summary(as.factor(dt1$result_analytical_reference_method)) 
summary(as.factor(dt1$laboratory_name)) 
summary(as.factor(dt1$laboratory_sample_id)) 
summary(as.factor(dt1$field_comment)) 
summary(as.factor(dt1$analysis_comment)) 
summary(as.factor(dt1$result_comment))
detach(dt1)               



inUrl2  <- "https://pasta.lternet.edu/package/data/eml/edi/1818/1/d29d69ba2b05d97246b2f6dd68733b6e" 
infile2 <- tempfile()
try(download.file(inUrl2,infile2,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
if (is.na(file.size(infile2))) download.file(inUrl2,infile2,method="auto")


dt2 <-read.csv(infile2,header=F 
               ,skip=1
               ,sep=","  
               ,quot='"' 
               , col.names=c(
                 "dataset_name",     
                 "monitoring_location_name",     
                 "records",     
                 "lat_dd",     
                 "lon_dd",     
                 "coord_horiz_accu_meas",     
                 "coord_horiz_accu_unit",     
                 "method_coord",     
                 "update_date"    ), check.names=TRUE)

unlink(infile2)

# Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings

if (class(dt2$dataset_name)!="factor") dt2$dataset_name<- as.factor(dt2$dataset_name)
if (class(dt2$monitoring_location_name)!="factor") dt2$monitoring_location_name<- as.factor(dt2$monitoring_location_name)
if (class(dt2$records)=="factor") dt2$records <-as.numeric(levels(dt2$records))[as.integer(dt2$records) ]               
if (class(dt2$records)=="character") dt2$records <-as.numeric(dt2$records)
if (class(dt2$lat_dd)=="factor") dt2$lat_dd <-as.numeric(levels(dt2$lat_dd))[as.integer(dt2$lat_dd) ]               
if (class(dt2$lat_dd)=="character") dt2$lat_dd <-as.numeric(dt2$lat_dd)
if (class(dt2$lon_dd)=="factor") dt2$lon_dd <-as.numeric(levels(dt2$lon_dd))[as.integer(dt2$lon_dd) ]               
if (class(dt2$lon_dd)=="character") dt2$lon_dd <-as.numeric(dt2$lon_dd)
if (class(dt2$coord_horiz_accu_meas)=="factor") dt2$coord_horiz_accu_meas <-as.numeric(levels(dt2$coord_horiz_accu_meas))[as.integer(dt2$coord_horiz_accu_meas) ]               
if (class(dt2$coord_horiz_accu_meas)=="character") dt2$coord_horiz_accu_meas <-as.numeric(dt2$coord_horiz_accu_meas)
if (class(dt2$coord_horiz_accu_unit)!="factor") dt2$coord_horiz_accu_unit<- as.factor(dt2$coord_horiz_accu_unit)
if (class(dt2$method_coord)!="factor") dt2$method_coord<- as.factor(dt2$method_coord)                                   
# attempting to convert dt2$update_date dateTime string to R date structure (date or POSIXct)                                
tmpDateFormat<-"%Y-%m-%d"
tmp2update_date<-as.Date(dt2$update_date,format=tmpDateFormat)
# Keep the new dates only if they all converted correctly
if(nrow(dt2[dt2$update_date != "",]) == length(tmp2update_date[!is.na(tmp2update_date)])){dt2$update_date <- tmp2update_date } else {print("Date conversion failed for dt2$update_date. Please inspect the data and do the date conversion yourself.")}                                                                    


# Convert Missing Values to NA for non-dates



# Here is the structure of the input data frame:
str(dt2)                            
attach(dt2)                            
# The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 

summary(dataset_name)
summary(monitoring_location_name)
summary(records)
summary(lat_dd)
summary(lon_dd)
summary(coord_horiz_accu_meas)
summary(coord_horiz_accu_unit)
summary(method_coord)
summary(update_date) 
# Get more details on character variables

summary(as.factor(dt2$dataset_name)) 
summary(as.factor(dt2$monitoring_location_name)) 
summary(as.factor(dt2$coord_horiz_accu_unit)) 
summary(as.factor(dt2$method_coord))
detach(dt2)               

write.csv(dt1,"/Users/renaudsrr/Desktop/STAGE_MTL/Scripts/Data/IISD-ELA/csv_traites/dt1.csv")
write.csv(dt2,"/Users/renaudsrr/Desktop/STAGE_MTL/Scripts/Data/IISD-ELA/csv_traites/dt2.csv")












































# Package ID: edi.1557.3 Cataloging System:https://pasta.edirepository.org.
# Data set title: IISD Experimental Lakes Area LTER: Phytoplankton Abundance, Biomass, and Community Structure, 1969 â 2023.
# Data set creator:    - IISD Experimental Lakes Area 
# Data set creator:  Scott Higgins - IISD Experimental Lakes Area 
# Contact:  Chris Hay - Scientific Data Officer IISD Experimental Lakes Area  - chay@iisd-ela.org
# Contact:  Scott Higgins - Senior Research Scientist IISD Experimental Lakes Area  - shiggins@iisd-ela.org
# Stylesheet v2.14 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu      
# Uncomment the following lines to have R clear previous work, or set a working directory
# rm(list=ls())      

# setwd("C:/users/my_name/my_dir")       



options(HTTPUserAgent="EDI_CodeGen")


inUrl1  <- "https://pasta.lternet.edu/package/data/eml/edi/1557/3/b802bab61e3f26591c3d5bffe0c23b6e" 
infile1 <- tempfile()
try(download.file(inUrl1,infile1,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
if (is.na(file.size(infile1))) download.file(inUrl1,infile1,method="auto")


dt1 <-read.csv(infile1,header=F 
               ,skip=1
               ,sep=","  
               ,quot='"' 
               , col.names=c(
                 "monitoring_location_name",     
                 "date_collected",     
                 "stratum",     
                 "depth_upper",     
                 "depth_lower",     
                 "cyanobacteria_biomass",     
                 "chlorophyte_biomass",     
                 "euglenophyte_biomass",     
                 "chrysophyte_biomass",     
                 "diatoms_biomass",     
                 "cryptophyte_biomass",     
                 "dinoflagellate_biomass",     
                 "total_biomass"    ), check.names=TRUE)

unlink(infile1)

# Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings

if (class(dt1$monitoring_location_name)!="factor") dt1$monitoring_location_name<- as.factor(dt1$monitoring_location_name)                                   
# attempting to convert dt1$date_collected dateTime string to R date structure (date or POSIXct)                                
tmpDateFormat<-"%Y-%m-%d"
tmp1date_collected<-as.Date(dt1$date_collected,format=tmpDateFormat)
# Keep the new dates only if they all converted correctly
if(nrow(dt1[dt1$date_collected != "",]) == length(tmp1date_collected[!is.na(tmp1date_collected)])){dt1$date_collected <- tmp1date_collected } else {print("Date conversion failed for dt1$date_collected. Please inspect the data and do the date conversion yourself.")}                                                                    

if (class(dt1$stratum)!="factor") dt1$stratum<- as.factor(dt1$stratum)
if (class(dt1$depth_upper)=="factor") dt1$depth_upper <-as.numeric(levels(dt1$depth_upper))[as.integer(dt1$depth_upper) ]               
if (class(dt1$depth_upper)=="character") dt1$depth_upper <-as.numeric(dt1$depth_upper)
if (class(dt1$depth_lower)=="factor") dt1$depth_lower <-as.numeric(levels(dt1$depth_lower))[as.integer(dt1$depth_lower) ]               
if (class(dt1$depth_lower)=="character") dt1$depth_lower <-as.numeric(dt1$depth_lower)
if (class(dt1$cyanobacteria_biomass)=="factor") dt1$cyanobacteria_biomass <-as.numeric(levels(dt1$cyanobacteria_biomass))[as.integer(dt1$cyanobacteria_biomass) ]               
if (class(dt1$cyanobacteria_biomass)=="character") dt1$cyanobacteria_biomass <-as.numeric(dt1$cyanobacteria_biomass)
if (class(dt1$chlorophyte_biomass)=="factor") dt1$chlorophyte_biomass <-as.numeric(levels(dt1$chlorophyte_biomass))[as.integer(dt1$chlorophyte_biomass) ]               
if (class(dt1$chlorophyte_biomass)=="character") dt1$chlorophyte_biomass <-as.numeric(dt1$chlorophyte_biomass)
if (class(dt1$euglenophyte_biomass)=="factor") dt1$euglenophyte_biomass <-as.numeric(levels(dt1$euglenophyte_biomass))[as.integer(dt1$euglenophyte_biomass) ]               
if (class(dt1$euglenophyte_biomass)=="character") dt1$euglenophyte_biomass <-as.numeric(dt1$euglenophyte_biomass)
if (class(dt1$chrysophyte_biomass)=="factor") dt1$chrysophyte_biomass <-as.numeric(levels(dt1$chrysophyte_biomass))[as.integer(dt1$chrysophyte_biomass) ]               
if (class(dt1$chrysophyte_biomass)=="character") dt1$chrysophyte_biomass <-as.numeric(dt1$chrysophyte_biomass)
if (class(dt1$diatoms_biomass)=="factor") dt1$diatoms_biomass <-as.numeric(levels(dt1$diatoms_biomass))[as.integer(dt1$diatoms_biomass) ]               
if (class(dt1$diatoms_biomass)=="character") dt1$diatoms_biomass <-as.numeric(dt1$diatoms_biomass)
if (class(dt1$cryptophyte_biomass)=="factor") dt1$cryptophyte_biomass <-as.numeric(levels(dt1$cryptophyte_biomass))[as.integer(dt1$cryptophyte_biomass) ]               
if (class(dt1$cryptophyte_biomass)=="character") dt1$cryptophyte_biomass <-as.numeric(dt1$cryptophyte_biomass)
if (class(dt1$dinoflagellate_biomass)=="factor") dt1$dinoflagellate_biomass <-as.numeric(levels(dt1$dinoflagellate_biomass))[as.integer(dt1$dinoflagellate_biomass) ]               
if (class(dt1$dinoflagellate_biomass)=="character") dt1$dinoflagellate_biomass <-as.numeric(dt1$dinoflagellate_biomass)
if (class(dt1$total_biomass)=="factor") dt1$total_biomass <-as.numeric(levels(dt1$total_biomass))[as.integer(dt1$total_biomass) ]               
if (class(dt1$total_biomass)=="character") dt1$total_biomass <-as.numeric(dt1$total_biomass)

# Convert Missing Values to NA for non-dates



# Here is the structure of the input data frame:
str(dt1)                            
attach(dt1)                            
# The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 

summary(monitoring_location_name)
summary(date_collected)
summary(stratum)
summary(depth_upper)
summary(depth_lower)
summary(cyanobacteria_biomass)
summary(chlorophyte_biomass)
summary(euglenophyte_biomass)
summary(chrysophyte_biomass)
summary(diatoms_biomass)
summary(cryptophyte_biomass)
summary(dinoflagellate_biomass)
summary(total_biomass) 
# Get more details on character variables

summary(as.factor(dt1$monitoring_location_name)) 
summary(as.factor(dt1$stratum))
detach(dt1)               



inUrl2  <- "https://pasta.lternet.edu/package/data/eml/edi/1557/3/f6e00309d88b6c726c4bbc7f917c70a0" 
infile2 <- tempfile()
try(download.file(inUrl2,infile2,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
if (is.na(file.size(infile2))) download.file(inUrl2,infile2,method="auto")


dt2 <-read.csv(infile2,header=F 
               ,skip=1
               ,sep=","  
               ,quot='"' 
               , col.names=c(
                 "monitoring_location_name",     
                 "date_collected",     
                 "stratum",     
                 "depth_upper",     
                 "depth_lower",     
                 "species_code",     
                 "volume_cell",     
                 "density",     
                 "biomass"    ), check.names=TRUE)

unlink(infile2)

# Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings

if (class(dt2$monitoring_location_name)!="factor") dt2$monitoring_location_name<- as.factor(dt2$monitoring_location_name)                                   
# attempting to convert dt2$date_collected dateTime string to R date structure (date or POSIXct)                                
tmpDateFormat<-"%Y-%m-%d"
tmp2date_collected<-as.Date(dt2$date_collected,format=tmpDateFormat)
# Keep the new dates only if they all converted correctly
if(nrow(dt2[dt2$date_collected != "",]) == length(tmp2date_collected[!is.na(tmp2date_collected)])){dt2$date_collected <- tmp2date_collected } else {print("Date conversion failed for dt2$date_collected. Please inspect the data and do the date conversion yourself.")}                                                                    

if (class(dt2$stratum)!="factor") dt2$stratum<- as.factor(dt2$stratum)
if (class(dt2$depth_upper)=="factor") dt2$depth_upper <-as.numeric(levels(dt2$depth_upper))[as.integer(dt2$depth_upper) ]               
if (class(dt2$depth_upper)=="character") dt2$depth_upper <-as.numeric(dt2$depth_upper)
if (class(dt2$depth_lower)=="factor") dt2$depth_lower <-as.numeric(levels(dt2$depth_lower))[as.integer(dt2$depth_lower) ]               
if (class(dt2$depth_lower)=="character") dt2$depth_lower <-as.numeric(dt2$depth_lower)
if (class(dt2$species_code)!="factor") dt2$species_code<- as.factor(dt2$species_code)
if (class(dt2$volume_cell)=="factor") dt2$volume_cell <-as.numeric(levels(dt2$volume_cell))[as.integer(dt2$volume_cell) ]               
if (class(dt2$volume_cell)=="character") dt2$volume_cell <-as.numeric(dt2$volume_cell)
if (class(dt2$density)=="factor") dt2$density <-as.numeric(levels(dt2$density))[as.integer(dt2$density) ]               
if (class(dt2$density)=="character") dt2$density <-as.numeric(dt2$density)
if (class(dt2$biomass)=="factor") dt2$biomass <-as.numeric(levels(dt2$biomass))[as.integer(dt2$biomass) ]               
if (class(dt2$biomass)=="character") dt2$biomass <-as.numeric(dt2$biomass)

# Convert Missing Values to NA for non-dates



# Here is the structure of the input data frame:
str(dt2)                            
attach(dt2)                            
# The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 

summary(monitoring_location_name)
summary(date_collected)
summary(stratum)
summary(depth_upper)
summary(depth_lower)
summary(species_code)
summary(volume_cell)
summary(density)
summary(biomass) 
# Get more details on character variables

summary(as.factor(dt2$monitoring_location_name)) 
summary(as.factor(dt2$stratum)) 
summary(as.factor(dt2$species_code))
detach(dt2)               



inUrl3  <- "https://pasta.lternet.edu/package/data/eml/edi/1557/3/81c5e704f55ccd388c80a20b6ad2be64" 
infile3 <- tempfile()
try(download.file(inUrl3,infile3,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
if (is.na(file.size(infile3))) download.file(inUrl3,infile3,method="auto")


dt3 <-read.csv(infile3,header=F 
               ,skip=1
               ,sep=","  
               ,quot='"' 
               , col.names=c(
                 "species_code",     
                 "group",     
                 "phylum",     
                 "class",     
                 "order",     
                 "family",     
                 "genus",     
                 "specific_epithet",     
                 "variety",     
                 "species"    ), check.names=TRUE)

unlink(infile3)

# Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings

if (class(dt3$species_code)!="factor") dt3$species_code<- as.factor(dt3$species_code)
if (class(dt3$group)!="factor") dt3$group<- as.factor(dt3$group)
if (class(dt3$phylum)!="factor") dt3$phylum<- as.factor(dt3$phylum)
if (class(dt3$class)!="factor") dt3$class<- as.factor(dt3$class)
if (class(dt3$order)!="factor") dt3$order<- as.factor(dt3$order)
if (class(dt3$family)!="factor") dt3$family<- as.factor(dt3$family)
if (class(dt3$genus)!="factor") dt3$genus<- as.factor(dt3$genus)
if (class(dt3$specific_epithet)!="factor") dt3$specific_epithet<- as.factor(dt3$specific_epithet)
if (class(dt3$variety)!="factor") dt3$variety<- as.factor(dt3$variety)
if (class(dt3$species)!="factor") dt3$species<- as.factor(dt3$species)

# Convert Missing Values to NA for non-dates

dt3$order <- as.factor(ifelse((trimws(as.character(dt3$order))==trimws(".")),NA,as.character(dt3$order)))
dt3$family <- as.factor(ifelse((trimws(as.character(dt3$family))==trimws(".")),NA,as.character(dt3$family)))
dt3$specific_epithet <- as.factor(ifelse((trimws(as.character(dt3$specific_epithet))==trimws(".")),NA,as.character(dt3$specific_epithet)))


# Here is the structure of the input data frame:
str(dt3)                            
attach(dt3)                            
# The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 

summary(species_code)
summary(group)
summary(phylum)
summary(class)
summary(order)
summary(family)
summary(genus)
summary(specific_epithet)
summary(variety)
summary(species) 
# Get more details on character variables

summary(as.factor(dt3$species_code)) 
summary(as.factor(dt3$group)) 
summary(as.factor(dt3$phylum)) 
summary(as.factor(dt3$class)) 
summary(as.factor(dt3$order)) 
summary(as.factor(dt3$family)) 
summary(as.factor(dt3$genus)) 
summary(as.factor(dt3$specific_epithet)) 
summary(as.factor(dt3$variety)) 
summary(as.factor(dt3$species))
detach(dt3)               



inUrl4  <- "https://pasta.lternet.edu/package/data/eml/edi/1557/3/00fd73139e11a63377da7170f54177ea" 
infile4 <- tempfile()
try(download.file(inUrl4,infile4,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
if (is.na(file.size(infile4))) download.file(inUrl4,infile4,method="auto")


dt4 <-read.csv(infile4,header=F 
               ,skip=1
               ,sep=","  
               ,quot='"' 
               , col.names=c(
                 "species_code",     
                 "trophic_type"    ), check.names=TRUE)

unlink(infile4)

# Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings

if (class(dt4$species_code)!="factor") dt4$species_code<- as.factor(dt4$species_code)
if (class(dt4$trophic_type)!="factor") dt4$trophic_type<- as.factor(dt4$trophic_type)

# Convert Missing Values to NA for non-dates



# Here is the structure of the input data frame:
str(dt4)                            
attach(dt4)                            
# The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 

summary(species_code)
summary(trophic_type) 
# Get more details on character variables

summary(as.factor(dt4$species_code)) 
summary(as.factor(dt4$trophic_type))
detach(dt4)               



inUrl5  <- "https://pasta.lternet.edu/package/data/eml/edi/1557/3/0231dbfd0d8e93265910efbe655c6cc7" 
infile5 <- tempfile()
try(download.file(inUrl5,infile5,method="curl",extra=paste0(' -A "',getOption("HTTPUserAgent"),'"')))
if (is.na(file.size(infile5))) download.file(inUrl5,infile5,method="auto")


dt5 <-read.csv(infile5,header=F 
               ,skip=1
               ,sep=","  
               ,quot='"' 
               , col.names=c(
                 "monitoring_location_name",     
                 "area_catchment",     
                 "area_surface",     
                 "depth_mean",     
                 "depth_max",     
                 "volume_total",     
                 "order_lake",     
                 "mixing_status",     
                 "latitude",     
                 "longitude"    ), check.names=TRUE)

unlink(infile5)

# Fix any interval or ratio columns mistakenly read in as nominal and nominal columns read as numeric or dates read as strings

if (class(dt5$monitoring_location_name)!="factor") dt5$monitoring_location_name<- as.factor(dt5$monitoring_location_name)
if (class(dt5$area_catchment)=="factor") dt5$area_catchment <-as.numeric(levels(dt5$area_catchment))[as.integer(dt5$area_catchment) ]               
if (class(dt5$area_catchment)=="character") dt5$area_catchment <-as.numeric(dt5$area_catchment)
if (class(dt5$area_surface)=="factor") dt5$area_surface <-as.numeric(levels(dt5$area_surface))[as.integer(dt5$area_surface) ]               
if (class(dt5$area_surface)=="character") dt5$area_surface <-as.numeric(dt5$area_surface)
if (class(dt5$depth_mean)=="factor") dt5$depth_mean <-as.numeric(levels(dt5$depth_mean))[as.integer(dt5$depth_mean) ]               
if (class(dt5$depth_mean)=="character") dt5$depth_mean <-as.numeric(dt5$depth_mean)
if (class(dt5$depth_max)=="factor") dt5$depth_max <-as.numeric(levels(dt5$depth_max))[as.integer(dt5$depth_max) ]               
if (class(dt5$depth_max)=="character") dt5$depth_max <-as.numeric(dt5$depth_max)
if (class(dt5$volume_total)=="factor") dt5$volume_total <-as.numeric(levels(dt5$volume_total))[as.integer(dt5$volume_total) ]               
if (class(dt5$volume_total)=="character") dt5$volume_total <-as.numeric(dt5$volume_total)
if (class(dt5$order_lake)=="factor") dt5$order_lake <-as.numeric(levels(dt5$order_lake))[as.integer(dt5$order_lake) ]               
if (class(dt5$order_lake)=="character") dt5$order_lake <-as.numeric(dt5$order_lake)
if (class(dt5$mixing_status)!="factor") dt5$mixing_status<- as.factor(dt5$mixing_status)
if (class(dt5$latitude)=="factor") dt5$latitude <-as.numeric(levels(dt5$latitude))[as.integer(dt5$latitude) ]               
if (class(dt5$latitude)=="character") dt5$latitude <-as.numeric(dt5$latitude)
if (class(dt5$longitude)=="factor") dt5$longitude <-as.numeric(levels(dt5$longitude))[as.integer(dt5$longitude) ]               
if (class(dt5$longitude)=="character") dt5$longitude <-as.numeric(dt5$longitude)

# Convert Missing Values to NA for non-dates



# Here is the structure of the input data frame:
str(dt5)                            
attach(dt5)                            
# The analyses below are basic descriptions of the variables. After testing, they should be replaced.                 

summary(monitoring_location_name)
summary(area_catchment)
summary(area_surface)
summary(depth_mean)
summary(depth_max)
summary(volume_total)
summary(order_lake)
summary(mixing_status)
summary(latitude)
summary(longitude) 
# Get more details on character variables

summary(as.factor(dt5$monitoring_location_name)) 
summary(as.factor(dt5$mixing_status))
detach(dt5)               



write.csv(dt1,"/Users/renaudsrr/Desktop/STAGE_MTL/Scripts/Data/IISD-ELA/csv_traites/dt3.csv")
write.csv(dt2,"/Users/renaudsrr/Desktop/STAGE_MTL/Scripts/Data/IISD-ELA/csv_traites/dt4.csv")
write.csv(dt3,"/Users/renaudsrr/Desktop/STAGE_MTL/Scripts/Data/IISD-ELA/csv_traites/dt5.csv")
write.csv(dt4,"/Users/renaudsrr/Desktop/STAGE_MTL/Scripts/Data/IISD-ELA/csv_traites/dt6.csv")
write.csv(dt5,"/Users/renaudsrr/Desktop/STAGE_MTL/Scripts/Data/IISD-ELA/csv_traites/dt7.csv")



