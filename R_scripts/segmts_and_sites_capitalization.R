  
## Script to correct/standardize sites names letter capitalization in the CSV
## files "date_time_diurnal.txt" and "locations_diurnal.txt" (in the folder
## "Data_input/Raw_data), which will become, respectively, the objects "SEGMTS"
## and "SITES" in the script "data_preparation_r.R".

# ------------------------------------------------------------------------------
## Load files:

# First we need to set the working directory using the command "setwd()", ex.:

# setwd("C:/Users/fabio/OneDrive/Documentos/Projetos_github/birds_data_preparation/R_scripts")

SEGMTS <- read.csv("../Data_input/Raw_data/date_time_diurnal.txt", head=TRUE, sep=",")
SITES <- read.csv("../Data_input/Raw_data/locations_diurnal.txt", head=TRUE, sep="\t", check.names = FALSE)

# ------------------------------------------------------------------------------
## Correct/standardize capitalization:

SEGMTS$Spot <- sub("(?<=^.{2})C", "c", SEGMTS$Spot, perl = TRUE)
SEGMTS$Spot <- sub("(?<=^.{2})M", "m", SEGMTS$Spot, perl = TRUE)
SEGMTS$Spot <- sub("(?<=^.{7})X", "x", SEGMTS$Spot, perl = TRUE)

SITES$Spot <- sub("(?<=^.{2})C", "c", SITES$Spot, perl = TRUE)
SITES$Spot <- sub("(?<=^.{2})M", "m", SITES$Spot, perl = TRUE)
SITES$Spot <- sub("(?<=^.{7})X", "x", SITES$Spot, perl = TRUE)

# ------------------------------------------------------------------------------
## Export corrected files:

write.table(SEGMTS, "../Data_input/date_time_diurnal_capitalization_r.txt", sep = ",", row.names=FALSE, quote=FALSE)
write.table(SITES, "../Data_input/locations_diurnal_capitalization_r.txt", sep = "\t", row.names=FALSE, quote=FALSE)
