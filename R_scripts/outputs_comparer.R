
## Script to compare the tables in "output_data_preparation_r.RData" and
## "output_data_preparation_python.RData".

# ------------------------------------------------------------------------------
## Load files:

# First we need to set the working directory using the command "setwd()", ex.:

# setwd("C:/Users/fabio/OneDrive/Documentos/Projetos_github/birds_data_preparation/R_scripts")

load("output_data_preparation_r.RData")
load("output_data_preparation_python.RData")

# ------------------------------------------------------------------------------
## Compare tables:

sum(DATSG != DATSG_py, na.rm = TRUE) # "na.rm = TRUE" removes the "NA" values from the sum

sum(EFFGmins != EFFGmins_py, na.rm = TRUE)

sum(EFFGsits != EFFGsits_py, na.rm = TRUE)

sum(NDETSG != NDETSG_py, na.rm = TRUE)
