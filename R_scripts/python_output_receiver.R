
# Code to receive the output from "data_preparation_python.py" (in the folder
# "Python_scripts") and prepare it to the same format as the output from
# "data_preparation_r.R" (in the folder "R_scripts)

# ------------------------------------------------------------------------------
## Create arguments:
nsps <- 63

# ------------------------------------------------------------------------------
## Load files:

# First we need to set the working directory using the command "setwd()", ex.:

# setwd("C:/Users/fabio/OneDrive/Documentos/Projetos_github/birds_data_preparation/R_scripts")

load("../Python_scripts/DATSG_py.RData")
load("../Python_scripts/EFFGmins_py.RData")
load("../Python_scripts/EFFGsits_py.RData")

for (i in 1:nsps) { # 1:nsps
  load(paste("../Python_scripts/NDETSG_py_",i,".RData", sep=""))
}

# ------------------------------------------------------------------------------
## Define output tables:

NDETSG_py <- array(data=NA,dim=c(nrow(NDETSG_py_1),ncol(NDETSG_py_1),nsps))
slices <- list()

## save "NDETSG_py" slices as a list:
for (i in 1:nsps) {
  name <- paste("NDETSG_py_",i, sep="")
  obj <- ls()[ls() %in% c(name)]
  slices[[i]] <- get(obj)
}

## fill 3D array "NDETSG_py" with slices:
for (i in 1:length(slices)) {
  for (j in 1:ncol(NDETSG_py)) {
    NDETSG_py[,j,i] <- slices[[i]][,j]
  }
}

## clean work space keeping only necessary objects
list=ls()
list <- list[-which(list%in%c("DATSG_py", "EFFGmins_py", "EFFGsits_py", "NDETSG_py"))]
rm(list=list, list)

## Turn "NaN" into "NA"
NDETSG_py[which(is.nan(NDETSG_py))] <- NA
EFFGmins_py <- apply(EFFGmins_py, c(1,2), function (x) ifelse(is.nan(x),x <- NA, x))
EFFGsits_py <- apply(EFFGsits_py, c(1,2), function (x) ifelse(is.nan(x),x <- NA, x))

# ------------------------------------------------------------------------------
## Export corrected files as an RData work space:

save.image("output_data_preparation_python.RData")
