
## Script for preparation of Amazonia birds vocalization data

## Receives detection raw data and returns:

## - 3D table (species X groups X dates) with number of detections by group of
##    sites, date and species (one detection corresponds to a recording segment
##    where the detection probability for the species is above the treshold);

## - 2D table (sites X dates) with dates corresponding to the surveys of each
##    site at each year;

## - 2D table (groups X dates) with effort (minutes of recording) for each group
##    of sites and days of each year;

## - 2D table (groups X dates) with effort (number of sites per group) for each
##    group of sites and days of each year;

# ------------------------------------------------------------------------------
## Load data:

# First we need to set the working directory using the command "setwd()", ex.:

# setwd("C:/Users/fabio/OneDrive/Documentos/Projetos_github/birds_data_preparation/R_scripts")

# Now we can load the input files:

# "Y" is a CSV file whose dimensions are:
#   - Rows: recording segments
#   - Columns: species
# where each cell has the probability of presence of the vocalization from that
# species in that segment.
# It is a small survey from the last 100000 rows from the original (much
# bigger) dataset.
Y <- read.csv("../Data_input/Y.csv", head=TRUE, sep=";")

# "SEGMTS" is a CSV file whose dimensions are:
# - Rows: recording segments
# - Columns: ID data of each recording segment (site, date, recording number and
#             segment number)
# It is a small survey from the last 100000 rows from the original (much
# bigger) dataset.
SEGMTS <- read.csv("../Data_input/date_time_diurnal_capitalization_r.txt", head=TRUE, sep=",")

# "SITES" is a CSV file whose dimensions are:
# - Rows: sites
# - Columns: ID data of each site (name, forest type (old growth = 0, secondary
#             forest = 1), geographic coordinates (X and Y))
SITES <- read.csv("../Data_input/locations_diurnal_capitalization_r.txt", head=TRUE, sep="\t")

# ------------------------------------------------------------------------------
## Create arguments:

# Species number:
nsps <- length(colnames(Y))

# Threshold to filter detection probabilities for the presence of vocalization
# at the recording segment:
thr <- 0.9
# thr <- 0.85
# thr <- 0.99

# Due to "SEGMTS" not having survey information about all site presents in
# "SITES", we have to filter the "SITES" object keeping only the sites present
# in "SEGMTS":
shortSITES<-SITES[which(SITES$Spot %in% SEGMTS$Spot),]


# Define maximum distance in meters necessary to consider two sites as belonging
# to the same cluster (group).
# It doesnt prevent some sites from the same group from having a bigger distance
# than this between them:
dmin <- 110

# Obtain groups with the "single" method, which clusterizes based on the
# maximum distance between sites from each cluster and saves this groups in a
# dendrogram object:
dendrog <- hclust(dist(shortSITES[,3:4]), method="single")

# Plot:
plot(dendrog,hang=-1)

# Create vector with groups of sites based on the dendrogram:
groups <- cutree(dendrog,h=dmin)

# Plot representing groups with colors:
plot(shortSITES[,3:4],col=groups)

# Create matrix of groups centroid coordinates with row number equal to the
# number of groups:
CORCEN <- matrix(NA,nrow=max(groups),ncol=2)
for(i in 1:max(groups)) {
  CORCEN[i,1] <- mean(shortSITES[which(groups==i),3])
  CORCEN[i,2] <- mean(shortSITES[which(groups==i),4])
}

# Add centroinds to the plot:
points(CORCEN,pch=4,cex=0.5)

# ------------------------------------------------------------------------------
## Define output dimensions:

# Sites names list:
sitel <- sort(unique(SEGMTS$Spot))

# Number of sites:
nsites <- length(unique(sitel))

# Number of sites groups:
ngroups <- length(table(groups))

# Vector of vectors of groups where each vector of groups has all sites
# belonging to that group:
sitegroup <- as.vector(rep(list(rep(NA, max(table(groups)))), ngroups))
for (i in 1:length(sitegroup)) { sitegroup[[i]] <- shortSITES[which(groups==i),1] }

# Groups names list (the name of each group is the name of the first site
# belonging to that group):
groupl <- rep(NA,ngroups)
for(i in 1:ngroups) {groupl[i]<-sitegroup[[i]][1]}

# Years list:
yearl <- sort(unique(floor(SEGMTS$Date/10000)))

# Dates list:
datel <- as.Date(as.character(SEGMTS$Date), format = "%Y%m%d")

# Obtain the maximum number of surveying days in each year based on groups:
yrgroup <- matrix(rep(NA,ngroups*length(yearl)),ncol=length(yearl))
for(i in 1:ngroups) {
  yrgroup[i,] <- tabulate(floor(sort(unique(SEGMTS$Date[which(SEGMTS$Spot %in% sitegroup[[i]])]))/10000)-2009, nbins=5)
}
maxdy <- apply(yrgroup,2,max)

# ------------------------------------------------------------------------------
## Define output tables:

# Table 1 - number of detections by group, day and species:
NDETSG <- array(data=NA,dim=c(ngroups,sum(maxdy),nsps))
dimnames(NDETSG)[[3]] <- colnames(Y)

# Table 2 - Dates of the surveying days by site over the years:
DATSG <- t(matrix(data=NA,nrow=ngroups,ncol=sum(maxdy)))
DATSG <- data.frame(DATSG)
for(i in 1:ngroups) {class(DATSG[,i])="Date"}

# Table 3 - Effort (minutes of recording) by day and group:
EFFGmins <- matrix(data=NA,nrow=ngroups,ncol=sum(maxdy))

# Table 4 - Effort (number of surveyed sites) by day and group:
EFFGsits <- matrix(data=NA,nrow=ngroups,ncol=sum(maxdy))

# ------------------------------------------------------------------------------
## Fill output tables:

# get column corresponding to current species
for(i in 1:nsps) {
	
	# define current year
	for(j in 1:length(yearl)) {
		cy <- yearl[j] # "cy" = "current year"
		
		# define current group
		for(k in 1:ngroups) {
			cat("\nespecie = ", i, "\nano = ", j, "\ngrupo =",k,"\n", "\n-------------")
			cg <- sitegroup[[k]] # "cg" = "current group"
			
			# get dates belonging to current group and year:			
			cdg <- sort(unique(datel[which(SEGMTS$Spot%in%cg & format(datel,'%Y')==cy)])) # "cdg" = "current day group"
			
			# skip to the next group if there are no surveyed days in "cy":
			if(length(cdg)==0) next
			
			# Obtain "fdc" (first day column), which is the first column to be filled
			# (related to the first day from current year)
			if(j==1) { fdc <- 1 } else { fdc<-sum(maxdy[1:(j-1)])+1 }
			
			# save the dates at "DATSG":
			if (i == 1) { # if it is the first species loop
			  DATSG[fdc:(fdc+length(cdg)-1),k] <- cdg
			}
			
			# save effort and number of detections, respectively, at the effort tables
			# and at NDETSG:			
			for(l in 1:length(cdg)) {
				cd <- cdg[l] # "cd" = "current day"
				
				# save effort in minutes of recording at "EFFGmins":
				EFFGmins[k,(fdc+l-1)] <- length(which(SEGMTS$Spot%in%cg & datel==cd))
				
				# save effort in number of sites at "EFFGsits":
				EFFGsits[k,(fdc+l-1)] <- length(unique(SEGMTS$Spot[which(SEGMTS$Spot%in%cg & datel==cd)]))

				# save number of detections at "NDETSG":
				ndet <- sum(datel==cd & SEGMTS$Spot%in%cg & Y[,i]>thr,na.rm=TRUE)
				NDETSG[k, fdc+l-1, i] <- ndet

  		} # l dates
		} # k sites
	} # j years	
} # i species

  # ------------------------------------------------------------------------------
## Save output

# Clean work space:
list=ls()
list <- list[-which(list%in%c("DATSG","EFFGmins","EFFGsits","shortSITES","groups","NDETSG"))]
rm(list=list, list)

# Save separately the CSV file "groups" to be used in the Python script:
write.table(groups, "../Data_input/groups.csv", sep = "\t", row.names=FALSE, quote=TRUE)

# Save output as an "RData" file:
save.image("output_data_preparation_r.RData")



