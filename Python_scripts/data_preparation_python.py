# Script for preparation of Amazonia birds vocalization data

# Receives detection raw data and returns:

# - 3D table (species X groups X dates) with number of detections by group of
#    sites, date and species (one detection corresponds to a recording segment
#    where the detection probability for the species is above the treshold);

# - 2D table (sites X dates) with dates corresponding to the surveys of each
#    site at each year;

# - 2D table (groups X dates) with effort (minutes of recording) for each group
#    of sites and days of each year;

# - 2D table (groups X dates) with effort (number of sites per group) for each
#    group of sites and days of each year;

# ----------------------------------------------------------------------------------------------------------------------
# Import dependencies:

# To import libraries first enter the command "pip install <library name here>" in the terminal.
# Maybe after that you will have to install it anyway clicking on the package name below and selecting to
# install package.

from math import floor
from time import time
import pandas as pd  # "as pd" just creates a nickname to the package
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from statistics import mean
from pyreadr import write_rdata

# ----------------------------------------------------------------------------------------------------------------------
# Start running time count:
start_time = time()

# the code will return at the end the running time difference.

# ----------------------------------------------------------------------------------------------------------------------
# Load CSV data as Pandas DataFrames:

# "Y" is a CSV file whose dimensions are:
#   - Rows: recording segments
#   - Columns: species
# where each cell has the probability of presence of the vocalization from that species in that segment.
Y = pd.read_csv("../Data_input/Raw_data/Y.csv", sep=";", dtype='float32')  # na_values=['NaN']

# "SEGMTS" is a CSV file whose dimensions are:
# - Rows: recording segments
# - Columns: ID data of each recording segment (site, date, recording number and segment number)
SEGMTS = pd.read_csv("../Data_input/date_time_diurnal_capitalization_python.txt", sep=",")

# "SITES" is a CSV file whose dimensions are:
# - Rows: sites
# - Columns: ID data of each site (name, forest type (old growth = 0, secondary forest = 1), geographic
#            coordinates (X and Y))
SITES = pd.read_csv("../Data_input/locations_diurnal_capitalization_python.txt", sep="\t")

# To extract a value at an index in a Pandas DataFrame we use the function "iloc".
# As an example we can access the first element (index 0) of the column "Spot" from dataFrame "SEGMTS" doing this:

# print(SITES["Spot"].iloc[0])

# We can also select the column number instead of column name, like this:

# print(SITES.iloc[0, 0])

# ----------------------------------------------------------------------------------------------------------------------
# Create arguments:

# Species number:
nsps = Y.shape[1]  # "shape[0]" is row number and "shape[1]" is column number

# Threshold to filter detection probabilities for the presence of vocalization at the recording segment:
thr = 0.9
# thr = 0.85
# thr = 0.99

# Due to "SEGMTS" not having survey information about all site presents in "SITES", we have to filter the "SITES"
# object keeping only the sites present in "SEGMTS":

# R original code (01):
"""
shortSITES = SITES[which(SITES$Spot %in% SEGMTS$Spot),]
"""
# Python version code (01):
indices = SITES["Spot"].isin(SEGMTS["Spot"])
shortSITES = SITES.loc[indices, :]

# Define maximum distance in meters necessary to consider two sites as belonging to the same cluster (group).
# It doesn't prevent some sites from the same group from having a bigger distance than this between them:
dmin = 110

# ----------------------------------------------------------------------------------------------------------------------
# Obtain groups with the "single" method, which clusterizes based on the maximum distance between sites from each
# cluster and saves this groups in a dendrogram object:

# R original code (02):
"""
#dendrog <- hclust(dist(shortSITES[,3:4]), method="single")
"""
# Python version code (02):
distance_matrix = pdist(shortSITES.iloc[:, 2:4])  # notice that in python the counting ends BEFORE the last index,
                                                  # in this case, before the index "4", in other words, at "3".
dendrogram_data = linkage(distance_matrix, method='single')

# Plot:

# R original code (03):
"""
#plot(dendrog,hang=-1)
"""
# Python version code (03):
"""
plt.figure(figsize=(10, 6))
dendrogram(dendrogram_data)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.title('Dendrogram')
plt.axhline(y=dmin)
plt.show()
"""

# Create vector with groups of sites based on the dendrogram:

# R original code (04):
"""
groups <- cutree(dendrog,h=dmin)
"""
# Python version code (04):
groups = fcluster(dendrogram_data, t=dmin, criterion='distance')

# We can see that the groups generated in Python have a different numeration from the equivalent object generated in R:
print("groups generated in python: ")
print(groups)
print("-"*50)

# Let's see if the groups sizes are equivalent to the object generated in R:

# At R we see the number of sites in each group with those commands:
"""
test <- rep(NA, length(unique(groups)))
for (i in 1:length(unique(groups))) {
  test[i] <- sum(groups == unique(groups)[i])
}
"""

# which returned this vector:
"""
> test
  [1]  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
 [36]  1  2  1  1  1  1  1  1  1  1  1  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
 [71]  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
[106]  1  1  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
[141]  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1
[176]  1  1  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
[211]  1  1  1 23
"""

# and filtering only the groups with more than one site it returns this:
"""
> test[test > 1]
[1]  2  2  2  2  2  2 23
"""

# Let's see the same in Python:

ug = sorted(np.unique(groups))   # vector with the number of groups in the object "groups":
groupsum = [None] * len(ug)      # empty vector

for i in range(0, len(groupsum)):
    groupsum[i] = sum(groups == ug[i])  # receives the number of sites in the current group

print("ug: ")
print(ug)
print("-"*50)

print("groupsum: ")
print(groupsum)
print("-"*50)

# again, lets filter only the groups with more than one site:
print("groups with more than one site: ")
for i in range(0, len(groupsum)):
    if groupsum[i] > 1:
        print(groupsum[i])
print("-"*50)

# We can see that despite the different numeration, both "groups" objects have the same number of groups with more than
# one site.

# Plot representing groups with colors:

# R original code (05):
"""
plot(shortSITES[,3:4],col=groups)
"""
# Python version code (05):
"""
plt.scatter(shortSITES["X"], shortSITES["Y"], c=groups)  # probably not working
plt.xlabel('Column 3')
plt.ylabel('Column 4')
plt.title('Scatter Plot with Colored Points')
plt.show()
"""

# ----------------------------------------------------------------------------------------------------------------------
# Due to the results not being exactly identical, lets import the file "groups.csv" which is the "groups" object
# generated in R. After that, lets generate again the objects "ug" and "groupsum":

groups = pd.read_csv("../Data_input/groups.csv")
groups = np.array(groups["x"])

ug = sorted(np.unique(groups))   # vector with the number of groups in the object "groups":
groupsum = [None] * len(ug)      # empty vector

for i in range(0, len(groupsum)):
    groupsum[i] = sum(groups == ug[i])  # receives the number of sites in the current group

print("groups generated in R: ")
print(groups)
print("-"*50)

print("length of groups generated in R: ")
print(str(len(groups)))
print("-"*50)

print("new 'ug':")
print(ug)
print("-"*50)

print("length of new 'ug':")
print(str(len(ug)))
print("-"*50)

print("groupsum:")
print(groupsum)
print("-"*50)

print("max(groupsum): ")
print(max(groupsum))
print("-"*50)

# ----------------------------------------------------------------------------------------------------------------------
# Create matrix of groups centroid coordinates with row number equal to the number of groups:

# R original code (06):
"""
CORCEN <- matrix(NA,nrow=max(groups),ncol=2)
 for(i in 1:max(groups)) {
   CORCEN[i,1] <- mean(shortSITES[which(groups==i),3])
   CORCEN[i,2] <- mean(shortSITES[which(groups==i),4])
}
"""
# Python version code (06):
CORCEN = pd.DataFrame(np.nan, index=range(ug[(len(ug)-1)]), columns=["X", "Y"], dtype=float)

for i in range(0, max(ug)):
    condition = (groups == ug[i])               # Define the condition for which to find the indices.
    indices = np.argwhere(condition).flatten()  # Get the indices where the condition is True (we use "flatten()" to
                                                # convert the resulting 2D array to a 1D array).
    CORCEN.iloc[i, 0] = mean(shortSITES.iloc[indices, 2])
    CORCEN.iloc[i, 1] = mean(shortSITES.iloc[indices, 3])

print("CORCEN: ")
print(CORCEN)
print("-"*50)

# ----------------------------------------------------------------------------------------------------------------------
# Define output dimensions:

# R original code (07):
"""
sitel <- sort(unique(SEGMTS$Spot)) # list of site names
nsites <- length(unique(sitel))    # number of sites
ngroups <- length(table(groups))   # number of groups of sites
sitegroup <- as.vector(rep(list(rep(NA, max(table(groups)))), ngroups))
for (i in 1:length(sitegroup)) { sitegroup[[i]] <- shortSITES[which(groups==i),1] }
groupl <- rep(NA,ngroups)          # list of group names
for(i in 1:ngroups) {groupl[i]<-sitegroup[[i]][1]}
yearl <- sort(unique(floor(SEGMTS$Date/10000)))
datel <- as.Date(as.character(SEGMTS$Date), format = "%Y%m%d")
"""
# Python version code (07):
sitel = sorted(np.unique(SEGMTS["Spot"]))  # list of site names
nsites = len(np.unique(sitel))             # number of sites
ngroups = len(np.unique(groups))           # number of groups of sites
print("ngroups: " + str(ngroups))
print("-"*50)

# Vector of vectors of groups where each vector of groups has all sites belonging to that group:
# Empty vector:
sitegroup = []
for i in range(0, ngroups):
    condition = (groups == (i+1))                           # Define the condition for which to find the indices.
    indices = np.argwhere(condition).flatten()              # Get the indices where the condition is True
    mylist = np.array(shortSITES.iloc[indices, 0]).tolist()   # Creates a Numpy array and turn it into a list object
    sitegroup.append(mylist)                                  # add list as an element in "sitegroup"

# Let's see the last (and biggest) element in "sitegroup" as an example:
print("sitegroup[len(sitegroup)-1]:")
print(sitegroup[len(sitegroup)-1])
print("-"*50)

# Groups names list (the name of each group is the name of the first site belonging to that group):
groupl = [None] * ngroups           # list of group names
for i in range(0, len(sitegroup)):
    groupl[i] = sitegroup[i][0]     # name of the first site in the group

print("groupl: ")
print(groupl)
print("-"*50)

# Years list:
yearl = sorted(np.unique(np.floor(SEGMTS["Date"]/10000).astype(int)))

print("yearl:")
print(yearl)
print("-"*50)

# Dates list:
datel = pd.to_datetime(SEGMTS["Date"], format="%Y%m%d")

print("datel: ")
print(datel)
print("-"*50)

# ----------------------------------------------------------------------------------------------------------------------
# Obtain the maximum number of surveying days in each year based on groups:

# Create matrix of groups by years:

# R original code (08):
"""
yrgroup <- matrix(rep(NA,ngroups*length(yearl)),ncol=length(yearl))
for(i in 1:ngroups) {
  yrgroup[i,] <- tabulate(floor(sort(unique(SEGMTS$Date[which(SEGMTS$Spot %in% sitegroup[[i]])]))/10000)-2009, nbins=5)
}
maxdy <- apply(yrgroup,2,max)
"""
# Python version code (08):
yrgroup = pd.DataFrame(index=range(ngroups), columns=range(len(yearl)))
for i in range(0, ngroups):
    condition = SEGMTS["Spot"].isin(sitegroup[i])
    indices = np.argwhere(condition).flatten()
    rawdates = sorted(np.unique(SEGMTS["Date"].iloc[indices]))
    rawdates = [floor(x/10000)-2009 for x in rawdates]
    # python doesn't have an identical equivalent of "tabulate" function in R
    # it has a similar function "np.bincount", but it can only receive a minimal number of bins, not maximum,
    # so we have to use a loop to count to a maximum of 5 bins
    tabulate = np.bincount(rawdates, minlength=6)  # we use 6 instead of 5 (nyears) because the 1st value is "0"
    tablist = [None] * len(yearl)

    for j in range(1, len(tabulate)):
        tablist[j-1] = tabulate[j]

    yrgroup.iloc[i, :] = tablist

print("yrgroup: ")
print(yrgroup)
print("-"*50)

maxdy = [None] * yrgroup.shape[1]  # "shape[0]" is rownumber and "shape[1]" is colnumber
for i in range(0, len(maxdy)):
    maxdy[i] = max(yrgroup.iloc[:, i])

# Notice that as we are using only the last 100000 rows from the original "Y" matrix and the sites we are using where
# surveyed only in the first year, so the 4 last years in "maxdy" will have the value "0" (it doesn't happen using the
# full dataset):

print("maxdy: ")
print(maxdy)
print("-"*50)

# ----------------------------------------------------------------------------------------------------------------------
# Define output tables:

# Table 1 - number of detections by group, day and species:

# R original code (09):
"""
NDETSG <- array(data=NA,dim=c(ngroups,sum(maxdy),nsps))
dimnames(NDETSG)[[3]] <- colnames(Y)
"""
# Python version code (09):

# Instead of a 3D array, we will create a series of 2D Dataframes which will be the equivalent of a 3D array.
nrow = ngroups
ncol = sum(maxdy)
depth = nsps

df = pd.DataFrame(index=range(nrow), columns=range(ncol), dtype='uint8')  # a 2D dataframe
NDETSG = {i: df.copy() for i in range(depth)}  # here we create nsps copies of de 2D dataframe "df"
                                               # where each copy is a "slice" of the 3D array

# to access an element at, for example, the position (0, 0, 1) we can do that:

# NDETSG[0].iloc[0, 1]

# where "NDETSG[0]" gives the slice at index 0 and ".iloc[0, 1]" gives the row at index 0 and column at index 1.

# Table 2 - Dates of the surveying days by site over the years:

# R original code (10):
"""
DATSG <- t(matrix(data=NA,nrow=ngroups,ncol=sum(maxdy)))
DATSG <- data.frame(DATSG)
for(i in 1:ngroups) {class(DATSG[,i])="Date"}
"""
# Python version code (10):
DATSG = pd.DataFrame(index=range(sum(maxdy)), columns=range(ngroups), dtype='datetime64[ns]')

# Table 3 - Effort (minutes of recording) by day and group:

# R original code (11):
"""
EFFGmins <- matrix(data=NA,nrow=ngroups,ncol=sum(maxdy))
"""
# Python version code (11):
EFFGmins = pd.DataFrame(index=range(ngroups), columns=range(sum(maxdy)), dtype='uint8')

# Table 4 - Effort (number of surveyed sites) by day and group:

# R original code (12):
"""
EFFGsits <- matrix(data=NA,nrow=ngroups,ncol=sum(maxdy))
"""
# Python version code (12):
EFFGsits = pd.DataFrame(index=range(ngroups), columns=range(sum(maxdy)), dtype='uint8')

# ----------------------------------------------------------------------------------------------------------------------
# Fill output tables (13):

# R original code:
"""
for(i in 1:nsps) {
	cat("i =",i,"\n")
	## pegar a coluna correspondente a especie da vez
	colsp <- which(colnames(Y)==cspecies[i])
	for(j in 1:length(yearl)) {
	  cat("j =",j,"\n")
		# definir o ano da vez
		cy <- yearl[j] 
		# loop through groups
		for(k in 1:ngroups) {
			cat("k =",k,"\n")
		  #cs1 <- sitel[k]
		  #cs2 <- sitel[5]
			cg <- sitegroup[[k]] # definir o grupo da vez
			
			# definir datas correspondentes ao group e ano da vez
			#cds1 <- sort(unique(datel[which(SEGMTS$Spot==cs1 & format(datel,'%Y')==cy)]))
			#cds2 <- sort(unique(datel[which(SEGMTS$Spot==cs2 & format(datel,'%Y')==cy)]))			
			cdg <- sort(unique(datel[which(SEGMTS$Spot%in%cg & format(datel,'%Y')==cy)]))
			# jump to the next group if there are no days with data for the current year
			if(length(cdg)==0) next
			# calcular o fdc
			if(j==1) { fdc <- 1 } else { fdc<-sum(maxdy[1:(j-1)])+1 }
			# guardar as datas no DATSG
			if (i == 1) { ## se for o loop da primeira especie
			  DATSG[fdc:(fdc+length(cdg)-1),k] <- cdg
			}
			
			# guardar o esforco e o numero de deteccoes, respetivamente,
			# nas matrizes de esforco e no NDETS			
			for(l in 1:length(cdg)) {
				cd <- cdg[l] # dia da vez
				# guardar esforco em sitios no EFFGsits
				EFFGsits[k,(fdc+l-1)] <- length(unique(SEGMTS$Spot[which(SEGMTS$Spot%in%cg & datel==cd)]))
				# mesmo para EFFGmins
				EFFGmins[k,(fdc+l-1)] <- length(which(SEGMTS$Spot%in%cg & datel==cd))

				# guardar o numero de deteccoes em NDETSG
				#ndet <- sum(datel==cd & SEGMTS$Spot%in%cg & Y[,colsp]>=thr,na.rm=TRUE) # >= thr
				ndet <- sum(datel==cd & SEGMTS$Spot%in%cg & Y[,colsp]>thr,na.rm=TRUE)   # > thr
				#ndet <- sum(datel==cd & SEGMTS$Spot==cs & Y[,colsp]>=thr) # numero de deteccoes no cs e cd
				NDETSG[k, fdc+l-1, i] <- ndet

  		} # l dates
		} # k sites
	} # j years	
} # i species
"""

# Python version code (13):

# get column corresponding to current species
for i in range(0, nsps):  # range(0, nsps)

    # define current year
    for j in range(0, len(yearl)):  # range(0, len(yearl))
        cy = yearl[j]  # "cy" = "current year"

        # define current group
        for k in range(0, ngroups):  # range(0, ngroups)
            print("\nspecies: " + str(i) + "\nyear: " + str(j) + "\ngroup:" + str(k) + "\n--------------------")
            cg = sitegroup[k]  # "cg" = "current group"

            # get dates belonging to current group and year:
            condition2 = (SEGMTS["Spot"].isin(cg)) & (datel.dt.year == cy)
            indices2 = np.argwhere(condition2).flatten()
            cdg = pd.Series(sorted(np.unique(datel.iloc[indices2])))

            # skip to the next group if there are no surveyed days in "cy":
            if len(cdg) == 0:
                continue
            else:
                # Obtain "fdc" (first day column), which is the first column to be filled
                # (related to the first day from current year)
                if j == 0:
                    fdc = 0
                else:
                    fdc = sum(maxdy[0:j])

                # save the dates at "DATSG":
                if i == 0:  # if it is the first species loop
                    DATSG.iloc[fdc:(fdc + len(cdg)), k] = cdg

                # save effort and number of detections, respectively, at the effort tables and at NDETSG:
                for l in range(0, len(cdg)):  # range(0, len(cdg))
                    cd = cdg.iloc[l]  # "cd' = current day

                    # save effort in minutes of recording at "EFFGmins":
                    condition4 = (SEGMTS["Spot"].isin(cg)) & (datel == cd)
                    indices4 = np.argwhere(condition4).flatten()
                    EFFGmins.iloc[k, (fdc + l)] = len(indices4)

                    # save effort in number of sites at "EFFGsits":
                    condition3 = (SEGMTS["Spot"].isin(cg)) & (datel == cd)
                    indices3 = np.argwhere(condition3).flatten()
                    EFFGsits.iloc[k, (fdc + l)] = len(np.unique(SEGMTS["Spot"][indices3]))

                    # save number of detections at "NDETSG":
                    ndet = sum((datel == cd) & (SEGMTS["Spot"].isin(cg)) & (Y.iloc[:, i] > thr))
                    NDETSG[i].iloc[k, (fdc + l)] = ndet

# ----------------------------------------------------------------------------------------------------------------------
# Change column names (starting in 1 instead of 0) to be similar to an R data frame:
DATSG.columns = range(1, (len(DATSG.columns)+1))
EFFGmins.columns = range(1, (len(EFFGmins.columns)+1))
EFFGsits.columns = range(1, (len(EFFGsits.columns)+1))

for i in range(0, 1):  # range(0, nsps)
    NDETSG[i].columns = range(1, (len(NDETSG[i].columns)+1))

# ----------------------------------------------------------------------------------------------------------------------
# Export arrays as RData (will be saved in the same directory where this script is) using the function "write_data"
# from pyreadr:
write_rdata('DATSG_py.RData', DATSG, df_name="DATSG_py", datetimeformat="%Y-%m-%d")
write_rdata('EFFGmins_py.RData', EFFGmins, df_name="EFFGmins_py")
write_rdata('EFFGsits_py.RData', EFFGsits, df_name="EFFGsits_py")

# NDETSG slices (species) will be exported separately:
for i in range(0, nsps):  # range(0, nsps)
    write_rdata('NDETSG_py_' + str(i+1) + '.RData', NDETSG[i], df_name="NDETSG_py_" + str(i+1))

# ----------------------------------------------------------------------------------------------------------------------
# End running time:

end_time = time()

print("running time (seconds):")
print(end_time - start_time)
