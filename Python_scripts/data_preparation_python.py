# Script de organizacao de dados de vocalizacao de aves da amazonia

# Funcao que recebe vector com nomes de especies e retorna tabela 2 ou 3-D com numero de deteccoes por sitio e
# dias agrupados por anos.
# Uma deteccao corresponde a um segmento com probabilidade de presenca de vocalizacao da especie acima do threshold.
# Tabela 2-D com datas correspondentes as amostras de cada sitio em cada ano.
# ----------------------------------------------------------------------------------------------------------------------

# import dependencies:
# to import libraries first enter the command "pip install <library name here>" in the terminal
# maybe after that you will have to install it anyway clicking on the package name and selecting to install package
import csv
import datetime

import pandas as pd  # "as pd" just creates a nickname to the package
import numpy as np
import datetime as dt
import re  # module for dealing with regular expressions
from scipy import cluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from statistics import mean

# ----------------------------------------------------------------------------------------------------------------------
# load data:
Y = pd.read_csv("../Data_input/shortY.csv", sep=";", dtype='float32', na_values=['NaN'])
SEGMTS = pd.read_csv("../Data_input/date_time_diurnal_capitalization_r.txt", sep=",")
SITES = pd.read_csv("../Data_input/locations_diurnal_capitalization_r.txt", sep="\t")
# SPSNAMES = pd.read_csv("C:/Users/fabio/OneDrive/NSpsOccDynManaus/Data Processing/Data/spnames.txt", sep="\t", header=None)

# lets see the dataFrames. The "print" function shows the head and tail of the dataFrame:

# R original code:
"""
print(Y)
print(SEGMTS)
print(SITES)
print(SPSNAMES)
"""

# as an example we can access the first element (index 0) of the column "Spot" from dataFrame "SEGMTS" doing this:
#print(SITES["Spot"].iloc[0])

# ----------------------------------------------------------------------------------------------------------------------
# Definir especie(s)

# R original code:
# cspecies = Y.columns.values.toList()[0] # "Attila_spadiceus"

# Python code:
cspecies = Y.columns.values.tolist()
nsps = len(cspecies)
thr = 0.9 # threshold de probabilidade de presença de vocalização no corte

# ----------------------------------------------------------------------------------------------------------------------
# Redefinir lista de sitios incluindo so os que aparecem no SEGMTS

# R original code:
"""
#shortSITES = SITES[which(SITES$Spot %in% SEGMTS$Spot),]
"""

# Python code:
indices = SITES["Spot"].isin(SEGMTS["Spot"])
shortSITES = SITES.loc[indices, :]

# ----------------------------------------------------------------------------------------------------------------------
# Definir distancia minima em metros necessaria para que dois pontos sejam considerados do mesmo cluster
# isto nao impede que alguns pontos dentro do mesmo cluster tenham uma distancia maior que a minima entre si.
dmin = 110

# ----------------------------------------------------------------------------------------------------------------------
# obter clusters pelo método "single" que agrupa com base na distancia minima entre pontos de cada cluster e
# gravar num objeto de dendrograma

# R original code:
"""
#dendrog <- hclust(dist(shortSITES[,3:4]), method="single")
"""

# Python code:
distance_matrix = pdist(shortSITES.iloc[:, 2:4]) # notice that in python the counting starts AFTER the first index,
                                                 # in this case, after the index "2", in other words, at the index "3"
dendrogram_data = linkage(distance_matrix, method='single')

# plotar

# R original code:
"""
#plot(dendrog,hang=-1)
"""

# Python code:
"""
plt.figure(figsize=(10, 6))
dendrogram(dendrogram_data)
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.title('Dendrogram')
plt.axhline(y=dmin)
plt.show()
"""

# ----------------------------------------------------------------------------------------------------------------------
# formar grupos de sitios com base no dendrograma

# R original code:
"""
groups <- cutree(dendrog,h=dmin)
"""

# Python code
groups = fcluster(dendrogram_data, t=dmin, criterion='distance')

# podemos ver que foram gerados grupos, porem com numeracao diferente do objeto equivalente no R
print("groups gerado no python: ")
print(groups)
print("-"*50)

# vamos verificar se os tamanhos dos grupos sao iguais aos do R:

# no R verificamos a quantidade de sitios em cada grupo assim:
"""
lixo <- rep(NA, length(unique(groups)))
for (i in 1:length(unique(groups))) {
  lixo[i] <- sum(groups == unique(groups)[i])
}
"""

#  o que nos retornou este vetor:
"""
> lixo
  [1]  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
 [36]  1  2  1  1  1  1  1  1  1  1  1  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
 [71]  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
[106]  1  1  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
[141]  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1
[176]  1  1  1  1  1  2  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
[211]  1  1  1 23
"""

# filtrando apenas os grupos com mais de um sitio:
"""
> lixo[lixo > 1]
[1]  2  2  2  2  2  2 23
"""

# vamos ver o mesmo no python:
ug = sorted(np.unique(groups))   # vetor com numeros dos grupos em "groups"
groupsum = [None] * len(ug)      # vetor vazio

for i in range(0, len(groupsum)):
    groupsum[i] = sum(groups == ug[i])  # quantidade de sítios no grupo da vez

print("ug: ")
print(ug)
print("-"*50)

print("groupsum: ")
print(groupsum)
print("-"*50)

# novamente, vamos filtrar apenas os grupos com mais de um sitio:
print("grupos com mais de um sitio: ")
for i in range(0, len(groupsum)):
    if groupsum[i] > 1:
        print(groupsum[i])
print("-"*50)

# representar os grupos com cores

# R original code:
#plot(shortSITES[,3:4],col=groups)

# Python code:
"""
plt.scatter(shortSITES["X"], shortSITES["Y"], c=groups)  # probably not working
plt.xlabel('Column 3')
plt.ylabel('Column 4')
plt.title('Scatter Plot with Colored Points')
plt.show()
"""

# ----------------------------------------------------------------------------------------------------------------------
# Como os resultados nao ficaram exatamente iguais, vamos importar o arquivo "groups.csv", que é o objeto "groups"
# que foi gerado no script em R e tambem gerar novamente os objetos "ug" e "groupsum":

groups = pd.read_csv("C:/Users/fabio/OneDrive/NSpsOccDynManaus/Data Processing/Data/groups.csv")
groups = np.array(groups["x"])

ug = sorted(np.unique(groups))   # vetor com numeros dos grupos em "groups"

groupsum = [None] * len(ug)          # vetor vazio
for i in range(0, len(groupsum)):
    groupsum[i] = sum(groups == ug[i])  # quantidade de sítios no grupo da vez

print("groups gerado no R: ")
print(groups)
print("-"*50)

print("novo ug:")
print(ug)
print("-"*50)

print("groupsum:")
print(groupsum)
print("-"*50)
print("max groupsum: ")
print(max(groupsum))
print("-"*50)

# ----------------------------------------------------------------------------------------------------------------------
# Criar matriz de coordenadas de centroide de grupo com tantas linhas quantos grupos

# encontrar maior grupo em "groups"
maxg = 0  # maxg eh o maior grupo
for i in range(0, len(ug)):
    if sum(groups == ug[i]) > sum(groups == maxg):
        maxg = ug[i]

print("maxg: ")
print(maxg)
print("-"*50)

# R original code:
"""
CORCEN <- matrix(NA,nrow=max(groups),ncol=2)
 for(i in 1:max(groups)) {
   CORCEN[i,1] <- mean(shortSITES[which(groups==i),3])
   CORCEN[i,2] <- mean(shortSITES[which(groups==i),4])
}
"""

# Python code:
CORCEN = pd.DataFrame(np.nan, index=range(ug[(len(ug)-1)]), columns=["X", "Y"], dtype=float)

for i in range(0, max(ug)):
    condition = (groups == ug[i])               # Define the condition for which to find the indices
    indices = np.argwhere(condition).flatten()  # get the indices where the condition is True (we use "flatten()" to
                                                # convert the resulting 2D array to a 1D array).
    CORCEN.iloc[i, 0] = mean(shortSITES.iloc[indices, 2])
    CORCEN.iloc[i, 1] = mean(shortSITES.iloc[indices, 3])

print("CORCEN: ")
print(CORCEN)
print("-"*50)

# ----------------------------------------------------------------------------------------------------------------------
# Achar dimensoes do output

# R original code:
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

# Python code:
sitel = sorted(np.unique(SEGMTS["Spot"]))  # list of site names
nsites = len(np.unique(sitel))             # number of sites
ngroups = len(np.unique(groups))           # number of groups of sites
print("ngroups: " + str(ngroups))
print("-"*50)
sitegroup = pd.DataFrame(index=range(ngroups), columns=range(max(groupsum)))  # sites (cols) in each group (rows)
for i in range(0, len(sitegroup)):  # from 0 to nrow of sitegroup
    condition = (groups == (i+1))
    indices = np.argwhere(condition).flatten()
    for j in range(0, len(indices-1)):
        sitegroup.iloc[i, j] = shortSITES.iloc[indices[j], 0]

groupl = [None] * ngroups             # list of group names
for i in range(0, len(sitegroup)):    # from 0 to nrow of sitegroup
    groupl[i] = sitegroup.iloc[i, 0]  # name of the first site in the group

yearl = sorted(np.unique(np.floor(SEGMTS["Date"]/10000).astype(int)))
#datel = SEGMTS["Date"].apply(lambda x: dt.datetime.strptime(str(x), "%Y%m%d"))
#datel = dt.datetime.strptime(SEGMTS["Date"], "%Y%m%d")
datel = pd.to_datetime(SEGMTS["Date"], format="%Y%m%d")

print("yearl:")
print(yearl)
# ----------------------------------------------------------------------------------------------------------------------
# achar o numero maximo de dias de amostragem em cada ano, com base em grupos
# Definir matriz de grupos por anos

# R original code:
"""
yrgroup <- matrix(rep(NA,ngroups*length(yearl)),ncol=length(yearl))
for(i in 1:ngroups) {
  yrgroup[i,] <- tabulate(floor(sort(unique(SEGMTS$Date[which(SEGMTS$Spot %in% sitegroup[[i]])]))/10000)-2009, nbins=5)
}
maxdy <- apply(yrgroup,2,max)
"""

# Python code:
yrgroup = pd.DataFrame(index=range(ngroups), columns=range(len(yearl)))
for i in range(0, ngroups):
    condition = SEGMTS["Spot"].isin(sitegroup.iloc[i, :])
    indices = np.argwhere(condition).flatten()
    rawdates = sorted(np.floor(np.unique(SEGMTS["Date"].iloc[indices]/10000)).astype(int)-2009)
    # python doesnt have an identical equivalent of "tabulate" function in R
    # it has a similar function "np.bincount", but it can only receive a minimal number of bins, not maximum
    # so we have to use a loop to count to a maximum of 5 bins
    tabulate = np.bincount(rawdates, minlength=6)  # we use 6 instead of 5 (nyears) because the 1st value is "0"
    tablist = [None] * len(yearl)
    for j in range(1, len(tabulate)):
        tablist[j-1] = tabulate[j]

    yrgroup.iloc[i, :] = tablist

maxdy = [None] * yrgroup.shape[1]  # index 0 is rownumber and index 1 is colnumber
for i in range(0, len(maxdy)):
    maxdy[i] = max(yrgroup.iloc[:, i])

# ----------------------------------------------------------------------------------------------------------------------
# Definir tabelas do output

# numero de deteccoes por grupo e dia

# R original code:
"""
NDETSG <- array(data=NA,dim=c(ngroups,sum(maxdy),nsps))
dimnames(NDETSG)[[3]] <- colnames(Y)
"""

# Python code:
nrow = ngroups
ncol = sum(maxdy)
depth = nsps
df = pd.DataFrame(index=range(nrow), columns=range(ncol), dtype='uint8')  # a 2D dataframe
NDETSG = {i: df.copy() for i in range(depth)}  # here we create nsps copies of de 2D dataframe "df"
                                               # where each copy is a "slice" of the 3D array
# to access an element at, for example, the position (0, 0, 1):
#   NDETSG[0].iloc[0, 1]
# where "NDETSG[0]" gives the slice at index 0 and ".iloc[0, 1]" gives the row at index 0 and column at index 1

# datas dos dias de amostragem por sítio ao longo dos anos

# R original code:
"""
DATSG <- t(matrix(data=NA,nrow=ngroups,ncol=sum(maxdy)))
DATSG <- data.frame(DATSG)
for(i in 1:ngroups) {class(DATSG[,i])="Date"}
"""

# Python code:
DATSG = pd.DataFrame(index=range(sum(maxdy)), columns=range(ngroups), dtype='datetime64[ns]')

# esforço por dia por grupo

# R original code:
"""
EFFGmins <- matrix(data=NA,nrow=ngroups,ncol=sum(maxdy))
EFFGsits <- matrix(data=NA,nrow=ngroups,ncol=sum(maxdy))
"""

# Python code:
EFFGmins = pd.DataFrame(index=range(ngroups), columns=range(sum(maxdy)), dtype='uint8')
EFFGsits = pd.DataFrame(index=range(ngroups), columns=range(sum(maxdy)), dtype='uint8')

# ----------------------------------------------------------------------------------------------------------------------
# Preencher tabelas do output

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

# Python code:
#chunksize = 1000
for i in range(0, 1):  # range(0, nsps)
    if i == 1:
        break
    print("species: " + str(i))
    # pegar a coluna correspondente a especie da vez
    #condition1 = Y.columns.values == cspecies[i]
    #colsp = np.argwhere(condition1).flatten()

    for j in range(0, 1):  # range(0, len(yearl))
        if j == 1:
            break
        # definir o ano da vez
        cy = yearl[j]

        # loop through groups
        for k in range(0, ngroups):  # range(0, ngroups)
            print("\nspecies: " + str(i) + "\nyear: " + str(j) + "\ngroup:" + str(k) + "\n--------------------")
            cg = sitegroup.iloc[k, :]  # definir o grupo da vez
            # definir datas correspondentes ao group e ano da vez
            condition2 = (SEGMTS["Spot"].isin(cg) & (datel.dt.year == cy))
            indices2 = np.argwhere(condition2).flatten()
            cdg = pd.Series(sorted(np.unique(datel.iloc[indices2])))

            # jump to the next group if there are no days with data for the current year
            if len(cdg) == 0:
                continue
            else:
                # calcular o fdc
                if j == 0:
                    fdc = 0
                else:
                    fdc = sum(maxdy[1:j]) + 1

                # guardar as datas no DATSG
                if i == 1:  # se for o loop da primeira especie
                    DATSG.iloc[fdc:(fdc + len(cdg)), k] = cdg

                # guardar o esforco e o numero de deteccoes, respetivamente, nas matrizes de esforco e no NDETSG
                for l in range(0, len(cdg)):  # range(0, len(cdg))
                    cd = cdg.iloc[l]  # dia da vez
                    # guardar esforco em sitios no EFFGsits
                    condition3 = (SEGMTS["Spot"].isin(cg)) & (datel == cd)
                    indices3 = np.argwhere(condition3).flatten()
                    EFFGsits.iloc[k, (fdc + l)] = len(np.unique(SEGMTS["Spot"][indices3]))

                    # mesmo para EFFGmins
                    condition4 = (SEGMTS["Spot"].isin(cg)) & (datel == cd)
                    indices4 = np.argwhere(condition4).flatten()
                    EFFGmins.iloc[k, (fdc + l)] = len(indices4)

                    # guardar o numero de deteccoes em NDETSG
                    #newY = "C:/Users/fabio/OneDrive/NSpsOccDynManaus/Data Processing/Data/Y.csv"
                    #for chunk in pd.read_csv(newY, sep=";", dtype='float32', na_values=['NaN'], chunksize=100000):
                    ndet = sum((datel == cd) & (SEGMTS["Spot"].isin(cg)) & (Y.iloc[:, i] > thr))
                    NDETSG[i].iloc[k, (fdc + l)] = ndet

print(NDETSG[0].iloc[:, :])

a = 1

















