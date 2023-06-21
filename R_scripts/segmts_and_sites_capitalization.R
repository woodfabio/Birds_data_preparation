  
## Script para correcao/uniformizacao da capitalizacao de letras nos nomes dos
## sitios nos arquivos .csv "date_time_diurnal.txt" e "locations_diurnal.txt",
## que se tornarao, respectivamente, os objetos"SEGMTS" e "SITES" no script
## "Formatadados_r.R"

# ------------------------------------------------------------------------------
## Carregar arquivos:

SEGMTS <- read.csv("../Data_input/date_time_diurnal.txt", head=TRUE, sep=",")
SITES <- read.csv("../Data_input/locations_diurnal.txt", head=TRUE, sep="\t", check.names = FALSE)

# ------------------------------------------------------------------------------
## Corrigir capitalizaçao:

SEGMTS$Spot <- sub("(?<=^.{2})C", "c", SEGMTS$Spot, perl = TRUE)
SEGMTS$Spot <- sub("(?<=^.{2})M", "m", SEGMTS$Spot, perl = TRUE)
SEGMTS$Spot <- sub("(?<=^.{7})X", "x", SEGMTS$Spot, perl = TRUE)

SITES$Spot <- sub("(?<=^.{2})C", "c", SITES$Spot, perl = TRUE)
SITES$Spot <- sub("(?<=^.{2})M", "m", SITES$Spot, perl = TRUE)
SITES$Spot <- sub("(?<=^.{7})X", "x", SITES$Spot, perl = TRUE)

# ------------------------------------------------------------------------------
## Exportar arquivos corrigidos:

write.table(SEGMTS, "../Data_input/date_time_diurnal_capitalization_r.txt", sep = ",", row.names=FALSE, quote=FALSE)
write.table(SITES, "../Data_input/locations_diurnal_capitalization_r.txt", sep = "\t", row.names=FALSE, quote=FALSE)
