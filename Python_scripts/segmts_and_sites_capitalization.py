    # import dependencies
# to import libraries first enter the command "pip install <library name here>" in the terminal
# maybe you will have to install it anyway clicking on the digited word "pandas" below and selecting to install package
import csv
import pandas as pd # "as pd" just creates a nickname to the package
import re # module for dealing with regular expressions

# ----------------------------------------------------------------------------------------------------------------------
# SEGMTS capitalization correction
# load file
SEGMTS = pd.read_csv("../Data_input/Raw_data/date_time_diurnal.txt", sep=",")

# correcting letter capitalization with regex
SEGMTS["Spot"] = SEGMTS["Spot"].str.replace(r'(^..)C', r'\1c', regex=True)
SEGMTS["Spot"] = SEGMTS["Spot"].str.replace(r'(^..)M', r'\1m', regex=True)
SEGMTS["Spot"] = SEGMTS["Spot"].str.replace(r'(^.{7})X', r'\1x', regex=True)

# save corrected txt file
SEGMTS.to_csv("../Data_input/date_time_diurnal_capitalization_python.txt", index=None, sep=',', mode='w')

# ----------------------------------------------------------------------------------------------------------------------
# SITES capitalization correction
# load file
SITES = pd.read_csv("../Data_input/Raw_data/locations_diurnal.txt", sep="\t")

# correcting letter capitalization with regex
SITES["Spot"] = SITES["Spot"].str.replace(r'(^..)C', r'\1c', regex=True)
SITES["Spot"] = SITES["Spot"].str.replace(r'(^..)M', r'\1m', regex=True)
SITES["Spot"] = SITES["Spot"].str.replace(r'(^.{7})X', r'\1x', regex=True)

# save corrected txt file
SITES.to_csv("../Data_input/locations_diurnal_capitalization_python.txt", index=None, sep='\t', mode='w')
