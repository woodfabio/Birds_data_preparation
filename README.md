# Birds_data_preparation 🐦
This project is composed by scripts in R and Python languages for Amazon birds vocalization data prepartion in 2D and 3D arrays.

As an undergraduate researcher at Ferraz Population Biology Lab (UFRGS) (http://ferrazlab.org/) I helped to prepare vocalization recording data from 63 species of Amazon birds for posterior analysis of bayesian hierarquical modeling of population biology, obtaining estimatives like probabilities of presence, extinction, colonization and detection.

The raw dataset had informations like probability of presence of each species at each recording segment in CSV files and the posterior analysis required all this data organized in 2D and 3D arrays, separating the data, for example, by site, date and species and also sites clustering for analysis optimization.  Besides, some sites names were not completely standardized, with some letters capitalization not matching. To prepare the data we first developed R scripts to standardize the letter capitalization and then prepare this data in organized arrays. However, this R script takes a very long time (about 2 days and a half) to complete the data preparation, and as we need to run this script many times (for example, using differend threshold values) I had to work on the translation of this script to Python, which usually has a faster running than R.

### Result:
In fact, the Python version of this script run the complete original dataset in only 4 hours, an enormous difference in comparison with the 2 days and a half time spent by the R script.

### Future updates:
I have the intention to translate this script to the Rust language, which is even faster than Python due to being compiled directly into machine code, without an interpreter or virtual machine between the code and the hardware.

Attention to the fact that in this repository I am using only a small survey from the original dataset beacuse it is really huge (almost 700000 rows and 63 columns) only for demonstration purpose. This survey is composed by the last 100000 rows from the original dataset.

In this project we had to develop some scripts and create some files, listed by folder:

## Data_input
It has the original "raw" dataset files in the folder ***Raw_data*** and also the versions of this dataset with standardized letter capitalization:

1. **date_time_diurnal_capitalization_python.txt**: the output from *segmts_and_sites_capitalization.py* (at the *Python_scipts* folder), which receives *date_time_diurnal.txt* (at the *Data_input/Raw_data* folder) as input and returns this file with standardized letter capitalization. It is identical to *date_time_diurnal_capitalization_r.txt*, only generated through different languages.

2.  **date_time_diurnal_capitalization_r.txt**: the output from *segmts_and_sites_capitalization.R* (at the *R_scripts* folder), which receives *date_time_diurnal.txt* (at the *Data_input/Raw_data* folder) as input and returns this file with standardized letter capitalization. It is identical to *date_time_diurnal_capitalization_python.txt*, only generated through different languages.

3.  **groups.csv**: a CSV file generated by the script *data_preparation_r.R* (at the *R_scripts* folder) with the groups names of each surveyed site. This file is used by the script *data_preparation_python.py* (at the *Python_scripts* folder).

4. **locations_diurnal_capitalization_python.txt**: the output from *segmts_and_sites_capitalization.py* (at the *Python_scripts* folder), which receives *locations_diurnal.txt* (at the *Data_input/Raw_data* folder) as input and returns this file with standardized letter capitalization. It is identical to *locations_diurnal_capitalization_r.txt*, only generated through different languages.

5.  **locations_diurnal_capitalization_r.txt**: the output from *segmts_and_sites_capitalization.R* (at the *R_scripts* folder), which receives *locations_diurnal.txt* (at the *Data_input/Raw_data* folder) as input and returns this file with standardized letter capitalization. It is identical to *locations_diurnal_capitalization_python.txt*, only generated through different languages.

### Data_input/Raw_data
It has only the original "raw" dataset files:

1. **date_time_diurnal.txt**: the original CSV file with ID data from each recording segment. It is a small survey from the last 100000 rows from the original (much bigger) dataset. It has the following dimensions:
            - Rows: recording segments
            - Columns: ID data of each recording segment (site, date, recording number and segment number)
            
2.  **locations_dirunal.txt**: the original CSV file with ID data from each site. It is a small survey from the last 100000 rows from the original (much bigger) dataset. It has the following dimensions:
           - Rows: sites
           - Columns: ID data of each site (name, forest type (old growth = 0, secondary forest = 1), geographic coordinates (X and Y))

3. **Y.csv**: the original CSV file where each cell has the probability of presence of the vocalization from that species in that segment. It is a small survey from the last 100000 rows from the original (much bigger) dataset. It has the following dimensions:
           - Rows: recording segments
           - Columns: species

## Python_scripts
It has all the Python scripts and some Python output files:

1. **data_preparation_python.py**: the script that makes sites clustering and organizes the raw data in three 2-dimensional arrays and one 3-dimensional array (exported as separated 2-dimensional "slices" which will be joined by *python_output_receiver.R* at the *R_scripts* folder) for posterior analysis of bayesian hierarquical modeling. The code has anotations explaining the process in details. It is the python version of the R langauge script *data_preparation_r.R* (at the *R_scripts* folder).

2. **segmts_and_sites_capitalization.py**: the script that receives *date_time_diurnal.txt* and *locations_dirunal.txt* (both at the *Data_input* folder) as input and returns *date_time_diurnal_capitalization_python.txt* and *locations_capitalization_python.txt* (both at the *Data_input* folder) as output, with standardized letter capitalization. It is the Python version of the file *segmts_and_sites_capitalization.R* (at the folder *R_scripts*).

This folder also contains the output from *data_preparation_python.py* in the form of 66 *RData* files, which will be further prepared by the script *python_output_receiver.R* (at the folder *R_scripts*). The files description is in the *data_preparation_python.py* script comments. Those files are:

      1. DATSG.RData
      2. EFFGmins.RData
      3. EFFGsits.RData
      4. NDETSG_py.RData (x63)

## R_scripts
It has all the R scripts and its outputs:

1. **data_preparation_r.R**: the script that makes sites clustering and organizes the raw data in three 2-dimensional arrays and one 3-dimensional array for posterior analysis of bayesian hierarquical modeling. The code has anotations explaining the process in details. It is the R language version of the Python script *data_preparation_python.py* (at the *Python_scripts* folder).

2. **output_data_preparation_python.RData**: an RData file with the output from *data_preparation_python.py* (at the folder *Python_scripts*) already prepared by the script *python_output_receiver.R* (at this folder).

3. **output_data_preparation_r.RData**: an RData file with the output from *data_preparation_r.R* (at this folder).

4. **outputs_comparer.R**: the script which compares the outputs from *data_preparation_r.R* and *data_preparation_python.py* (at the folder *Python_scripts*) to check if they are identical.

5. **python_output_receiver.R**: the script which receives the output files from *data_preparation_python.py* (at the folder *Python_scripts*) which are the at the folder *Python_scripts*, and prepare them to further analysis in the R environment. It basically get some 2-dimensional arrays exported by the python script and join them as a 3-dimensional array and also turn all the "NaN" values into "NA" values.

6. **segmts_and_sites_capitalization.R**: the script that receives *date_time_diurnal.txt* and *locations_dirunal.txt* (both at the *Data_input* folder) as input and returns *date_time_diurnal_capitalization_r.txt* and *locations_capitalization_r.txt* (both at the *Data_input* folder) as output, with standardized letter capitalization. It is the R language version of the file *segmts_and_sites_capitalization.py* (at the folder *Python_scripts*).


