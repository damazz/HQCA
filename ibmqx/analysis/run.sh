#/bin/bash 
# $1 INPUT $2 INPUT_LOCATION $OUTPUT_LOCATION

module load anaconda/4.0.0
source activate scott
python3 analyze.py $1
source deactivate
