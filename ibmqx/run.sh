#!/bin/bash 

#
# $1 INPUT $2 INPUT_LOCATION $OUTPUT_LOCATION
#
# Be sure to specifiy the input, input_location, and out_put_location if doing different experiments.

def_out_loc="./output/"
def_in_loc="./input/"
def_input="test.qc"
if [ "$1" == "" ]; then { 
echo "input file is unset...using default: $def_input";
} 
else 
{
def_input=$1;
echo "input file is:  $def_input";
} fi
echo $1

if [ "$2" == "" ]; then { 
echo "input location is unset...using default: $def_in_loc";
} 
else 
{
def_in_loc=$2;
echo "input location is: $def_in_loc";
} fi
echo $2

#if [ -z ${3+def_out_loc} ]; then echo "output loc var is unset"; else echo "var is set to '$3'"; fi
if [ "$3" == "" ]; then { 
echo "output location is unset...using default: $def_out_loc";
} 
else 
{
def_out_loc=$3;
echo "input location is: $def_out_loc";
} fi
echo $3


#First, write the config file
python3 prepare.py $def_input $def_in_loc
#Second, execute the main program
python3 main.py $def_input $def_out_loc
