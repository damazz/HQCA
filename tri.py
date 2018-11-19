
'''
tri.py 

Supplemental program for obtaining the triangulation for a projective run. Can be
used to simply check the ibm mechanics as well. 


'''
import subprocess
import pickle
import os, sys
import numpy as np
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from tools import Energy as energy
from tools.QuantumFramework import add_to_config_log
import datetime
from datetime import date
import sys
np.set_printoptions(precision=8)


# Setting the input file and molecular locations  

try:
    filename = sys.argv[1]
    tri = sys.argv[2]
    with open('./config.txt','w') as fp:
        fp.write('#\n')
        fp.write('# ./config.txt \n')
        fp.write('#\n')
        fp.write('# Generated from main.py, and tells main.py where to\n')
        fp.write('# look for parameters and the pyscf type mol file.\n')
        fp.write('#\n')
        fp.write('# Pointer for input file \n')
        fp.write('input_file= {} \n'.format(filename))
        fp.write('# Pointer for mol file \n')
        fp.write('mol_file= {} \n'.format('VOID'))
except IndexError:
    tri = '{}'.format('default')

today = date.timetuple(date.today())
today = '{:04}{:02}{:02}.'.format(today[0],today[1],today[2])
tri_name = './results/tri/{}{}.tri'.format(today,tri)
log_name = './results/tri/{}{}.txt'.format(today,tri)

# Now, read in the variables
print('----------')
print('--START---')
print('----------')
try:
    print('Computational parameters are taken from: {}'.format(pre.filename))
except:
    print('Taking parameters from config file. Must be specified elsewhere.')
print('Importing run parameters.')
print('----------')
print('Run on: {}'.format(datetime.datetime.now().isoformat()))
print('----------')
sys.stdout=open(log_name,'w')

import pre
sys.stdout = sys.__stdout__
print('----------')
if pre.occ_energy=='qc':
    print('Hello. Measuring the triangle. ')
elif pre.occ_energy=='classical':
    print('Hello. You have the wrong input file. Goodbye!')
    sys.exit()
print('Let\'s begin!')
print('----------')

if pre.qc_connect:
    print('Run is connected to the IBMQuantumExperience.')
    print('Checking for config file.')
else:
    print('Running locally. Checking for config file.')    
add_to_config_log(pre.qc_use_backend,pre.qc_connect)

print('----------')
print('Quantum algorithm: {}'.format(pre.qc_algorithm))
print('Quantum backend: {}'.format(pre.qc_use_backend))
print('----------')
# Setting mapping for system. Should be size specific. 

#
#
# Now, beginning optimization procedure. 
#
#

if pre.occ_energy=='qc':
    # Energy function is computed through the quantum computer 
    keys = {
        'wf_mapping':None,
        'ints_1e_no':None,
        'ints_2e_no':None,
        'E_ne': None,
        'algorithm':pre.qc_algorithm,
        'backend':pre.qc_use_backend,
        'order':pre.qc_qubit_order,
        'num_shots':pre.qc_num_shots,
        'split_runs':pre.qc_combine_run,
        'connect':pre.qc_connect,
        'method':pre.occ_method,
        'print_run':pre.print_extra,
        'energy':pre.occ_energy,
        'verbose':pre.qc_verbose,
        'wait_for_runs':pre.wait_for_runs,
        'store':energy.Storage()
        }

if pre.occ_method in ['stretch','diagnostic']:
    print('Measuring triangle for generating an  affine transformation.')
    print('Please make sure your circuit rotation is in 2D.')
    print('----------')
    try:
        triangle=energy.find_triangle(
                Ntri=pre.occ_method_Ntri,
                **keys)
    except Exception as e:
        print('Error in start. Goodbye!')
        traceback.print_exc()
    print('Succefully measured the triangle.',
            ' Proceeding with optimization.')
    print('----------')
    with open(tri_name, 'wb') as fp:
        pickle.dump(
                triangle,
                fp,
                pickle.HIGHEST_PROTOCOL
                )
else:
    pass


