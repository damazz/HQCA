# main.py
# 
# Executes the main program. Loads settings from settings.py and then allows 
# for results. 

#
# 1. IMPORT MODULES AND LOAD FUNCTIONS
# 

from IBMQuantumExperience import IBMQuantumExperience
#import qfunc.rdm as rdmf
import settings as st
from qfunc import rand
from qfunc import quantum
import numpy as np
import traceback
import sys
from pprint import pprint
np.set_printoptions(suppress=True,precision=6)
from qiskit import QuantumProgram
import Qconfig
import time
from simul import run as sim
import json
import pickle

#
# Determine if the check function should be utilized, which requires user input
# for the run. 
#

try:
    st.pass_true
    if st.use_backend=='check':
        from qfunc.rand import check
    elif st.pass_true=='yes' or st.pass_true=='True' or st.pass_true=='y':
        from qfunc.rand import check_pass as check
    else:
        from qfunc.rand import check
except Exception:
    from qfunc.rand import check

#
# Test the quantum program by checking the QC or backend is available. 
#
if st.connect:
    test_program = QuantumProgram() 
    test_program.register
    test_program.set_api(Qconfig.APItoken,Qconfig.config['url'])
    #print('Here are the available programs: ')
    #pprint(test_program.available_backends())
    if st.use_backend=='check':
        st.use_backend = input('Please provide available backend: ')
        found = 0
        while not found:
            try:
                test_program.get_backend_status(st.use_backend)
                found = 1
            except:
                traceback.print_exc()
                st.use_backend = input('Please try again: ')
    #print('Here is the status of your backend:')
    #pprint(test_program.get_backend_status(st.use_backend))
    #pprint(test_program.get_backend_configuration(st.use_backend)) 

#
# Determine the number of qubits needed in your system - varies from simulator
# to experimental computers. 
#


try:
    n_qubits = test_program.get_backend_configuration(st.use_backend)['n_qubits'] 
except Exception: 
    n_qubits = 3
if n_qubits>15:
    n_qubits = 3
if st.connect:
    api = IBMQuantumExperience(Qconfig.APItoken)
    pprint(api.get_my_credits())
    check('Proceed?: ', 'Goodbye!')

file_name = sys.argv[1]
file_name = file_name.split('.')[0]

#
# 2. EXECUTE CIRCUITS
#

#
# Output 'data' object is a dictionary with relevant information, but
# primarily has the counts and other information for the runs. 
#


data = quantum.Run(st.run_type,st.algorithm,st.parameters,st.qubit_order,n_qubits,st.num_shots,Qconfig.APItoken,Qconfig.config['url'],name=file_name, backend=st.use_backend,tomography=st.run_tomography,combine=st.combine_run,ibm_connect=st.connect,verbose=st.verbose)

#
# 3. SAVE RESULTS
#

#
# Dumps the pickle dictionary to your output file location. 
#


if (st.save_file == 'yes' or st.save_file == 'True' or st.save_file=='y'):
    #print('Saving file to {}.'.format(sys.argv[2]))
    with open('{}{}.dat'.format(sys.argv[2],file_name),'wb') as fp:
        pickle.dump(data,fp,0)
