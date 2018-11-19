import sys
sys.path.insert(0,'./')
from tools.energy import energy_eval_qc
import numpy as np


'''
Sub-program to find an energy functional and scan over the range of parameters.
If the data set is good, we can plot over it. 
'''
import pre 

par1 = np.linspace(0,45,10)
par2 = np.linspace(0,45,10)
keys = {
    wf_mapping = {
        0:0, 1:1, 3:2,
        2:3, 4:4, 5:5},
    E_ne = 


'wf_mapping':mapping,
'ints_1e_no':ints_1e,
'ints_2e_no':ints_2e,
'E_ne': E_ne,
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
'store':Store
