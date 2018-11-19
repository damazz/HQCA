# pre.py
#
# Reads the configuration file, which has been set by the main program, and then
# read in all of the parameters. Then, if any parameters have not been assigned,
# critical ones which must be assigned as set to default values. 
# 
#
import sys
import numpy as np
import traceback
try:
    with open('./config.txt','r') as fp:
        for line in fp:
            if line[0]=='#':
                continue
            elif line[0]=='\n':
                continue
            line = line.split()
            if line[0]=='input_file=':
                filename = line[1]
            if line[0]=='mol_file=': 
                mol_loc = line[1]

    with open(filename) as fp:
        for line in fp:
            line = line.replace('\n',' ')
            if line[0]=='#':
                print(line[:-1])
                continue
            elif line[0]=='\n':
                continue
            print(line[:-1])
            line = line.split(' ')
            # IBM inputs
            if line[0]=='qc_backend=':
                qc_use_backend = line[1]
                continue
            if line[0]=='qc_run_type=':
                qc_run_type = line[1]
                continue  
            if line[0]=='qc_algorithm=':
                qc_algorithm = line[1]
                continue
            if line[0]=='qc_num_shots=':
                qc_num_shots = int(line[1])
                continue 
            if line[0]=='qc_order=':
                qc_qubit_order = line[1]
                continue
            if line[0]=='qc_combine=':
                qc_combine_run = line[1]
                continue
            if line[0]=='qc_tomography=':
                qc_tomography = line[1]
                continue
            if line[0]=='qc_verbose=':
                qc_verbose = line[1]
                continue
            if line[0]=='qc_connect=':
                qc_connect= line[1]
                continue

            # General inputs

            if line[0]=='max_iter=':
                max_iter= int(line[1])
                continue
            if line[0]=='restart=':
                restart_run = line[1]
                continue
            if line[0]=='wait=':
                wait_for_runs = line[1]
                continue
            if line[0]=='print=':
                print_extra= line[1]
                continue
            if line[0]=='chem_orb=':
                chem_orbitals= line[1]
                continue
            if line[0]=='mapping=':
                mapping=line[1]
                continue
            if line[0]=='opt_crit=':
                opt_crit=float(line[1])
                continue

            # Occupation number inputs
            if line[0]=='occ_seed=':
                occ_seed_run = line[1]
                continue
            if line[0]=='occ_energy=':
                occ_energy= line[1]
                continue
            if line[0]=='occ_opt_crit=':
                occ_opt_conv_criteria= line[1]
                continue
            if line[0]=='occ_increase=':
                occ_increase_runs= line[1]
                continue
            if line[0]=='occ_optimizer=':
                occ_opt_method= line[1]
                continue
            if line[0]=='occ_nm_simplex=':
                if (not line[1]=='default'):
                    occ_nm_simplex= float(line[1])
                else:
                    occ_nm_simplex='default'
            if line[0]=='occ_method=':
                occ_method= line[1]
                continue
            if line[0]=='occ_max_iter=':
                occ_max_iter= int(line[1])
                continue
            if line[0]=='occ_opt_thresh=':
                occ_opt_thresh = line[1]
                continue
            if line[0]=='occ_gd_gradient=':
                occ_gd_gradient=line[1]
                continue
            if line[0]=='occ_gd_grad_distance=':
                occ_gd_grad_dist=line[1]
                continue
            if line[0]=='occ_method_Ntri=':
                occ_method_Ntri = line[1]
                continue
            if line[0]=='occ_load_triangle=':
                occ_load_triangle = line[1]
                continue

            # Orbital optimizer inputs
            if line[0]=='orb_max_iter=':
                orb_max_iter= int(line[1])
                continue
            if line[0]=='orb_opt_crit=':
                orb_opt_conv_criteria= line[1]
                continue
            if line[0]=='orb_optimizer=':
                orb_opt_method= line[1]
                continue
            if line[0]=='orb_nm_simplex=':
                if (not line[1]=='default'):
                    orb_nm_simplex= float(line[1])
                else:
                    orb_nm_simplex='default'
            if line[0]=='orb_method=':
                orb_method= line[1]
                continue
            if line[0]=='orb_gd_gradient=':
                orb_gd_gradient=line[1]
                continue
            if line[0]=='orb_gd_grad_dist=':
                orb_gd_grad_dist=line[1]
                continue
            if line[0]=='orb_opt_thresh=':
                orb_opt_thresh = line[1]
                continue
            if line[0]=='orb_print=':
                orb_print = line[1]
                continue
            if line[0]=='orb_opt_region=':
                orb_opt_region = line[1]
                continue
            if line[0]=='orb_seed=':
                orb_seed = line[1]
                continue

except:
    traceback.print_exc() 

# Here, we set the basic parameters of the run. Could use some more formatting
# to be honest, but it is okay for the moment.
try:
    if occ_seed_run=='user':
        parameters = []
        ind = 0 
        with open(filename) as fp:
            for line in fp:
                if line[0]=='$':
                    pass
                else:
                    continue
                parameters.append([])
                line = line.split()
                for obj in line:
                    if not (obj=='\n' or obj=='$'):
                        parameters[ind].append(float(obj))
                ind += 1
except:
    traceback.print_exc()


# Finally, test the relevant variables and assign default or appropriate values
# if necessary. 
try:
    print_extra
    if print_extra in ['yes','y','true','True']:
        print_extra=True
    else:
        print_extra=False
except:
    print_extra=False
try:
    if wait_for_runs in ['yes','y','true','True']:
        wait_for_runs = True
    else:
        wait_for_runs=False
except:
    wait_for_runs=False
try:
    if restart_run in ['yes','y','true','True']:
        restart_run = True
    else:
        restart_run=False
except:
    restart_run=False
try:
    occ_opt_method
except:
    occ_opt_method='NM'
try:
    orb_opt_method
except:
    orb_opt_method='NM'
try:
    qc_use_backend
except:
    qc_use_backend = 'local_qasm_simulator'
try:
    qc_algorithm
except:
    qc_algorithm = 'ry2p'
try:
    qc_num_shots
except:
    qc_num_shots = 1024
try:
    qc_qubit_order
except:
    qc_qubit_order = 'default'
try:
    qc_combine_run
    if (qc_combine_run in ['yes','Y','y','True']):
        qc_combine_run = True
    else:
        qc_combine_run = False
except:
    qc_combine_run = True
try:
    qc_tomography
except:
    qc_tomography = False
try:
    nre
except:
    nre= 0 
try:
    parameters[0]
except:
    parameters = [[0,0,0]]
try:
    occ_opt_conv_criteria
except:
    occ_opt_conv_criteria='default'
try:
    orb_opt_conv_criteria
except:
    orb_opt_conv_criteria='default'
try:
    if occ_increase_runs in ['y','yes','True']:
        occ_increase_runs = True
    else:
        occ_increase_runs = False
except:
    occ_increase_runs= False
try:
    opt_crit
except:
    opt_crit = 'default'
try:
    occ_method
except:
    occ_method='project'
try:
    orb_method
except:
    orb_method='givens'
try:
    chem_orbitals
except:
    chem_orbitals='FCI'
try:
    occ_energy
except:
    occ_energy='qc'
try:
    type(max_iter)==type(10)
except:
    max_iter=50
try:
    type(orb_max_iter)==type(10)
except:
    orb_max_iter=5000
try:
    type(occ_max_iter)==type(10)
except:
    occ_max_iter=50
try:
    occ_gd_gradient
except:
    occ_gd_gradient='numerical'
try:
    occ_gd_grad_dist
except:
    occ_gd_grad_dist='default'
try:
    mapping
except:
    mapping='zeta'
try:
    if qc_verbose in ['yes','Y','y','True']:
        qc_verbose=True
    else:
        qc_verbose=False
except:
    qc_verbose=False
try:
    occ_nm_simplex
except:
    occ_nm_simplex='default'
try:
    orb_nm_simplex
except:
    orb_nm_simplex='default'

try:
    orb_gd_gradient
except:
    orb_gd_gradient='numerical'
try:
    orb_gd_grad_dist
except:
    orb_gd_grad_dist='default'
try:
    orb_opt_thresh
except:
    orb_opt_thresh='default'
try:
    orb_opt_region
except:
    orb_opt_region='active_space'
try:
    occ_method_Ntri=int(occ_method_Ntri)
except:
    occ_method_Ntri='default'
try:
    occ_opt_thresh
except:
    occ_opt_thresh='default'
try:
    qc_connect
    if qc_connect in ['y','Y','True','yes']:
        qc_connect=True
    else:
        qc_connect=False
except:
    qc_connect=False
try:
    orb_seed
    if orb_seed in ['y','Y','True','yes']:
        orb_seed=True
    else:
        orb_seed=False
except:
    orb_seed=False
try:
    if occ_load_triangle in ['y','yes','True','Y']:
        occ_load_triangle = True
    else: 
        occ_load_triangle = False
except:
    occ_load_triangle = False
try:
    if orb_print in ['y','yes','True','Y']:
        orb_print=True
    else: 
        orb_print=False
except:
    orb_print=False
