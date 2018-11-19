# settings.py
#
# Reads the configuration file, which has been set by prepare.py, and then loads
# the parameters to be called by the main program. 
#
#
import sys
import numpy as np
import traceback
try:
    try:
        filename = './ibmqx/config.txt'
        test = open(filename)
    except FileNotFoundError:
        filename = './config.txt'
    with open(filename) as fp:
        for line in fp:
            if line[0]=='#':
                continue
            elif line[0]=='\n':
                continue
            line = line.split()
            if line[0]=='backend=':
                use_backend = line[1]
                continue
            if line[0]=='run_type=':
                run_type = line[1]
                continue  
            if line[0]=='algorithm=':
                algorithm = line[1]
                continue
            if line[0]=='num_shots=':
                num_shots = int(line[1])
                continue 
            if line[0]=='order=':
                qubit_order = line[1]
                continue
            if line[0]=='save_file=':
                save_file = line[1]
                continue
            if line[0]=='pass=':
                pass_true = line[1]
                continue
            if line[0]=='date_time=':
                date_time = line[1]
                continue
            if line[0]=='order_shift=':
                order_shift = line[1]
                continue
            if line[0]=='simulator=':
                use_simulator = line[1]
                continue
            if line[0]=='off_diag=':
                calc_off_diag = line[1]
                continue
            if line[0]=='combine=':
                combine_run = line[1]
                continue
            if line[0]=='connect=':
                connect = line[1]
                continue
            if line[0]=='tomography=':
                run_tomography = line[1]
                continue
            if line[0]=='verbose=':
                verbose = line[1]
                continue

except:
    traceback.print_exc() 
# now, establish paramteres based on the run

try:
    try:
        filename = './ibmqx/config.txt'
        test = open(filename)
    except FileNotFoundError:
        filename = './config.txt'
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
# finally, test the relevant variables

try:
    use_backend
except:
    use_backend = 'local_qasm_simulator'
try:
    run_type
except:
    run_type = 'single'
try:
    algorithm
except:
    algorithm = 'alpha'
try:
    num_shots
except:
    num_shots = 1024
try:
    qubit_order
except:
    qubit_order = 'default'
try:
    save_file
except:
    save_file = 'yes'
try:
    pass_true
except:
    pass_true = 'yes'
try:
    date_time
except:
    date_time = 'no'
try:
    order_shift
except:
    order_shift = 'no'
try:
    calc_off_diag
except:
    calc_off_diag = 'yes'
try:
    #print('From setting, st.connect: {}'.format(connect))
    if connect=='no' or connect=='False':
        connect=False
except Exception as e:
    #print(e)
    connect = True
try:
    combine_run
    if combine_run in ['yes','y','Y','True','true','Yes']:
        combine_run = True
    else:
        combine_run = False
except:
    combine_run = True
try:
    run_tomography
    if (run_tomography=='y' or run_tomography=='yes' or run_tomography=='True'):
        run_tomography = True
    elif run_tomography=='both':
        pass
    else:
        run_tomography = False
except:
    run_tomography = False
try:
    parameters[0]
except Exception as e:
    print(e)
    parameters = [[0,0,0]]
try: 
    verbose
    if verbose in ['yes','y','True','Yes','true']:
        verbose=True
    else:
        verbose=False
except:
    verbose=False
