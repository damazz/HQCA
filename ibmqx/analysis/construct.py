import numpy as np
import sys
import os
from pprint import pprint
# file to string together a bunch of files
import pickle
import json
sys.path.append('../gpc/')
from gpcf import gpc
from gpcf import rdm
# Construct.py 
#
# Part 1: construct file
# Part 2: check file for similarities
# Part 3: save to new file
# ---------------------------------------
#
# Part 1: File Construction
#

# Setting input location

try:
    sys.argv[1]
    default_in_loc = sys.argv[1]
except Exception:
    default_in_loc = './../results/'
stop = False
file_list = []

default_out_loc = './compiled/'

# Getting files names

print('Print \'stop\' or \'end\' to signal end of file input.')
print_files = input('Print available files? y/n ')
if (print_files=='y' or print_files=='Y'):
    holder_2 = 0
    for item in os.listdir(default_in_loc):
        print(item)
while stop==False:
    check = input('File to add: ')
    if (check=='stop' or check=='end'):
        stop=True
    else:
        file_list.append(default_in_loc+check)

# Check files names

ok=False
while not ok:
    holder_1 = 0
    for name in file_list:    
        try:
            test = open(name+'.dat')
            file_type='.dat'
            ok=True
        except Exception as e:
            print(e)
            try:
                test = open(name+'.iqd')
                file_type = '.iqd'
                print('File type is an ibm data type.')
            except:
                print('You should replace this filename - it can\'t be opened.')
                check = input('File to add: ')
                if (check=='stop' or check=='end'):
                    file_list.remove(name)  
                    break
                file_list[holder_1]=default_in_loc+check
                ok=False
                break

        holder_1+=1

#
# Start processing
#

# Functions

def check_dictionaries(one,two,keywords):
    not_okay = False
    for word in keywords:
        if one[word]==two[word]:
            pass
        else:
            not_okay=True
    return not_okay

def combine_dict(one,two):
    for key,val in two.items():
        try:
            one[key] = int(one[key]) + int(val)
        except:
            one[key] = int(val)
    return one

    



# ask if you would like to combine


if file_type=='.iqd':
    combine = input('Would you like to combine and discard redundant files? y/n ')
    # structure of data: long array with dictionary items
    # dicitonary has following keys:
    #   id, usedCredits, userId, backend, maxCredits, deleted, calibration, status, creationDate, qasms, shots
    # qasms has array objects, and then in each array, the following dictionary keys: 
    #   qasm, status, executionId, result
    # the array in qasms are the circuit executions, so the diag and off diag error circuits
    # 

    combine = input('Would you like to combine runs? y/n ')
    while cont==False:
        if combine=='y' or combine=='yes' or combine=='no' or combine=='n':
            cont=True
        else:
            combine = input('Try again: ')
    if combine=='y' or combine=='yes':
        max_N = input('Max number of runs? ')
        err   = input('Error circuit? y/n ')
        new_data = []
        i = 0
        count = 0
        for item in compiled_data:
            if i%max_N==0:
                new_data.append([])

                    
                 
                
            
            
    

elif file_type=='.dat':
    # 
    # Filtering through data: selecting which data is available and what can be filtered. 
    #
    combine = input('Would you like to combine and discard redundant files? y/n ')
    max_N = 0
    compiled_data = []
    for name in file_list:
        with open(name+'.dat','rb') as current:
            data = pickle.load(current)
            for run in data:
                compiled_data.append(run)
                max_N = max(run['run'],max_N)
    max_N+= 1
    
    cont=False
    while cont==False:
        if combine=='y' or combine=='yes' or combine=='no' or combine=='n':
            cont=True
        else:
            combine = input('Try again: ')
    if combine=='y' or combine=='yes':
    
        print('Looking for similarities')
        print('Number of runs = {}'.format(max_N))
        print('How many would you like to combine?')
        try:
            n_combine = int(input(':'))
        except:
            n_combine = 0
        print('How many would you like to discard?')
        try:
            n_discard = int(input(':'))
        except:
            n_discard = 0
        cont = False
        while cont==False:
            if not (n_combine+n_discard)==max_N:
                print('Error in number.')     
                print('How many would you like to combine?')
                n_combine = int(input(':'))
                print('How many would you like to discard?')
                n_discard = int(input(':'))
            else:
                cont=True
        
        print('Which ones would you like to combine and discard?')
        print(list(range(max_N)))
        hold_comb = []
        hold_disc = []
        cont=False
        while cont==False:  
            while len(hold_comb)<n_combine:
                hold_comb.append(int(input('Combine: ')))
            while len(hold_disc)<n_discard:
                hold_disc.append(int(input('Discard: ')))
            if (set(hold_comb)&set(hold_disc))==set([]) and (set(hold_comb).union(set(hold_disc)))==set(list(range(max_N))):
                cont=True
            else:
                hold_comb = []
                hold_disc = []
                print('Try again. Error in indices requested.')
        #
        # Now, begin the process of filtering and discarding. 
        #
        i=0
        compare = []
        for item_1 in compiled_data:
            j = 0
            for item_2 in compiled_data:
                if j>i:
                    pass
                else:
                    j += 1 
                    continue
                val = check_dictionaries(item_1,item_2,['backend','combined','parameters','shots','order'])
                if not val:
                    if ((item_1['run'] in hold_comb) and (item_2['run'] in hold_comb) and (item_1['run']!=item_2['run'])):
                        pass
                    else:
                        val=True
                compare.append([val,i,j])
                j+= 1
            i+=1 
        for pair in reversed(compare):
            print(pair)
            if not pair[0]:
                #print(compiled_data[pair[1]]['main-counts'])
                #print(compiled_data[pair[2]])
                if compiled_data[pair[2]]['run']=='used':
                    continue
                compiled_data[pair[1]]['main-counts'] = combine_dict(compiled_data[pair[1]]['main-counts'],compiled_data[pair[2]]['main-counts'])
                compiled_data[pair[1]]['err-counts'] = combine_dict(compiled_data[pair[1]]['err-counts'],compiled_data[pair[2]]['err-counts'])
                compiled_data[pair[1]]['shots'] += compiled_data[pair[2]]['shots']
                compiled_data[pair[1]]['combined']=True
                compiled_data[pair[1]]['run']='done'
                compiled_data[pair[2]]['run']='used'
                compiled_data[pair[1]]['construct']=True
        n = len(compiled_data)
        for item in reversed(compiled_data):
            n-=1 
            if not item['run']=='done':
                del compiled_data[n]
    i = 0 
    n_qb = input('Number of qubits?')
    recalc = input('Would you like to recalculate the unitary transformations?')
    for item in compiled_data:
        try:
            algorithm = item['algorithm']
        except:
            algorithm = 'ry6p'
        main = item['main-counts']['counts']
        print(main)
        err  = item['err-counts']['counts']
        ONrdm, ON, ONvec = rdm.construct_rdm(
            rdm.rdm(
                rdm.filt(main,trace=[3,4])
                ),
            rdm.rdm(
                rdm.filt(err,trace=[3,4])
                )
            )
        ON.sort()
        compiled_data[i]['exp-ON'] = ON.tolist()
        if recalc=='yes' or recalc=='y' or recalc=='True' or recalc=='Y':
            new_circuit = gpc.Psuedo_Quant_Algorithm(main,err,int(n_qb))
            parameters = item['parameters']
            qorder = item['order']
            new_circuit.get_unitary(algorithm,parameters,qorder)
            compiled_data[i]['ideal'] = new_circuit.unit_ON.tolist()
        i+= 1

#
#
# Part 3: Saving new file 
#

print('Here are the files in the save directory. \n')
for item in os.listdir(default_out_loc):
    print(item)
save_filename = input('\nSave file as?')
save_filename = './compiled/' + save_filename  
with open(save_filename,'wb') as save_file:
    pickle.dump(compiled_data,save_file,0)
