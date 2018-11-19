import numpy as np
import sys
import os
from pprint import pprint
sys.path.append('../main/')
from gpcf import rdm 
from simul import run
# file to string together a bunch of files
from 
# part 1: construct file
# part 2: check file for similatrities
# part 3: save to new file

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
print('Do you want to look for .txt/.err (0) or .dat/.dict (1) files? 0/1 ')
method = input('Type: ')
cont = False
while cont==False:
    if method=='0' or method=='1':
        cont = True
    else:
        print('Wrong input. Please try again.')
        method = input('Type: ')

if method=='0':
    filetype = ['.txt','.err']
elif method=='1':
    filetype = ['.dat','.dict']
    

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
            test = open(name+filetype[0])
            ok=True
        except Exception as e:
            print(e)
            print('You should replace this filename - it can\'t be opened.')
            check = input('File to add: ')
            if (check=='stop' or check=='end'):
                file_list.remove(name)  
                break
            file_list[holder_1]=default_in_loc+check
            ok=False
            break

        holder_1+=1

if method=='0':
    # check for sim files
    proc_sim=False
    use_sim_files = input('Do you want .sim or .chk files? n/.sim/.chk ')
    while proc_sim==False:
        proc_sim=True
        if use_sim_files=='n' or use_sim_files=='.chk' or use_sim_files=='.sim':
            pass
        else:
            use_sim_files=input('Invalid option. Please try again.')
            proc_sim=False
    # Reading files 
    
    main_data = []
    row = 0
    count_sim = 0
    for name in file_list:
    
        with open(name+'.txt','r') as current:
            file_start = row
            for line in current:
                #print(line)
                if (not line.strip)==False:
                    line.split()
                    main_data.append([])  
                    main_data[file_start].append(line[:-1]) #omits the '\n' eol element
                    file_start+=1 
    
        #pprint(main_data)
        with open(name+'.err','r') as current:
            file_start = row
            write_here = True
            increase_row = False
            for line in current:
                #print(file_start,line,not line.strip())    
                if not line.strip():
                    continue
                if line.strip()=='###### run details ######':
                    # reached a stopping point
                    print('run details')
                    write_here = False 
                    continue
                if line.strip()=='###### angles and order used #####':
                    print('angles used')
                    write_here = True
                    file_start = row
                    increase_row = True
                    continue
                if line.strip()=='###### error analysis ######':
                    print('error analysis')
                    write_here = False
                    continue
                if write_here==True:
                    main_data[file_start].append(line[:-1])
                    file_start+=1
                    if increase_row==True:
                        row+=1  
        if use_sim_files=='.sim' or use_sim_files=='.chk':
            print('proceed')
            with open(name+use_sim_files,'r') as current:
                file_start = count_sim
                for line in current:
                    if (not line.strip)==False: # checking to make the sure the line isnt empty
                        line.split()
                        main_data[file_start].append(line[:-1])
                        file_start +=1 
                        count_sim  +=1
            
    print('Finished with main_data. Yay!' ) 
    #
    # Part 2: Checking file for similarities
    #
    

elif method=='1':
    
    def key_reader(key,dictionary):
        cout = dictionary[0][int(key[1])]
        nout = dictionary[1][int(key[3])]
        qout = dictionary[2][int(key[5])]
        p1out = dictionary[3][int(key[7])]
        p2out = dictionary[4][int(key[8])]
        p3out = dictionary[5][int(key[9])]
        return cout,nout,qout,p1out,p2out,p3out
    
    # Assemble into new ordered dictionary
    main_data = []
    f_ind = 0
    max_N = 1
    for name in file_list:
        main = [{},{},{},{},{},{}]
        with open(name+filetype[1],'r') as current:
            for line in current:
                if line.strip()=='###### run details ######':
                    break
                if line[0]=='#':
                    if line[1]=='C':
                        sub_main = main[0]
                    elif line[1]=='N':
                        sub_main = main[1]
                    elif line[1]=='Q':
                        sub_main = main[2]
                    elif line[1:3]=='P1':
                        sub_main = main[3]
                    elif line[1:3]=='P2':
                        sub_main = main[4]
                    elif line[1:3]=='P3':
                        sub_main = main[5]
                    marker = line[0:3]
                    line = next(current)
                    items = line.split()                
                    for un in items:
                        temp = un.split(':')
                        try:
                            ans = sub_main[temp[0]]
                            if ans==temp[1]:
                                continue  
                            else:
                                raise Exception  
                        except:
                            d_ind = len(sub_main)
                            try:
                                sub_main[d_ind]=temp[1]
                            except:
                                if int(temp[0])>max_N:
                                    max_N = int(temp[0])
                                
                                    
                    
                    if marker[1]=='C':
                        main[0] = sub_main
                    elif marker[1]=='N':
                        main[1] = sub_main
                    elif marker[1]=='Q':
                        main[2] = sub_main
                    elif marker[1:3]=='P1':
                        main[3] = sub_main
                    elif marker[1:3]=='P2':
                        main[4] = sub_main
                    elif marker[1:3]=='P3':
                        main[5] = sub_main
        main[1]=dict(zip(range(0,max_N),range(0,max_N)))       
        with open(name+filetype[0],'r') as current:
            for line in current:
                main_data.append([])
                hold_dat = line.split()
                key = hold_dat[0]
                n_state = len(hold_dat)
                coder = key_reader(key,main)
                main_data[f_ind].append(coder[0])
                main_data[f_ind].append(coder[1])
                main_data[f_ind].append(coder[2])
                main_data[f_ind].append(coder[3])
                main_data[f_ind].append(coder[4])
                main_data[f_ind].append(coder[5])
                sub_dict = {} 
                for i in range(1,n_state):
                    pair = hold_dat[i].split(':')
                    sub_dict[pair[0]] = pair[1]
                main_data[f_ind].append(sub_dict)
                f_ind+=1 
                
    #for i in main_data:
    #    print(i)
            
    
    def combine_dict(one,two):
        for key,val in two.items():
            try:
                one[key] = int(one[key]) + int(val)
            except:
                one[key] = int(val)
        return one
    
    combine = input('Would you like to combine and discard redundant files? y/n ')
    
    cont=False
    while cont==False:
        if combine=='y' or combine=='yes' or combine=='no' or combine=='n':
            cont=True
        else:
            combine = input('Try again: ')
    
    if combine=='y' or combine=='yes':
    
        print('Looking for similarities')
        print('Number of runs = '+str(max_N))
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
    
        ind = 0
        hol_1 = []
        ind1 = 0
        for item in main_data:
            #print(item[1])
            if int(item[1]) in hold_disc:
                hol_1.append(ind)
            ind+=1 
        for item in reversed(hol_1):
            del main_data[item]
        #print(hol_1)
        for i in main_data:
            print(i)
        skip=[]
        for item1 in main_data:
            temp = item1[:]
            del temp[6]
            del temp[1]
            ind2 = 0
            for item2 in main_data:
                if ind2 in skip:
                    ind2+=1 
                    continue
                elif ind2>ind1:
                    pass
                else:
                    ind2+=1 
                    continue
                par1 = item1[0:6]
                par2 = item2[0:6]
                if [i for i,j in zip(par1,par2) if i==j]==temp:
                    #print(temp)
                    item1[6] = combine_dict(item1[6],item2[6])
                    skip.append(ind2)
                ind2 +=1 
            ind1 += 1
        for index in reversed(skip):
            del main_data[index]
    
    print(main_data)        
    cont = input('Proceed? y/n ')
    if cont=='y' or cont=='yes':
        pass
    else:
        sys.exit()
    
    print('On to the data analysis!')
    calc_diag = False
    for item in main_data:
        if item[0]=='err':
            calc_diag = True    

    # now, going to convert main data into a new form 
    new_data = []
    ndata = len(main_data)
    if calc_diag==True:
        ndata*= int(0.5)
    ind_3 = 0 
    for item in range(0,ndata):
        new_data.append([])
        if calc_diag:
            theta = main_data[2*item][2:5]
            main_r= rdm.rdm(main_data[2*item][6])
            err_r = rdm.rdm(main_data[2*item+1][6])
        else:
            theta = main_data[2*item][2:5]
            main_r= rdm.rdm(main_data[item][6])
            err_r = np.array([0,0,0])
        main_r = main_r[n_qubits-3:n_qubits]
        err_r  = err_r[n_qubits-3:n_qubits]
        rdm_r, occ_r, vec_r = rdm.construct_rdm(main_r,err_r)
        wf = run.single_run_c3(float(theta[0]),float(theta[1]),float(theta[2]),'020121')


        for i in occ_r:
            new_data[ind_3].append(i)
        for i in err_r:
            new_data[ind_3].append(i)
        for i in theta:
            new_data[ind_3].append(i)
        
        ind_3+= 1

            
    
#
#
# Part 3: Saving new file 
#

print('Here are the files in the save directory. \n')
for item in os.listdir(default_out_loc):
    print(item)
save_filename = input('\nSave file as?')
save_filename = './compiled/' + save_filename  
with open(save_filename,'w') as save_file:
    for row in main_data:
        for item in row:   
            save_file.write(item+' ')
        save_file.write('\n')
# Part 3: Saving new file 
#

print('Here are the files in the save directory. \n')
for item in os.listdir(default_out_loc):
    print(item)
save_filename = input('\nSave file as?')
save_filename = './compiled/' + save_filename  
with open(save_filename,'w') as save_file:
    for row in main_data:
        for item in row:   
            save_file.write(item+' ')
        save_file.write('\n')



