import run as rn
import numpy as np
import numpy.linalg as LA
import sys
import os
import random
import warnings
warnings.filterwarnings('ignore')
# stochastic.py

# should take an input sys.argv[1], which is used for the filename with no suffix, i.e. the name you get


# using a certain function, see how far it can spread stochastically
# need a calculate euclidean distance to other points function

# set up paramters
step = 10
ulim = 45
llim = 0
cut_off = 0.0625

#order = [1,0,1,2,2,0]
order = [0,2,0,1,0,0]
#run
##
'''
data = np.loadtxt('./output/exp_04.dat')
use_data = np.zeros(len(data))
'''
occ_list = [[1,1,1]]
full_list = [[0,0,0,0,0,0]]
index_occ = 0


add_count = 0 

hold = 0
done = False
total = 0 
add_count = 0 
check_stop = True
while done==False:
    
    # generate random variables
    a = []
    for i in range(0,6):
        if i<4 and i%2==0:
            val = round(ulim*random.random()+llim,1)
        else:
            val = 0 
        a.append(val)
        #a.append(0)
        
    rdm = rn.construct_rdm(rn.single_full_run_c2(a[0],a[1],a[0],a[1],order))
    noc,nor = LA.eig(rdm)
    noc.sort()
    noc = noc.tolist()
    snoc = noc[3:]
    near_dist = rn.nearest_neighbor(snoc,occ_list)
    if near_dist>cut_off:
        print(near_dist)
        occ_list.append([])
        full_list.append([])
        index_occ += 1
        for i in range(0,6):
            full_list[index_occ].append(a[i])
        for i in range(0,3):
            occ_list[index_occ].append(snoc[i])
        print(noc,add_count)
        add_count = 0
        #print(occ_list)
    else:
        add_count+= 1  
    if (add_count>5000 and check_stop==True):
        filling = 96*(len(occ_list))*np.pi*(cut_off**3)/6
        print('There are {} items with an approximate filling of: {:.5f}.'.format(len(occ_list),filling))
        stop = input('Would you like to stop? y/n ')
        if stop=='y':
            done=True
        else:
            check_stop=False
    if  add_count%10000==0:
        check_stop = True
    hold +=1 
     
print(hold)
occ_list = np.asmatrix(occ_list)
full_list = np.asmatrix(full_list)
#print(occ_list)
try:
    name = sys.argv[1]
except:
    name = 'test_ON'

np.savetxt(name+'.txt',occ_list)
np.savetxt(name+'.dat',full_list)
np.savetxt(name+'.num',[hold])
np.savetxt(name+'.use',use_data)





