import run as rn
import numpy as np
import numpy.linalg as LA
import sys
import os

# using a certain function, see how far it can spread stochastically
# need a calculate euclidean distance to other points function

# set up paramters
step = 10
ulim = 45
llim = 0
cut_off = 0.05
par1 = np.linspace(llim,ulim,step)
print(par1)
par2 = np.linspace(llim,ulim,step)
par3 = np.linspace(llim,ulim,step)
par4 = np.linspace(llim,ulim,step)
par5 = np.linspace(llim,ulim,step)
par6 = np.linspace(llim,ulim,step)
print(len(par4))
order = [0,2,2,1,1,0]
#run
occ_list = [[1,1,1]]
full_list = [[1,1,1,0,0,0]]
index_occ = 0
dim = len(par1)
dim *= len(par2)
dim *= len(par3)
dim *= len(par4)
dim *= len(par5)
dim *= len(par6)
print(dim)
hold = 0
tick = int(round(dim/1000))
for a in par1:
    for b in par2:
        for c in par3:
            for d in par4:
                for e in par5:
                    for f in par6:
                        hold += 1                 
                        rdm = rn.construct_rdm(rn.single_full_run_c3(a,b,c,d,e,f,order))
                        noc,nor = LA.eig(rdm)
                        noc.sort()
                        noc = noc.tolist()
                        snoc = noc[3:]
                        near_dist = rn.nearest_neighbor(snoc,occ_list)
                        if hold%tick==0:
                            print('{:.2f}% of the way done.'.format(hold/dim*100))
                        if near_dist>cut_off:
                            occ_list.append([])
                            full_list.append([])
                            index_occ += 1
                            for i in range(0,6):
                                full_list[index_occ].append(noc[i])
                            for i in range(0,3):
                                occ_list[index_occ].append(snoc[i])
                            print(noc)
                        else:
                            continue

occ_list = np.asmatrix(occ_list)
#print(occ_list)
try:
    name = sys.argv[1]
except:
    name = 'test_ON'

np.savetxt(name+'.txt',occ_list)





