import numpy as np
from hqca.tools import *

dat = np.loadtxt('h4_qc_pure.csv',delimiter=',')
print(dat)



new = np.zeros((8,8,8,8))
alp = [0,1,2,3]
bet = [4,5,6,7]

dat = np.reshape(dat,(4,4,4,4))


#  # take spatial? 
for i in alp:
    for j in alp:
        for k in bet:
            K = k%4
            for l in bet:
                L = l%4
                new[i,k,j,l]+=dat[i,K,j,L]
                new[k,i,j,l]-=dat[i,K,j,L]
                new[k,i,l,j]+=dat[i,K,j,L]
                new[i,k,l,j]-=dat[i,K,j,L]

test = RDM(order=2,Ne=4,S=0,S2=0,alpha=[0,1,2,3],
        beta=[4,5,6,7],state='given',rdm=new)
aa = test*test
print(aa.rdm)
test.contract()
print(test.rdm)
