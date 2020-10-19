import numpy as np
from hqca.tools import *
import matplotlib.pyplot as plt
import sys

#data = np.load('yorktown_rdms.npy')
data = np.load('yorktown_scan.npy')
print(data.shape)

def d1s(rdm):
    es = np.sort(np.linalg.eigvalsh(rdm.rdm))
    a1 = 0.5*(es[0]+es[1])
    a2 = 0.5*(es[2]+es[3])
    vec = np.zeros(4)
    vec[0]=es[0]-a1
    vec[1]=es[1]-a1
    vec[2]=es[2]-a2
    vec[3]=es[3]-a2
    return np.linalg.norm(vec)

norm =np.zeros((4,4))
cumulant = np.zeros((4,4))
hartree = np.zeros((4,4))
d2eigs = np.zeros((4))
g2eigs = np.zeros((4))
q2eigs = np.zeros((4))
d1eigs = np.zeros((4))
N = np.zeros((4))

rdme = np.zeros((4,data.shape[0]))
rdme1 = np.zeros((4,data.shape[0]))
rdme2 = np.zeros((4,data.shape[0]))
rdmeig = np.zeros((4,data.shape[0]))
cval = np.zeros((4,data.shape[0]))

for t in range(data.shape[0]):
    print('T: {}'.format(t))
    d2_0 = RDM(2,alpha=[0,1],
            beta=[2,3],
            state='given',
            Ne=2,
            S=0,
            rdm=data[t,0,:,:]
            )
    d2_q = RDM(2,alpha=[0,1],
            beta=[2,3],
            state='given',
            Ne=2,
            S=0,
            rdm=data[t,1,:,:]
            )
    d2_z = RDM(2,
            alpha=[0,1],beta=[2,3],
            state='given',
            Ne=2,
            S=0,rdm=data[t,3,:,:]
            )
    d2_s = RDM(2,
            alpha=[0,1],beta=[2,3],
            state='given',
            Ne=2,
            S=0,rdm=data[t,2,:,:]
            )
    d2_s.expand()
    d2_z.expand()
    for i in range(4):
        for j in range(4):

            d2_s.rdm[i,j,i,j]=d2_z.rdm[i,j,i,j]
            d2_s.rdm[i,j,j,i]=d2_z.rdm[i,j,j,i]
    for ni,i in enumerate([d2_0,d2_q,d2_z,d2_s]):
        d1_i = i.reduce_order()
        ci = i.cumulant()
        i.contract()
        #print(np.linalg.norm(ci.rdm))
        for nj,j in enumerate([d2_0,d2_q,d2_z,d2_s]):
            cj = j.cumulant()
            j.contract()
            hi = i-ci
            hj = j-cj
            norm[ni,nj]+= np.linalg.norm(i.rdm-j.rdm)
            cumulant[ni,nj]+= np.linalg.norm(ci.rdm-cj.rdm)
            hartree[ni,nj]+= np.linalg.norm(hi.rdm-hj.rdm)
            if ni==2 and nj==3:
                zed = i-j
                mat = zed._get_ab_block()
                #np.fill_diagonal(mat,0)
                print(mat)
                a,b = np.linalg.eig(mat)
                print(a)
                print(b)

        v = np.zeros(16)
        v[0]=2
        i.contract()
        d2eigs[ni]+= np.linalg.norm(np.sort(np.linalg.eigvalsh(i.rdm))-np.sort(v))
        d1eigs[ni]+= d1s(d1_i)
        rdmeig[ni,t]= np.linalg.eigvalsh(i.rdm)[-1]
        rdme[ni,t]= i.rdm[2,7]
        rdme1[ni,t]= d1_i.rdm[0,0]
        rdme2[ni,t]= d1_i.rdm[1,1]
        cval[ni,t] = np.linalg.norm(ci.rdm)
        q = i.get_Q_matrix()
        g = i.get_G_matrix()
        q.contract()
        g.contract()
        i.contract()
        print('D,Q,G')
        print(np.linalg.eigvalsh(i.rdm))
        print(np.linalg.eigvalsh(q.rdm))
        print(np.linalg.eigvalsh(g.rdm))

        #print(ci._get_ab_block())

sys.exit()



print('-------------------')
print('Error in density matrices')
print('-------------------')
print(' ideal   qc     z_symm od_symm')
norm = norm*(1/25)
print(norm)
print('-------------------')
print('Error in cumulants')
print('-------------------')
print(' ideal   qc     z_symm od_symm')
cumulant = cumulant/25
print(cumulant)
print('-------------------')
print('Error in reconstructed 2D')
print('-------------------')
print(' ideal   qc     z_symm od_symm')
hartree = hartree/25
print(hartree)
print('-------------------')
print('Error in 2-RDM eigenvalues')
print('-------------------')
print(' ideal   qc     z_symm od_symm')
d2eigs = d2eigs/25
print('',d2eigs)
print('Error in 1-RDM eigenvalues from pairing')
d1eigs = d1eigs/25
print('',d1eigs)


x = np.arange(0,25)
fig, ax = plt.subplots(2,2)

for n,l in enumerate(['ideal','qc','z only', 'zz symm']):
    ax[0,0].plot(x,rdme[n,:],label=l)
    ax[0,1].plot(x,rdmeig[n,:],label=l)
    ax[1,0].plot(x,rdme1[n,:],label=l)
    ax[1,0].plot(x,rdme2[n,:],label=l)
    ax[1,1].plot(x,cval[n,:],label=l)
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()
plt.show()


