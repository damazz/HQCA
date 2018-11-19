from pyscf import scf,gto,mcscf,ao2mo,fci
from functools import reduce
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=6,suppress=True,linewidth=200)
#dist = [0.537,0.687,0.937,1.187,1.387,1.637,1.937,2.437,3.237#]
#dist = np.linspace(1.175,1.275,11)

# eq dist for BH+ is like....1.2076
# now, testing distance to make a good PES
dist =[0.7576,0.8576,1.0576,1.2076,1.3576,1.6076,2.0576,2.5076,3.0076]
#dist =[0.8576,1.0576,1.2076,1.3576,1.6076,2.0576]
h = []
E = np.zeros((len(dist),6))
i=0
gs = gto.Mole()
gs.atom =[['B',(0,0,0)],['H',(1.2076,0,0)]]
gs.basis='sto-3g'
gs.charge=1
gs.spin=1
gs.build()
hf0 = scf.ROHF(gs)
hf0.kernel()
for x in dist:
    mol = gto.Mole()
    mol.atom = [
            ['B',(0.0,0.0,0.0)],
            ['H',(x,0.0,0.0)],
            ]
    mol.basis = 'sto-3g'
    mol.spin=1
    mol.charge=1
    mol.verbose=1
    mol.build()

    m = scf.ROHF(mol)
    m.max_cycle=100
    m.kernel()
    mcscf0 = mcscf.CASSCF(m,3,3)
    mcscf0.kernel()
    print('Finished MCSCF for dist {}'.format(x))
    orb=mcscf.project_init_guess(mcscf0,hf0.mo_coeff,gs)
    mcscf1 = mcscf.CASSCF(m,3,3)
    mcscf1.kernel(orb)
    print('Finished projected MCSCF for dist {}'.format(x))
    mcci0 = mcscf.CASCI(m,3,3)
    mcci0.kernel()
    print('Finished CASCI for dist {}'.format(x))
    mcci1 = mcscf.CASCI(m,3,3)
    mcci1.kernel(orb)
    print('Finished CASCI for projected mo, dist {}'.format(x))
    mcci2 = mcscf.CASCI(m,3,3)
    mcci2.kernel(mcscf0.mo_coeff)
    print('Finished CASCI for CASCF projected mo, dist {}'.format(x))
    E[i,0]=0#m.e_tot
    E[i,3]=mcci0.e_tot-mcscf0.e_tot
    E[i,4]=mcci1.e_tot-mcscf0.e_tot
    E[i,5]=mcci2.e_tot-mcscf0.e_tot
    E[i,1]=0#mcscf0.e_tot
    E[i,2]=0#mcscf1.e_tot-mcscf0.e_tot
    i+=1 


from matplotlib import cm
fig = plt.figure()
labels = [
        'HF',
        'CASSCF',
        'CASSCF in projected basis',
        'CASCI',
        'CASCI in projected basis',
        'CASCI in proj. CASSCF basis']

for i in range(0,len(dist)):
    print('Dist.:{}, Energies: {}'.format(dist[i],E[i,:]))
mark=['o','x','x','+','+','+']
for i in range(2,6):
    if i==4:
        continue
    ax = plt.plot(dist,E[:,i],marker=mark[i],label=labels[i])
r = np.max(E)-np.min(E)
plt.ylim(np.min(E)-0.1*r,np.max(E)+0.1*r)
plt.legend()
plt.show()
