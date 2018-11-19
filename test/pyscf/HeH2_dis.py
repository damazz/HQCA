from pyscf import scf,gto,mcscf,ao2mo,fci
from functools import reduce
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=3,suppress=True,linewidth=200)
mol = gto.Mole()
#dist = [0.537,0.687,0.937,1.187,1.387,1.637,1.937,2.437,3.237#]
dist = np.linspace(1.142,1.145,10)
#dist = [1.671,1.821,1.971,2.121,2.271]
dist = [0.943,1.043,1.143,1.241,1.341]
theta=[53.66]
h = []
E = np.zeros(len(dist))
i=0


for x in dist:
    j=0
    for y in theta:
        c = np.cos(y*np.pi/180)
        s = np.sin(y*np.pi/180)
        mol.atom = [
                ['He',( 0.000, 0.000,0.000)],
                ['H',(x,0.000,0)],
                ['H',(-x,0.000,0)],
                ]
        mol.basis = 'sto-3g'
        mol.spin=1
        mol.charge=1
        if i==0:
            mol.verbose=4
        else:
            mol.verbose=1
        mol.build()

        m = scf.ROHF(mol)
        m.max_cycle=100
        m.kernel()
        if i==0:
            m.analyze()
        h.append(m.mo_coeff)
        print('Done with distance {}.'.format(x))
        mc = mcscf.CASSCF(m,3,3)
        mc.verbose=4
        mc.max_cycle_macro=100
        mc.kernel()
        E[i] = mc.e_tot
        j+=1
    i+=1 
for a in range(0,len(m.mo_coeff)):
    j = 0 
    for state in h:
        print('AO coeff for MO #{} at dist {}:'.format(a+1,dist[j]))
        print(state[:,a])
        print('')
        j+=1 
    print('###########')
    print('###########')
    print('###########')
from matplotlib import cm
fig = plt.figure()
ax = plt.scatter(dist,E)
#surf = ax.plot_surface(X,Y,Z, linewidth=0,cmap=cm.coolwarm)
#fig.colorbar(surf)
#ax.set_zlim(-115,-113)
r = max(E)-min(E)
plt.ylim(min(E)-0.1*r,max(E)+0.1*r)
plt.show()
