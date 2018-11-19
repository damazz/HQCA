from pyscf import scf,gto,mcscf,ao2mo,fci
from functools import reduce
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=3,suppress=True,linewidth=200)
mol = gto.Mole()
#dist = [0.537,0.687,0.937,1.187,1.387,1.637,1.937,2.437,3.237]

theta=[26.828]
h = []
dist = np.linspace(1.2,3.0,11)
E = np.zeros(len(dist))
i=0
r = np.sqrt(
        (
            (-1.867+1.305)**2+
            (-1.581+0.660)**2
        )
        )
mol0 = gto.Mole()
mol0.atom ='''
    C  0.000  0.000 0.000;
    C -1.305 -0.660 0.000;
    C -1.305  0.660 0.000;
    H -1.867 -1.581 0.000;
    H -1.867  1.581 0.000;
    H  0.757  0.000 0.787
    '''
mol0.basis = 'sto-3g'
mol0.spin=1
mol0.verbose=4
mol0.build()
m0 = scf.ROHF(mol0)
m0.max_cycle=100
m0.kernel()
m0.analyze()


for x in dist:
    j=0
    for y in theta:
        c = np.cos(y*np.pi/180)
        s = np.sin(y*np.pi/180)
        mol.atom = [
                ['C',( 0.000, 0.000,0.000)],
                ['C',(+c*x,-x*s,0.000)],
                ['C',(-c*x,-x*s,0.000)],
                ['H',(0,0.757,0.787)],
                ['H',(+(x+r)*c,-(x+r)*s,0)],
                ['H',(-(x+r)*c,-(x+r)*s,0)]
                ]
        mol.basis = 'sto-3g'
        mol.spin=1
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
        print(c*x,x*s,(x+r)*c,(x+r)*s)
        mc = mcscf.CASSCF(m,3,3)
        mc.max_cycle_macro=100
        mc.verbose=4
        orb=mcscf.project_init_guess(mc,m0.mo_coeff,mol0)
        mc.kernel(orb)
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
plt.show()
