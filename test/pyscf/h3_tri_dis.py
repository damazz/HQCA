from pyscf import scf,gto,mcscf,ao2mo,fci
from functools import reduce
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mol = gto.Mole()
#dist = [0.537,0.687,0.937,1.187,1.387,1.637,1.937,2.437,3.237]
dist = np.linspace(0.9,3.0,15)
theta = np.linspace(60,85,15)
X,Y = np.meshgrid(dist,theta)
Z = np.zeros(
        (
            len(theta),
            len(dist)
            )
        )
i=0
for x in dist:
    j=0 
    for y in theta:
        c = np.cos(y*np.pi/180)
        s = np.sin(y*np.pi/180)
        mol.atom = [['H0',(0,0,0)], ['H',(0,-x*c,x*s)], ['H',(0,x*c,x*s)]]
        mol.basis = 'sto-3g'
        mol.spin=1
        mol.verbose=1
        mol.build()

        m = scf.ROHF(mol)
        m.max_cycle=100
        m.kernel()
        mc = mcscf.CASSCF(m,3,3)
        mc.verbose=4
        mc.kernel()
        Z[j,i] = mc.e_tot
        j+=1
    i+=1 
print(Z)
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Z, linewidth=0,cmap=cm.coolwarm)
fig.colorbar(surf)
ax.set_zlim(-1.75,-1.25)
plt.show()
