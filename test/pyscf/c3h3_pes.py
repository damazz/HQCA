from pyscf import scf,gto,mcscf,ao2mo,fci
from functools import reduce
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mol = gto.Mole()
#dist = [0.537,0.687,0.937,1.187,1.387,1.637,1.937,2.437,3.237]
dist = np.linspace(1.0,2.5,6)
theta = [53.66]
X,Y = np.meshgrid(dist,theta)
Z = np.zeros(
        (
            len(theta),
            len(dist)
            )
        )
i=0
r = np.sqrt(
        (
            (-1.867+1.305)**2+
            (-1.581+0.660)**2
        )
        )
for x in dist:
    j=0 
    for y in theta:
        c = np.cos(y*np.pi/180)
        s = np.sin(y*np.pi/180)
        mol.atom = [
                ['C',( 0.000, 0.000,0.000)],
                ['C',(+c*x,-x*s,0.000)],
                ['C',(-c*x,-x*s,0.000)],
                ['H0',(0,0.757,0.787)],
                ['H',(+(x+r)*c,-(x+r)*s,0)],
                ['H',(-(x+r)*c,-(x+r)*s,0)]
                ]
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
ax.set_zlim(-115,-113)
plt.show()
