from pyscf import scf,gto,mcscf,ao2mo,fci
from functools import reduce
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision=3,suppress=True,linewidth=200)
mol = gto.Mole()
#dist = [0.537,0.687,0.937,1.187,1.387,1.637,1.937,2.437,3.237#]
dist =[0.8076,0.9076,1.00076,1.1076,1.2076,1.3076,
        1.4076,1.5076,1.6076,1.7076,1.8076,1.9076,2.0076
        ]

from pyscf import scf,gto,mcscf,ao2mo
from functools import reduce
import numpy as np

mol = gto.Mole()
mol.atom = '''B 0 0 0; H 0 0 1.3076'''
mol.basis = 'sto-3g'
mol.spin=1
mol.charge=+1
mol.verbose=9
mol.build()

m = scf.ROHF(mol)
m.verbose=9
m.max_cycle=100
m.kernel()
mc = mcscf.CASSCF(m,6,5)
mc.verbose=9
mc.kernel()
mc.analyze()
#red = reduce(np.dot, (mc.mo_coeff.T, mc.get_hcore(), mc.mo_coeff))

ints_1e_scf = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
ints_1e_mcscf = reduce(np.dot, (mc.mo_coeff.T, mc.get_hcore(), mc.mo_coeff))
ne = mol.energy_nuc()


ints_2e_scf = ao2mo.kernel(mol,m.mo_coeff,compact=False,verbose=1)
ints_2e_mcscf = ao2mo.kernel(mol,mc.mo_coeff,compact=False,verbose=1)
#mc.analyze(verbose=9)

#mc.mc2step()
np.set_printoptions(suppress=True,precision=3)
print(mc.ci)
d1,d2 = mc.fcisolver.make_rdm12(mc.ci,6,5)
N = ints_1e_scf.shape[0]
d2 = np.reshape(d2,(N**2,N**2))


a = reduce(np.dot, (ints_2e_mcscf,d2)).trace()
b = reduce(np.dot, (ints_1e_mcscf,d1)).trace()
print(a*0.5)
print(b)
print(ne)
print('Total: {}'.format(a*0.5+b+ne))


