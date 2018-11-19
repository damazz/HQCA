from pyscf import scf,gto,mcscf,ao2mo,molden
from functools import reduce
import numpy as np
from sympy import pprint
np.set_printoptions(linewidth=200,precision=4,suppress=True)

mol = gto.Mole()
mol.atom ='''
    H 0.0  0.0 0.000;
    H -0.7 0.0  0.000;
    '''
mol.basis = 'sto-3g'
mol.spin=0
mol.verbose=4
mol.build()

m = scf.ROHF(mol)
m.max_cycle=100
m.kernel()
m.analyze()
for a in range(0,len(m.mo_coeff)):
    print('AO coeff for MO #{}: '.format(a+1))
    pprint(m.mo_coeff[:,a])
pprint(m.mo_occ)
pprint(m.mo_energy)

mc = mcscf.CASSCF(m,2,2)
mc.verbose=4
mc.kernel()
mc.analyze()
print(mc.mo_coeff)
