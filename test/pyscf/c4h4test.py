from pyscf import scf,gto,mcscf,ao2mo,molden,fci
from functools import reduce
import numpy as np
from sympy import pprint
np.set_printoptions(linewidth=200,precision=4,suppress=True)

mol = gto.Mole()
mol.atom ='''
    C  0.0000  0.6759  0.7844;
    C  0.0000  0.6759 -0.7844;
    C  0.0000 -0.6759  0.7844;
    C  0.0000 -0.6759 -0.7844;
    H  0.0000  1.4498  1.5475;
    H  0.0000  1.4498 -1.5475;
    H  0.0000 -1.4498  1.5475;
    H  0.0000 -1.4498 -1.5475
    '''
mol.basis = 'sto-3g'
mol.spin=0
mol.symmetry=True
mol.verbose=4
mol.build()

m = scf.RHF(mol)
m.max_cycle=100
m.kernel()
m.analyze()
mo = m.mo_coeff.T
h1 = reduce(np.dot, (mo,m.get_hcore(),mo.T))
eri = ao2mo.kernel(mol,m.mo_coeff)


#fc = fci.direct_spin0.FCISolver(mol=mol)
#e, ci = fc.kernel(h1,eri,h1.shape[1],mol.nelec,ecor=mol.energy_nuc())
#print('Full CI energy: {}'.format(e))
#print('Full CI coefficient matrix: ')
#print(ci)
mc = mcscf.CASSCF(m,14,14)
#mc.verbose=4
mc.kernel()
mc.analyze()
#print(mc.mo_coeff)
