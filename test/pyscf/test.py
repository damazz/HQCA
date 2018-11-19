#
# ./rdm2.py
#
# Generic molecular input for use with main.py 
#

from pyscf import scf,gto,mcscf,ao2mo,molden
from functools import reduce
import numpy as np
from sympy import pprint
np.set_printoptions(linewidth=200,precision=4,suppress=True)


from pyscf import gto
mol = gto.Mole()
#mol.atom = '''H 0.5 0 0; H 0 0.5 0; H 0 0 0.5  '''
mol.atom = '''N 0 0 0 '''
mol.basis = 'sto-3g'
mol.spin=1
mol.verbose=9
mol.charge=0
mol.build()
print('Starting HF.')


m = scf.ROHF(mol)
m.max_cycle=100
m.kernel()
m.analyze()

print('CASSCF')


mc = mcscf.CASCI(m,3,1)
mc.verbose=4
mc.kernel()
mc.analyze()


