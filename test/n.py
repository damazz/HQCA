from pyscf import gto,ao2mo, mcscf, scf
from functools import reduce
import numpy as np
from sympy import pprint
mol = gto.Mole()
mol.atom = '''N 0 0 0'''
mol.basis = 'sto-3g'
mol.spin=1
mol.verbose=2
mol.build()


m = scf.ROHF(mol)
m.verbose=9
m.max_cycle=100
m.kernel()
mc = mcscf.CASSCF(m,5,7)
mc.verbose=9
mc.kernel()
#red = reduce(np.dot, (mc.mo_coeff.T, mc.get_hcore(), mc.mo_coeff))

ints_1e_scf = reduce(np.dot, (m.mo_coeff.T, m.get_hcore(), m.mo_coeff))
ints_1e_mcscf = reduce(np.dot, (mc.mo_coeff.T, mc.get_hcore(), mc.mo_coeff))
ne = mol.energy_nuc()

ints_2e_scf = ao2mo.kernel(mol,m.mo_coeff,compact=False,verbose=1)
ints_2e_mcscf = ao2mo.kernel(mol,mc.mo_coeff,compact=False,verbose=1)
#mc.analyze(verbose=9)

#mc.mc2step()

d1,d2 = mc.fcisolver.make_rdm12(mc.ci,5,7)
#d1 = mc.fcisolver.make_rdm1(mc.ci,3,3)
#on, onv = np.linalg.eig(d1)

#ao2no = reduce(np.dot, (m.mo_coeff, onv))
#ao4no = np.kron(ao2no,ao2no)
d2 = np.reshape(d2,(5**2,5**2))
#d2 = reduce(np.dot, (ao4no.T,d2,ao4no))
#ints_1e_no = reduce(np.dot, (ao2no.T,ints_1e_scf,ao2no))
#ints_2e_no = ao2mo.kernel(mol,ao2no,compact=False)
#rdm = reduce(np.dot, (onv.T,d1,onv))


a = reduce(np.dot, (ints_2e_mcscf,d2)).trace()
b = reduce(np.dot, (ints_1e_mcscf,d1)).trace()
print(a*0.5)
print(b)
print(ne)
print('Total: {}'.format(a*0.5+b+ne))


