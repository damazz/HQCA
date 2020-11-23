'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''

from hqca.tools.fermions import *
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from pyscf import gto
from copy import deepcopy as copy
from math import pi
mol = gto.Mole()

d = 0.5
#mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
mol.atom=[['H',(0,0,0)],['H',(d,0,0)],['H',(-d,0,0)]]
mol.basis='sto-3g'
mol.spin=1
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol)
Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=6,
        provider='Aer')
tomoRe = ReducedTomography(qs)
tomoIm = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,method='gt',strategy='lf')
tomoIm.generate(real=False,imag=True,method='gt',strategy='lf')

acse = RunACSE(
        st,qs,Ins,
        method='newton',
        update='quantum',
        trotter=1,
        ansatz_depth=1,
        opt_thresh=1e-10,
        quantS_max=1e-10,
        classS_max=1e-10,
        classS_thresh_rel=1e-6,
        quantS_thresh_rel=1e-6,
        tr_objective_criteria=1e-10,
        tr_taylor_criteria=1e-10,
        propagation='trotter',
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.0001,
        max_iter=125,
        initial_trust_region=0.5,
        newton_step=-1,
        restrict_S_size=1.0,
        commutative_ansatz=True,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        verbose=False,
        )
acse.build()
acse.run()

