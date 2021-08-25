'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
from math import pi
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from hqca.vqe import *
from hqca.opts import *
from pyscf import gto
from functools import partial
import numpy as np
from hqca.transforms import *
mol = gto.Mole()

d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol,transform=JordanWigner)
ins = PauliSet
st = StorageVQE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=4,
        provider='Aer')
tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,
        transform=Qubit,
        )

qs.transform = Qubit
opt = partial(Optimizer,
        optimizer='bfgs',
        unity=pi/2,
        verbose=True)

vqe = RunVQE(st,opt,qs,ins,
        ansatz='ucc',
        tomography=tomoRe,
        gradient=True,
        opt_thresh=1e-4,
        verbose=False,
        )
vqe.build()
vqe.run()

print(vqe.T.evaluate(vqe.para))

#acse = RunACSE(
#        st,qs,Ins,
#        method='newton',
#        update='quantum',
#        opt_thresh=1e-10,
#        trotter=1,
#        ansatz_depth=1,
#        quantS_thresh_rel=1e-6,
#        propagation='trotter',
#        use_trust_region=True,
#        convergence_type='trust',
#        hamiltonian_step_size=0.01,
#        max_iter=100,
#        initial_trust_region=0.1,
#        newton_step=-1,
#        restrict_S_size=0.5,
#        tomo_S = tomoIm,
#        tomo_Psi = tomoRe,
#        verbose=False,
#        )
#acse.run()
#print(acse.log_rdm)

