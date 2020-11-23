'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''

import numpy as np
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from hqca.transforms import *
from pyscf import gto
mol = gto.Mole()

d = 1.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
#mol.atom=[['H',(0,0,0)],['H',(d,0,0)],['H',(-d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol,transform=JordanWigner)
Ins = PauliSet
st = StorageACSE(ham,S_depth=1)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        #backend='qasm_simulator',
        backend_initial_layout=[0,1,2,3],
        Nq=4,
        num_shots=8192,
        provider='Aer')
qs.set_error_mitigation(mitigation='ansatz_shift',coeff=1.0)

tomoRe = ReducedTomography(qs)
tomoIm = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,transform=JordanWigner)
tomoIm.generate(real=False,imag=True,transform=JordanWigner)

acse = RunACSE(
        st,qs,Ins,
        method='line',
        update='classical',
        opt_thresh=1e-10,
        trotter=1,
        ansatz_depth=1,
        quantS_thresh_rel=1e-6,
        propagation='trotter',
        use_trust_region=True,
        convergence_type='iterations',
        hamiltonian_step_size=0.01,
        max_iter=10,
        initial_trust_region=0.1,
        commutative_ansatz=True,
        newton_step=-1,
        restrict_S_size=1,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        verbose=True,
        )

acse.build(log=True)
acse.run()
acse.save('test_save')


