'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''


from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from hqca.tools.fermions import *
from pyscf import gto
import numpy as np
import sys
mol = gto.Mole()
mol.atom = [
        ['C', (-0.60000000, 0.0, 0)],
        ['C', (0.60000000, 0.0, 0)],
        ['H', (-1.12000,0.0, 0.000000)],
        ['H', (1.12000, 0.0, 0.000000)]
        ]
mol.atom = [
        ['H', (0.0000000, 1.120, 0)],
        ['H', (0.0000000, -1.120, 0)],
        ['H', (-1.12000,0.0, 0.000000)],
        ['H', (1.12000, 0.0, 0.000000)]
        ]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
bkSet = BravyiKitaevSet(8,Ne=[0,0],reduced=True)
ham = MolecularHamiltonian(mol,
        Ne_active_space=4,
        No_active_space=4,
        int_thresh=1e-3,
        orbitals='hf',
        mapping='bravyi-kitaev',
        kw_mapping={'bkSet':bkSet},
        )
Ins = PauliSet
st1 = StorageACSE(ham)
st2 = StorageACSE(ham)
qs1 = QuantumStorage()
qs1.set_algorithm(st1)
qs1.set_backend(
        backend='statevector_simulator',
        Nq=6,
        provider='Aer',
        )
qs2 = QuantumStorage()
qs2.set_algorithm(st2)
qs2.set_backend(
        backend='statevector_simulator',
        Nq=6,
        provider='Aer',
        )
tomoRe1 = ReducedTomography(qs2)
tomoIm1 = ReducedTomography(qs2)
print('Generating real and imaginary orbitals.')
tomoRe1.generate(real=True,imag=False,mapping='bk',bkSet=bkSet)
tomoIm1.generate(real=False,imag=True,mapping='bk',bkSet=bkSet)
acse = RunACSE(
        st1,qs1,Ins,
        #update='classical',
        update='quantum',
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.01,
        classS_thresh_rel=0.5,
        quantS_thresh_rel=0.75,
        max_iter=3,
        initial_trust_region=5.0,
        newton_step=-1,
        tomo_S = tomoIm1,
        tomo_Psi = tomoRe1,
        restrict_S_size=1.0,
        verbose=True,
        commutative_ansatz=True,
        )
acse.build()
acse.run()
