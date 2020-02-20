'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''


from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from hqca.tools.fermions import *
from hqca.tools import *
from pyscf import gto
import sys
mol = gto.Mole()
mol.atom = [
        ['H', (-0.5000,0.0, 0.00000000)],
        ['H', (0.5000, 0.0, 0.00000000)]
        ]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
bkSet = BravyiKitaevSet(4,Ne=[1,2],reduced=True)
ham = MolecularHamiltonian(mol,
        Ne_active_space=2,
        No_active_space=2,
        int_thresh=1e-5,
        #orbitals='active',
        orbitals='hf',
        mapping='bravyi-kitaev',
        kw_mapping={'bkSet':bkSet},
        )


Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=2,
        provider='Aer',
        )
#tomoRe = StandardTomography(qs)
tomoRe = ReducedTomography(qs)
tomoIm = ReducedTomography(qs)
print('Generating real orbitals.')
tomoRe.generate(real=True,imag=False,mapping='bk',bkSet=bkSet)
tomoIm.generate(real=False,imag=True,mapping='bk',bkSet=bkSet)
print(tomoIm.mapping)
print(tomoIm.op)
print(tomoRe.op)
acse = RunACSE(
        st,qs,Ins,
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.01,
        max_iter=5,
        initial_trust_region=0.1,
        newton_step=-1,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        restrict_S_size=1.0,
        verbose=True,
        commutative_ansatz=True,
        )
acse.build()
acse.run()

print(tomoIm.mapping)
