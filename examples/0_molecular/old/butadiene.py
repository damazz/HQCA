'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''


from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from hqca.tools.fermions import *
from pyscf import gto
import sys
mol = gto.Mole()

#d = 1.401000
d = 2.5
#mol.atom = [
#        ['C', (-0.60220000, 0.39720000, 0)],
#        ['C', (0.60240000, -0.39750000, 0)],
#        ['C', (-1.83150000, -0.13050000, 0)],
#        ['C', (1.83140000, 0.13080000, 0)],
#        ['H', (-0.49750000, 1.47890000, 0.00010000)],
#        ['H', (0.49790000, -1.47920000, 0.00010000)],
#        ['H', (-2.70350000, 0.51510000, 0)],
#        ['H', (-1.99750000, -1.20270000, 0)],
#        ['H', (2.70360000, -0.51430000, 0)],
#        ['H', (1.99690000, 1.20300000, 0)]
#        ]
mol.atom = [
        ['C', (-0.60000000, 0.0, 0)],
        ['C', (0.60000000, 0.0, 0)],
        ['H', (-1.12000,0.0, 0.00010000)],
        ['H', (1.12000, 0.0, 0.00010000)]
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
        orbitals='active',
        mapping='bravyi-kitaev',
        kw_mapping={'bkSet':bkSet},
        )
Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=6,
        provider='Aer',
        )
tomoRe = StandardTomography(qs)
tomoIm = StandardTomography(qs)
sys.exit()
tomoRe.generate(real=True,imag=False,mapping='bk',bkSet=bkSet)
tomoIm.generate(real=False,imag=True,mapping='bk',bkSet=bkSet)
acse = RunACSE(
        st,qs,Ins,
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.01,
        max_iter=100,
        initial_trust_region=0.1,
        newton_step=-1,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        restrict_S_size=1.0,
        verbose=False,
        commutative_ansatz=True,
        )
acse.build()
acse.run()

