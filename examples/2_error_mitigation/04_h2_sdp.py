'''

04_h2_sdp

Molecular test case of H2 with a sdp purification step.

'''
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.transforms import *
from hqca.acse import *
from pyscf import gto
mol = gto.Mole()

mol.atom=[['H',(0,0,0)],['H',(2.0,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()

ham = MolecularHamiltonian(mol,transform=JordanWigner)
Ins = PauliSet
st = StorageACSE(ham,S_depth=5)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='qasm_simulator',
        Nq=4,
        provider='Aer')
qs.set_error_mitigation(mitigation='sdp',
        path_to_maple='/home/scott/maple2020/bin/maple',
        spin_rdm=True)
tomoRe = ReducedTomography(qs)
tomoIm = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,transform=JordanWigner)
acse = RunACSE(
        st,qs,Ins,
        method='newton',
        update='classical',
        opt_thresh=1e-2,
        trotter=1,
        ansatz_depth=1,
        propagation='trotter',
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.01,
        max_iter=100,
        initial_trust_region=0.1,
        newton_step=-1,
        restrict_S_size=0.5,
        commutative_ansatz=True,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        verbose=False,
        )
acse.build()
acse.run()

