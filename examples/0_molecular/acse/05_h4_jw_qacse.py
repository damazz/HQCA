'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''

from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from pyscf import gto
from hqca.transforms import *
mol = gto.Mole()

d = 1.5
#mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
mol.atom=[
        ['H',(+0,0,0)],
        ['H',(+d,+0,0)],
        ['H',(-d,+0,0)],
        ['H',(+2*d,0,0)],
        ]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
T = JordanWigner
ham = MolecularHamiltonian(mol,transform=T)
Ins = PauliSet
st = StorageACSE(ham,closed_ansatz=True)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=8,
        provider='Aer')
tomoRe = ReducedTomography(qs)
tomoIm = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,transform=T)
tomoIm.generate(real=False,imag=True,transform=T)


acse = RunACSE(
        st,qs,Ins,
        method='newton',
        update='quantum',
        opt_thresh=1e-4,
        trotter=1,
        S_min=1e-6,
        S_thresh_rel=0.5,
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.01,
        max_iter=100,
        initial_trust_region=1.0,
        newton_step=-1,
        restrict_S_size=0.5,
        commutative_ansatz=True,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        verbose=True,
        )
acse.build()
acse.run()
#print(acse.log_rdm)

