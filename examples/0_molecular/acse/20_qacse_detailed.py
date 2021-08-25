
'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''

from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from pyscf import gto
from hqca.transforms import *
from hqca.operators import *
mol = gto.Mole()

d = 2.0
mol.atom=[
        ['H',(0,0,0)],
        ['H',(+d,+0,0)],
        ['H',(-d,+0,0)],
        #['H',(-d,-d,0)],
        #['H',(+d,-d,0)],
        #['H',(+d,+d,0)],
        ]
mol.basis='sto-3g'
mol.spin=1
mol.verbose=0
mol.build()
Tr, iTr = JordanWigner,JordanWigner
ham = MolecularHamiltonian(mol,transform=Tr,print_transformed=False)
op = ham._qubOp
o1 = Operator()
o2 = Operator()
for n,i in enumerate(op):
    if n%2:
        o1+=i
    else:
        o2+= i 
Ins = PauliSet
st = StorageACSE(ham,closed_ansatz=-1,trotter='second')
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=6,
        provider='Aer')
qs.initial_transform = iTr
tomoRe = ReducedTomography(qs)
tomoIm = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,transform=Tr)
tomoIm.generate(real=False,imag=True,transform=Tr)

hss = 0.1
acse = RunACSE(
        st,qs,Ins,
        method='newton',
        update='quantum',
        opt_thresh=1e-3,
        S_min=1e-8,
        S_thresh_rel=0.0,
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=hss,
        separate_hamiltonian=[o1,o2],
        #separate_hamiltonian=[op],
        expiH_approximation='first',
        max_iter=2,
        initial_trust_region=1.0,
        newton_step=-1,
        restrict_S_size=0.5,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        verbose=False,
        )
acse.build()
acse.run()
#print(acse.log_rdm)
st = StorageACSE(ham,closed_ansatz=-1,trotter='second')
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=6,
        provider='Aer')
qs.initial_transform = iTr

acse2 = RunACSE(
        st,qs,Ins,
        method='newton',
        update='quantum',
        opt_thresh=1e-3,
        S_min=1e-8,
        S_thresh_rel=0.0,
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=hss,
        expiH_approximation='first',
        max_iter=2,
        initial_trust_region=1.0,
        newton_step=-1,
        restrict_S_size=0.5,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        verbose=False,
        )
acse2.build()
acse2.run()

print(acse.A-acse2.A)
