import qiskit
import qiskit.providers.aer.noise as noise
from deconstruct import generateNoiseModel
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.processes import *
from hqca.acse import *
from pyscf import gto
from functools import partial

nm = generateNoiseModel(0.1)
mol = gto.Mole()
mol.atom=[['H',(0,0,0)],['H',(2.0,0,0)]]
mol.basis='sto-3g'
mol.build()

# run 1
ham = MolecularHamiltonian(mol)
Ins = PauliSet
InsSymm = partial(Stabilizer,
        **{'stabilize':'spin'}
        #{'stabilize':'simple'}
        )
Combine = partial(Combine,
        **{'Ins':[Ins,InsSymm]}
        )
st = StorageACSE(ham)
qs = QuantumStorage()
#pr = StabilizerProcess(stabilizer='simple')
pr = StabilizerProcess(stabilizer='spin')
qs.set_algorithm(st)
qs.set_backend(
        backend='qasm_simulator',
        Nq=4,
        Nq_ancilla=2,
        num_shots=8192,
        provider='Aer')
qs.set_noise_model(custom=True,
        noise_model=nm)
symm = []
tomoRe = StandardTomography(qs)
tomoIm = StandardTomography(qs)
tomoRe.generate(real=True,imag=False,
        symmetries=symm,simplify=True,
        method='gt',strategy='lf')
tomoIm.generate(real=False,imag=True,
        symmetries=symm,
        simplify=True,method='gt',
        strategy='lf')
acse = RunACSE(
        st,qs,Combine,
        method='euler',
        #method='newton',
        update='quantum',
        opt_thresh=1e-10,
        trotter=1,
        ansatz_depth=1,
        quantS_thresh_rel=1e-6,
        classS_thresh_rel=1e-3,
        propagation='trotter',
        use_trust_region=True,
        #convergence_type='trust',
        convergence_type='iterations',
        hamiltonian_step_size=0.1,
        max_iter=3,
        initial_trust_region=0.1,
        newton_step=-1,
        restrict_S_size=1.0,
        tomo_S = tomoIm,
        tomo_Psi = tomoRe,
        statistics='N',
        verbose=False,
        processor=pr,
        )
acse.build()
acse.run()
print('')
print('-----------------------------------')
print(' # # # # # # # # # # # # # # # # #')
print('-----------------------------------')
print(' # # # # # # # # # # # # # # # # #')
print('-----------------------------------')
print('')
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='qasm_simulator',
        Nq=4,
        Nq_ancilla=2,
        num_shots=8192,
        provider='Aer')
qs.set_noise_model(custom=True,
        noise_model=nm)

tomoRe = ReducedTomography(qs)
tomoIm = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,
       simplify=True,method='gt',strategy='lf',
       symmetries=symm,
       )
tomoIm.generate(real=False,imag=True,
       simplify=True,method='gt',strategy='lf',
       symmetries=symm,
       )
acse_red = RunACSE(
        st,qs,Combine,
        method='euler',
        update='quantum',
        opt_thresh=1e-10,
        trotter=1,
        ansatz_depth=1,
        quantS_thresh_rel=1e-6,
        classS_thresh_rel=1e-3,
        propagation='trotter',
        use_trust_region=True,
        #convergence_type='trust',
        convergence_type='iterations',
        hamiltonian_step_size=0.1,
        max_iter=3,
        initial_trust_region=0.1,
        newton_step=-1,
        restrict_S_size=1.0,
        tomo_S = tomoIm,
        statistics='N',
        tomo_Psi = tomoRe,
        verbose=False,
        processor=pr,
        )
acse_red.build()
acse_red.run()

