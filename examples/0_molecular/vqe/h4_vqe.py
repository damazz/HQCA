'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
from math import pi
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.tools.fermions import *
from hqca.acse import *
from hqca.vqe import *
from optss import *
from pyscf import gto
from functools import partial
mol = gto.Mole()

d = 2.0
mol.atom=[['H',(d*i,0,0)] for i in range(4)]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
ins = PauliSet
MapSet = ParitySet(6,reduced=True,Ne=[0,1])
kw_mapping = {'MapSet':MapSet}
#ham = MolecularHamiltonian(mol,mapping='parity',kw_mapping=kw_mapping)
ham = MolecularHamiltonian(mol)
st = StorageVQE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        #backend='qasm_simulator',
        Nq=8,
        provider='Aer')
tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,simplify=True,
        mapping='jw',MapSet=MapSet,
        method='gt',
        strategy='lf',
        )
opt = partial(Optimizer,
        optimizer='bfgs',
        unity=pi/2,
        verbose=True)

vqe = RunVQE(st,opt,qs,ins,
        ansatz='ucc',
        tomography=tomoRe,
        gradient=True,
        opt_thresh=1e-4,
        )
vqe.build()
vqe.run()

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

