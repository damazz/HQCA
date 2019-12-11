from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from pyscf import gto
import numpy as np
import sys
mol = gto.Mole()
d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol,casci=True)

maps  = {0:0,1:0,2:1,3:1}
qubit = {0:0,1:1,2:0,3:1}

ham = TwoQubitHamiltonian(sq=True,
        fermi=True,
        en_c = ham._en_c,
        ferOp=ham.fermi_operator,
        mapOrb=maps,
        mapQub=qubit,
        )

print(ham.matrix)
Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=2,
        provider='Aer')

#tomoRe.generate(real=True,imag=False)
#tomoIm.generate(real=False,imag=True)

acse = RunACSE(
        st,qs,Ins,
        #method='euler',
        method='newton',
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.1,
        commutative_ansatz=True,
        quantS_thresh_rel=0.25,
        max_iter=1,
        initial_trust_region=0.1,
        newton_step=-1,
        restrict_S_size=0.2,
        )
acse.build()
acse.run()



