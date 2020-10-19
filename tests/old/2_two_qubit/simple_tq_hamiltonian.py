from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from pyscf import gto
import numpy as np
import sys
mol = gto.Mole()

mat = np.matrix([
    #[1, 0,-1, 0],
    #[0, 0, 0,-1],
    #[-1,0,-3, 0],
    #[0,-1, 0,-5]])
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0 ,0, 0, 0],
    [0, 0, 0,-3]])

ham = TwoQubitHamiltonian(sq=False,
        matrix=mat)

Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=2,
        provider='Aer')
tomoRe = StandardTomography(qs,operator_type='qubit')
print(dir(tomoRe))

#tomoRe.generate(real=True,imag=False)
#tomoIm.generate(real=False,imag=True)

acse = RunACSE(
        st,qs,Ins,
        #method='euler',
        method='newton',
        use_trust_region=True,
        convergence_type='trust',
        hamiltonian_step_size=0.1,
        quantS_thresh_rel=0.0001,
        max_iter=100,
        initial_trust_region=0.1,
        newton_step=-1,
        restrict_S_size=0.2,
        )
acse.build()
acse.run()



