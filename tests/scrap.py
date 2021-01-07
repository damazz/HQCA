from hqca.tools.quantum_strings import PauliString as Pauli
from hqca.tools.quantum_strings import FermiString as Fermi
from hqca.tools.quantum_strings import QubitString as Qubit
from hqca.tools._operator import Operator as Op
from hqca.acse import *
import numpy as np
import pickle
from _generic import *
from hqca.hamiltonian import *
import timeit


ham,st,qs,ins,proc,tR,tI = advanced_acse_objects()
acse1 = RunACSE(
        st,qs,ins,processor=proc,
        method='euler',
        update='classical',
        opt_thresh=1e-10,
        S_thresh_rel=1e-6,
        S_min=1e-6,
        use_trust_region=True,
        convergence_type='norm',
        hamiltonian_step_size=0.000001,
        max_iter=10,
        initial_trust_region=1.5,
        newton_step=-1,
        restrict_S_size=0.5,
        tomo_S = tI,
        tomo_Psi = tR,
        verbose=True,
        )
acse1.build()
acse1.run()
#val =timeit.timeit(acse1.build,number=5)
#print('Time: {}, Time per iteration: {}'.format( val, val/5))

#print('Quantum S: ')
#print(acse2.A)
#print('Diff: ')
#print(acse2.A-acse1.A)



