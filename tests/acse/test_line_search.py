from delayed_assert import delayed_assert as da
from hqca.acse._ansatz_S import *
from hqca.operators.quantum_strings import PauliString as Pauli
from copy import deepcopy as copy
from hqca.acse._line_search_acse import LineSearchACSE

import hqca.config as config
config._use_multiprocessing=False
from _generic_acse import *

def test_line_search_gradient():
    '''
    test that the gradient information is correct;
    '''
    _no_multiprocessing = True
    ham,st,qs,ins,proc,tR,tI,t3 = advanced_acse_objects()
    st = StorageACSE(ham,closed_ansatz=0)
    acse = RunACSE(
            st,qs,ins,processor=proc,
            method='bfgs',
            update='quantum',
            opt_thresh=1e-10,
            S_thresh_rel=1e-6,
            S_min=1e-6,
            use_trust_region=True,
            convergence_type='norm',
            hamiltonian_step_size=0.000001,
            max_iter=1,
            initial_trust_region=1.5,
            newton_step=-1,
            epsilon=0.5,
            tomo_A = tI,
            tomo_psi = tR,
            verbose=False,
            )
    acse.build()
    ls = LineSearchACSE(acse,p=-acse.A)

    f0  =ls.phi(0)
    g_0,g0 = ls.dphi(0)
    print(f0)
    print(g0)
    fe = ls.phi(0.001)
    assert abs((fe-f0)/(g0*0.001) - 1 )<0.01

test_line_search_gradient()


#fepsilon = self.phi(0.01) #evaluate phi(alp_i) 
#print('Test')
#print(self.c1)
#print(fepsilon)
#print(f0+0.01*g0)
