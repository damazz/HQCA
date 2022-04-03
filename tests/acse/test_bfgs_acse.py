from hqca.acse import *
from _generic_acse import *
import hqca.config as config
config._use_multiprocessing=False
from delayed_assert import delayed_assert as da
from copy import deepcopy as copy

def test_quantum_bfgs_euler_newton_acse_moderate():
    _no_multiprocessing = True
    ham,st,qs,ins,proc,tR,tI,t3 = advanced_acse_objects()
    st = StorageACSE(ham,closed_ansatz=True)
    qacse = RunACSE(
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
            tomo_A = tI,
            tomo_psi = tR,
            verbose=False,
            )
    qacse.build()
    da.expect(abs(qacse.e0+1.210100865453)<=1e-8)
    #qacse.run()
    #da.expect(abs(qacse.e0+1.396744763667)<=1e-8)
    print('Energy: ',qacse.e0.real)
    stc = StorageACSE(ham,closed_ansatz=True)
    qc_acse = RunACSE(
            stc,qs,ins,
            method='euler',
            update='quantum',
            opt_thresh=1e-10,
            S_thresh_rel=1e-6,
            S_min=1e-6,
            use_trust_region=True,
            convergence_type='norm',
            hamiltonian_step_size=0.000001,
            max_iter=1,
            tomo_A = tI,
            tomo_psi = tR,
            verbose=False,
            )
    qc_acse.build()
    #qc_acse.run()
    #da.expect(abs(qc_acse.e0+1.396744763667)<=1e-8)
    print(qacse.norm)
    print(qc_acse.norm)

    #print('Energy: ',qc_acse.e0.real)

    #diff = copy(qacse.A)-copy(qc_acse.A)
    #da.expect(abs(diff.norm())<1e-8)
    da.expect(qacse.norm-qc_acse.norm <1e-8)
    da.assert_expectations()


test_quantum_bfgs_euler_newton_acse_moderate()
