from hqca.acse import *
from pyscf import gto
from hqca.hamiltonian import *
from hqca.instructions import *
from delayed_assert import delayed_assert as da
from hqca.acse._ansatz_S import *
from _generic import *
from hqca.tools.quantum_strings import PauliString as Pauli
from hqca.tools.quantum_strings import FermiString as Fermi
from hqca.tools.quantum_strings import QubitString as Qubit
from hqca.tools._operator import Operator as Op

def test_ansatz():
    Sc = Ansatz(closed=True)
    So = Ansatz(closed=False)
    op1 = Operator(
            [
                Pauli('XY',1j),
                Pauli('YX',1j),
        ])
    op2 = Operator(
            [
                Pauli('YZ',1j),
                Pauli('ZY',-1j)
                ]
            )
    So+= op1
    So+= op2
    So+= op1
    So+= op1
    da.expect(len(Sc)==2)
    print(Sc)
    print(len(Sc))
    for i in Sc:
        print(i)
    da.assert_expectations()


def test_acse():
    st,qs,ins,proc,tR,tI = generic_acse_objects()
    acse = RunACSE(
            st,qs,ins,
            method='newton',
            update='quantum',
            opt_thresh=1e-10,
            trotter=1,
            quantS_thresh_rel=1e-6,
            propagation='trotter',
            use_trust_region=True,
            convergence_type='trust',
            hamiltonian_step_size=0.01,
            max_iter=5,
            initial_trust_region=0.1,
            newton_step=-1,
            restrict_S_size=0.5,
            tomo_S = tI,
            tomo_Psi = tR,
            verbose=False,
            )
    acse.build()
    da.expect(abs(acse.e0+0.783792654277353)<=1e-10)
    da.assert_expectations()
    print(acse.e0)

test_ansatz()
#test_acse()
