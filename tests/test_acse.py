from delayed_assert import delayed_assert as da
from _generic import *
from hqca.acse._ansatz_S import *
from hqca.operators.quantum_strings import PauliString as Pauli

def test_ansatz():
    # test addition
    Sc = Ansatz(closed=True)
    So = Ansatz(closed=False)
    Sm = Ansatz(closed=False)
    Sl1 = Ansatz(closed=-1)
    Sl2 = Ansatz(closed=-2)
    Sl3 = Ansatz(closed=-3)
    op1 = Operator([
                Pauli('XY',1j),
                Pauli('YX',1j),])
    op2 = Operator([
                Pauli('YZ',1j),
                Pauli('ZY',-1j)])
    op3 = Operator([
                Pauli('YI',1j),
                Pauli('IY',-1j)])
    Sc+= op1
    Sc+= op2
    Sc+= op2
    So+= op1
    So+= op2
    So+= op2
    Sm+= op1
    Sm-= op1
    Sl1+= op1
    Sl1+= op2
    Sl1+= op1

    Sl2+= op1
    Sl2+= op2
    Sl2+= op3
    Sl2+= op1

    Sl3+= op1
    Sl3+= op2
    Sl3+= op3
    Sl3+= op1
    da.expect(len(Sc)==3)
    da.expect(len(So)==2)
    da.expect(len(Sm)==0)
    da.expect(len(Sl1)==3)
    da.expect(len(Sl2)==4)
    da.expect(len(Sl3)==3)
    da.assert_expectations()


def test_quantum_euler_acse():
    ham,st,qs,ins,proc,tR,tI = generic_acse_objects()
    st = StorageACSE(ham,closed_ansatz=-1)
    acse = RunACSE(
            st,qs,ins,processor=proc,
            method='euler',
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
            restrict_S_size=0.5,
            tomo_S = tI,
            tomo_Psi = tR,
            verbose=False,
            )
    acse.build()
    da.expect(abs(acse.e0+0.783792654277353)<=1e-10)
    acse.run()
    da.expect(abs(acse.e0+0.846147736093)<=1e-8)
    da.assert_expectations()


def test_quantum_opt_acse():
    ham,st,qs,ins,proc,tR,tI = generic_acse_objects()
    st = StorageACSE(ham,closed_ansatz=-1)
    acse = RunACSE(
            st,qs,ins,processor=proc,
            method='line',
            update='quantum',
            opt_thresh=1e-10,
            optimizer_threshold=0.75,
            S_thresh_rel=1e-6,
            S_min=1e-6,
            use_trust_region=True,
            convergence_type='norm',
            hamiltonian_step_size=0.000001,
            max_iter=1,
            initial_trust_region=1.5,
            newton_step=-1,
            restrict_S_size=0.5,
            tomo_S = tI,
            tomo_Psi = tR,
            verbose=False,
            )
    acse.build()
    da.expect(abs(acse.e0+0.783792654277353)<=1e-10)
    acse.run()
    da.expect(abs(acse.e0+0.9469262276063557)<=1e-8)
    da.assert_expectations()


def test_quantum_classical_acse():
    ham,st,qs,ins,proc,tR,tI = generic_acse_objects()
    st = StorageACSE(ham,closed_ansatz=-1)
    qacse = RunACSE(
            st,qs,ins,processor=proc,
            method='newton',
            update='quantum',
            opt_thresh=1e-10,
            S_thresh_rel=1e-6,
            S_min=1e-6,
            use_trust_region=True,
            convergence_type='trust',
            hamiltonian_step_size=0.000001,
            max_iter=1,
            initial_trust_region=1.5,
            newton_step=-1,
            restrict_S_size=0.5,
            tomo_S = tI,
            tomo_Psi = tR,
            verbose=False,
            )
    qacse.build()
    da.expect(abs(qacse.e0+0.783792654277353)<=1e-10)
    qacse.run()
    da.expect(abs(qacse.e0+0.930729301813997)<=1e-10)
    print('Energy: ',qacse.e0.real)
    stc = StorageACSE(ham,closed_ansatz=True)
    qc_acse = RunACSE(
            stc,qs,ins,
            method='newton',
            update='classical',
            opt_thresh=1e-10,
            S_thresh_rel=1e-6,
            S_min=1e-6,
            use_trust_region=True,
            convergence_type='iterations',
            max_iter=1,
            initial_trust_region=1.5,
            newton_step=-1,
            restrict_S_size=0.5,
            tomo_S = tI,
            tomo_Psi = tR,
            verbose=False,
            )
    qc_acse.build()
    qc_acse.run()
    da.expect(abs(qc_acse.e0+0.930729301813997)<=1e-10)
    print('Energy: ',qc_acse.e0.real)
    diff = qacse.S[0]-qc_acse.S[0]
    print('Norm: ',diff.norm())
    da.expect(abs(diff.norm())<1e-8)
    da.assert_expectations()

def test_quantum_classical_acse_moderate():
    ham,st,qs,ins,proc,tR,tI = advanced_acse_objects()
    st = StorageACSE(ham,closed_ansatz=-1)
    qacse = RunACSE(
            st,qs,ins,processor=proc,
            method='newton',
            update='quantum',
            opt_thresh=1e-10,
            S_thresh_rel=1e-6,
            S_min=1e-6,
            use_trust_region=True,
            convergence_type='trust',
            hamiltonian_step_size=0.000001,
            max_iter=1,
            initial_trust_region=1.5,
            newton_step=-1,
            restrict_S_size=0.5,
            tomo_S = tI,
            tomo_Psi = tR,
            verbose=True,
            )
    qacse.build()
    da.expect(abs(qacse.e0+1.210100865453)<=1e-8)
    qacse.run()
    da.expect(abs(qacse.e0+1.396744763667)<=1e-8)
    print('Energy: ',qacse.e0.real)
    stc = StorageACSE(ham,closed_ansatz=True)
    qc_acse = RunACSE(
            stc,qs,ins,
            method='newton',
            update='classical',
            opt_thresh=1e-10,
            S_thresh_rel=1e-6,
            S_min=1e-6,
            use_trust_region=True,
            convergence_type='iterations',
            max_iter=1,
            initial_trust_region=1.5,
            newton_step=-1,
            restrict_S_size=0.5,
            tomo_S = tI,
            tomo_Psi = tR,
            verbose=False,
            )
    qc_acse.build()
    qc_acse.run()
    da.expect(abs(qc_acse.e0+1.396744763667)<=1e-8)

    print('Energy: ',qc_acse.e0.real)

    diff = qacse.S[0]-qc_acse.S[0]
    print(qacse.S[0])
    print('')
    print(qc_acse.S[0])
    print('')
    print(diff)
    print('')
    print('Norm: ',diff.norm())
    da.expect(abs(diff.norm())<1e-8)
    da.assert_expectations()

