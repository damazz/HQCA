from hqca.acse import *
from pyscf import gto
from hqca.hamiltonian import *
from hqca.instructions import *
from delayed_assert import delayed_assert as da
from hqca.acse._ansatz_S import *
from _generic_acse import *
from hqca.operators.quantum_strings import PauliString as Pauli
from hqca.operators.quantum_strings import FermiString as Fermi
from hqca.operators.quantum_strings import QubitString as Qubit
from hqca.operators._operator import Operator as Op
from hqca.tools._stabilizer import Stabilizer
from hqca.transforms import *
from functools import partial
import hqca.config as config
config._use_multiprocessing=False
import pytest

def test_uacse_h2():
    Tr,iTr = get_transform_from_symmetries(
            Transform=JordanWigner,
            symmetries=['ZIIZ','ZIZ','ZZ'],
            qubits=[3,2,1],
            eigvals=[-1,+1,-1],
            )
    qTr,iqTr = get_transform_from_symmetries(
            Transform=Qubit,
            symmetries=['ZIIZ','ZIZ','ZZ'],
            qubits=[3,2,1],
            eigvals=[-1,+1,-1],
            )
    ham = MolecularHamiltonian(mol=generic_mol(),transform=Tr)
    ins = PauliSet
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=1,
            provider='Aer')
    qs.initial_transform = iTr
    tomoRe = StandardTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=Tr,use_multiprocessing=False,)
    tomoIm = QubitTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=qTr,use_multiprocessing=False,)
    proc = StandardProcess()
    norm = []
    for method in ['newton','euler','bfgs','cg','lbfgs',]:
        st = StorageACSE(ham,closed_ansatz=-1)
        acse = RunACSE(
                st,qs,ins,processor=proc,
                method=method,
                update='para',
                opt_thresh=1e-3,
                S_thresh_rel=1e-6,
                S_min=1e-6,
                convergence_type='norm',
                hamiltonian_step_size=0.000001,
                use_trust_region=True,
                max_iter=20,
                epsilon=0.1,
                tomo_A = tomoIm,
                tomo_psi = tomoRe,
                transform_psi=qTr,
                verbose=True,
                )
        acse.build()
        norm.append(acse.norm)
        print(norm)
        da.expect(abs(acse.e0+0.783792654277353)<=1e-10)
        da.expect(abs(norm[-1]-norm[0])<1e-6,'Norm should be the same!')
        acse.run()
        da.expect(abs(acse.e0-acse.store.H.ef)<=1e-4)

    da.assert_expectations()

test_uacse_h2()

#@pytest.mark.skip('Longer test!')
def test_uacse_h3():
    Tr,iTr = get_transform_from_symmetries(
            Transform=JordanWigner,
            symmetries=['IZIZIZ','IZIIZ','ZZZI'],
            qubits=[5,4,2],
            eigvals=[+1,-1,+1],
            )
    qTr,iqTr = get_transform_from_symmetries(
            Transform=Qubit,
            symmetries=['IZIZIZ','IZIIZ','ZZZI'],
            qubits=[5,4,2],
            eigvals=[+1,-1,+1],
            )
    ham = MolecularHamiltonian(mol=advanced_mol(),transform=Tr)
    ins = PauliSet
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='unitary_simulator',
            Nq=3,
            provider='Aer')
    qs.initial_transform = iTr
    tomoRe = StandardTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=Tr,use_multiprocessing=False,)
    tomoIm = QubitTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=qTr,use_multiprocessing=False,)
    proc = StandardProcess()
    for method in ['newton','euler','bfgs','cg','lbfgs']:
        st = StorageACSE(ham)
        acse = RunACSE(
                st,qs,ins,processor=proc,
                method=method,
                update='para',
                opt_thresh=1e-2,
                S_thresh_rel=1e-6,
                S_min=1e-6,
                convergence_type='norm',
                hamiltonian_step_size=0.000001,
                use_trust_region=True,
                max_iter=40,
                epsilon=0.15,
                tomo_A = tomoIm,
                tomo_psi = tomoRe,
                bfgs_limited=3,
                transform_psi = qTr,
                verbose=True,
                )
        acse.build()
        da.expect(abs(acse.e0-acse.store.ei)<=1e-10)
        acse.run()
        msg = 'error in {} method'.format(method)
        da.expect(abs(acse.e0-acse.store.H.ef)<=1e-3,msg)
    da.assert_expectations()

