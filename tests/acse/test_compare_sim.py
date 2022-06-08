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

def test_acse_h2_sim():
    Tr,iTr = get_transform_from_symmetries(
            Transform=JordanWigner,
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
    tomoIm = StandardTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=Tr,use_multiprocessing=False,)
    proc = StandardProcess()
    norm = []
    st = StorageACSE(ham,closed_ansatz=-1)
    acse = RunACSE(
            st,qs,ins,processor=proc,
            method='bfgs',
            update='quantum',
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
            verbose=True,
            )
    acse.build()
    acse.run()

    st2 = StorageACSE(ham)
    qs2 = QuantumStorage()
    qs2.set_algorithm(st2)
    qs2.set_backend(
            backend='unitary_simulator',
            Nq=1,
            provider='Aer')
    qs2.initial_transform = iTr
    norm = []
    st2 = StorageACSE(ham,closed_ansatz=-1)
    acse2 = RunACSE(
            st2,qs2,ins,
            method='bfgs',
            update='quantum',
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
            verbose=True,
            )
    acse2.build()
    acse2.run()
    for i,j in zip(acse.log_E,acse2.log_E):
        assert abs(i-j)<1e-8

test_acse_h2_sim()


def test_acse_h3_sim():
    Tr,iTr = get_transform_from_symmetries(
            Transform=JordanWigner,
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
            backend='statevector_simulator',
            Nq=3,
            provider='Aer')
    qs.initial_transform = iTr
    tomoRe = StandardTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=Tr,use_multiprocessing=False,)
    tomoIm = StandardTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=Tr,use_multiprocessing=False,)
    proc = StandardProcess()
    norm = []
    st = StorageACSE(ham,closed_ansatz=-1)
    acse = RunACSE(
            st,qs,ins,processor=proc,
            method='bfgs',
            update='quantum',
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
            verbose=True,
            )
    acse.build()
    acse.run()

    st2 = StorageACSE(ham)
    qs2 = QuantumStorage()
    qs2.set_algorithm(st2)
    qs2.set_backend(
            backend='unitary_simulator',
            Nq=3,
            provider='Aer')
    qs2.initial_transform = iTr
    norm = []
    st2 = StorageACSE(ham,closed_ansatz=-1)
    acse2 = RunACSE(
            st2,qs2,ins,
            method='bfgs',
            update='quantum',
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
            verbose=True,
            )
    acse2.build()
    acse2.run()
    for i,j in zip(acse.log_E,acse2.log_E):
        assert abs(i-j)<1e-8


test_acse_h3_sim()
