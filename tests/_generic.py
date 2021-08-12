from pyscf import gto
import numpy as np
from hqca.hamiltonian import *
from hqca.opts import *
from hqca.transforms import *
from hqca.acse import *
from hqca.tools import *
from hqca.instructions import *
from hqca.core import *
from hqca.vqe import *
from hqca.core.primitives import *

class genericIns(Instructions):
    def __init__(self,coeff):
        self._gates =[[(coeff,),self._test]]

    def _test(self,Q,coeff):
        Q.si(0)
        Q.Cx(1,0)
        Q.Cx(2,1)
        Q.Cx(3,2)
        Q.Rx(3,coeff[0])
        Q.Rx(1,coeff[1])
        Q.Cx(3,2)
        Q.Cx(2,1)
        Q.Cx(1,0)
        Q.s(0)
        Q.si(2)
        Q.Cx(3,2)
        Q.Rx(3,coeff[2])
        Q.Cx(3,2)
        Q.s(2)

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a


def generic_mol():
    mol = gto.Mole()
    mol.atom=[['H',(0,0,0)],['H',(2.0,0,0)]]
    mol.basis='sto-3g'
    mol.spin=0
    mol.verbose=0
    mol.build()
    return mol

def advanced_mol():
    mol = gto.Mole()
    mol.atom=[
            ['H',(0,0,0)],
            ['H',(2.0,0,0)],
            ['H',(-2.0,0,0)],
            ]
    mol.basis='sto-3g'
    mol.spin=1
    mol.verbose=0
    mol.build()
    return mol

def tomo_mol():
    mol = gto.Mole()
    mol.atom=[
            ['H',(0,0,0)],
            ['H',(2.0,0,0)],
            ['H',(+2.0,1,0)],
            ['H',(-2.0,-1,0)],
            ['H',(+2.0,0,1)],
            ['H',(-2.0,0,-1)],
            ]
    mol.basis='sto-3g'
    mol.spin=0
    mol.verbose=0
    mol.build()
    return mol


def generic_molecular_hamiltonian():
    return MolecularHamiltonian(
            mol=generic_mol(),
            verbose=False,
            transform=JordanWigner)

def generic_fermionic_hamiltonian():
    mol = generic_mol()
    return FermionicHamiltonian(
            proxy_mol=mol,
            ints_1e=np.load('./store/ints_1e.npy'),
            ints_2e=np.load('./store/ints_2e.npy'),
            ints_spatial=False,
            transform=JordanWigner,
            verbose=False,
            Ne_active_space=2,
            No_active_space=2,
            en_con=mol.energy_nuc(),
            )

def generic_acse_storage():
    return StorageACSE(generic_molecular_hamiltonian())

def generic_quantumstorage():
    ham = generic_molecular_hamiltonian()
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=4,
            provider='Aer')
    return qs

def generic_quantumstorage():
    ham = generic_molecular_hamiltonian()
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=4,
            provider='Aer')
    return qs

def tomo_quantumstorage():
    ham =  MolecularHamiltonian(
            mol=tomo_mol(),
            generate_operators=False,
            transform=JordanWigner)
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=12,
            provider='Aer')
    return qs

def generic_acse_objects():
    ham = generic_molecular_hamiltonian()
    ins = PauliSet
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=4,
            provider='Aer')
    tomoRe = StandardTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=JordanWigner)
    tomoIm = StandardTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=JordanWigner)
    proc = StandardProcess()
    return ham,st,qs,ins,proc,tomoRe,tomoIm

def generic_vqe_objects():
    ham = generic_molecular_hamiltonian()
    ins = PauliSet
    st = StorageVQE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=4,
            provider='Aer')
    tomoRe = StandardTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=JordanWigner)
    proc = StandardProcess()
    return ham,st,qs,ins,proc,tomoRe


def advanced_mol():
    mol = gto.Mole()
    mol.atom=[
            ['H',(0,0,0)],
            ['H',(2.0,0,0)],
            ['H',(-2.0,0,0)],
            ]
    mol.basis='sto-3g'
    mol.spin=1
    mol.verbose=0
    mol.build()
    return mol

def expert_mol():
    mol = gto.Mole()
    mol.atom=[
            ['H',(0.0,0.0,0.0)],
            ['H',(2.0,0.0,0.0)],
            ['H',(0.0,2.0,0.0)],
            ['H',(2.0,2.0,0.0)],
            ]
    mol.basis='sto-3g'
    mol.spin=0
    mol.verbose=0
    mol.build()
    return mol

def advanced_acse_objects():
    ham =  MolecularHamiltonian(
            mol=advanced_mol(),
            transform=JordanWigner)
    ins = PauliSet
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=6,
            provider='Aer')
    tomoRe = ReducedTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=JordanWigner)
    tomoIm =ReducedTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=JordanWigner)
    proc = StandardProcess()
    return ham,st,qs,ins,proc,tomoRe,tomoIm

def expert_acse_objects():
    ham =  MolecularHamiltonian(
            mol=expert_mol(),
            transform=JordanWigner)
    ins = PauliSet
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=8,
            provider='Aer')
    tomoRe = ReducedTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=JordanWigner)
    tomoIm =ReducedTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=JordanWigner)
    proc = StandardProcess()
    return ham,st,qs,ins,proc,tomoRe,tomoIm


