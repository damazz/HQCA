
from pyscf import gto
import numpy as np
from hqca.hamiltonian import *
from hqca.opts import *
from hqca.transforms import *
from hqca.acse import *
from hqca.tools import *
from hqca.instructions import *
from hqca.storage import *
from hqca.core import *
from hqca.vqe import *
from hqca.core.primitives import *

def generic_mol():
    mol = gto.Mole()
    mol.atom=[['H',(0,0,0)],['H',(2.0,0,0)]]
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

def large_mol():
    mol = gto.Mole()
    mol.atom=[
            ['H',(0,0,0)],
            ['H',(2.0,0,0)],
            ['H',(-2.0,0,0)],
            ['H',(-4.0,0,0)],
            ]
    mol.basis='sto-3g'
    mol.spin=0
    mol.verbose=0
    mol.build()
    return mol

def generic_acse_storage():
    return StorageACSE(generic_molecular_hamiltonian())

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
    tomoD3 = ReducedTomography(qs,order=3)
    tomoD3.generate(real=True,imag=False,transform=JordanWigner)
    proc = StandardProcess()
    return ham,st,qs,ins,proc,tomoRe,tomoIm,tomoD3

def expert_acse_objects():
    Tr,iTr = parity_free(4,4,1,1,JordanWigner)
    ham =  MolecularHamiltonian(
            mol=expert_mol(),
            transform=Tr)
    ins = PauliSet
    st = StorageACSE(ham)
    qs = QuantumStorage()
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=6,
            provider='Aer')
    qs.initial_transform = iTr
    tomoRe = StandardTomography(qs)
    tomoRe.generate(real=True,imag=False,transform=Tr)
    tomoIm =StandardTomography(qs)
    tomoIm.generate(real=False,imag=True,transform=Tr)
    tomoD3 = StandardTomography(qs,order=3)
    tomoD3.generate(real=True,imag=False,transform=Tr)
    proc = StandardProcess()
    return ham,st,qs,ins,proc,tomoRe,tomoIm,tomoD3


