'''
Example highlighting transformations of 

'''
from pyscf import gto,scf,mcscf
import sys
import numpy as np
from functools import reduce
from hqca.hamiltonian import *
from hqca.transforms import *
from hqca.tools import *
from hqca.acse import *

mol = gto.Mole()
d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)],
        ['H',(0,d,0)],
        ]
mol.basis='sto-3g'
mol.spin=1
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol,
        transform=JordanWigner,
        )
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=6,
        provider='Aer')
tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,simplify=True,
        transform=JordanWigner,
        )

par = ParityCheckMatrix(ham._qubOp,verbose=True)
par.gaussian_elimination()

