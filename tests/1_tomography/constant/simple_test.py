'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
from hqca.hamiltonian import *
from functools import partial
from hqca.instructions import *
from hqca.state_tomography import *
from hqca.transforms import *
from hqca.acse import *
from pyscf import gto
import sys

d = 2.0
mol = gto.Mole()
mol.atom = [['H',(0,0,0)],['H',(d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
Q = 4
tr = JordanWigner

mol.build()
mol.verbose=0
ham = MolecularHamiltonian(mol,verbose=True,transform=tr)
Ins = PauliSet
st = StorageACSE(ham)
qs = QuantumStorage(verbose=False)
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        #backend='qasm_simulator',
        Nq=Q,
        provider='Aer')

tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,simplify=True,transform=tr,criteria='mc',rel='mc')

