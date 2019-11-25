from hqca.hamiltonian import *
from hqca.instructions import *
from pyscf import gto

mol = gto.Mole()
d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol)
ins = PauliSet




