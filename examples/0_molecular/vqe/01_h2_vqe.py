'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
from math import pi
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from hqca.vqe import *
from hqca.opts import *
from pyscf import gto
from functools import partial
from hqca.transforms import *
mol = gto.Mole()

d = 2.0
mol.atom=[['H',(0,0,0)],['H',(d,0,0)],['H',(-d,0,0)]]
mol.basis='sto-3g'
mol.spin=1
mol.verbose=0
mol.build()
ham = MolecularHamiltonian(mol,transform=JordanWigner)
ins = PauliSet
st = StorageVQE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=6,
        provider='Aer')
tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,
        transform=JordanWigner,
        )

opt = partial(Optimizer,
        optimizer='bfgs',
        unity=pi/2,
        verbose=True)

vqe = RunVQE(st,opt,qs,ins,
        ansatz='ucc',
        tomography=tomoRe,
        gradient=True,
        opt_thresh=1e-6,
        verbose=True,
        )
vqe.build()
vqe.run()
