'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
from math import pi
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.transforms import *
from hqca.vqe import *
from opts import *
from pyscf import gto
from functools import partial
mol = gto.Mole()

d = 2.0
mol.atom=[['H',(d*i,0,0)] for i in range(4)]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()

ins = PauliSet
Tr,iTr = parity_free(4,4,+1,+1,JordanWigner)

ham = MolecularHamiltonian(mol,transform=Tr)
st = StorageVQE(ham)

qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=6,
        provider='Aer')

qs.initial_transform=iTr
tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,transform=Tr
        )

opt = partial(Optimizer,
        optimizer='bfgs',
        unity=pi/2,
        verbose=True)

vqe = RunVQE(st,opt,qs,ins,
        ansatz='ucc',
        tomography=tomoRe,
        gradient=True,
        opt_thresh=1e-4,
        )
vqe.build()
vqe.run()

