'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.transforms import *
from hqca.acse import *
from pyscf import gto
import sys
import timeit

d = 2.0
molecules = [
        [['H',(j,0,0)] for j in range(2)],
        [['H',(j,0,0)] for j in range(3)],
        [['H',(j,0,0)] for j in range(4)],
        [['H',(j,0,0)] for j in range(5)],
        [['H',(j,0,0)] for j in range(6)],
        ]
qubits = [4,6,8,10,12]
spins = [
        0,1,0,1,0,1,0,1,
        ]
for atoms,Q,S in zip(molecules,qubits,spins):
    mol = gto.Mole()
    mol.atom=atoms
    mol.basis='sto-3g'
    mol.spin=S
    mol.verbose=0
    mol.build()
    print('Starting Hamiltonian...')
    t1 = timeit.default_timer()
    ham = MolecularHamiltonian(mol,verbose=False,generate_operators=False)
    t2 = timeit.default_timer()
    print('Time for hamiltonian: {:.2f}'.format(t2-t1))
    Ins = PauliSet
    st = StorageACSE(ham)
    qs = QuantumStorage(verbose=False)
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            Nq=Q,
            provider='Aer')
    print('###############')
    #for maps in ['bk']:
    T = JordanWigner
    print('Standard Tomography for {} Qubits'.format(Q))
    tomoRe = StandardTomography(qs)
    tomoRe.generate(
            real=True,imag=False,
            simplify=True,
            transform=T,verbose=True,
            rel='qwc',
            )
    print('Projected Tomography for {} Qubits'.format(Q))
    tomoRe = ReducedTomography(qs)
    tomoRe.generate(
            real=True,imag=False,
            transform=T,verbose=True,
            rel='qwc',
            )
