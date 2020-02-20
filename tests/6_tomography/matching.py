'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.tools.fermions import *
from hqca.acse import *
from pyscf import gto
import sys

d = 2.0
molecules = [
        [
            ['H',(0,0,0)],
            ['H',(0,1,0)],
            ['H',(1,0,0)],
            ['H',(-1,0,0)],
            ['H',(1,1,0)]],
            ['H',(1,1,0)]],
        ]
qubits = [
        12,
        ]
spins = [
        0,
        ]
for atoms,Q,S in zip(molecules,qubits,spins):
    mol = gto.Mole()
    mol.atom=atoms
    mol.basis='sto-3g'
    mol.spin=S
    mol.verbose=0
    mol.build()
    ham = MolecularHamiltonian(mol,verbose=False)
    Ins = PauliSet
    st = StorageACSE(ham)
    qs = QuantumStorage(verbose=False)
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            #backend='qasm_simulator',
            Nq=Q,
            provider='Aer')
    print('###############')
    #for maps in ['jw','parity','bk']:
    for maps in ['bk']:
        print('Mapping: {}'.format(maps))
        if maps=='bk':
            bkSet = BravyiKitaevSet(Q,reduced=True,Ne=[0,0])
        else:
            bkSet=None

        tomoRe = ReducedTomography(qs)
        tomoRe.generate(real=True,imag=False,simplify=False,mapping=maps,bkSet=bkSet)
        print('Reduced Tomography for {} Qubits:'.format(Q))
        print('Naive: {}'.format(len(tomoRe.op)))

        tomoRe.generate(real=True,imag=False,simplify=True,mapping=maps,bkSet=bkSet)
        print('Grouped: {}'.format(len(tomoRe.op)))
        print(tomoRe.op)

        tomoRe = ReducedTomography(qs,match_aa_bb=True)
        tomoRe.generate(real=True,imag=False,simplify=False,mapping=maps,bkSet=bkSet)
        print('Reduced Tomography with matching for {} Qubits:'.format(Q))
        print('Naive: {}'.format(len(tomoRe.op)))

        tomoRe.generate(real=True,imag=False,simplify=True,mapping=maps,bkSet=bkSet)
        print('Grouped: {}'.format(len(tomoRe.op)))
        print(tomoRe.op)

