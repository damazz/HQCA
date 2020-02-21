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
import timeit

d = 2.0
molecules = [
        #[
        #    ['H',(0,0,0)],
        #    ['H',(0,1,0)],
        #    ],
        [
            #['H',(1,0,0)],
            #['H',(1,1,0)],
            ['H',(1,-1,0)],
            ['H',(-1,-1,0)],
            ['H',(2,1,0)],
            ['H',(1,3,0)],
            ['H',(4,1,0)],
            ['H',(6,1,0)],
            ['H',(3,1,0)],
            ]
        ]
qubits = [14]
spins = [
        1,
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
    ham = MolecularHamiltonian(mol,verbose=False,operators=False)
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
    #for maps in ['jw','parity','bk']:
    for maps in ['jw']:
        print('Mapping: {}'.format(maps))
        if maps=='bk':
            MapSet = BravyiKitaevSet(Q,reduced=False,Ne=[0,0])
        elif maps=='parity':
            MapSet = ParitySet(Q,reduced=False,Ne=[0,0])
        else:
            MapSet=None
        #tomoRe = StandardTomography(qs)
        #print('Default Tomography for {} Qubits:'.format(Q))
        #tomoRe.generate(
        #    real=True,imag=False,simplify=False,
        #    mapping=maps,bkSet=bkSet,verbose=True)
        #print('Naive: {}'.format(len(tomoRe.op)))
        #tomoRe.generate(
        #    real=True,imag=False,verbose=True,
        #    simplify=True,mapping=maps,bkSet=bkSet)
        #print('Grouped, QWC: {}'.format(len(tomoRe.op)))
        tomoRe = ReducedTomography(qs)
        print('Randomly sampled sets....')
        tomoRe.generate(
                real=True,imag=False,simplify=True,
                mapping=maps,bkSet=bkSet,verbose=True,
                weight=['I'],rel='qwc',stochastic=True,
                threshold=0.2,

                )
        print('Grouped, QWC: {}'.format(len(tomoRe.op)))

        tomoRe.generate(
            real=True,imag=False,verbose=True,
            simplify=False,mapping=maps,MapSet=MapSet)
        print('Reduced Tomography for {} Qubits:'.format(Q))
        print('Naive: {}'.format(len(tomoRe.op)))
        tomoRe.generate(
                real=True,imag=False,simplify=True,
                mapping=maps,MapSet=MapSet,verbose=True,
                weight=['I'],rel='qwc',
                )
        print('Grouped, QWC: {}'.format(len(tomoRe.op)))

