'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.tools.fermions import *
from hqca.acse import *
from pyscf import gto
from math import log2
import sys

d = 2.0
molecules = [
        #[['H',(0,0,0)],['H',(d,0,0)]],
        #[['H',(0,0,0)],['H',(d,0,0)],['H',(-d,0,0)]],
        #[['H',(0,0,0)],['H',(0,1,0)],['H',(1,0,0)],['H',(1,1,0)]],
        [['H',(0,0,0)],['H',(0,1,0)],['H',(1,0,0)],['H',(1,1,0)],['H',(0,-1,0)]],
        #[['H',(0,0,0)],['H',(0,1,0)],['H',(1,0,0)],['H',(1,1,0)],['H',(0,-1,0)],['H',(-1,-1,0)]],
        ]
qubits = [
        #4,
        #6,
        #8,
        10,
        #12
        ]
spins = [
        #0,
        #1,
        #0,
        1,
        #0
        ]
for atoms,Q,S in zip(molecules,qubits,spins):
    mol = gto.Mole()
    mol.atom=atoms
    mol.basis='sto-3g'
    mol.spin=S
    mol.verbose=0
    mol.build()
    print('Initiating')
    ham = MolecularHamiltonian(mol,verbose=False)
    print('Done with Hamiltonian')
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
    bkSet = BravyiKitaevSet(qs)
    for maps in ['jw','parity','bk']:
        tomoRe = StandardTomography(qs)
        print('* * * * *')
        print('Mapping: {}'.format(maps))
        print('* * * * *')
        print('-------')
        print('Default Tomography for {} Qubits:'.format(Q))
        print('-------')
        tomoRe.generate(real=True,imag=False,simplify=True,rel='mc',mapping=maps,bkSet=bkSet,verbose=True)
        print('MC Grouped: {}'.format(len(tomoRe.op)))
        print('-------')
        print('Reduced Tomography for {} Qubits:'.format(Q))
        print('-------')
        tomoRe = ReducedTomography(qs)
        tomoRe.generate(real=True,imag=False,simplify=True,mapping=maps,rel='mc',bkSet=bkSet,weight=['I'],verbose=True)
        print('MC Grouped: {}'.format(len(tomoRe.op)))
        tomoRe.generate(real=True,imag=False,simplify=True,mapping=maps,rel='mc',bkSet=bkSet,weight=['I'],verbose=True,criteria='mc')
        print('MC Rel Grouped: {}'.format(len(tomoRe.op)))
        #print(tomoRe.op)

