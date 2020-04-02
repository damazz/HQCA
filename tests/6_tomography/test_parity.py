'''
Molecular test case of H2, and H3, with under the Newton optimization with the
quantum ACSE method.
'''
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.acse import *
from hqca.tools.fermions import *
from pyscf import gto

d = 2.0
molecules = [
        #[['H',(0,0,0)],['H',(d,0,0)]],
        #[['H',(0,0,0)],['H',(d,0,0)],['H',(-d,0,0)]],
        [['H',(0,0,0)],['H',(0,1,0)],['H',(1,0,0)],['H',(1,1,0)]],
        #[['H',(0,0,0)],['H',(0,1,0)],['H',(1,0,0)],['H',(1,1,0)],['H',(0,-1,0)]],
        #[['H',(0,0,0)],['H',(0,1,0)],['H',(1,0,0)],['H',(1,1,0)],['H',(0,-1,0)],['H',(-1,-1,0)]],
        ]
qubits = [
        #4,
        #6,
        8,
        #10,
        #12
        ]
spins = [
        #0,
        #1,
        0,
        #1,
        #0
        ]
for atoms,Q,S in zip(molecules,qubits,spins):
    mol = gto.Mole()
    mol.atom=atoms
    mol.basis='sto-3g'
    mol.spin=S
    mol.verbose=0
    mol.build()
    ham = MolecularHamiltonian(mol,verbose=False,generate_operators=False,
            mapping='parity',)
    Ins = PauliSet
    st = StorageACSE(ham)
    MapSet = ParitySet(Q,
            Ne=[mol.nelec[0]%2,(mol.nelec[0]+mol.nelec[1])%2],
            reduced=True)
    qs = QuantumStorage(verbose=False)
    qs.set_algorithm(st)
    qs.set_backend(
            backend='statevector_simulator',
            #backend='qasm_simulator',
            Nq=Q-2,
            provider='Aer')
    print('###############')
    #print('Default Tomography for {} Qubits:'.format(Q))
    #tomoRe = StandardTomography(qs)
    #tomoRe.generate(real=True,imag=False,simplify=False,mapping='parity')
    ##print(tomoRe.op)
    #print('Naive: {}'.format(len(tomoRe.op)))
    #tomoRe.generate(real=True,imag=False,simplify=True,mapping='parity')
    ##print(tomoRe.op)
    #print('Grouped: {}'.format(len(tomoRe.op)))

    tomoRe = ReducedTomography(qs)
    tomoRe.generate(
            real=True,imag=False,
            simplify=False,
            mapping='parity',MapSet=MapSet)
    print('Reduced Tomography for {} Qubits:'.format(Q))
    print('Naive: {}'.format(len(tomoRe.op)))
    #print(tomoRe.op)
    tomoRe.generate(
            real=True,imag=False,
            simplify=True,
            mapping='parity',
            weight=['I'],MapSet=MapSet)
    #print(tomoRe.op)
    print('Grouped: {}'.format(len(tomoRe.op)))

