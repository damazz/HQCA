'''
Noiseless run of the meta-benzyne isomer
'''

from pyscf import gto
import numpy as np
from math import pi,sqrt
np.set_printoptions(suppress=True,precision=8)
from copy import deepcopy as copy
import sys
from functools import reduce,partial
from hqca.hamiltonian import FermionicHamiltonian
from hqca.tools import Operator as Op
from hqca.tools import PauliString as Pauli
from hqca.transforms import JordanWigner
from hqca.acse import RunACSE,StorageACSE,QuantumStorage,PauliSet,ReducedTomography

# example run for meta-isomer

# load EIs 
ei1 = np.load('EI1.npy')
ei2 = np.load('EI2.npy')

mol = gto.Mole()
mol.atom =[['H',(i,0,0)] for i in range(4)]
mol.build()
e_fci = -3.291031448342
#
# apply symmetries to generate qubit-tapered state
#
c = 1/sqrt(2)
U1 = Op()+Pauli('IIIIIIIX',c)+Pauli('IIIIZIIZ',c)
U2 = Op()+Pauli('IIIIIIX',c)+Pauli('ZZIIZIZ',c)
U3 = Op()+Pauli('IIIIIX',c)+Pauli('ZZIIZZ',c)
U4 = Op()+Pauli('IIIXI',c)+Pauli('ZIIZI',c)
U5 = Op()+Pauli('IIXI',c)+Pauli('IZZI',c)
Us = [U1,U2,U3,U4,U5]
tap_qb = [7,6,5,3,2]
symm_eval = [-1,-1,+1,-1,-1]

def modify(
        ops,fermi,U,Ut,
        qubits,paulis,eigvals,
        initial=False,
        ):
    '''
    Return modified fermionic transformation. 
    '''
    new = fermi(ops,initial=initial)
    new = change_basis(new,U,Ut)
    new = trim_operator(new,
            qubits=qubits,
            paulis=paulis,
            null=int(initial),
            eigvals=eigvals)
    return new
#
tran = JordanWigner
for qb,ev,U in zip(tap_qb,symm_eval,Us):
    tran = partial(
            modify,
            fermi=copy(tran),
            U=U,Ut=U,
            qubits=[qb],
            paulis=['X'],
            eigvals=[ev])
tr_init = partial(tran,initial=True)
#
# calculate molecular Hamiltonian from EIs and tapered transformation
#
ham = FermionicHamiltonian(mol,
        ints_1e=ei1,ints_2e=ei2,normalize=False,
        transform=tran,verbose=True,en_fin=e_fci,
        )
#
# initialize remaining components and run qACSE 
Ins = PauliSet
st = StorageACSE(ham,closed_ansatz=-1)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=3,
        num_shots=8192,
        provider='Aer')
qs.initial_transform = tr_init
proc = StandardProcess()
#
tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,transform=tran)
tomoIm = ReducedTomography(qs)
tomoIm.generate(real=False,imag=True,transform=tran)

acse = RunACSE(
        st,qs,Ins,method='newton',update='quantum',
        hamiltonian_step_size=0.5,opt_thresh=0.01,
        S_thresh_rel=0.1,S_min=1e-6,convergence_type='norm',
        processor=proc,use_trust_region=True,max_iter=10,
        newton_step=-1,restrict_S_size=1.5,initial_trust_region=1,
        tomo_Psi = tomoRe,tomo_S = tomoIm,
        )
acse.build()
acse.run()
