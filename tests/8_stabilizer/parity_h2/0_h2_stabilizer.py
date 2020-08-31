'''
Simple molecular example demonstrating how to used MSES, or the method of
symmetry encoded stabilizers
'''

from pyscf import gto,scf,mcscf
import sys
import numpy as np
from functools import reduce,partial
from hqca.hamiltonian import *
from hqca.core.primitives import *
from hqca.tools import *
from hqca.transforms import *
from hqca.processes import *
from hqca.processes import *
from hqca.acse import *
from math import pi
from hqca.instructions import *
import pickle
np.set_printoptions(suppress=True,precision=4)

# build mol objects

def mod(ops,fermi,U,qubits,paulis,eigvals,initial=False):
    new = fermi(ops,initial=initial)
    new = change_basis(new,U,U)
    new = trim_operator(new,
            qubits=qubits,
            paulis=paulis,
            null=int(initial),
            eigvals=eigvals)
    return new

def clifford(ops,fermi,U,**kw):
    new = fermi(ops,**kw)
    return new.clifford(U)

U1 = Operator()
U2 = Operator()

x1 = 'IIIX'
z1 = 'IIIZ'

x2 = 'IXI'
z2 = 'IZI'
Nq = 2
c = 1/np.sqrt(2)
U1+= PauliString(x1,c)
U1+= PauliString(z1,c)
U2+= PauliString(x2,c)
U2+= PauliString(z2,c)
tr0 = partial(mod,
        fermi=Parity,
        U=U1,
        qubits=[3],
        paulis=['X'],
        eigvals=[+1],
        )
tr1 = partial(mod,
        fermi=tr0,
        U=U2,
        qubits=[1],
        paulis=['X'],
        eigvals=[-1],
        )
mol = gto.Mole()
mol.atom=[['H',(0,0,0)],['H',(0.73482,0,0)],
#mol.atom=[['H',(0,0,0)],['H',(3.0,0,0)],
        ]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
#tr = Parity
tr = tr1
tr_init = partial(tr,initial=True)
ham = MolecularHamiltonian(mol,
        transform=tr,
        )

#qham = QubitHamiltonian(operator='pauli',
#        op=ham._qubOp,
#        qubits=4,order=4)
#print(qham._matrix)
#print(np.linalg.eigvalsh(qham._matrix[0])+mol.energy_nuc())
#sys.exit()
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='qasm_simulator',
        #backend='ibmq_5_yorktown',
        backend_initial_layout=[0,1],
        transpiler_keywords={'optimization_level':0},
        num_shots=8192,Nq=Nq,
        #provider='IBMQ',
        provider='Aer',
        )
qs0 = QuantumStorage()
qs0.set_algorithm(st)
qs0.set_backend(
        backend_initial_layout=[0,1],
        transpiler_keywords={'optimization_level':0},
        num_shots=8192,Nq=Nq,provider='Aer'
        )

## ## ## ## ## ## ## ## ## ## ## ## ##
class UCC_Instruct:
    def __init__(self,c1,c2,c3):
        self.gates = [
                [(1,),apply_h],
                [(1,0,),apply_cx],
                [(0,c1),apply_ry],
                [(1,c1),apply_ry],
                [(1,0,),apply_cx],
                [(1,),apply_h],
                [(0,c2),apply_ry],
                [(1,-c3),apply_ry],
                ]
#
# IDEAL
# 
tomo0 = ReducedTomography(qs0,verbose=False)
tomo0.generate(real=True,imag=True,transform=tr,verbose=False)
ins = UCC_Instruct(np.pi/4,np.pi/6,np.pi/6)
proc = StandardProcess()
tomo0.set(ins)
tomo0.simulate()
tomo0.construct(procesor=proc)
tomo0.rdm.analysis()


print('Tomo1')
tomo1 = ReducedTomography(qs,verbose=False)
tomo1.generate(real=True,imag=True,transform=tr,verbose=False)

tomo1.set(ins)
tomo1.simulate()
tomo1.construct(procesor=proc)
tomo1.rdm.analysis()

check = tomo1.build_stabilizer()
qs.set_error_mitigation(mitigation='MSES',
        stabilizer_map=check)

print('Tomo2')
tomo2 = ReducedTomography(qs,method='stabilizer',
        preset=True,
        Tomo=tomo1)

proc = StabilizerProcess(stabilizer='encoded')
tomo2.set(ins)
tomo2.simulate()
tomo2.construct(processor=proc)
tomo2.rdm.analysis()

print('Tomo3')
tomo3 = ReducedTomography(qs,method='local')
tomo3.generate(real=True,imag=True,transform=tr,verbose=False)
proc_z = StabilizerProcess(stabilizer='filter_diagonal')
tomo3.set(ins)
tomo3.simulate()
tomo3.construct(processor=proc_z)
tomo3.rdm.analysis()

norm = []
N = []
eig = []

e0 = np.linalg.eigvalsh(tomo0.rdm.rdm)
e1 = np.linalg.eigvalsh(tomo1.rdm.rdm)
e2 = np.linalg.eigvalsh(tomo2.rdm.rdm)
d01 = tomo0.rdm-tomo1.rdm
d02 = tomo0.rdm-tomo2.rdm
d12 = tomo1.rdm-tomo2.rdm
d01.contract()
d12.contract()
d02.contract()
N01 = np.linalg.norm(d01.rdm,ord='fro') 
N02 = np.linalg.norm(d02.rdm,ord='fro')
N12 = np.linalg.norm(d12.rdm,ord='fro')
'''
for l,d in zip(['01','02','12'],[d01,d02,d12]):
    print('Distance between {}'.format(l))
    n0 = np.linalg.norm(d.rdm,ord='fro')
    print(n0)
    nn,ne,ee = split_matrix(d)
    t1 = np.nonzero(nn)
    t2 = np.nonzero(ne)
    t3 = np.nonzero(ee)
    n1 = np.linalg.norm(nn,ord='fro')
    n2 = np.linalg.norm(ne,ord='fro')
    n3 = np.linalg.norm(ee,ord='fro')
    print('Number operator')
    print(np.linalg.norm(nn,ord='fro'))
    print('Number excitation')
    print(np.linalg.norm(ne,ord='fro'))
    print('Double excitation')
    print(np.linalg.norm(ee,ord='fro'))
    #print(np.real(ee))
    #print(np.imag(ee))
    if l=='12':
        norm.append([N01,N02,N12,n1,n2,n3])
'''
