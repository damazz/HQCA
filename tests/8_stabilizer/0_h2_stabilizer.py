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
from hqca.acse import *
from math import pi
from hqca.instructions import *
import pickle
np.set_printoptions(suppress=True,precision=4)

# build mol objects

mol = gto.Mole()
mol.atom =[
        ['H',(0,0,0,)],['H',(1.0,0,0)]]
mol.basis = 'sto-3g'
mol.spin=0
mol.verbose=0
mol.build()

# set up hqca run with quantumstorage

Nq = 4
ham = MolecularHamiltonian(mol,
        int_thresh=1e-5,
        transform=JordanWigner,
        verbose=False,
        )
st = StorageACSE(ham,
        )
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='ibmq_5_yorktown',
        backend_initial_layout=[2,0,3,4],
        transpiler_keywords={'optimization_level':0},
        num_shots=8192,Nq=Nq,provider='IBMQ'
        )
qs0 = QuantumStorage()
qs0.set_algorithm(st)
qs0.set_backend(
        backend_initial_layout=[0,1,2,3],
        transpiler_keywords={'optimization_level':0},
        num_shots=8192,Nq=Nq,provider='Aer'
        )

## ## ## ## ## ## ## ## ## ## ## ## ## ##

class UCC_Instruct:
    def __init__(self,c1,c2):
        self.gates = [
                [(0,),apply_h],
                [(1,),apply_h],
                [(2,),apply_h],
                [(3,),apply_si],
                [(3,),apply_h],
                [(1,0,),apply_cx],
                [(0,2,),apply_cx],
                [(2,3,),apply_cx],
                [(3,c1),apply_rz],
                [(2,3,),apply_cx],
                [(0,2,),apply_cx],
                [(1,0,),apply_cx],
                [(0,),apply_h],
                [(1,),apply_h],
                [(2,),apply_h],
                [(3,),apply_h],
                [(3,),apply_s],
                #
                [(1,),apply_si],
                [(0,),apply_h],
                [(1,),apply_h],
                [(2,),apply_h],
                [(3,),apply_si],
                [(3,),apply_h],
                [(1,0,),apply_cx],
                [(0,c2),apply_rz],
                [(1,0,),apply_cx],
                [(2,3,),apply_cx],
                [(3,c2),apply_rz],
                [(2,3,),apply_cx],
                [(0,),apply_h],
                [(1,),apply_h],
                [(1,),apply_s],
                [(2,),apply_h],
                [(3,),apply_h],
                [(3,),apply_s],
                ]

#################################

#
# IDEAL

tomo0 = ReducedTomography(qs0,verbose=False)
tomo0.generate(real=True,imag=True,transform=JordanWigner,verbose=False)
ins = UCC_Instruct(np.pi/4,np.pi/6)
proc = StandardProcess()
tomo0.set(ins)
tomo0.simulate()
tomo0.construct(procesor=proc)
tomo0.rdm.analysis()



tomo1 = ReducedTomography(qs,verbose=False)
tomo1.generate(real=True,imag=True,transform=JordanWigner,verbose=False)

tomo1.set(ins)
tomo1.simulate()
tomo1.construct(procesor=proc)
tomo1.rdm.analysis()

check = tomo1.build_stabilizer()
qs.set_error_mitigation(mitigation='MSES',
        stabilizer_map=check)

tomo2 = ReducedTomography(qs,method='stabilizer',
        preset=True,
        Tomo=tomo1)

proc = StabilizerProcess(stabilizer='encoded')
tomo2.set(ins)
tomo2.simulate()
tomo2.construct(processor=proc)
tomo2.rdm.analysis()
#
def split_matrix(rdm):
    N = rdm.rdm.shape[0]
    R = int(np.sqrt(N))
    nn = np.zeros(rdm.rdm.shape,dtype=np.complex_)
    ne = np.zeros(rdm.rdm.shape,dtype=np.complex_)
    ee = np.zeros(rdm.rdm.shape,dtype=np.complex_)
    for i in range(N):
        p,r = i//R,i%R
        for j in range(N):
            q,s = j//R,j%R
            ind = tuple([p,q,r,s])
            if len(set(ind))==2:
                nn[i,j]=rdm.rdm[i,j]
            elif len(set(ind))==3:
                ne[i,j]=rdm.rdm[i,j]
            elif len(set(ind))==4:
                ee[i,j]=rdm.rdm[i,j]
    return nn,ne,ee

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
