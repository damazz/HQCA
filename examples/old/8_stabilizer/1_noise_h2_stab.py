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
import qiskit.providers.aer.noise as noise
from deconstruct import *

# build mol objects


## ## ## ## ## ## ## ## ## ## ## ## ## ##

class UCC_Instruct:
    def __init__(self,c1,c2=0,c3=0):
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
                ]
'''
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
                [(3,c3),apply_rz],
                [(2,3,),apply_cx],
                [(0,),apply_h],
                [(1,),apply_h],
                [(1,),apply_s],
                [(2,),apply_h],
                [(3,),apply_h],
                [(3,),apply_s],
'''
#################################
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



n = 0
# IDEAL
#while n<25:
#    n+=1
#    c1 = np.random.random()*np.pi/2-np.pi/4
#    c2 = np.random.random()*np.pi/2-np.pi/4
#    c3 = np.random.random()*np.pi/2-np.pi/4
#    coeffs.append([c1,c2,c3])
coeffs = np.linspace(-np.pi/2,np.pi/2,25)
#errors = [0.0,0.25,0.5,1,2,4]
errors = [0.0]
dat = np.zeros((len(coeffs),4,16,16),dtype=np.complex_)
dat_avg = np.zeros((len(errors),15))
##for ne,error in enumerate(errors):
#    print('--=--=--=--=--=--=--=--=--')
#    print('Error Strength: {}'.format(error))
#    print('--=--=--=--=--=--=--=--=--')
#    nm = generateNoiseModel(error)
mol = gto.Mole()
mol.atom =[
        ['H',(0,0,0,)],['H',(1.0,0,0)]]
mol.basis = 'sto-3g'
mol.spin=0
mol.verbose=0
mol.build()
N = []
eig = []
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
        #backend='ibmq_5_yorktown',
        #backend='ibmq_5_yorktown',
        backend='qasm_simulator',
        backend_initial_layout=[1,0,2,3],
        transpiler_keywords={'optimization_level':0},
        num_shots=8192,Nq=Nq,
        #provider='IBMQ'
        )
#qs.set_noise_model(custom=True,noise_model=nm)
qs0 = QuantumStorage()
qs0.set_algorithm(st)
qs0.set_backend(
        backend_initial_layout=[0,1,2,3],
        transpiler_keywords={'optimization_level':0},
        num_shots=8192,Nq=Nq,provider='Aer'
        )
for nc, coeff in enumerate(coeffs):
    tomo0 = ReducedTomography(qs0,verbose=False)
    tomo0.generate(real=True,imag=True,transform=JordanWigner,verbose=False)
    ins = UCC_Instruct(coeff)
    proc = StandardProcess()
    tomo0.set(ins)
    tomo0.simulate()
    tomo0.construct(procesor=proc)
    t01,t02,t03,t04,t05 = tomo0.rdm.analysis()
    # 
    tomo1 = ReducedTomography(qs,verbose=False)
    tomo1.generate(real=True,imag=True,transform=JordanWigner,verbose=False)
    # 
    tomo1.set(ins)
    tomo1.simulate()
    tomo1.construct(procesor=proc)
    t11,t12,t13,t14,t15 = tomo1.rdm.analysis()
    # 
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
    t21,t22,t23,t24,t25 = tomo2.rdm.analysis()


    tomo3 = ReducedTomography(qs,method='local')
    tomo3.generate(real=True,imag=True,transform=JordanWigner,verbose=False)
    proc_z = StabilizerProcess(stabilizer='filter_diagonal')
    tomo3.set(ins)
    tomo3.simulate()
    tomo3.construct(processor=proc_z)
    t31,t32,t33,t34,t35 = tomo3.rdm.analysis()
    #
    tomo0.rdm.contract()
    tomo1.rdm.contract()
    tomo2.rdm.contract()
    tomo3.rdm.contract()
    dat[nc,0,:,:]=tomo0.rdm.rdm
    dat[nc,1,:,:]=tomo1.rdm.rdm
    dat[nc,2,:,:]=tomo2.rdm.rdm
    dat[nc,3,:,:]=tomo3.rdm.rdm

np.save('yorktown_scan',dat)
'''
        e0d = np.linalg.norm(np.sort(t03)-np.sort(np.array([0,0,0,1])))
        e1d = np.linalg.norm(np.sort(t13)-np.sort(np.array([0,0,0,1])))
        e2d = np.linalg.norm(np.sort(t23)-np.sort(np.array([0,0,0,1])))
        def d1s(rdm):
            rdm1 = rdm.reduce_order()
            es = np.sort(np.linalg.eigvalsh(rdm1.rdm))
            a1 = 0.5*(es[0]+es[1])
            a2 = 0.5*(es[2]+es[3])
            vec = np.zeros(4)
            vec[0]=es[0]-a1
            vec[1]=es[1]-a1
            vec[2]=es[2]-a2
            vec[3]=es[3]-a2
            return np.linalg.norm(vec)

        c0 = tomo0.rdm.cumulant()
        c1 = tomo1.rdm.cumulant()
        c2 = tomo2.rdm.cumulant()

        h0 = tomo0.rdm-c0
        h1 = tomo1.rdm-c1
        h2 = tomo2.rdm-c2


        d0 = tomo0.rdm
        d1 = tomo1.rdm
        d2 = tomo2.rdm
        for d in [d0,d1,d2,c0,c1,c2,h0,h1,h2]:
            d.contract()
        D = [d0,d1,d2]
        C = [c0,c1,c2]
        H = [h0,h1,h2]
        for nn,(i,j) in enumerate([[0,1],[0,2],[1,2]]):
            dat[ne,nc,3*nn+0]=np.linalg.norm(D[i].rdm-D[j].rdm)
            dat[ne,nc,3*nn+1]=np.linalg.norm(C[i].rdm-C[j].rdm)
            dat[ne,nc,3*nn+2]=np.linalg.norm(H[i].rdm-H[j].rdm)
            print('Distance between {},{}'.format(i,j))
            print(dat[ne,nc,3*nn])
            print('Error in Cumulant Matrix:')
            print(dat[ne,nc,3*nn+1])
            print('Error in 1-Body Matrix:')
            print(dat[ne,nc,3*nn+2])
        dat[ne,nc,9 ]=e0d
        dat[ne,nc,10]=e1d
        dat[ne,nc,11]=e2d
        dat[ne,nc,12]=d1s(d0)
        dat[ne,nc,13]=d1s(d1)
        dat[ne,nc,14]=d1s(d2)
    for i in range(dat.shape[2]):
        dat_avg[ne,i] = np.average(dat[ne,:,i])
    #print(dat[ne,:,:])
print(dat_avg)
'''

