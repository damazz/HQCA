'''
Second example calculation from:

    Smart, S. E., & Mazziotti, D. A. (2021). Lowering tomography costs in quantum simulation 
    with a symmetry projected operator basis. Physical Review A, 103(1), 012420. 
    https://doi.org/10.1103/PhysRevA.103.012420

Here we are simuatling a noisy quantum system using a tunable noise model provided from an actual quantum device, and comparing the tomography of the 2-RDM under the default and symmetry projected techniques with the ideal 2-RDM.

'''

import numpy as np
import sys
from math import pi
import qiskit.providers.aer.noise as noise
from noise_model.deconstruct import *
from hqca.hamiltonian import *
from hqca.instructions import *
from hqca.processes import *
from hqca.acse import *
from hqca.core import *
from hqca.core.primitives import *
from pyscf import gto
from hqca.transforms import *
from functools import partial
from hqca.tools import *
from hqca.state_tomography import *
np.set_printoptions(precision=3)
import qiskit

class Ins(Instructions):
    def __init__(self,coeff):
        self._gates =[[(coeff,),self._test]]

    def _test(self,Q,coeff):
        Q.si(0)
        Q.Cx(1,0)
        Q.Cx(2,1)
        Q.Cx(3,2)
        Q.Rx(3,coeff[0])
        Q.Rx(1,coeff[1])
        Q.Cx(3,2)
        Q.Cx(2,1)
        Q.Cx(1,0)
        Q.Cx(3,2)
        Q.Ry(3,coeff[2])
        Q.Cx(3,2)
        Q.s(0)

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a

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
# generate mol object
mol = gto.Mole()
mol.atom=[['H',(0,0,0)],['H',(2.0,0,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.build()
N = []
eig = []
norm = []
ham = MolecularHamiltonian(mol,transform=JordanWigner)
st = StorageACSE(ham)
qs = QuantumStorage()
qs0 = QuantumStorage()
pr = StandardProcess()
qs0.set_algorithm(st)

# set Nq, number of shots, and error strength

Nq = 4
Ns = 8192
error = 0.0


# qs0, ideal
# qs, noisy simulated 
qs0.set_backend(
        backend='statevector_simulator',
        Nq=Nq,
        Nq_ancilla=0,
        num_shots=Ns,
        provider='Aer')
qs.set_algorithm(st)
# can specify provider='IBMQ' and an appropriate backend if desired
qs.set_backend(
        backend='qasm_simulator',
        Nq=Nq,
        num_shots=Ns,
        provider='Aer')
nm = model_v2(scaling=error,name='./noise_model/110220_ibmq_bogota')
qs.set_noise_model(custom=True,
        noise_model=nm)
tomo = []
tomo_sim = []
coefficients = np.load('./noise_model/coefficients.npy')
# runs the tomography in sets of 5...suited for particular constraints on quantum device access
# but can be easily modified
for q in range(5):
    coeffs = coefficients[q*5:q*5+5,:]
    for coeff in coeffs:
        print(coeff)
        # run 1
        tomo0 = StandardTomography(qs0,verbose=False)
        tomo0.generate(real=True,imag=True,
                simplify=True,transform=JordanWigner,
                method='gt',strategy='lf')
        ins0 = Ins(coeff)
        tomo0.set(ins0)

        tomo1 = StandardTomography(qs,verbose=False)
        tomo1.generate(real=True,imag=True,
                simplify=True,transform=JordanWigner,
                method='gt',strategy='lf')
        ins = Ins(coeff)
        tomo1.set(ins)
    
        tomo2 = ReducedTomography(qs,verbose=False)
        tomo2.generate(real=True,imag=True,
                simplify=True,transform=JordanWigner,
                method='gt',strategy='lf')
        ins = Ins(coeff)
        tomo2.set(ins)
        tomo_sim.append(tomo0)
        tomo.append(tomo1)
        tomo.append(tomo2)
    run_multiple(tomo[q*10:(q*10+10)],qs)
    run_multiple(tomo_sim[q*5:(q*5+5)],qs0)
for item in tomo:
    print(item.counts['ZZZZ'])
print('Constructing..')
for t in tomo:
    t.construct(processor=pr)
for t in tomo_sim:
    t.construct(processor=pr)
for i in range(len(coefficients)):
    print(coefficients[i,:])
    tomo0 = tomo_sim[i]
    tomo1 = tomo[i*2]
    tomo2 = tomo[i*2+1]
    st.analysis(tomo0.rdm)
    st.analysis(tomo1.rdm)
    st.analysis(tomo2.rdm)
    tomo0.rdm.contract()
    tomo1.rdm.contract()
    tomo2.rdm.contract()
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
    print('Difference D0-D1: {}'.format(N01))
    print('Difference D0-D2: {}'.format(N02))
    print('Difference D1-D2: {}'.format(N12))
    norm.append([N01,N02,N12])

print('--- --- --- --- --- ---')
print('Frombenius norm of D01, D02, and D12 for each run')
norm = np.asmatrix(norm)
print(norm)
print('--- --- --- --- --- ---')
print('     average  (std dev)')
for i,l in zip(range(norm.shape[1]),['D01','D02','D12']):
    print('{}: {:.6f} {:.6f}'.format(l,np.average(norm[:,i]),np.std(norm[:,i])))


