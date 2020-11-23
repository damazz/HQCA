from hqca.tools import *
from hqca.hamiltonian import *
from hqca.transforms import *
import numpy as np

a = Circ(4)
Sp = Operator()
Sm = Operator()
Sz = Operator()
A = Operator()
B = Operator()

Sz+= FermiString(0.5,indices=[0],ops='p',N=4)
Sz+= FermiString(0.5,indices=[1],ops='p',N=4)
Sz+= FermiString(-0.5,indices=[2],ops='p',N=4)
Sz+= FermiString(-0.5,indices=[3],ops='p',N=4)
Sz2 = Sz*Sz

Sp+= FermiString(1,indices=[2,0],ops='+-',N=4)
Sp+= FermiString(1,indices=[3,1],ops='+-',N=4)
Sm+= FermiString(1,indices=[0,2],ops='+-',N=4)
Sm+= FermiString(1,indices=[1,3],ops='+-',N=4)


N = Operator()
N+= FermiString(1,indices=[0],ops='p',N=4)
N+= FermiString(1,indices=[1],ops='p',N=4)
N+= FermiString(1,indices=[2],ops='p',N=4)
N+= FermiString(1,indices=[3],ops='p',N=4)

new = Sp*Sm+Sz2-Sz
qub_Sz2 = new.transform(JordanWigner)
qub_Sz = Sz.transform(JordanWigner)
qub_N = N.transform(JordanWigner)

op_S2 = QubitHamiltonian(qubits=4,operator='pauli',order=4,
        pauli=qub_Sz2)
op_Sz = QubitHamiltonian(qubits=4,operator='pauli',order=4,
        pauli=qub_Sz)
op_N  = QubitHamiltonian(qubits=4,operator='pauli',order=4,
        pauli=qub_N)
mat_S2 = op_S2._matrix[0]
mat_Sz = op_Sz._matrix[0]
mat_N  = op_N._matrix[0]

A = np.zeros((16,16),dtype=np.complex_)
eigs = [[0,0.75,2],[-1.0,-0.5,0,0.5,1],[0,1,2,3,4]]
w1,v1 = np.linalg.eig(mat_S2)
w2,v2 = np.linalg.eig(mat_Sz)
w3,v3 = np.linalg.eig(mat_N)
print(w1)
print(w2)
print(w3)
for i in eigs[0]:
    for j in eigs[1]:
        for k in eigs[2]:
            b1 = np.zeros((16,16),dtype=np.complex_)
            b2 = np.zeros((16,16),dtype=np.complex_)
            b3 = np.zeros((16,16),dtype=np.complex_)
            for a in range(16):
                if w1[a]==i:
                    b1+= np.outer(v1[:,a],v1[:,a])
                if w2[a]==j:
                    b2+= np.outer(v2[:,a],v2[:,a])
                if w3[a]==k:
                    b3+= np.outer(v3[:,a],v3[:,a])
            t = np.zeros((16,16),dtype=np.complex_)
            t+=1
            U = np.dot(b1,np.dot(b2,b3))
            A+= np.dot(U,np.dot(t,np.transpose(U)))
print(np.real(A))

C = Circ(4)
C.m = A
print(np.real(C.get_cN()))
print(C.cNb)
