from hqca.tools import *
from hqca.hamiltonian import *
from hqca.transforms import *
import numpy as np
import sys


#a = Circ(3)
#a.h(0)
#a.Cx(0,1)
#a.h(1)
#b = a.trace_operator(qb=[2,1])
#print(a.m)
#print(b.m)
#sys.exit()


a = Circ(6)
Sp = Operator()
Sm = Operator()
Sz = Operator()
A = Operator()
B = Operator()

Sz+= FermiString(0.5,indices=[0],ops='p',N=8)
Sz+= FermiString(0.5,indices=[1],ops='p',N=8)
Sz+= FermiString(0.5,indices=[2],ops='p',N=8)
Sz+= FermiString(0.5,indices=[3],ops='p',N=8)
#Sz+= FermiString(0.5,indices=[4],ops='p',N=10)
Sz+= FermiString(-0.5,indices=[5],ops='p',N=8)
Sz+= FermiString(-0.5,indices=[6],ops='p',N=8)
Sz+= FermiString(-0.5,indices=[7],ops='p',N=8)
Sz+= FermiString(-0.5,indices=[4],ops='p',N=8)
#Sz+= FermiString(-0.5,indices=[9],ops='p',N=10)
Sz2 = Sz*Sz

Sp+= FermiString(1,indices=[4,0],ops='+-',N=8)
Sp+= FermiString(1,indices=[5,1],ops='+-',N=8)
Sp+= FermiString(1,indices=[6,2],ops='+-',N=8)
Sp+= FermiString(1,indices=[7,3],ops='+-',N=8)
#Sp+= FermiString(1,indices=[9,4],ops='+-',N=8)
Sm+= FermiString(1,indices=[0,4],ops='+-',N=8)
Sm+= FermiString(1,indices=[1,5],ops='+-',N=8)
Sm+= FermiString(1,indices=[2,6],ops='+-',N=8)
Sm+= FermiString(1,indices=[3,7],ops='+-',N=8)
#Sm+= FermiString(1,indices=[4,9],ops='+-',N=8)

N = Operator()
N+= FermiString(1,indices=[0],ops='p',N=8)
N+= FermiString(1,indices=[1],ops='p',N=8)
N+= FermiString(1,indices=[2],ops='p',N=8)
N+= FermiString(1,indices=[3],ops='p',N=8)
#N+= FermiString(1,indices=[4],ops='p',N=8)
N+= FermiString(1,indices=[4],ops='p',N=8)
N+= FermiString(1,indices=[5],ops='p',N=8)
N+= FermiString(1,indices=[6],ops='p',N=8)
N+= FermiString(1,indices=[7],ops='p',N=8)
#N+= FermiString(1,indices=[9],ops='p',N=8)

new = Sp*Sm+Sz2-Sz
qub_Sz2 = new.transform(JordanWigner)
qub_Sz = Sz.transform(JordanWigner)
qub_N = N.transform(JordanWigner)

op_S2 = QubitHamiltonian(qubits=8,operator='pauli',order=8,
        pauli=qub_Sz2)
op_Sz = QubitHamiltonian(qubits=8,operator='pauli',order=8,
        pauli=qub_Sz)
op_N  = QubitHamiltonian(qubits=8,operator='pauli',order=8,
        pauli=qub_N)
mat_S2 = op_S2._matrix[0]
mat_Sz = op_Sz._matrix[0]
mat_N  = op_N._matrix[0]
qb = [1,2,3,4,7]
a = Circ(8)
a.m = mat_S2
b = a.trace_operator(qb=qb)
c = Circ(8)
c.m = mat_Sz
d = c.trace_operator(qb=qb)
e = Circ(8)
e.m = mat_N
f = e.trace_operator(qb=qb)

mat_S2 = b.m
mat_Sz = d.m
mat_N = f.m
T = Circ(3)
T.z(0)
T.x(1)
T.x(2)
#T.x(4)
#T.x(5)
#T.x(3)
A = np.zeros(T.m.shape,dtype=np.complex_)
w1,v1 = np.linalg.eig(mat_S2)
w2,v2 = np.linalg.eig(mat_Sz)
w3,v3 = np.linalg.eig(mat_N)
print(w1)
print(w2)
print(w3)
temp1 = np.zeros(T.m.shape,dtype=np.complex_)
for i in set(w1):
    b1 = np.zeros(T.m.shape,dtype=np.complex_)
    for a in range(T.m.shape[0]):
        if w1[a]==i:
            b1+= np.outer(v1[:,a],v1[:,a])
    temp1+=  np.dot(b1,np.dot(T.m,b1))


temp2 = np.zeros(T.m.shape,dtype=np.complex_)
for j in set(w2):
    b2 = np.zeros(T.m.shape,dtype=np.complex_)
    for a in range(T.m.shape[0]):
        if w2[a]==j:
            b2+= np.outer(v2[:,a],v2[:,a])
    temp2+=  np.dot(b2,np.dot(temp1,b2))

temp3 = np.zeros(T.m.shape,dtype=np.complex_)
for k in set(w3):
    b3 = np.zeros(T.m.shape,dtype=np.complex_)
    for a in range(T.m.shape[0]):
        if w3[a]==k:
            b3+= np.outer(v3[:,a],v3[:,a])
    temp3+=  np.dot(b3,np.dot(temp2,b3))

print(np.real(temp3))
print(np.imag(temp3))
print(np.count_nonzero(temp3))

C = Circ(T.n)
C.m = temp3
print(C.get_cN())
print(C.cNb)
