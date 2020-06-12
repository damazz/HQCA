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


a = Circ(8)
Sp = Operator()
Sm = Operator()
Sz = Operator()

Sz+= FermiString(0.5,indices=[0],ops='p',N=6)
Sz+= FermiString(0.5,indices=[1],ops='p',N=6)
Sz+= FermiString(0.5,indices=[2],ops='p',N=6)
Sz+= FermiString(0.5,indices=[3],ops='p',N=6)
Sz+= FermiString(-0.5,indices=[4],ops='p',N=6)
Sz+= FermiString(-0.5,indices=[5],ops='p',N=6)
Sz2 = Sz*Sz

N = Operator()
N+= FermiString(1,indices=[0],ops='p',N=6)
N+= FermiString(1,indices=[1],ops='p',N=6)
N+= FermiString(1,indices=[2],ops='p',N=6)
N+= FermiString(1,indices=[3],ops='p',N=6)
N+= FermiString(1,indices=[4],ops='p',N=6)
N+= FermiString(1,indices=[5],ops='p',N=6)

qub_Sz = Sz.transform(JordanWigner)
qub_N = N.transform(JordanWigner)

op_Sz = QubitHamiltonian(qubits=6,operator='pauli',order=6,
        pauli=qub_Sz)
op_N  = QubitHamiltonian(qubits=6,operator='pauli',order=6,
        pauli=qub_N)
mat_Sz = op_Sz._matrix[0]
mat_N  = op_N._matrix[0]
qb = []
c = Circ(6)
c.m = mat_Sz
d = c.trace_operator(qb=qb)
e = Circ(6)
e.m = mat_N
f = e.trace_operator(qb=qb)

mat_Sz = d.m
mat_N = f.m
T = Circ(6)
T.x(0)
T.x(1)
T.x(2)
T.x(3)
T.x(4)
T.x(5)
#T.x(3)
#T.x(4)
#T.x(3)
A = np.zeros(T.m.shape,dtype=np.complex_)
w2,v2 = np.linalg.eig(mat_Sz)
w3,v3 = np.linalg.eig(mat_N)
print(w2)
print(w3)
for j in set(w2):
    for k in set(w3):
        b2 = np.zeros(T.m.shape,dtype=np.complex_)
        b3 = np.zeros(T.m.shape,dtype=np.complex_)
        for a in range(T.m.shape[0]):
            if w2[a]==j:
                b2+= np.outer(v2[:,a],v2[:,a])
            if w3[a]==k:
                b3+= np.outer(v3[:,a],v3[:,a])
        t =  np.zeros(T.m.shape)
        t+=1
        U = np.dot(b2,b3)
        '''
        if np.count_nonzero(U)>0:
            print(j,k)
            z = np.nonzero(U)
            for a,b in zip(z[0],z[1]):
                val = U[a,b]
                if abs(val)>1e-5:
                    print(val,'inds: ',a,b,'eigs: ',j,k)
            print('U')
            print(np.real(U))
        '''
        A+= np.dot(U,np.dot(T.m,np.transpose(U)))
print(np.real(A))
print(np.imag(A))
print(np.count_nonzero(A))

C = Circ(T.n)
C.m = A
print(C.get_cN())
print(C.cNb)
