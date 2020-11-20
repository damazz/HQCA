import numpy as np
from numpy import kron as k
from functools import reduce


'''
State class......similar to circuit class, but returns vectors instead of matrices

and

Density Matrix class
'''

class State:
    def __init__(self,size=3,symbolic=False):
        if symbolic:
            pass
            from sympy import cos,sin
        elif not symbolic:
            from numpy import cos,sin
        else:
            pass
        self.cos = cos
        self.sin = sin
        self.n = size
        self.N = 2**self.n
        self.b = ['{:0{}b}'.format(i,self.n)[::1] for i in range(0,self.N)]
        self.m = np.zeros((self.N,1),dtype=np.complex_)
        self.m[0,0]=1
        self._cN_basis()

    def __str__(self):
        z = ''
        for b,s in zip(self.b,np.asarray(self.m[:,0])):
            try:
                s = s[0]
            except IndexError:
                pass
            if abs(s)>1e-6:
                if abs(s.real)<1e-6:
                    z+= 'i{:+.8f} |{}> \n'.format(s.imag,b)
                elif abs(s.imag)<1e-6:
                    z+= ' {:+.8f} |{}> \n'.format(s.real,b)
                else:
                    z+= ' {:+.8f} + i{:.8f} |{}> \n'.format(s.real,s.imag,b)
        return z

    def _cN_basis(self):
        self.cNb = []
        self.cNi = []
        for i in range(0,self.n+1):
            for n,b in enumerate(self.b):
                temp = 0
                for s in b:
                    temp+=int(s)
                if temp==i:
                    self.cNb.append(b)
                    self.cNi.append(n)

    def __mul__(self,circ):
        temp = np.dot(circ.m,self.m).T
        N = temp.shape[1]
        real, imag,bas = '','',''
        lb = len(self.b[0])
        try:
            add = 0
            for k in range(N):
                real, imag,bas = '','',''
                for i,j in enumerate(np.asarray(temp).tolist()[0][k:k]):
                    #if np.count_nonzero(j)==1:
                    if abs(j)>1e-6:
                        add+=1 
                        real += '{:6.3f}   '.format(np.real(j))
                        imag += '{:6.3f}   '.format(np.imag(j))
                        bas  += '|{}>{:{}}'.format(self.b[i+k*16],'',7-lb)
                print(real)
                print(imag)
                print(bas)
                print('')
        except Exception as e:
            print(e)
        self.m = temp

class DensityMatrix:
    def __init__(self,
            state=None,
            size=2,
            ):
        if type(state)==type(None):
            self.n = size
            self.N = 2**self.n
            self.b = ['{:0{}b}'.format(i,self.n)[::1] for i in range(0,self.N)]
            self.m = np.zeros((self.N,self.N),
                    dtype=np.complex_)
        else:
            self.n = state.n
            self.N = state.N
            self.b = state.b
            self.m = np.outer(state.m,np.conj(state.m).T)
        self.gb = {k:v for v,k in enumerate(self.b)}
    
    @property
    def rho(self):
        return self.m

    @rho.setter
    def rho(self,new):
        self.m = new

    def _cN_basis(self):
        self.cNb = []
        self.cNi = []
        for i in range(0,self.n+1):
            for n,b in enumerate(self.b):
                temp = 0
                for s in b:
                    temp+=int(s)
                if temp==i:
                    self.cNb.append(b)
                    self.cNi.append(n)

    def partial_trace(self,qubits=[]):
        def reduce_str(b,q):
            for i in q[::-1]:
                b = b[:i]+b[i+1:]
            return b
        rn = self.n - len(qubits)
        new = DensityMatrix(size=rn)
        rb = [new.gb[reduce_str(b,qubits)] for b in self.b]
        for n,i in enumerate(self.b):
            for m,j in enumerate(self.b):
                use=True
                for q in qubits:
                    if not i[q]==j[q]:
                        use=False
                if use:
                    new.m[rb[n],rb[m]]+= self.m[n,m]
        return new

    def observable(self,circuit):
        return np.dot(self.m,circuit.m).trace()[0,0]

