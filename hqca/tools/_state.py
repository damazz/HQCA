import numpy as np
from numpy import kron as k
from functools import reduce

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
        self.m = np.zeros((self.N,1))
        self.m[0,0]=1
        self._cN_basis()

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
        for k in range(0,N//16+1):
            real, imag,bas = '','',''
            for i,j in enumerate(np.asarray(temp).tolist()[0][k*16:k*16+16]):
                #if np.count_nonzero(j)==1:
                real += '{:6.3f}   '.format(np.real(j))
                imag += '{:6.3f}   '.format(np.imag(j))
                bas  += '|{}>{:{}}'.format(self.b[i+k*16],'',7-lb)
            if real=='':
                break
            print(real)
            print(imag)
            print(bas)
            print('')
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

