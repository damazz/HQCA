from hqca.tools.quantum_strings._quantum_string import *
import numpy as np
from copy import deepcopy as copy

class PauliString(QuantumString):
    '''
    Simple Pauli operator
    '''
    def __init__(self,
            pauli='I',
            coeff=1,
            add=True,
            get='default',
            symbolic=False,
            ):
        self.s = pauli #string
        self.c = coeff
        self.qCo = np.copy(coeff)
        self.add=add
        if get=='default':
            self.g = pauli #get 
        else:
            self.g = get
        self.norm = self.c*np.conj(self.c)
        self.sym = symbolic
        self.n = len(self.s)
        r1 = [int(self.s[i]=='X' or self.s[i]=='Y') for i in range(self.n)]
        r2 = [int(self.s[i]=='Z' or self.s[i]=='Y') for i in range(self.n)]
        self.r = r1+r2
        self.shi = 1j**(sum([r1[i]*r2[i] for i in range(self.n)]))

    def __eq__(self,obj):
        return self.s==obj.s

    def __ne__(self,obj):
        return self.s==obj.s and self.c==obj.c

    def symm(self,P):
        val = 0
        for i in range(self.n):
            val+= self.r[i]*P.r[i+self.n]
            val+= P.r[i]*P.r[i+self.n]
            val+= P.r[i]*self.r[i+self.n]
            val+= self.r[i]*self.r[i+self.n]
        return -val

    def skew(self,P):
        val = 0
        for i in range(self.n):
            val+= self.r[i]*P.r[i+self.n]-P.r[i]*self.r[i+self.n]
        return val


    def __mul__(self,P):
        ph = 1
        for i in range(self.n):
            t = self.s[i]+P.s[i]
            if t in ['XY','YZ','ZX']:
                ph*=1j
            elif t in ['YX','ZY','XZ']:
                ph*=-1j
        c = ph*self.c*P.c
        pauli = [(self.r[i]+P.r[i])%2 for i in range(2*self.n)]
        st =''
        for i in range(self.n):
            if pauli[i] and pauli[i+self.n]:
                st+='Y'
            elif pauli[i]:
                st+='X'
            elif pauli[i+self.n]:
                st+='Z'
            else:
                st+='I'
        return PauliString(st,c)

    def __add__(self,P):
        new = copy(self)
        if P==self:
            new.c += P.c 
        else:
            sys.exit('Why are you adding these operators?')
        return new

    def comm(self,P):
        c = 0
        for i in range(self.n):
            c+= self.r[i]*P.r[i+self.n]
            c+= self.r[i+self.n]*P.r[i]
        return (c+1)%2


    def isSame(self,a):
        return self.s==a.p

    def isHermitian(self,a):
        return self.s==a.p

    def clear(self):
        pass

    def __str__(self):
        if abs(self.c.real)>1e-10:
            z = '{}: {:.8f}'.format(self.s,self.c.real)
        else:
            z = '{}: {:+.8f}j'.format(self.s,self.c.imag)
        return z

    def partial_trace(self,qb=[0]):
        qb = sorted(qb)[::-1]
        s = ''
        c = copy(self.c)
        for q in reversed(range(len(self.s))):
            if q in qb:
                if not self.s[q]=='I':
                    c*= 0
            else:
                s+= self.s[q]
        return PauliString(pauli=s[::-1],coeff=c)





