from hqca.operators.quantum_strings._quantum_string import *
import numpy as np
from copy import deepcopy as copy
import sys



class PauliString(QuantumString):
    '''
    Simple Pauli operator
    '''
    def __init__(self,
            pauli='I',
            coeff=1,
            add=True,
            symbolic=False,
            vec=None,
            ):
        if type(vec)==type(None):
            self.s = pauli #string
            self.c = coeff
            self.sym = symbolic
            self.n = len(pauli)
            self.l = np.array([int(pauli[i]=='X' or pauli[i]=='Y') for i in range(self.n)])
            self.r = np.array([int(pauli[i]=='Z' or pauli[i]=='Y') for i in range(self.n)])
            self.p = 1j**np.sum((self.l&self.r)) #phase associated with symplectic representation
        else:
            self.l = vec[0]
            self.r = vec[1]
            self.p = 1j**np.sum((self.l&self.r))
            self.c = coeff
            self.n = len(vec[0])
            self.s = self._construct()
            self.sym = symbolic

    #@property
    #def s(self):
    #    return self._construct()

    def _construct(self):
        key = {
                '00':'I',
                '10':'X',
                '01':'Z',
                '11':'Y',
                }
        s = [key[''.join([str(self.l[i]),str(self.r[i])])] for i in range(self.n)]
        return ''.join(s)


    def __hash__(self):
        return hash(self.s)

    def __eq__(self,obj):
        #return not (self.l!=obj.l).any() and not (self.r!=obj.r).any()
        return self.s==obj.s

    def __ne__(self,obj):
        return self==obj and self.c==obj.c

    def symm(self,P):
        val = 0
        for i in range(self.n): # for ab, cd
            val+= self.r[i]*P.r[i+self.n] #  ad
            val+= P.r[i]*P.r[i+self.n] # cd
            val+= P.r[i]*self.r[i+self.n] # cb
            val+= self.r[i]*self.r[i+self.n] # ad
        return -val

    def skew(self,P):
        val = 0
        for i in range(self.n):
            val+= self.r[i]*P.r[i+self.n]-P.r[i]*self.r[i+self.n]
        return val

    def __mul__(self,P):
        '''
        multiplication done with symplectic definition
        '''
        if isinstance(P,float) or isinstance(P,int):
            return PauliString(vec=[self.l,self.r],coeff=self.c*P)
        l = self.l^P.l  #mod 2 addition
        r = self.r^P.r
        L = np.add(self.l,P.l) #mod 4 quantity
        R = np.add(self.r,P.r)
        phase = (1j)**np.sum(np.subtract(self.r&P.l, self.l&P.r))
        phase*= (1j)**(np.sum(np.subtract(np.multiply(L,R),l&r)))
        '''
        for i in range(self.n):
            t = self.s[i]+P.s[i]
            if t in ['XY','YZ','ZX']:
                ph*=1j
            elif t in ['YX','ZY','XZ']:
                ph*=-1j
        c = ph*self.c*P.c
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
        if self.sym or P.sym:
            sym=True
        else:
            sym=False
        '''
        #return PauliString(st,c,symbolic=sym)
        return PauliString(vec=[l,r],coeff=self.c*P.c*phase,symbolic=P.sym or self.sym)

    def __add__(self,P):
        new = copy(self)
        if P==self:
            new.c += P.c 
        else:
            sys.exit('Why are you adding these operators?')
        return new



    def __str__(self):
        try:
            if abs(self.c.real)>1e-10:
                z = '{}: {:.8f}'.format(self.s,self.c.real)
            else:
                z = '{}: {:+.8f}j'.format(self.s,self.c.imag)
        except Exception:
            z = '{}: {}'.format(self.s,self.c)
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


