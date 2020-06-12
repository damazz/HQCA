import numpy as np
from copy import deepcopy as copy
from hqca.tools.quantum_strings._quantum_string import *
from sympy import symbols
from sympy import re,im

class FermiString(QuantumString):
    '''
    Class of operators, obeys simple fermionic statistics.

    Also used to generate raw lists of Pauli strings and subterms, which can be
    compiled for circuits and tomography.

    has a key word defining the type 
    '''
    def __init__(self,
            coeff=1,
            s=None, # string
            indices=[0,1,2,3], #orbital indices
            ops='-+-+', #second quantized operators
            N=10,
            antisymmetric=True,
            add=True,
            symbolic=False,
            ):
        self.fermi = antisymmetric
        if type(s)==type(None):
            self._generate_from_sq(coeff,inds=indices,sqop=ops,N=N)
        else:
            # generate from string
            self.s = string
            self.c = coeff
        self.sym = symbolic
        self.add = add

    def __str__(self):
        s = ''.join([j for j in self.s if not j=='i'])
        i = ''.join([str(n) for n in range(len(self.s)) if not self.s[n]=='i'])
        #z = '{} {} {} '.format(self.c,s,i)
        z = '{}: {}'.format(self.s,self.c)
        return z

    def __len__(self):
        return len(self.ops())

    def __add__(self,A):
        new = copy(self)
        if A==self:
            new.c += A.c
        else:
            sys.exit('What are you adding these operators?')
        return new

    def __eq__(self,A):
        return A.s==self.s

    def inds(self):
        return [n for n in range(len(self.s)) if not self.s[n]=='i']

    def ops(self):
        return  [j for j in self.s if not j=='i']

    def N(self):
        return len(self.s)

    def __mul__(self,A):
        s = ''
        ind = []
        for j in [self,A]:
            for i in range(len(j.s)):
                if not j.s[i]=='i':
                    s+= j.s[i]
                    ind.append(i)
        #print(s,ind,self,A)
        return FermiString(
                coeff=self.c*A.c,
                indices=ind,
                ops=s,
                antisymmetric=self.fermi,
                add=self.add,
                N=max(len(self.s),len(A.s)),
                symbolic=self.sym)

    def _generate_from_sq(self,coeff=1,inds=[],sqop='',N=10):
        sort = False
        while not sort:
            sort=True
            for i in range(len(sqop)-1):
                if inds[i]>inds[i+1]:
                    if self.fermi:
                        if sqop[i] in ['p','h']:
                            pass
                        elif sqop[i+1] in ['p','h']:
                            pass
                        else:
                            coeff*=-1
                    inds = inds[:i]+[inds[i+1]]+[inds[i]]+inds[i+2:]
                    sqop = sqop[:i]+sqop[i+1]+sqop[i]+sqop[i+2:]
                    sort=False
                    break
        #print(sqop,inds)
        if len(set(inds))==len(sqop):
            pass
        else:
            zeros = ['h+','-h','p-','+p','++','--','ph','hp']
            simple = {
                    'h-':'-','+h':'+','p+':'+','-p':'-',
                    'pp':'p','hh':'h','+-':'p','-+':'h',}
            done = False
            while not done:
                done=True
                for i in range(len(inds)-1):
                    if inds[i]==inds[i+1]:
                        inds.pop(i+1)
                        if sqop[i:i+2] in zeros:
                            coeff=0
                            sqop= ''
                            break
                        else:
                            key = simple[sqop[i:i+2]]
                            sqop = sqop[0:i]+key+sqop[i+2:]
                        done=False
                        break
        if coeff==0:
            s = 'i'*N
        else:
            s = 'i'*inds[0]
            #print(sqop,inds,s)
            for i in range(len(inds)-1):
                s+= sqop[i]
                s+='i'*(inds[i+1]-inds[i]-1)
            s+= sqop[-1]+'i'*(N-inds[-1]-1)
        self.s = s
        self.c = coeff



    def hasSameInd(self,b):
        '''
        is equal indices
        '''
        if self.qInd==b.qInd:
            return True
        else:
            return False


    def hasSameOp(self,b):
        return self.qOp==b.qOp

    def hasHermitianOp(self,b):
        # if all unique elements
        if len(b.qOp)==len(set(b.qOp)):
            for l in range(len(b.qOp)):
                if b.qOp[l]==self.qOp[l]:
                    return False
        else:
            N = []
            for l in range(len(b.qOp)-1):
                if b.qInd[l]==b.qInd[l+1]:
                    N.append(l)
                    N.append(l+1)
            for l in range(len(b.qOp)):
                if l in N:
                    pass
                else:
                    if b.qOp[l]==self.qOp[l]:
                        return False
        return True

    def isSame(self,b):
        if self.hasSameInd(b) and self.hasSameOp(b):
            return True
        else:
            return False

    def isHermitian(self,b):
        if self.hasSameInd(b) and self.hasHermitianOp(b):
            return True
        else:
            return False


    def _simplify(self):
        '''
        check if there are any similar indices  and simplify to number operator
        '''
        zeros = [
                'h+','-h','p-','+p',
                '++','--','ph','hp']
        simple = {
                'h-':'-',
                '+h':'+',
                'p+':'+',
                '-p':'-',
                'pp':'p',
                'hh':'h',
                '+-':'p',
                '-+':'h',
                }
        if len(set(self.qInd))==len(self.qInd):
            pass
        else:
            done = False
            while not done:
                done=True
                for i in range(self.order-1):
                    if self.qInd[i]==self.qInd[i+1]:
                        self.qInd.pop(i+1)
                        if self.qOp[i:i+2] in zeros:
                            self.c=0
                            self.qCo=0
                            self.qOp=''
                            break
                        else:
                            key = simple[self.qOp[i:i+2]]
                            self.qOp = self.qOp[0:i]+key+self.qOp[i+2:]
                        self.order-=1
                        done=False
                        break


    ##
    #
    # Hermitian Excitation Operators, such as in exp(iHt)
    #
    ##


    def generateTomography(self,**kw):
        self.generateOperators(**kw)

    def generateOperators(self,
            Nq,
            real=True,
            imag=True,
            mapping='jw',
            **kw):
        if mapping in ['jw','jordan-wigner']:
            self.pPauli,self.pCoeff = JordanWignerTransform(
                    self,Nq,**kw)
        elif mapping=='parity':
            self.pPauli,self.pCoeff = ParityTransform(
                    self,Nq,**kw)
        elif mapping in ['bravyi-kitaev','bk']:
            self.pPauli,self.pCoeff = BravyiKitaevTransform(
                    self,Nq,**kw)
        else:
            print('Incorrect mapping: {}. Goodbye!'.format(mapping))
        if self.sym:
            self._complex = [im(i) for i in self.pCoeff]
            self._real = [re(i) for i in self.pCoeff]
            pass
        else:
            self._complex  = [i.imag for i in self.pCoeff]
            self._real = [i.real for i in self.pCoeff]

    def _commutator_relations(self,lp,rp):
        if rp=='I':
            return 1,lp
        elif rp==lp:
            return 1,'I'
        elif rp=='Z':
            if lp=='X':
                return -1j,'Y'
            elif lp=='Y':
                return 1j,'X'
        elif rp=='Y':
            if lp=='X':
                return 1j,'Z'
            elif lp=='Z':
                return -1j,'X'
        elif rp=='X':
            if lp=='Y':
                return -1j,'Z'
            elif lp=='Z':
                return 1j,'Y'
        elif rp=='h':
            if lp=='Z':
                return 1,'h'
        elif rp=='p':
            if lp=='Z':
                return -1,'p'
        else:
            sys.exit('Incorrect paulis: {}, {}'.format(lp,rp))

    def formOperator(self):
        new = Operator()
        for p,c in zip(self.pPauli,self.pCoeff):
            new+=PauliOperator(p,c,add=self.add,symbolic=self.sym)
        return new

