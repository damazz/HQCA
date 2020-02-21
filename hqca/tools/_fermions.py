import numpy as np
from hqca.tools._operator import *
from hqca.tools._qubit_operator import PauliOperator
from hqca.tools.fermions import *

class FermionicOperator:
    '''
    Class of operators, obeys simple fermionic statistics.

    Also used to generate raw lists of Pauli strings and subterms, which can be
    compiled for circuits and tomography.
    '''
    def __init__(self,
            coeff=1,
            indices=[0,1,2,3], #orbital indices
            sqOp='-+-+', #second quantized operators
            spin='abba',  # spin, for help.
            antisymmetric=True,
            add=True,
            ):
        self.c =coeff
        self.fermi = antisymmetric
        self.ind =indices
        self.op = sqOp
        self.add = add
        self.sp = spin
        self.norm = self.c*np.conj(self.c)
        self.order = len(sqOp)
        self.as_set = set(indices)
        self._qubit_order()
        self._simplify()
        self._classify()

    #def __str__(self):
    #    z = '{}, {}, {}: {}'.format(self.ind,self.op,self.sp,self.qCo)
    #    return z

    def __str__(self):
        z1 = '{}, {}, {}: {}, '.format(self.ind,self.op,self.sp,self.qCo)
        z2 = '<< {},{} >>'.format(self.qInd,self.qOp)
        return z1+z2

    def hasSameInd(self,b):
        '''
        is equal indices
        '''
        if self.qInd==b.qInd:
            return True
        else:
            return False

    def clear(self):
        self.pPauli = []
        self.pGet = []
        self.pCoeff = []

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

    def _classify(self):
        if self.order%2==0:
            l = len(set(self.ind))
            if l==1 and self.order==2:
                self.opType = 'no' #number operator
            elif l==2 and self.order==2:
                self.opType = 'se' # single excitation
            elif l==2 and self.order==4:
                self.opType = 'nn' # double number operator
            elif l==3 and self.order==4:
                self.opType = 'ne' # number-excitation operator
            elif l==4 and self.order==4:
                self.opType = 'de' # double excitation operator
            else:
                self.opType = 'none'
            self.rdm = self.order//2
            Nb,Na = 0,0
            for i in self.sp:
                if i=='a':
                    Na+=1 
                elif i=='b':
                    Nb+=1
            if Na-Nb==0:
                self.spBlock = 'ab'
            elif Na-Nb==-1:
                self.spBlock = 'b'
            elif Na-Nb==1:
                self.spBlock = 'a'
            elif Na-Nb==-2:
                self.spBlock = 'bb'
            elif Na-Nb==2:
                self.spBlock = 'aa'
        else:
            pass

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

    def _qubit_order(self):
        sort = False
        self.no = self.ind[:]
        to = self.op[:]
        ts = self.sp[:]
        ti = self.ind[:]
        tc = self.c
        order = self.order
        while not sort:
            sort=True
            for i in range(order-1):
                if ti[i]>ti[i+1]:
                    if self.fermi:
                       tc*=-1
                    to = to[:i]+to[i+1]+to[i]+to[i+2:]
                    ti = ti[:i]+[ti[i+1]]+[ti[i]]+ti[i+2:]
                    ts = ts[:i]+ts[i+1]+ts[i]+ts[i+2:]
                    sort=False
                    break
        self.qOp = to
        self.qInd = ti
        self.qSp = ts
        self.qCo = tc
        self.q2rdm = tc

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
                    self,Nq)
        elif mapping=='parity':
            self.pPauli,self.pCoeff = ParityTransform(
                    self,Nq,**kw)
        elif mapping in ['bravyi-kitaev','bk']:
            self.pPauli,self.pCoeff = BravyiKitaevTransform(
                    self,Nq,**kw)
        else:
            print('Incorrect mapping: {}. Goodbye!'.format(mapping))
        self._complex  = [i.imag for i in self.pCoeff]
        self._real = [i.real for i in self.pCoeff]
        for n in reversed(range(len(self.pPauli))):
            if not real:
                if abs(self._complex[n])<1e-10:
                    self.pPauli.pop(n)
                    self.pCoeff.pop(n)
            elif not imag:
                if abs(self._real[n])<1e-10:
                    self.pPauli.pop(n)
                    self.pCoeff.pop(n)

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
            new+=PauliOperator(p,c,add=self.add)
        return new
