from hqca.tools.quantum_strings._quantum_string import *
import numpy as np


#class PauliOperator:
#    '''
#    Simple Pauli operator
#    '''
#    def __init__(self,
#            pauli,
#            coeff,
#            add=True,
#            get='default',
#            symbolic=False,
#            ):
#        self.p = pauli
#        self.c = coeff
#        self.qCo = np.copy(coeff)
#        self.add=add
#        if get=='default':
#            self.g = pauli #get 
#        else:
#            self.g = get
#        self.norm = self.c*np.conj(self.c)
#        self.sym = symbolic
#
#    def isSame(self,a):
#        return self.p==a.p
#
#    def isHermitian(self,a):
#        return self.p==a.p
#
#    def clear(self):
#        pass
#
#    def __str__(self):
#        z = '{}: {}'.format(self.p,self.c)
#        return z

class QubitString:
    '''
    Fermionic operator without the antisymmetry requirements, and more general
    tomography options
    '''
    def __init__(self,
            coeff=1,
            indices=[0,1,2,3], #orbital indices
            sqOp='-+-+', #second quantized operators
            add=True,
            symbolic=False,
            ):
        self.c = coeff
        self.ind = indices
        self.sqOp = sqOp
        self.sym = symbolic
        self.norm = self.c*np.conj(self.c)
        self.order = len(indices)
        self.as_set = set(indices)
        self._qubit_order()
        self._simplify()
        self.add=add

    def hasSameInd(self,b):
        '''
        is equal indices
        '''
        if self.qInd==b.qInd:
            return True
        else:
            return False

    def _qubit_order(self):
        sort = False
        to = self.sqOp[:]
        ti = self.ind[:]
        order = self.order
        while not sort:
            sort=True
            for i in range(order-1):
                if ti[i]>ti[i+1]:
                    to = to[:i]+to[i+1]+to[i]+to[i+2:]
                    ti = ti[:i]+[ti[i+1]]+[ti[i]]+ti[i+2:]
                    sort=False
                    break
        self.qOp = to
        self.qInd = ti
        self.qCo = np.copy(self.c)

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
                            self.qOp=''
                            break
                        else:
                            key = simple[self.qOp[i:i+2]]
                            self.qOp = self.qOp[0:i]+key+self.qOp[i+2:]
                        self.order-=1
                        done=False
                        break

    def __str__(self):
        z = '{} ({}): {}'.format(self.qInd,self.qOp,self.c)
        return z

    def clear(self):
        self.pauliExp = []
        self.pauliGet = []
        self.pauliCoeff = []
        self.pauliGates = []

    def hasSameOp(self,b):
        return self.qOp==b.qOp

    def hasHermitianOp(self,b):
        # check if operators are hermitian
        # if all unique elements
        if len(b.qOp)==len(self.qOp):
            for l in range(len(b.qOp)):
                if b.qOp[l]=='p' and not self.qOp[l]=='h':
                    return False
                elif b.qOp[l]=='h' and not self.qOp[l]=='p':
                    return False
                elif b.qOp[l]=='+' and not self.qOp[l]=='-':
                    return False
                elif b.qOp[l]=='-' and not self.qOp[l]=='+':
                    return False
            return True
        else:
            return False

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

    def _jw_op_to_paulis(self):
        '''
        op is something like... +, -, +-, ++,0,1
        '''
        coeff = [self.c]
        pauli = ['']
        for o in self.qOp:
            p1 = []
            c1 = []
            p2 = []
            c2 = []
            for p,c in zip(pauli,coeff):
                if o=='+':
                    p1.append(p+'X')
                    c1.append(c*0.5)
                    p2.append(p+'Y')
                    c2.append(-1j*c*0.5)
                elif o=='-':
                    p1.append(p+'X')
                    c1.append(c*0.5)
                    p2.append(p+'Y')
                    c2.append(1j*c*0.5)
                elif o in ['1','p']:
                    p1.append(p+'I')
                    c1.append(c*0.5)
                    p2.append(p+'Z')
                    c2.append(-1*c*0.5)
                elif o in ['0','h']:
                    p1.append(p+'I')
                    c1.append(c*0.5)
                    p2.append(p+'Z')
                    c2.append(+1*c*0.5)
            pauli = p1+p2
            coeff = c1+c2
        self.pPauli = pauli
        self.pCoeff = coeff
        self._complex = [i.imag for i in self.pCoeff]
        self._real = [i.real for i in self.pCoeff]

    def _fill_out_paulis(self,Nq):
        for n,item in enumerate(self.pPauli):
            # item is a string
            c = 0
            new = ''
            for g,ind in zip(item,self.qInd):
                for i in range(c,ind):
                    new+='I'
                new+=g
                c = ind+1
            for i in range(c,Nq):
                new+='I'
            self.pPauli[n]=new


    def generateExponential(self,**kw):
        self.generateOperators(**kw)

    def generateTomography(self,**kw):
        self.generateOperators(**kw)

    def generateOperators(self,Nq,real=True,imag=True,mapping='jw',**kw):
        self._jw_op_to_paulis()
        if mapping=='jw':
            self._fill_out_paulis(Nq)
        try:
            for n in reversed(range(len(self.pPauli))):
                if not real:
                    if abs(self._complex[n])<1e-10:
                        self.pPauli.pop(n)
                        self.pCoeff.pop(n)
                elif not imag:
                    if abs(self._real[n])<1e-10:
                        self.pPauli.pop(n)
                        self.pCoeff.pop(n)
        except Exception:
            pass

    def formOperator(self,**kw):
        new = Operator()
        for p,c in zip(self.pPauli,self.pCoeff):
            new+=PauliOperator(p,c,add=self.add)
        return new
