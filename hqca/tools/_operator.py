from copy import deepcopy as copy
import sys
import traceback

class Operator:
    def __init__(self,
            ops = [],
            antihermitian=True,
            ):
        self._op = ops
        self.ah=antihermitian
        self.sym = False

    def reordering(self,method='magnitude',**kw):
        if method=='magnitude':
            l = []
            def func(n):
                s =self._op[n].c
                return abs(s)

            for n in range(len(self._op)):
                l.append(n)
            l = sorted(l,reverse=True,key=func)
            new = Operator()
            for n in l:
                new+= self._op[n]
            self._op = new.op
        elif method=='hamiltonian':
            self._hamiltonian_ordering(**kw)
        
    def _hamiltonian_ordering(self,qubOpH=None):
        keys = {}
        ind = 1
        for n,aOp in enumerate(self._op):
            for m,bOp in enumerate(qubOpH.op):
                if aOp.isSame(bOp):
                    keys[aOp.p]=m
                    break
            ind+=1
        def f(op):
            return keys[op.p]
        self._op = sorted(self._op,key=f)

    def _switch_ops(i,j):
        pass

    def _update(self):
        pass
    
    def __str__(self):
        z = ''
        for i in self._op:
            z += i.__str__()
            z += '\n'
        return z[:-1]

    def __add__(self,A):
        Old = copy(self._op)
        ''')
        Commutative addition
        '''
        try:
            if A.sym:
                self.sym=True
        except Exception:
            pass
        for old_op in Old:
            old_op.clear()
        try:
            for new in A._op:
                new_op = True
                for old in Old:
                    if old.isSame(new) and old.add:
                        try:
                            old.qCo = new.qCo+old.qCo
                        except Exception:
                            pass
                        old.c  = new.c+old.c
                        new_op = False
                        break
                if new_op:
                    Old.append(new)
        except AttributeError:
            new_op =True
            for old in Old:
                if old.isSame(A) and old.add:
                    try:
                        old.qCo = A.qCo+old.qCo
                    except Exception:
                        pass
                    old.c  = A.c+old.c
                    new_op = False
                    break
            if new_op:
                Old.append(A)
        except Exception:
            traceback.print_exc()
        New = Operator(
                ops=Old,
                antihermitian=self.ah)
        return New

    def clean(self):
        done = False
        while not done:
            done=True
            for i in range(len(self._op)):
                try:
                    if abs(self._op[i].c)<1e-12:
                        self._op.pop(i)
                        done=False
                        break
                except TypeError:
                    if self.op==0:
                        self._op.pop(i)
                        done=False
                        break


    def generateSkewExpOp(self):
        for items in self._op:
            items.generateAntiHermitianExcitationOperators()

    def generateHermExpOp(self):
        for items in self._op:
            items.generateHermitianExcitationOperators()

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self,nop):
        self._op = nop

