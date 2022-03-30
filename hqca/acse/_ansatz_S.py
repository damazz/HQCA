from hqca.tools import *
from hqca.operators import *
from copy import deepcopy as copy
import sys
from hqca.core import *

class Ansatz:
    def __init__(self,
            closed=True,
            trotter='first',
            ):

        '''
        Compilation of operators. Designed for iterative ansatz, where different layers
        of ansatz terms can be defined.

        A closed ansatz is one in which the inner layers of the ansatz cannot be accessed
        '''
        self.A = [] #instead of strings, holds operators at each place
        self.closed = closed
        self.trotter = trotter
        self._store = []
        if closed>1:
            sys.exit('Need to specify closed <=1')

    def truncate(self,d='default'):
        '''
        return truncated form of the ansatz with depth d
         
        from initial to final (i.e., first to last)
        '''
        if d =='default':
            d = len(self)
        self.A = self.A[:d]
        self._store = self._store[:d]


    def norm(self):
        return [a.norm() for a in self]

    def __getitem__(self,k):
        return self.A[k]

    def op_form(self):
        if self.trotter=='first':
            return [p for o in self.A for p in o]
        else:
            raise QuantumRunError('Unspecified trotterization: {}'.format(self.trotter))

    def __iter__(self):
        return self.A.__iter__()

    def __next__(self):
        return self.A.__next__()

    def __len__(self):
        return len(self.A)

    def __str__(self):
        z = '- - -'
        for i in range(len(self)):
            z+= '\nS_{}:\n'.format(i)
            z+= self.A[i].__str__()
        z+= '\n- - -'
        return z

    def clean(self):
        for a in self.A:
            a.clean()
        for a in reversed(range(len(self.A))):
            if len(self.A[a])==0:
                self.A.pop(a)

    def __mul__(self,A):
        sys.exit('Multiplication not defined for Ansatz Class')

    def get_lim(self):
        if self.closed==0: 
            return -1*(len(self.A))
        elif self.closed==1:
            return 0
        else:
            return max(self.closed,-1*len(self))

    def __add__(self,O):
        new = copy(self)
        if isinstance(O,type(Operator())):
            if len(self.A)==0:
                # if first step, append new operator O
                new.A.append(O)
                return new
            newOp = Operator()
            if self.closed==1: 
                # if closed==True, also just append new operator
                new.A.append(O)
            else:
                if self.closed==0:
                    # false, set to length of ansatz
                    lim = -1*(len(self.A))
                else:
                    lim = max(self.closed,-1*len(self))
                    # set minimum of self.closed, len of ansatz
                #print([i for i in range(lim,0)])
                for o in O:
                    # for each operator in O, search previous steps in self
                    added=False
                    for d in reversed(range(lim,0)):
                        if o in self.A[d]:
                            new.A[d]+= o
                            added=True
                            break #move on to next operator o 
                    if not added:
                        newOp+= o
                if len(newOp)>0:
                    new.A.append(newOp)
        else:
            pass
            sys.exit(r'Can\'t add non-Operator to S. ')
        new.clean()
        return new

    def __sub__(self,A):
        new = Operator()
        for o in A:
            O = copy(o)
            O.c = O.c*(-1)
            new+= O
        return self+new

