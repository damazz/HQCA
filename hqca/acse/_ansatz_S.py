from hqca.tools import *
from copy import deepcopy as copy
import sys

class Ansatz:
    def __init__(self,
            depth_to_add=1,
            **kw):
        '''
        Close to an operator class, but really an operator of operator? Yeah.
        Addition, mulptiplication are defined differently though

        need to update ACSE as well :( o.o o.p O.D 
        '''
        self.A = [] #instead of strings, holds operators at each place
        self.depth=  depth_to_add #if 1, will go back 1 step
        self.d = 0

    def truncate(self,d=0):
        self.A = self.A[:d]
        self.d = d

    def __getitem__(self,k):
        return self.A[k]

    def _op_form(self):
        return [p for o in self.A for p in o]

    def __iter__(self):
        return self._op_form().__iter__()

    def __next__(self):
        return self._op_form().__next__()

    def __contains__(self,A):
        for n in range(self.d):
            if A in self.op:
                return True
        return False

    def __len__(self):
        return [len(o) for o in self.A]

    def __str__(self):
        z = '- - -'
        for i in range(self.d):
            z+= '\nS_{}:\n'.format(i)
            z+= self.A[i].__str__()
        z+= '\n- - -'
        return z

    def clean(self):
        for a in self.A:
            a.clean()

    def __mul__(self,A):
        sys.exit('Multiplication not defined for Ansatz Class')

    def __add__(self,O):
        # rawr
        new = copy(self)
        if isinstance(O,type(Operator())):
            if len(self.A)==0:
                new.A.append(O)
                new.d+=1
                return new
            newOp = Operator(commutative_addition=True)
            O.ca=True
            for o in O:
                added=False
                for d in range(min(self.depth,self.d)):
                    if o in self.A[-d-1]:
                        new.A[-d-1]+= o
                        added=True
                        break
                if not added:
                    newOp+= o
            if len(newOp)>0:
                new.A.append(newOp)
                new.d+=1
        else:
            pass
            sys.exit(r'Can\'t add non-Operator to S. ')
        new.clean()
        return new

