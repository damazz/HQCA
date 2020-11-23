from hqca.tools import *
from copy import deepcopy as copy
import sys

class Ansatz:
    def __init__(self,
            closed=False,
            ):
        '''
        Compilation of operators. Designed for iterative ansatz, where different layers
        of ansatz terms can be defined. 

        A closed ansatz is one in which the inner layers of the ansatz cannot be accessed.
        I.e., it is non commutative with respect to addition. 
        '''
        if closed==False:
            self.depth= 0
        elif closed:
            self.depth= len(self)
        else:
            self.depth = int(closed)
        self.A = [] #instead of strings, holds operators at each place
        self.d = 0

    def truncate(self,d='default'):
        '''
        return truncated form of the ansatz with depth d
        '''
        if d =='default':
            d = len(self)
        self.A = self.A[:d]
        self.d = d

    def norm(self):
        return [a.norm for a in self]

    def __getitem__(self,k):
        return self.A[k]

    def _op_form(self):
        return [p for o in self.A for p in o]

    def __iter__(self):
        return self.A.__iter__()

    def __next__(self):
        return self.A.__next__()

    def __contains__(self,A):
        for n in range(self.d):
            if A in self.op:
                return True
        return False

    def __len__(self):
        return len(self.A)

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

