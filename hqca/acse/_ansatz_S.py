from hqca.tools import *
import sys

class Ansatz:
    def __init__(self,
            depth_to_add=1,
            **kw):
        '''
        Close to an opeartor class, but really an operator of operator? Yeah.
         Addition, mulptiplication are defined differently though

         need to update ACSE as well :( o.o o.p O.D 
        '''
        self.A = [] #instead of strings, holds operators at each place
        self.depth=  depth_to_add #if 1, will go back 1 step
        self.d = 0


    def __len__(self):
        return [len(o) for o in self.A]

    def __str__(self):
        for i in range(self.d):
            print('O_{}'.format(i))
            print(self.A[i])

    def __mul__(self,A):
        sys.exit('Multiplication not defined for Ansatz Class')

    def __add__(self,O):
        # rawr
        new = copy(self)
        if isinstance(O,type(Operator())):
            newOp = Operator()
            for o in O:
                added=False
                for d in range(self.depth):
                    if o in self.A[-d]:
                        self.A[-d]+= o
                        added=True
                        break
                if not added:
                    newOp+= o
            if len(newOp)>0:
                new.A.append(newOp)
                new.A.d+=1
        else:
            pass
            sys.exit(r'Can\'t add non-Operator to S. ')
        new.clean()
        return new

    



