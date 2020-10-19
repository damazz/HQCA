from copy import deepcopy as copy
import sys
import traceback
from hqca.tools.quantum_strings import *

class Operator:
    '''
    Can construct mathematical operators using different 'string'.
    Aggregate collection of smaller strings, typically either qubit, 
    fermionic, or Pauli strings. Addition or multiplication follows the 
    rules for the component string. Can also be iterated through, or accessed
    through indices. 

    transform(Transformation) returns a new operator.   

    '''
    def __init__(self,
            commutative_addition=True
            ):
        self.op = []
        self.ca = commutative_addition

    def __str__(self):
        z = ''
        for i in self.op:
            z += i.__str__()
            z += '\n'
        return z[:-1]

    def __next__(self):
        return self.op.__next__()

    def __iter__(self):
        return self.op.__iter__()

    def __getitem__(self,key):
        return self.op[key]

    def __contains__(self,A):
        return A in self.op

    def __len__(self):
        return len(self.op)

    def __add__(self,A):
        new = copy(self)
        if isinstance(A ,type(QuantumString())):
            # we are adding a string
            if not self.ca:
                add=True
            else:
                add=True
                for n,i in enumerate(self):
                    if i==A:
                        new.op[n]+=A
                        add=False
                        break
            if add:
                new.op.append(A)
            return new
        elif isinstance(A,type(Operator())):
            for o in A:
                if not self.ca or not A.ca:
                    add=True
                else:
                    add=True
                    for n,i in enumerate(self):
                        if i==o:
                            new.op[n]+=o
                            add=False
                            break
                if add:
                    new.op.append(o)
        new.clean()
        return new

    def __mul__(self,A):
        new = Operator()
        if isinstance(A,type(QuantumString())):
            for i in self:
                new+= i*A
            # we are adding a string
            new.clean()
            return new
        elif isinstance(A,type(Operator())):
            for o in A:
                for p in self:
                    new+= o*p
        else:
            raise TypeError
        new.clean()
        return new

    def null(self):
        for i in self:
            if abs(i.c)>=1e-10:
                return False
        return True

    def transform(self,
            T=None,
            *args,**kwargs):
        '''
        perform transformation on some operators
        '''
        new = Operator()
        new += T(self,*args,**kwargs)
        return new

    def __sub__(self,A):
        new = copy(self)
        if isinstance(A ,type(QuantumString())):
            A.c *=-1
            # we are adding a string
            if not self.ca:
                add=True
            else:
                add=True
                for n,i in enumerate(self):
                    if i==A:
                        new.op[n]+=A
                        add=False
                        break
            if add:
                new.op.append(A)
            return new
        elif isinstance(A,type(Operator())):
            for n,o in enumerate(A):
                A[n].c*=-1
                if not self.ca or not A.ca:
                    add=True
                else:
                    add=True
                    for n,i in enumerate(self):
                        if i==o:
                            new.op[n]+=o
                            add=False
                            break
                if add:
                    new.op.append(o)
        new.clean()
        return new
    def clean(self):
        done = False
        while not done:
            done = True
            for n,i in enumerate(self):
                if abs(i.c)<1e-12:
                    self.op.pop(n)
                    done=False
                    break

    def commutator(self,A):
        try:
            return self*A - A*self
        except Exception: 
            new = Operator()
            new+= A
            return self*A-A*self

    def clifford(self,U):
        '''
        applies clifford unitaries...note these are in terms of qubit orderings
        '''
        cliff = {
                'H':{
                    'X':['Z',1],
                    'Y':['Y',-1],
                    'Z':['X',1],
                    'I':['I',1],
                    },
                'S':{
                    'X':['Y',-1],
                    'Y':['X',-1],
                    'Z':['Z',1],
                    'I':['I',1],
                    },
                'V':{   # SHS
                    'X':['X',-1],
                    'Y':['Z',-1],
                    'Z':['Y',-1],
                    'I':['I',1],
                    },
                }
        new = Operator()
        #print('U: ',U)
        for op in self:
            if not isinstance(op,type(PauliString())):
                sys.exit('Can not apply Clifford groups to non-Pauli strings.')
            temp = []
            c = copy(op.c)
            for s,u in zip(op.s,U):
                temp.append(cliff[u][s][0])
                c*= cliff[u][s][1]
            #print(PauliString(''.join(temp),c))
            #print('----')
            new+= PauliString(''.join(temp),c)
        return new

