from copy import deepcopy as copy
import sys
from hqca.operators.quantum_strings import PauliString,FermiString
from hqca.operators.quantum_strings import QubitString,QuantumString
from hqca.core import OperatorError
import hqca.config as config
import multiprocessing as mp
from collections import OrderedDict

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
            op=None,
            ):
        self.op = OrderedDict()
        if type(op)==type(None):
            pass
        elif isinstance(op, type(QuantumString())):
            self.op[op.s]=op
        elif isinstance(op,type([])):
            for i in op:
                self.op[i.s]=i

    def norm(self):
        '''
        calculates the l2 norm of operator in the respective basis
        '''
        n = 0
        for o in self:
            n+= (o.c.real+1j*o.c.imag)*(o.c.real-1j*o.c.imag)
        return (n.real)**(0.5)

    def __str__(self):
        z = ''
        for i in self:
            z += i.__str__()
            z += '\n'
        return z[:-1]

    def items(self):
        return self.op.items()

    def values(self):
        return self.op.values()

    def keys(self):
        return self.op.keys()

    def __iter__(self):
        return iter(self.op.values())

    def __getitem__(self,key):
        return self.op[key]

    def __contains__(self,A):
        return A.s in self.op

    def __len__(self):
        return len(self.op)

    def __add__(self,A):
        if isinstance(A ,type(QuantumString())): # we are adding a string
            if A in self:
                self.op[A.s]+=A
            else:
                self.op[A.s]=A
        elif isinstance(A,type(Operator())):
            for op in A:
                if op in self:
                    self.op[op.s]+=op
                else:
                    self.op[op.s]=op
            self.clean()
        else:
            print('')
            print('# # # # # ')
            print('Something wrong with class, cannot add type')
            print(type(A),' to Operator class.')
            print('# # # # # ')
            sys.exit()
        return self

    def __rmul__(self,A):
        new = Operator()
        c1 = isinstance(A,type(QuantumString()))
        c2 = isinstance(A,type(Operator()))
        if (not c1) and (not c2):
            for o in self:
                new+= A*self.op[o]
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
            for a in A:
                for s in self:
                    new+= s*a
        else:
            try:
                for o in self:
                    new+= o*A
            except Exception as e:
                print(e)
                sys.exit('Multiplication error type Ib.')
        new.clean()
        return new

    def null(self):
        for i in self:
            if abs(i.c)>=1e-10:
                return False
        return True

    def transform(self,
            T=None):
        """
        perform transformation on some operators

        TODO: add global attribute for triggering multiprocessing?

        """
        if len(self)>4 and config._use_multiprocessing:
            pool = mp.Pool(mp.cpu_count())
            results = pool.map(T,self.values())
            pool.close()
        else:
            results = [T(o) for o in self.values()]
        new = Operator()
        for r in results:
            new+= r
        return new

    def __sub__(self,A):
        if isinstance(A ,type(QuantumString())):
            A.c *=-1
            return self.__add__(A)
        elif isinstance(A,type(Operator())):
            for n in (A):
                A[n.s].c*=-1
            return self.__add__(A)

    def clean(self):
        remove = []
        for n in self:
            if n.sym:
                if n.c.equals(0):
                    remove.append(n)
            else:
                if abs(n.c)<=1e-12:
                    remove.append(n)
        for n in remove:
            del self.op[n.s]

    def remove(self,key):
        try:
            del self.op[key]
        except Exception as e:
            pass


        #L =len(self)
        #for n,i in enumerate(reversed(self)):
        #    if i.sym:
        #        try:
        #            if i.c.equals(0):
        #                self.op.pop(L-n-1)
        #        except AttributeError as e:
        #            if abs(i.c)<1e-12:
        #                self.op.pop(L-n-1)
        #    else:
        #        if abs(i.c)<1e-12:
        #            self.op.pop(L-n-1)

    def truncate(self,threshold=1e-10):
        for i in reversed(range(len(self))):
            if abs(self.op[i].c)<threshold:
                self.op.pop(i)

    def commutator(self,A):
        try:
            return self*A - A*self
        except Exception: 
            new = Operator()
            new+= A
            return self*A-A*self

    def clifford(self,U):
        '''
        TODO: need ot change this to symplectic form, in line with self.l and self.r
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
                'V':{   # SHSdag
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
                raise OperatorError('Can not apply Clifford groups to non-Pauli strings.')
            temp = []
            c = copy(op.c)
            for s,u in zip(op.s,U):
                temp.append(cliff[u][s][0])
                c*= cliff[u][s][1]
            #print(PauliString(''.join(temp),c))
            #print('----')
            new+= PauliString(''.join(temp),c,symbolic=self.sym)
        return new

