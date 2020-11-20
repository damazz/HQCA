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
            op=None,
            commutative_addition=True
            ):
        self.op = []
        self.ca = commutative_addition
        if type(op)==type(None):
            pass
        elif isinstance(op, type(QuantumString())):
            self.op = self.__add__(op).op
        elif isinstance(op,type([])):
            new = copy(self)
            for o in op:
                new+= o
            self.op = new.op


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
            for a in A:
                for s in self:
                    new+= s*a
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

    def simplify(self):
        if not isinstance(self.op[0],type(FermiString())):
            sys.exit('Can not simplify non Fermionic strings.')
        done = False
        def sub1(self):
            for j in range(len(self.op)):
                for i in range(j):
                    for k in range(len(self.op[i].s)):
                        s1 = self.op[i].s[:k]+self.op[i].s[k+1:]
                        s2 = self.op[j].s[:k]+self.op[j].s[k+1:]
                        c1,c2 = copy(self.op[i].c),copy(self.op[j].c)
                        if s1==s2 and set([self.op[i].s[k],self.op[j].s[k]])==set(['p','h']):
                            self.op[i].s = self.op[i].s[:k]+'i'+self.op[i].s[k+1:]
                            self.op[j].c = c2-c1
                            return False
            return True

        def sub2(self):
            for j in range(len(self.op)):
                for i in range(j):
                    if self.op[i]==self.op[j]:
                        #print(self.op[i],self.op[j])
                        self.op[i].c+= self.op[j].c
                        del self.op[j]
                        return False
            return True

        def sub3(self):
            for j in range(len(self.op)):
                for i in range(j):
                    for k in range(len(self.op[i].s)):
                        s1 = self.op[i].s[:k]+self.op[i].s[k+1:]
                        s2 = self.op[j].s[:k]+self.op[j].s[k+1:]
                        k1,k2 = self.op[i].s[k], self.op[j].s[k]
                        c1,c2 = copy(self.op[i].c),copy(self.op[j].c)
                        if s1==s2 and set([self.op[i].s[k],self.op[j].s[k]])==set(['p','i']):
                            if abs(c1+c2)<1e-6:
                                c = abs(c1)
                                if self.op[i].s[k]=='p':
                                    self.op[i].c = c2
                                elif self.op[i].s[k]=='i':
                                    self.op[i].c = c1
                                self.op[i].s = self.op[i].s[:k]+'h'+self.op[i].s[k+1:]
                                del self.op[j]
                                return False
                            elif k1=='i' and abs(c1+c2*0.5)<1e-6:
                                # c1 is half as large as c2
                                # i.e., c1 = c2*0.5
                                # c1 I - 2c1 P = * 
                                self.op[i].s = self.op[i].s[:k]+'h'+self.op[i].s[k+1:]
                                self.op[i].c = c1
                                self.op[j].s = self.op[i].s[:k]+'p'+self.op[i].s[k+1:]
                                self.op[j].c = 0.5*c2
                            elif k2=='i' and abs(c1*0.5+c2)<1e-6:
                                # 
                                #
                                self.op[i].s = self.op[i].s[:k]+'p'+self.op[i].s[k+1:]
                                self.op[i].c = c1*0.5
                                self.op[j].s = self.op[i].s[:k]+'h'+self.op[i].s[k+1:]
                                self.op[j].c = c2

                        elif s1==s2 and set([self.op[i].s[k],self.op[j].s[k]])==set(['h','i']):
                            if abs(c1+c2)<1e-6:
                                c = abs(c1)
                                if self.op[i].s[k]=='h':
                                    self.op[i].c = c2
                                elif self.op[i].s[k]=='i':
                                    self.op[i].c = c1
                                self.op[i].s = self.op[i].s[:k]+'p'+self.op[i].s[k+1:]
                                del self.op[j]
                                return False
                            elif k1=='i' and abs(c1+c2*0.5)<1e-6:
                                # c1 is half as large as c2
                                # i.e., c1 = c2*0.5
                                # c1 I - 2c1 P = * 
                                self.op[i].s = self.op[i].s[:k]+'p'+self.op[i].s[k+1:]
                                self.op[i].c = c1
                                self.op[j].s = self.op[i].s[:k]+'h'+self.op[i].s[k+1:]
                                self.op[j].c = 0.5*c2
                            elif k2=='i' and abs(c1*0.5+c2)<1e-6:
                                # 
                                #
                                self.op[i].s = self.op[i].s[:k]+'h'+self.op[i].s[k+1:]
                                self.op[i].c = c1*0.5
                                self.op[j].s = self.op[i].s[:k]+'p'+self.op[i].s[k+1:]
            return True
        pre = False
        #print(len(self.op))
        while not pre:
            pre = sub2(self)
        #print(len(self.op))
        l1 = False
        l2 = False
        while not (l1 and l2):
            l1 = False
            while not l1:
                l1 = sub1(self)
                almost = False
                while not almost:
                    almost = sub2(self)
            l2 = False
            while not l2:
                l2 = sub3(self)
        self.clean()
        return self

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

