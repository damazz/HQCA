import numpy as np
from copy import deepcopy as copy
from hqca.operators.quantum_strings._quantum_string import *
from sympy import symbols
from sympy import re,im
import sys

class FermiString(QuantumString):
    '''
    Class of operators, obeys simple fermionic statistics.

    Also used to generate raw lists of Pauli strings and subterms, which can be
    compiled for circuits and tomography.

    has a key word defining the type 
    '''
    def __init__(self,
            coeff=1,
            s=None, # string
            indices=[0,1,2,3], #orbital indices
            ops='+-+-', #second quantized operators
            N=10,
            antisymmetric=True,
            add=True,
            symbolic=False,
            ):
        self.fermi = antisymmetric
        if type(s)==type(None):
            self._generate_from_sq(coeff,inds=indices,sqop=ops,N=N)
        else:
            # generate from string
            self.s = s
            self.c = coeff
        self.sym = symbolic
        self.add = add

    def __str__(self):
        if self.sym:
            z = '{}: {}'.format( self.s,self.c.__str__())
        else:
            z = '{}: {} + i*{}'.format(self.s,self.c.real,self.c.imag)
        return z

    def __len__(self):
        return len(self.ops())

    def rank(self):
        l = 0
        for i in self.ops():
            if i in ['p','h']:
                l+=2
            if i in ['+','-']:
                l+=1 
        return l

    def __add__(self,A):
        new = copy(self)
        if A==self:
            new.c += A.c
        else:
            sys.exit('Huh?')
        return new

    def iszero(self):
        return abs(self.c)<=1e-15

    def __eq__(self,A):
        return A.s==self.s

    def __ne__(self,A):
        return A.s==self.s and A.c ==self.c

    def inds(self):
        return [n for n in range(len(self.s)) if not self.s[n]=='i']

    def ops(self):
        return  [j for j in self.s if not j=='i']

    def N(self):
        return len(self.s)

    def order_as_D_matrix(self):
        '''
        generate ordering, indices, and coefficient for a particle D matrix
        '''
        # 
        temp = []
        def subroutine(op):
            c = copy(op.c)
            s = copy(op.s)
            N = len(s)
            ops,inds = [],[]
            for n,i in enumerate(s):
                if i in ['p','h']:
                    if i=='p':
                        ops += ['+','-']
                    elif i=='h':
                        ops += ['-','+']
                    inds+= [n,n]
                elif i in ['+','-']:
                    ops.append(i)
                    inds.append(n)
            if len(inds)==0 and len(ops)==0:
                done = True
            else:
                done = False
            while not done:
                # check cre-ann order
                # evaluate number of times we switch charactes
                # this should only happen once
                switch = 0
                for i in range(len(ops)-1):
                    if not ops[i]==ops[i+1]:
                        switch+=1 
                if switch==1 and ops[0]=='+':
                    done = True
                    break
                #
                #
                for i in range(len(ops)-1):
                    if not ops[i]==ops[i+1] and ops[i]=='-':
                        if inds[i]==inds[i+1]:

                            nind = inds[:i]+inds[i+2:]
                            nops = ''.join(ops[:i]+ops[i+2:])
                            new = FermiString(coeff=copy(c),
                                    indices=nind,
                                    ops=nops,
                                    N=N
                                    )
                            subroutine(new)

                            c*= - 1
                            ops[i],ops[i+1]=ops[i+1],ops[i]
                        else:
                            ops[i],ops[i+1]=ops[i+1],ops[i]
                            inds[i],inds[i+1]=inds[i+1],inds[i]
                        c*= -1
                        done = False
                        break
            temp.append([c,inds,ops,N])
        subroutine(self)
        return temp






    def __rmul__(self,A):
        # A * self
        if not isinstance(A,type(FermiString)):
           try:
                new = FermiString(
                        coeff=self.c*A,
                        s=self.s)
                return new
           except Exception as e:
               print(e)
               sys.exit()

    def __mul__(self,A):
        # self * A
        if isinstance(A,type(FermiString())):
            s = ''
            ind = []
            for j in [self,A]:
                for i in range(len(j.s)):
                    if not j.s[i]=='i':
                        s+= j.s[i]
                        ind.append(i)
            return FermiString(
                    coeff=self.c*A.c,
                    indices=ind,
                    ops=s,
                    antisymmetric=self.fermi,
                    add=self.add,
                    N=max(len(self.s),len(A.s)),
                    symbolic=self.sym)
        else:
            try:
                new = FermiString(
                        coeff=self.c*A,
                        s=self.s)
            except Exception as e:
                print(e)
                sys.exit()
            return new

    def _generate_from_sq(self,coeff=1,inds=[],sqop='',N=10):
        sort = False
        if len(inds)==0 and sqop=='':
            self.s = 'i'*N
            self.c = coeff
        else:
            while not sort:
                #print(coeff,inds,sqop)
                sort=True
                for i in range(len(sqop)-1):
                    if inds[i]>inds[i+1]:
                        if self.fermi:
                            if sqop[i] in ['p','h']:
                                pass
                            elif sqop[i+1] in ['p','h']:
                                pass
                            else:
                                coeff*=-1
                        inds = inds[:i]+[inds[i+1]]+[inds[i]]+inds[i+2:]
                        sqop = sqop[:i]+sqop[i+1]+sqop[i]+sqop[i+2:]
                        sort=False
                        break
            #print(sqop,inds)
            if len(set(copy(inds)))==len(sqop):
                pass
            else:
                zeros = ['h+','-h','p-','+p','++','--','ph','hp']
                simple = {
                        'h-':'-','+h':'+','p+':'+','-p':'-',
                        'pp':'p','hh':'h','+-':'p','-+':'h',}
                done = False
                while not done:
                    done=True
                    for i in range(len(inds)-1):
                        if inds[i]==inds[i+1]:
                            inds.pop(i+1)
                            if sqop[i:i+2] in zeros:
                                coeff=0
                                sqop= ''
                                break
                            else:
                                key = simple[sqop[i:i+2]]
                                sqop = sqop[0:i]+key+sqop[i+2:]
                            done=False
                            break
            if coeff==0:
                s = 'i'*N
            else:
                s = 'i'*inds[0]
                #print(sqop,inds,s)
                for i in range(len(inds)-1):
                    s+= sqop[i]
                    s+='i'*(inds[i+1]-inds[i]-1)
                s+= sqop[-1]+'i'*(N-inds[-1]-1)
            self.s = s
            self.c = coeff

