import sys
from math import floor
import numpy as np
from functools import reduce
from math import floor,ceil,log2
from copy import deepcopy as copy
from hqca.operators import *

class BravyiKitaevMap:
    def __init__(self,
            Nq,
            alternating=False,
            ):
        '''
        alternating
        '''
        self.Nq=Nq
        self.Nq_tot=copy(Nq)
        if floor(log2(self.Nq))==ceil(log2(self.Nq)):
            N = self.Nq
        else:
            N = int(2**(ceil(log2(self.Nq))))
        self._find_flip_set(N)
        self._find_update_set(N)
        self._find_parity_set(N)
        self.remainder = [
                self.parity[i]-self.flip[i] for i in range(self.Nq)]
        self.rho = []
        for i in range(self.Nq):
            if i%2==0:
                self.rho.append(self.parity[i])
            elif i%2==1:
                self.rho.append(self.remainder[i])
        if alternating:
            self.map_to_spin()


    def map_to_spin(self):
        Na = int(self.Nq/2)
        mapping = {i:i//2+(i%2)*(Na) for i in range(self.Nq)}
        rev_map = {k:v for v,k in mapping.items()}
        new = [[set() for i in range(self.Nq)] for j in range(5)]
        old = [self.flip,self.update,self.parity,self.remainder,self.rho]
        for n,o in enumerate(old):
            for m,i in enumerate(o):
                # also need to swap index here.....
                M = rev_map[m]
                for j in i:
                    new[n][M].add(rev_map[j])
        self.flip = new[0]
        self.update =new[1]
        self.parity = new[2]
        self.remainder = new[3]
        self.rho = new[4]

    def _find_flip_set(self,N):
        def recursive_update(j,n):
            j,n = int(j),floor(n)
            if n<=1:
                return set()
            if j<n/2:
                return set(recursive_update(j,n/2))
            elif j>=n/2 and j<(n-1):
                return set([int(i+n/2) for i in recursive_update(j-n/2,n/2)])
            elif j==(n-1):
                return set([int(i+n/2) for i in recursive_update(j-n/2,n/2)]+[int(n/2-1)])
        self.flip = [recursive_update(j,N) for j in range(self.Nq)]
        for i in self.flip:
            done=False
            while not done:
                done=True
                for j in i:
                    if j>=self.Nq:
                        i.remove(j)
                        done=False
                        break

    def _find_update_set(self,N):
        def recursive_update(j,n):
            j,n = int(j),floor(n)
            if n<=1:
                return set()
            if j<n/2:
                return set(list(recursive_update(j,n/2))+[n-1])
            else:
                return set([int(i + n/2) for i in recursive_update(j-n/2,n/2)])
        self.update = [recursive_update(j,N) for j in range(self.Nq)]
        for i in self.update:
            done=False
            while not done:
                done=True
                for j in i:
                    if j>=self.Nq:
                        i.remove(j)
                        done=False
                        break

    def _find_parity_set(self,N):
        self.parity = []
        #def recursive_update(j,n):
        #    j,n = int(j),floor(n)
        #    if n<=1:
        #        return set()
        #    if j>=n/2:
        #        return set([int(i+n/2) for i in recursive_update(j-n/2,n/2)]+[int(n/2-1)])
        #    else:
        #        return set(recursive_update(j,n/2))
        #self.parity = [recursive_update(j,N) for j in range(self.Nq)]
        #for i in self.parity:
        #    done=False
        #    while not done:
        #        done=True
        #        for j in i:
        #            if j>=self.Nq:
        #                i.remove(j)
        #                done=False
        #                break

def BravyiKitaevTransform(op,**kw):
    Nq = len(op.s)
    MapSet = BravyiKitaevMap(Nq)
    pauli = ['I'*Nq]
    new = Operator()+PauliString('I'*Nq,op.c)
    for q,o in enumerate(op.s):
        if o=='i':
            continue
        if o in ['+','-']:
            s1 = 'I'*Nq
            s2 = 'I'*Nq
            for i in MapSet.update[q]:
                s1= s1[:i]+'X'+s1[i+1:]
                s2= s2[:i]+'X'+s2[i+1:]
            for i in MapSet.parity[q]:
                s1= s1[:i]+'Z'+s1[i+1:]
            for i in MapSet.rho[q]:
                s2= s2[:i]+'Z'+s2[i+1:]
            s1 = s1[:q]+'X'+s1[q+1:]
            s2 = s2[:q]+'Y'+s2[q+1:]
            s1 = Operator()+PauliString(s1,0.5)
            s2 = Operator()+PauliString(s2,1j*((o=='-')-0.5))
        elif o in ['p','h']:
            f = 'I'*Nq
            for i in MapSet.flip[q]:
                f = f[:i]+'Z'+f[i+1:]
            f = f[:q]+'Z'+f[q+1:]
            s1 = Operator()+PauliString('I'*Nq,0.5)
            s2 = Operator()+PauliString(f,(o=='h')-0.5)
        temp = Operator()
        temp+= s1
        temp+= s2
        new = temp*new
    return new


def BravyiKitaev(operator,
        **kw
        ):
    if isinstance(operator,type(QuantumString())):
        return BravyiKitaevTransform(operator,**kw)
    else:
        new = Operator()
        for op in operator:
            new+= BravyiKitaevTransform(op,**kw)
        return new


