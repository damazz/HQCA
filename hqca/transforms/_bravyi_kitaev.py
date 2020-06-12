import sys
from math import floor
import numpy as np
from functools import reduce
from math import floor,ceil,log2
from copy import deepcopy as copy
from hqca.tools._operator import *
from hqca.tools.quantum_strings import *

def _commutator_relations(lp,rp):
    if rp=='I':
        return 1,lp
    elif rp==lp:
        return 1,'I'
    elif rp=='Z':
        if lp=='X':
            return -1j,'Y'
        elif lp=='Y':
            return 1j,'X'
        else:
            sys.exit('Incorrect paulis: {}, {}'.format(lp,rp))
    elif rp=='Y':
        if lp=='X':
            return 1j,'Z'
        elif lp=='Z':
            return -1j,'X'
        else:
            sys.exit('Incorrect paulis: {}, {}'.format(lp,rp))
    elif rp=='X':
        if lp=='Y':
            return -1j,'Z'
        elif lp=='Z':
            return 1j,'Y'
        else:
            sys.exit('Incorrect paulis: {}, {}'.format(lp,rp))
    elif rp=='h':
        if lp=='Z':
            return 1,'h'
        elif lp=='X':
            return 1,'+'
        elif lp=='Y':
            return 1j,'+'
        else:
            sys.exit('Incorrect paulis: {}, {}'.format(lp,rp))
    elif rp=='p':
        if lp=='Z':
            return -1,'p'
        elif lp=='X':
            return 1, '-'
        elif lp=='Y':
            return -1j,'-'
        else:
            sys.exit('Incorrect paulis: {}, {}'.format(lp,rp))
    else:
        sys.exit('Incorrect paulis: {}, {}'.format(lp,rp))

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
        def recursive_update(j,n):
            j,n = int(j),floor(n)
            if n<=1:
                return set()
            if j>=n/2:
                return set([int(i+n/2) for i in recursive_update(j-n/2,n/2)]+[int(n/2-1)])
            else:
                return set(recursive_update(j,n/2))
        self.parity = [recursive_update(j,N) for j in range(self.Nq)]
        for i in self.parity:
            done=False
            while not done:
                done=True
                for j in i:
                    if j>=self.Nq:
                        i.remove(j)
                        done=False
                        break

def BravyiKitaevTransform(op,**kw):
    Nq = len(op.s)
    MapSet = BravyiKitaevMap(Nq)
    pauli = ['I'*Nq]
    new = Operator()+PauliString('I'*Nq,1)
    for q,o in enumerate(op.s):
        if o=='i':
            continue
        if o in ['+','-']:
            u,p,r = 'I'*Nq,'I'*Nq,'I'*Nq
            for i in MapSet.update[q]:
                u= u[:i]+'X'+u[i+1:]
            for i in MapSet.parity[q]:
                p= p[:i]+'Z'+p[i+1:]
            for i in MapSet.rho[q]:
                r= r[:i]+'Z'+r[i+1:]
            p = Operator()+PauliString(p,1)
            r = Operator()+PauliString(r,1)
            u = Operator()+PauliString(u,1)
            x = 'I'*q +'X'+(Nq-q-1)*'I'
            y = 'I'*q +'Y'+(Nq-q-1)*'I'
            s1 = Operator()+PauliString(x,0.5)
            s2 = Operator()+PauliString(y,1j*((o=='-')-0.5))
            s1 = (s1*u)*p
            s2 = (s2*u)*r
        elif o in ['p','h']:
            f = 'I'*Nq
            for i in MapSet.flip[q]:
                f = f[:i]+'Z'+f[i+1:]
            f = Operator()+PauliString(f,1)
            s1 = Operator()+PauliString('I'*Nq,0.5)
            t = 'I'*q+'Z'+(Nq-q-1)*'I'
            s2 = Operator()+PauliString(t,(o=='h')-0.5)
            s1 = s1*f
            s2 = s2*f
        temp = Operator()
        temp+= s1
        temp+= s2
        new = new*temp
    return new

def OldBravyiKitaevTransform(op,
        Nq,
        Nq_tot='default',
        MapSet=None,
        **kw):
    if Nq_tot=='default':
        Nq_tot = copy(Nq)
    coeff = [op.qCo]
    if MapSet.reduced and Nq==MapSet.Nq:
        pauli=['I'*(Nq_tot)]
    else:
        pauli = ['I'*MapSet.Nq_tot]
    if type(MapSet)==type(None):
        print('Bravyi-Kitaev transform not initiated properly!')
        sys.exit()
    for q,o in zip(op.qInd[::-1],op.qOp[::-1]):
        p1s,c1s,p2s,c2s = [],[],[],[]
        def create(q,p,c,MapSet):
            c1,c2 = [],[]
            tc1,tp1 = _commutator_relations(
                    'X',p[q])
            tc2,tp2 = _commutator_relations(
                    'Y',p[q])
            p1 = p[:q]+tp1+p[q+1:]
            p2 = p[:q]+tp2+p[q+1:]
            c1.append(c*0.5*tc1)
            c2.append(-1j*c*0.5*tc2)
            for i in MapSet.update[q]:
                nc1,np1 = _commutator_relations(
                        'X',p1[i])
                nc2,np2 = _commutator_relations(
                        'X',p2[i])
                p1 = p1[:i]+np1+p1[i+1:]
                p2 = p2[:i]+np2+p2[i+1:]
                c1[0]*=nc1
                c2[0]*=nc2
            for i in MapSet.parity[q]:
                nc1,np1 = _commutator_relations(
                        'Z',p1[i])
                p1 = p1[:i]+np1+p1[i+1:]
                c1[0]*=nc1
            for i in MapSet.rho[q]:
                nc2,np2 = _commutator_relations(
                        'Z',p2[i])
                p2 = p2[:i]+np2+p2[i+1:]
                c2[0]*=nc2
            return p1,p2,c1,c2

        def annihilate(q,p,c,MapSet):
            c1,c2 = [],[]
            tc1,tp1 = _commutator_relations(
                    'X',p[q])
            tc2,tp2 = _commutator_relations(
                    'Y',p[q])
            p1 = p[:q]+tp1+p[q+1:]
            p2 = p[:q]+tp2+p[q+1:]
            c1.append(c*0.5*tc1)
            c2.append(1j*c*0.5*tc2)
            for i in MapSet.update[q]:
                nc1,np1 = _commutator_relations(
                        'X',p1[i])
                nc2,np2 = _commutator_relations(
                        'X',p2[i])
                p1 = p1[:i]+np1+p1[i+1:]
                p2 = p2[:i]+np2+p2[i+1:]
                c1[0]*=nc1
                c2[0]*=nc2
            for i in MapSet.parity[q]:
                nc1,np1 = _commutator_relations(
                        'Z',p1[i])
                p1 = p1[:i]+np1+p1[i+1:]
                c1[0]*=nc1
            for i in MapSet.rho[q]:
                nc2,np2 = _commutator_relations(
                        'Z',p2[i])
                p2 = p2[:i]+np2+p2[i+1:]
                c2[0]*=nc2
            return p1,p2,c1,c2


        for p,c in zip(pauli,coeff):
            c1,c2 = [],[]
            if o=='+':
                p1,p2,c1,c2 = create(q,p,c,MapSet)
            elif o=='-':
                p1,p2,c1,c2 = annihilate(q,p,c,MapSet)
            elif o=='p':
                p1,p2,c1,c2 = particle(q,p,c,MapSet)
            elif o=='h':
                p1,p2,c1,c2 = hole(q,p,c,MapSet)
            p1s.append(p1)
            p2s.append(p2)
            c1s+= c1
            c2s+= c2
        pauli = p1s+p2s
        coeff = c1s+c2s
    if MapSet.reduced:
        q1,q2 = MapSet._reduced_set[0],MapSet._reduced_set[1]
        c1,c2 = MapSet._reduced_coeff[q1],MapSet._reduced_coeff[q2]
        for n,(p,c) in enumerate(zip(pauli,coeff)):
            c = copy(c)
            if p[q1]=='Z':
                c*=c1
            if p[q2]=='Z':
                c*=c2
            pauli[n]=p[:q1]+p[(q1+1):q2]
            coeff[n] = c
    return pauli,coeff


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


