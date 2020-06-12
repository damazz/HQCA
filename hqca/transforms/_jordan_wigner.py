import sys
from copy import deepcopy as copy
from hqca.tools._operator import *
from hqca.tools.quantum_strings import *


class JordanWignerMap:
    '''
    Class for modifying Jordan Wigner Mapping
    '''
    def __init__(self,
            Nq,
            Nq_tot='default',
            reduced=False,
            qubits=[],
            paulis=[],
            eigvals=[],
            U=None,
            Ut=None,
            **kw
            ):
        self.Nq = Nq
        if Nq_tot=='default':
            self.Nq_tot=copy(Nq)
        else:
            self.Nq_tot = Nq_tot
        self.red_q = qubits
        self.red_p = paulis
        self.red_e = eigvals
        self.reduced= reduced
        self.U = U
        self.Ut = Ut

def JordanWignerTransform(op):
    '''
    transforms a fermistrings into a operators of paulistrings
    '''
    Nq = len(op.s)
    pauli = ['I'*Nq]
    new = Operator()
    new+= PauliString('I'*Nq,op.c)
    # define paulis ops
    for qi,o in enumerate(op.s[::-1]):
        # revrersed is because of the order in which we apply cre/ann ops
        q = Nq-qi-1
        if o=='i':
            continue
        if o in ['+','-']:
            s1 = 'Z'*q+'X'+(Nq-q-1)*'I'
            s2 = 'Z'*q+'Y'+(Nq-q-1)*'I'
            c1,c2 = 0.5,((o=='-')-0.5)*1j
        elif o in ['p','h']:
            s1 = 'I'*q+'I'+(Nq-q-1)*'I'
            s2 = 'I'*q+'Z'+(Nq-q-1)*'I'
            c1,c2 = 0.5,(o=='h')-0.5
        tem = Operator()
        tem+= PauliString(s1,c1)
        tem+= PauliString(s2,c2)
        new = new*tem
    return new

def ModifiedJordanWignerTransform(op):
    '''
    transforms a fermistrings into a operators of paulistrings
    '''
    Nq = len(op.s)
    pauli = ['I'*Nq]
    new = Operator()
    new+= PauliString('I'*Nq,op.c)
    # define paulis ops
    for qi,o in enumerate(op.s[::-1]):
        # revrersed is because of the order in which we apply cre/ann ops
        q = Nq-qi-1
        if o=='i':
            continue
        if o in ['+','-']:
            s1 = 'Y'*q+'X'+(Nq-q-1)*'I'
            s2 = 'Y'*q+'Z'+(Nq-q-1)*'I'
            c1,c2 = 0.5,((o=='-')-0.5)*1j
        elif o in ['p','h']:
            s1 = 'I'*q+'I'+(Nq-q-1)*'I'
            s2 = 'I'*q+'Y'+(Nq-q-1)*'I'
            c1,c2 = 0.5,(o=='h')-0.5
        tem = Operator()
        tem+= PauliString(s1,c1)
        tem+= PauliString(s2,c2)
        new = new*tem
    return new

def old_JordanWignerTransform(op,
        Nq_tot='default',
        MapSet=None,
        **kw):
    Nq = len(op.s)
    if Nq_tot=='default':
        Nq_tot=copy(Nq)
    if type(MapSet)==type(None):
        MapSet = JordanWignerSet(Nq,Nq_tot,reduced=False)
    coeff = [op.qCo]
    if MapSet.reduced and Nq==MapSet.Nq:
        pauli=['I'*(Nq_tot)]
    else:
        pauli = ['I'*MapSet.Nq_tot]
    for q,o in zip(op.qInd[::-1],op.qOp[::-1]):
        p1s,c1s,p2s,c2s = [],[],[],[]
        for p,c in zip(pauli,coeff):
            c1,c2 = [],[]
            if o=='+':
                tc1,tp1 = _commutator_relations(
                        'X',p[q])
                tc2,tp2 = _commutator_relations(
                        'Y',p[q])
                p1 = p[:q]+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(-1j*c*0.5*tc2)
            elif o=='-':
                tc1,tp1 = _commutator_relations(
                        'X',p[q])
                tc2,tp2 = _commutator_relations(
                        'Y',p[q])
                p1 = p[:q]+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(1j*c*0.5*tc2)
            if o in ['+','-']:
                for i in range(q):
                    nc1,np1 = _commutator_relations(
                            'Z',p1[i])
                    nc2,np2 = _commutator_relations(
                            'Z',p2[i])
                    p1 = p1[:i]+np1+p1[i+1:]
                    p2 = p2[:i]+np2+p2[i+1:]
                    c1[0]*=nc1
                    c2[0]*=nc2
            elif o in ['1','p']:
                tc1,tp1 =1,p[q]
                tc2,tp2 = _commutator_relations(
                        'Z',p[q])
                p1 = p[:q]+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(-1*c*0.5*tc2)
            elif o in ['0','h']:
                tc1,tp1 =1,p[q]
                tc2,tp2 = _commutator_relations(
                        'Z',p[q])
                p1 = p[:q]+tp1+p[q+1:]
                p2 = p[:q]+tp2+p[q+1:]
                c1.append(c*0.5*tc1)
                c2.append(c*0.5*tc2)
            p1s.append(p1)
            p2s.append(p2)
            c1s+= c1
            c2s+= c2
        pauli = p1s+p2s
        coeff = c1s+c2s
    '''
    if MapSet.reduced:
        # first, transform Paulis
        Op = Operator()
        for p,c in zip(pauli,coeff):
            Op+= PauliOperator(p,c)
        OpT = (MapSet.U*Op)*MapSet.Ut
        pauli = []
        coeff = []
        for op in OpT._op:
            pauli.append(op.p)
            coeff.append(op.c)
        for q,p,e in zip(MapSet.red_q,MapSet.red_p,MapSet.red_e):
            npauli =[]
            ncoeff= []
            for n in range(len(pauli)):
                tc = copy(coeff[n])
                if pauli[n][q]==p:
                    tc = tc*e
                elif pauli[n][q]=='I':
                    pass
                else:
                    tc = 0
                npauli.append(pauli[n][:q]+pauli[n][q+1:])
                ncoeff.append(tc)
            pauli = npauli[:]
            coeff = ncoeff[:]
    '''
    return pauli,coeff


def ModifiedJW(operator,**kw):
    if isinstance(operator,type(QuantumString())):
        return ModifiedJordanWignerTransform(operator)
    else:
        new = Operator()
        for op in operator:
            new+= ModifiedJordanWignerTransform(op)
        return new

def JordanWigner(operator,

        **kw
        ):
    if isinstance(operator,type(QuantumString())):
        return JordanWignerTransform(operator)
    else:
        new = Operator()
        for op in operator:
            new+= JordanWignerTransform(op)
        return new

