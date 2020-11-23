import sys
from copy import deepcopy as copy
from hqca.tools._operator import *
from hqca.tools.quantum_strings import *

def QubitTransform(op):
    '''
    transforms a qubit operator into a Pauli one 
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
            s1 = 'I'*q+'X'+(Nq-q-1)*'I'
            s2 = 'I'*q+'Y'+(Nq-q-1)*'I'
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


def Qubit(operator,
        **kw
        ):
    if isinstance(operator,type(QuantumString())):
        return QubitTransform(operator)
    else:
        new = Operator()
        for op in operator:
            new+= QubitTransform(op)
        return new

