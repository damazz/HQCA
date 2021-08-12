import sys
from copy import deepcopy as copy
from hqca.operators import *
import numpy as np
from numpy import zeros,int8,complex128
from timeit import default_timer as dt
import multiprocessing as mp
from hqca.core import TransformError
#from multiprocessing import set_start_method
#set_start_method('spawn')


def JordanWignerTransform(op):
    '''
    transforms a fermistrings into a operators of paulistrings
    '''
    Nq = len(op.s)
    pauli = ['I'*Nq]
    new = Operator()
    new+= PauliString('I'*Nq,op.c,symbolic=op.sym)
    # define paulis ops
    for qi,o in enumerate(op.s[::-1]):
        # reversed is because of the order in which we apply cre/ann ops
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
        else:
            message = 'Not recognized string {} in op: {}'.format(o,op.s)
            raise TransformError(message)
        tem = Operator()
        tem+= PauliString(s1,c1,symbolic=op.sym)
        tem+= PauliString(s2,c2,symbolic=op.sym)
        new = tem*new
    return new

def symplectic_product(l1,r1,l2,r2,c1,c2):
    l = l1^l2  #mod 2 addition
    r = r1^r2
    L = np.add(l1,l2) #mod 4 quantity
    R = np.add(r1,r2)
    phase = (1j)**np.sum(np.subtract(r1&l2, r2&l1))
    phase*= (1j)**(np.sum(np.subtract(np.multiply(L,R),l&r)))
    return l,r,c1*c2*phase

# 

def JordanWigner(operator,
        **kw
        ):
    if isinstance(operator,type(QuantumString())):
        return JordanWignerTransform(operator)
    else:
        raise TransformError("Can not feed operator into transform anymore.")
