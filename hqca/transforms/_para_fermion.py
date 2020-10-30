
import sys
from copy import deepcopy as copy
from hqca.tools._operator import *
from hqca.tools.quantum_strings import *

def ParaFermionTransform(op):
    '''
    transforms a Qubit string into a Fermionic Operator
    '''
    Nq = len(op.s)
    pauli = ['I'*Nq]
    new = Operator()
    new+= FermiString(
            coeff=op.c,s='i'*Nq)
    # define paulis ops
    for qi,o in enumerate(op.s[::-1]):
        # reversed is because of the order in which we apply cre/ann ops
        q = Nq-qi-1
        if o=='I':
            continue
        if o in ['+','-']:
            # # #
            s1 = 'i'*q +o+(Nq-q-1)*'i'
            c1 = 1
            tem1 = Operator()
            tem1+= FermiString(s=s1,coeff=c1)
            for qj in range(q):
                r = Nq-qj-1-1
                t1 = 'i'*r +'h'+(Nq-r-1)*'i'
                t2 = 'i'*r +'p'+(Nq-r-1)*'i'
                d1,d2 = 1,-1
                tem2 = Operator()
                tem2+= FermiString(s=t1,coeff=d1)
                tem2+= FermiString(s=t2,coeff=d2)
                tem1 = tem2*tem1
        elif o in ['p','h']:
            s1 = 'i'*q+o+(Nq-q-1)*'i'
            c1 = 1
            tem1 = Operator()
            tem1+= FermiString(s=s1,coeff=c1)
        new = tem1*new
    return new

def ParaFermion(operator,
        **kw
        ):
    if isinstance(operator,type(QuantumString())):
        return ParaFermionTransform(operator)
    else:
        new = Operator()
        for op in operator:
            new+= ParaFermionTransform(op)
        return new

