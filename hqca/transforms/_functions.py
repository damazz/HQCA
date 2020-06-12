from copy import deepcopy as copy
import sys
from hqca.tools._operator import *
from hqca.tools.quantum_strings import *

def trim_operator(ops,
        qubits=[],
        paulis=[],
        eigvals=[],
        null=0,
        ):
    new = Operator()
    if not qubits==sorted(qubits)[::-1]:
        sys.exit('Reorder your trimming operations to ensure qubit ordering.')
    for op in ops:
        s,c = op.s,op.c
        for q,p,e in zip(qubits,paulis,eigvals):
            if s[q]==p:
                c*= e
            elif s[q]=='I':
                pass
            else:
                c*=null
            s = s[:q]+s[q+1:]
        new+= PauliString(s,c)
    return new

def change_basis(op,
        U,
        Ut=None,
        **kw):
    if type(Ut)==type(None):
        Ut = copy(U)
    return (U*op)*Ut

