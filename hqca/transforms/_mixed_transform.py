import sys
from copy import deepcopy as copy
from hqca.operators import *
from functools import partial

def get_mixed_transform(
        **kwargs
        ):
    return partial(
            MixedTransform,
            **kwargs)

def MixedTransform(operator,
        qubit_transform=None,
        fermi_transform=None,
        pauli_transform=None,
        ):
    if isinstance(operator,type(FermiString())):
        return fermi_transform(operator)
    elif isinstance(operator,type(QubitString())):
        return qubit_transform(operator)
    elif isinstance(operator,type(PauliString())):
        return pauli_transform(operator)
    elif isinstance(operator,type(OrderedString())):
        new = Operator()
        for n,i in enumerate(operator.op):
            if n==0:
                if i.stype=='p':
                    new+= pauli_transform(i)
                elif i.stype=='q':
                    new+= qubit_transform(i)
                elif i.stype=='f':
                    new+= fermi_transform(i)
            else:
                if i.stype=='p':
                    new*= pauli_transform(i)
                elif i.stype=='q':
                    new*= qubit_transform(i)
                elif i.stype=='f':
                    new*= fermi_transform(i)
        return new
    else:
        return None



