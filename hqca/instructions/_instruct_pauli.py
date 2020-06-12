from hqca.core.primitives import *
import sys
from hqca.core import *
from sympy import re,im
import numpy as np

class PauliSet(Instructions):
    '''
    Simple object - generate gates from an operator.

    Default interpretation here is to generate the exponential of the operators
    Pauli strings which are in each gates operator object.

    If no simplification has happened beforehand, then none is applied here.
    '''
    def __init__(self,
            operator,
            Nq,
            propagate=False,
            initial_state=[],
            **kw
            ):
        self._gates = []
        self._applyOp(operator,Nq,**kw)
        if propagate:
            self._applyH(**kw)

    def clear(self):
        self._gates = []
        pass

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a

    def _set_initial(self):
        pass

    def _applyOp(self,
            operator,Nq,depth=1,
            **kw):
        for d in range(depth):
            for item in operator:
                if item.sym:
                    if abs(im(item.c))<1e-10:
                        c = np.real(item.c)
                    elif abs(re(item.c))<1e-10:
                        c =  np.imag(item.c)
                    if abs(c)>1e-10:
                        self._gates.append(
                                [(
                                    c/depth,
                                    item.s,
                                    ),
                                    generic_Pauli_term
                                    ]
                                )
                else:
                    try:
                        item.s
                        if abs(item.c.imag)<1e-10:
                            c = item.c.real
                        elif abs(item.c.real)<1e-10:
                            c =  item.c.imag
                        if abs(c)>1e-10:
                            self._gates.append(
                                    [(
                                        c/depth,
                                        item.s,
                                        ),
                                        generic_Pauli_term
                                        ]
                                    )
                    except AttributeError:
                        sys.exit('Something wrong in instructions.')
                    except Exception as e:
                        print(e)
                        sys.exit()

    def _applyH(self,
            HamiltonianOperator,
            trotter_steps=1,
            scaleH=0.5,**kw):
        for i in range(trotter_steps):
            for item in HamiltonianOperator:
                self._gates.append(
                        [(
                            (1/trotter_steps)*scaleH*item.c.real,
                            item.s,
                            ),
                            generic_Pauli_term
                            ]
                        )

