from hqca.core.primitives import *
import sys
from hqca.core import *


class RestrictiveSet(Instructions):
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
        self._propagate=propagate
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
        self._xy,self._yx= 0.001,0.001
        for item in operator.op:
            if item.p=='XY':
                self._xy=item.c
            elif item.p=='YX':
                self._yx= item.c

        if not self._propagate:
            self._gates.append(
                    [(
                        self._xy,self._yx
                        ),
                        xy_yx_simple
                        ]
                    )

    def _applyH(self,
            HamiltonianOperator,
            trotter_steps=1,
            scaleH=0.5,**kw):
        '''
        applying a real Hamiltonian for H2....only ZI,IZ,ZZ, and XX terms
        '''

        accepted =['IZ','ZI','ZZ','XX']
        for item in HamiltonianOperator.op:
            if item.p=='ZI':
                self._zi = item.c
            elif item.p=='IZ':
                self._iz = item.c
            elif item.p=='ZZ':
                self._zz = item.c
            elif item.p=='XX':
                self._xx = item.c

        self._gates.append(
                [(
                    self._xy,
                    self._yx,
                    self._zi,
                    self._iz,
                    self._zz,
                    self._xx,
                    scaleH,
                    ),
                    xy_yx_ih_simple
                    ]
                )


