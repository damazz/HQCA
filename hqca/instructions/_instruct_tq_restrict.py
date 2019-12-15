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
        xy,yx= 0,0
        for item in operator.op:
            if item.p=='XY':
                xy=item.c
            elif item.p=='YX':
                yx= item.c
        
        self._gates.append(
                [(
                    xy,yx
                    ),
                    xy_yx_gate
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
                zi = item.c
            elif item.p=='IZ':
                iz = item.c
            elif item.p=='ZZ':
                zz = item.c
            elif item.p=='XX':
                xx = item.c

        self._gates.append(
                [(
                    zi,iz,
                    zz,xx,
                    scaleH,
                    ),
                    h2_hamiltonian
                    ]
                )


