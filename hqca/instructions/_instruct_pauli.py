from hqca.core.primitives import *
from hqca.core import *


class PauliSet(Instructions):
    '''
    Simple object - generate gates
    '''
    def __init__(self):
        self._gates = []

    def _applyAns(
            Ansatz,
            apply_H=True,
            order=1
            **kw):
        for item in Ansatz.ansatz:
            self._gates.append(
                    [(
                        item.PauliCoeff,
                        item.pauliExp,
                        ),
                        _generic_Pauli_term
                        ]
                    )
        if apply_H:
            self._applyH(method,**kw)


    def _applyH(self,
            trotter_steps,
            Hamiltonian,
            scaleH):
        for i in range(trotter_steps):
            for item in Hamiltonian.operator:
                self._gates.append(
                        [(
                            (1/trotter_steps)*scaleH*item.pauliCoeff,
                            item.pauliExp
                            ),
                            _generic_Pauli_term
                            ]
                        )
