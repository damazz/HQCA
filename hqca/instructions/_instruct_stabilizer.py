from hqca.core.primitives import *
import sys
from hqca.core import *


class StabilizerInstruct(Instructions):
    '''
    Instructions for a particular type of error mitigation. I.e., will act only
    on ancilla qubits to perform certain types of checks.

    '''
    def __init__(self,
            stabilize='simple',
            **kw
            ):
        self._gates = []
        if stabilize=='simple':
            self._apply_parity_check_simple(**kw)
        elif stabilize=='spin':
            self._apply_parity_check_spin(**kw)

    @property
    def gates(self):
        return self._gates

    @gates.setter
    def gates(self,a):
        self._gates = a




    

