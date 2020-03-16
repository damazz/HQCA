from hqca.core.primitives import *
import sys
from hqca.core import *


class Stabilizer(Instructions):
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

    def _apply_parity_check_simple(self,quantstore=None,**kw):
        if not quantstore.Nq_anc==1:
            print('Error in number of ancilla needed for parity check.')
            print('Need 1 ancilla.')
            sys.exit()
        elif not quantstore.mapping in ['jordan-wigner','jw']:
            sys.exit('Stabilizer not configured for non-JW.')
        else:
            Nanc = quantstore.Nq
            for i in range(quantstore.Nq):
                self.gates.append([(i,Nanc),Cx])


    def _apply_parity_check_spin(self,quantstore=None,**kw):
        if not quantstore.Nq_anc==2:
            print('Error in number of ancilla needed for parity check.')
            print('Need 1 ancilla.')
            sys.exit()
        elif not quantstore.mapping in ['jordan-wigner','jw']:
            print(quantstore.mapping)
            sys.exit('Stabilizer not configured for non-JW.')
        else:
            Nanc1 = quantstore.Nq
            Nanc2 = quantstore.Nq+1
            Nso = quantstore.Nq//2
            for i in range(Nso):
                self.gates.append([(i,Nanc1),Cx])
            for i in range(Nso,2*Nso):
                self.gates.append([(i,Nanc2),Cx])

