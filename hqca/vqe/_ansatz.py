import numpy as np
from copy import deepcopy as copy
from hqca.acse._ansatz_S import Ansatz
from hqca.core import *
from hqca.operators import *

class VariationalAnsatz:
    # T is a qubit operator (ordered)
    def __init__(self,
                 T,variables
                 ):
        self.T = T
        self.xi = variables

    def __str__(self):
        return self.T.__str__()

    def evaluate(self,parameters,T):
        psi = copy(self.T)
        var = self.xi
        for n in psi.keys():
            psi[n].c = psi[n].c.subs([(x,v) for x,v in zip(var,parameters)])
            psi[n].c = np.complex(psi[n].c)
            psi[n].sym=False
        psi.clean()
        return psi.transform(T)

class ADAPTAnsatz(Ansatz):
    '''
    using a modified form of the ACSE ansatz

    how do we want to do it

    - ansatz is an operator, need it to be variable?
    '''

    def add_term(self,
            indices=None):
        self.A.append(indices)


    def assign_variables(self,parameters,T=None,fermi=True,N=4):
        '''
        adds parameters to list?
        '''
        ansatz = []
        for a,p in zip(self.A,parameters):
            new = Operator()
            new += FermiString(
                p,
                indices=a,
                ops='++--',
                N=N,
                )
            new -= FermiString(
                p,
                indices=a[::-1],
                ops='++--',
                N=N,
                )
            for op in new.transform(T):
                ansatz.append(op)
        return ansatz

