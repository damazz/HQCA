import numpy as np
from copy import deepcopy as copy
from hqca.acse._ansatz_S import Ansatz
from hqca.core import *
from hqca.operators import *

class VariationalAnsatz:
    # T is a qubit operator (ordered)
    def __init__(self,
            indices=None,
                 ):
        self.T = indices

    def __str__(self):
        return self.T.__str__()

    def assign(self,parameters,T,N=4):
        psi = []
        for t,p in zip(self.T,parameters):
            dim = len(t)//2
            new = Operator()
            if abs(p)<1e-10:
                continue
            new += FermiString(
                p,
                indices=t,
                ops='+'*dim+'-'*dim,
                N=N,
                )
            new -= FermiString(
                p,
                indices=t[::-1],
                ops='+'*dim+'-'*dim,
                N=N,
                )
            for op in new.transform(T):
                psi.append(op)
        return psi

class ADAPTAnsatz(Ansatz):
    '''
    using a modified form of the ACSE ansatz

    how do we want to do it

    - ansatz is an operator, need it to be variable?
    '''
    def __init__(self,*args,**kwargs):
        Ansatz.__init__(self,*args,**kwargs)
        self.A_ind = []

    def add_term(self,
            indices=None,
            ind_key = 0):
        self.A.append(indices)
        self.A_ind.append(ind_key)

    def assign_variables(self,parameters,T=None,fermi=True,N=4):
        '''
        adds parameters to list?
        '''
        ansatz = []
        if fermi:
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
                nop = new.transform(T)
                for op in nop:
                    ansatz.append(op)
        else:
            for a,p in zip(self.A,parameters):
                new = Operator()
                new += QubitString(
                    p,
                    indices=a,
                    ops='++--',
                    N=N,
                    )
                new -= QubitString(
                    p,
                    indices=a[::-1],
                    ops='++--',
                    N=N,
                    )
                nop = new.transform(T)
                for op in nop:
                    ansatz.append(op)
        return ansatz

