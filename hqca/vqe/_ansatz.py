import numpy as np
from copy import deepcopy as copy




class VariationalAnsatz:
    def __init__(self,
                 T,variables
                 ):
        self.T = T
        self.xi = variables

    def __str__(self):
        return self.T.__str__()

    def evaluate(self,parameters):
        psi = copy(self.T)
        var = self.xi
        for n in psi.keys():
            psi[n].c = psi[n].c.subs([(x,v) for x,v in zip(var,parameters)])
            psi[n].c = np.complex(psi[n].c)
            psi[n].sym=False
        psi.clean()
        return psi


