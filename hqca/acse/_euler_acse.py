from copy import copy as copy
import numpy as np
from functools import partial
from hqca.tomography import *
from hqca.operators import *

def _euler_step(acse):
    ''' carries out the Euler step for the ACSE

    Euler step following the gradient direction (-A, where A
    is the gradient or residuals of the ACSE)

    '''
    testS = copy(acse.A)
    for s in testS:
        s.c*= -1*acse.epsilon
    acse.psi = acse.psi+testS
    circ = acse._generate_circuit()
    #
    en = np.real(acse.store.evaluate(circ.rdm))
    acse.store.update(circ.rdm)
    acse.circ = circ
    nz = np.nonzero(circ.rdm.rdm)
