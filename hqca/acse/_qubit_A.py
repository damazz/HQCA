import numpy as np
import sys
from hqca.tools import *
from hqca.tomography import *
from hqca.operators import *
import traceback

'''
/hqca/acse/_quant_S_acse.py

Will generate elements of the A matrix according to the quantum solution. Requires tomography of the auxillary 2-RDM, aquired with an additional propagator sequence appended to the ansatz.
'''
from hqca.core import *
from timeit import default_timer as dt
from copy import deepcopy as copy
import warnings


def solvepACSE(
        acse,
        H=None,
        S_min=1e-6,
        expiH_approx='first',
        matrix=False,
        verbose=False,
        norm='fro',
        **kw
        ):
    '''
    provides a solution to the qACSE

    expiH_approx refers to a first or second order approximation in e^iHx 
    H indicates the Hamiltonian operator provided

    TODO: Change imag/real tomography to be system specific or implied from the Hamiltonian operator (currently second order is redundant)

    additionally, we can perform a summand over many H operators
    '''
    kw['matrix']=matrix
    if type(H)==type(None):
        raise HamiltonianError('Cannot run qACSE without a qubit Hamiltonian.')
    elif isinstance(H,type(Operator())):
        if expiH_approx=='first':
            rdm = _runexpiH(acse,HamiltonianOperator=H,**kw)
    elif isinstance(H,list):
        if expiH_approx=='first':
            rdm = _runexpiH(acse,HamiltonianOperator=H[0],**kw)
            for h in H[1:]:
                rdm+= _runexpiH(acse,HamiltonianOperator=h,**kw)
    else:
        raise QuantumRunError(print(type(H)))
    #
    if matrix:
        return -rdm
    else:
        # form fermionic operator
        #if verbose:
        #    print('Elements of S from quantum generation: ')
        new = Operator()
        nz = np.nonzero(rdm)
        for i,k,j,l in zip(nz[0],nz[1],nz[2],nz[3]):
            term = rdm[i,k,j,l]
            if abs(term)>=S_min:
                print(i,k,j,l,term)
                new += QubitString(
                        coeff=-term,
                        indices=[i,k,l,j],
                        ops='++--',
                        N = rdm.shape[0],
                        )
        print(np.linalg.norm(rdm))
        return new,0.5*np.linalg.norm(rdm,ord=norm)

def _runexpiH(
        acse,
        HamiltonianOperator=None,
        operator=None,
        instruct=None,
        process=None,
        store=None,
        quantstore=None,
        verbose=False,
        S_min=1e-10,
        hamiltonian_step_size=0.1,
        tomo=None,
        transform=None,
        matrix=False,
        **kw
        ):
    '''
    subroutine for running an exp iO circuit where O is an operator
    '''
    t0 = dt()
    if verbose:
        print('Generating new S pairs with Hamiltonian step.')
    circ = acse._generate_circuit(operator,
            tomo=tomo,
            compact=matrix,
            ins_kwargs={
                'scaleH':hamiltonian_step_size,
                'propagate':'real',
                'HamiltonianOperator':HamiltonianOperator,
                }
            )
    hss = (1 / hamiltonian_step_size)
    if matrix:
        rdm = np.imag(circ.rdm.rdm) * hss
    else:
        rdm = np.imag(circ.rdm.rdm) * hss
    return rdm

