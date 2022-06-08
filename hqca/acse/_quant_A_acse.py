import numpy as np
from hqca.core import *
import sys
from hqca.tools import *
from hqca.operators import *
from hqca.tomography import *
import traceback
from timeit import default_timer as dt
from copy import deepcopy as copy
import warnings

def solveqACSE(
        acse,
        H=None,
        S_min=1e-6,
        expiH_approx='first',
        matrix=False,
        verbose=False,
        norm='fro',
        **kw
        ):
    ''' Solves the ACSE using a quantum computer with 2-RDM scaling
    provides a solution to the qACSE

    Note, by applying e^iH, we strictly obtain the residuals which are 
    traditionally defined as:

    That it, 
    < e^-idH G_k a^idH > = <G_k> - i*d <[H,G_k]>
                         = <G_k> + i*d <[G_k,H]>
        
    Or, similar to the classical ACSE. 
    A^ik_jl = < [a_i+ a_k+ a_l a_j ,H] >

    For continuity across all of the solvers, we instead return the gradient,
    or minus the above expression (also obtained by applying e^{-i*delta*H}).
    '''
    kw['matrix']=matrix
    if type(H)==type(None):
        raise HamiltoninError('Cannot run qACSE without a qubit Hamiltonian.')
    elif isinstance(H,type(Operator())):
        rdm = _runexpiH(acse,HamiltonianOperator=H,**kw)
    elif isinstance(H,list):
        rdm = _runexpiH(acse,HamiltonianOperator=H[0],**kw)
        for h in H[1:]:
            rdm+= _runexpiH(acse,HamiltonianOperator=h,**kw)
    else:
        raise QuantumRunError(print(type(H)))
    #
    if matrix:
        return -rdm
    else:
        new = Operator()
        nz = np.nonzero(rdm)#np.transpose(np.nonzero(rdm))
        norm = 0
        for i,k,j,l in zip(nz[0],nz[1],nz[2],nz[3]):
            term = rdm[i,k,j,l]
            if abs(term)>=S_min:
                #print(i,k,j,l,term)
                new+= FermiString(
                        coeff=-term,
                        indices=[i,k,l,j],
                        ops='++--',
                        N = rdm.shape[0],
                        )
                norm+= np.real(np.conj(term)*term)
        assert abs(np.sqrt(norm)-np.linalg.norm(rdm)) <1e-8
        return new,0.5*np.sqrt(norm)

def _runexpiH(
        acse,
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
        HamiltonianOperator=None,
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


