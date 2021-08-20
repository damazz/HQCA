import numpy as np
import sys
from hqca.tools import *
from hqca.operators import *
from hqca.tomography import *
import traceback
from timeit import default_timer as dt
from copy import deepcopy as copy
import warnings


def solveqACSE(
        H=None,
        S_min=1e-6,
        expiH_approx='first',
        matrix=False,
        verbose=False,
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
        raise HamiltoninError('Cannot run qACSE without a qubit Hamiltonian.')
    elif isinstance(H,type(Operator())):
        if expiH_approx=='first':
            rdm = _runexpiH(HamiltonianOperator=H,**kw)
        elif expiH_approx=='second':
            warnings.warn('Second order method is equivalent to first order for the real case!')
            rdm1 = _runexpiH(HamiltonianOperator=H,**kw)
            Hp = copy(H)*(-1)
            rdm2 = _runexpiH(HamiltonianOperator=Hp,**kw)
            rdm = (rdm1-rdm2)/2
    elif isinstance(H,list):
        if expiH_approx=='first':
            rdm = _runexpiH(HamiltonianOperator=H[0],**kw)
            for h in H[1:]:
                rdm+= _runexpiH(HamiltonianOperator=h,**kw)
        elif expiH_approx=='second':
            warnings.warn('Second order method is equivalent to first order for the real case!')
            rdm1 = _runexpiH(HamiltonianOperator=H[0],**kw)
            for h in H[1:]:
                rdm1+= _runexpiH(HamiltonianOperator=h,**kw)
            Hp = H*(-1)
            rdm2 = _runexpiH(HamiltonianOperator=Hp[0],**kw)
            for h in H[1:]:
                rdm2 += _runexpiH(HamiltonianOperator=h,**kw)
            rdm = (rdm1-rdm2)/2
    if matrix:
        return rdm
    else:
        # form fermionic operator
        #if verbose:
        #    print('Elements of S from quantum generation: ')
        newF = Operator()
        nz = np.transpose(np.nonzero(rdm))
        for index in nz:
            ind = tuple(index)
            val = rdm[ind]
            if abs(val)>S_min:
                l = len(ind)
                sop = l//2*'+'+l//2*'-'
                newF+= FermiString(
                        -val,
                        indices=list(ind),
                        ops=sop,
                        N=rdm.shape[0],
                        )
        #if verbose:
        #    print('S operator (pre-truncated)...')
        #    print('Fermionic S operator:')
        #    print(newF)
        return newF

def _runexpiH(
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
    newPsi = instruct(
            operator=operator,
            Nq=quantstore.Nq,
            quantstore=quantstore,
            scaleH=hamiltonian_step_size,
            propagate=True,
            **kw
            )
    t0 = dt()
    if type(tomo)==type(None):
        print('Recalculating tomography....')
        newCirc = StandardTomography(
                quantstore,
                verbose=verbose,
                )
        newCirc.generate(real=False,imag=True)
    else:
        newCirc = StandardTomography(
                quantstore,
                preset=True,
                Tomo=tomo,
                verbose=verbose,
                )
    newCirc.set(newPsi)
    newCirc.simulate(verbose=verbose)
    hss = (1 / hamiltonian_step_size)
    if matrix:
        newCirc.construct(processor=process,compact=True)
        rdm = np.imag(newCirc.rdm) * hss
    else:
        newCirc.construct(processor=process)
        rdm = np.imag(newCirc.rdm.rdm) * hss
    return rdm


