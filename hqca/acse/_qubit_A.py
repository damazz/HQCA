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
    else:
        raise QuantumRunError(print(type(H)))
    #
    if matrix:
        return -rdm
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
                #print(ind,val)
                l = len(ind)
                sop = l//2*'+'+l//2*'-'
                newF+= QubitString(
                        val,
                        indices=list(ind),
                        ops=sop,
                        N=rdm.shape[0],
                        )
        #print(newF)
        #if verbose:
        #    print('S operator (pre-truncated)...')
        #    print('Fermionic S operator:')
        #    print(newF)
        return newF,np.linalg.norm(rdm,ord=norm)

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
        newCirc = QubitTomography(
                quantstore,
                verbose=verbose,
                )
        newCirc.generate(real=False,imag=True)
    else:
        newCirc = QubitTomography(
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

def findQubitAQuantum(
        operator=None,
        instruct=None,
        process=None,
        store=None,
        quantstore=None,
        verbose=False,
        S_min=1e-10,
        hamiltonian_step_size=1.0,
        depth=1,
        parallel=False,
        commutative=True,
        tomo=None,
        transform=None,
        matrix=False,
        **kw
        ):
    '''
    need to do following:
        3. find S from resulting matrix

    '''
    if verbose:
        print('Generating new S pairs with Hamiltonian step.')
    newPsi = instruct(
            operator=operator,
            Nq=quantstore.Nq,
            quantstore=quantstore,
            propagate=True,
            HamiltonianOperator=store.H.qubit_operator,
            scaleH=hamiltonian_step_size,
            depth=depth,
            **kw
            )
    if type(tomo)==type(None):
        newCirc = QubitTomography(
                quantstore,
                verbose=verbose,
                )
        newCirc.generate(real=False,imag=True)
    else:
        newCirc = QubitTomography(
                quantstore,
                preset=True,
                Tomo=tomo,
                verbose=verbose,
                )
    newCirc.set(newPsi)
    hss = (1/hamiltonian_step_size)
    if verbose:
        print('Running circuits...')
    newCirc.simulate(verbose=verbose)
    if verbose:
        print('Constructing the RDMs...')
    if matrix:
        newCirc.construct(processor=process, compact=True)
        rdm = np.imag(newCirc.rdm) * hss
        return rdm
    else:
        newCirc.construct(processor=process)
        rdm = np.imag(newCirc.rdm.rdm) * hss

    #rdm = newCirc.rdm.rdm-store.rdm.rdm

    rdm = np.imag(newCirc.rdm.rdm)
    new = np.transpose(np.nonzero(rdm))
    if verbose:
        print('Elements of A from quantum generation: ')
    newF = Operator()
    for index in new:
        ind = tuple(index)
        val = rdm[ind]*hss
        if abs(val)>S_min:
            l = len(ind)
            sop = l//2*'+'+l//2*'-'
            newF+= QubitString(
                    val,
                    indices=list(ind),
                    ops=sop,
                    N=quantstore.dim,
                    )
    #fullS = newF.transform(quantstore.transform)
    if verbose:
        print('Qubit A operator:')
        print(newF)
    return newF

