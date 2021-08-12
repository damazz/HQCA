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

