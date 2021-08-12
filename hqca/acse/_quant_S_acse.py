import numpy as np
import sys
from hqca.tools import *
from hqca.operators import *
from hqca.tomography import *
import traceback
from timeit import default_timer as dt

'''
/hqca/acse/_quant_S_acse.py

Will generate elements of the A matrix according to the quantum solution. Requires tomography of the auxillary 2-RDM, aquired with an additional propagator sequence appended to the ansatz. 

'''

def findSPairsQuantum(
        op_type,
        **kw
        ):
    newS = _findFermionicSQuantum(**kw)
    return newS

def _findFermionicSQuantum(
        operator=None,
        instruct=None,
        process=None,
        store=None,
        quantstore=None,
        verbose=False,
        trotter_steps=1,
        S_min=1e-10,
        hamiltonian_step_size=1.0,
        ordering='default',
        depth=1,
        commutative=True,
        tomo=None,
        transform=None,
        propagate='first',
        matrix=False,
        **kw
        ):
    '''
    need to do following:
        3. find S from resulting matrix

    '''
    t0 = dt()
    if propagate=='first':
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
        #('-- ins A: {}'.format(dt() - t0))
        t0 = dt()
        if type(tomo)==type(None):
            print('Recalculating? ')
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
        #print('-- tomo A: {}'.format(dt() - t0))
        t0 = dt()
        newCirc.set(newPsi)
        #print('-- set A: {}'.format(dt()-t0))
        t0 = dt()
        if verbose:
            print('Running circuits...')
        newCirc.simulate(verbose=verbose)
        #print('-- sim A: {}'.format(dt()-t0))
        t0 = dt()
        if verbose:
            print('Constructing the RDMs...')
        hss = (1 / hamiltonian_step_size)
        if matrix:
            newCirc.construct(processor=process,compact=True)
            rdm = np.imag(newCirc.rdm) * hss
        else:
            newCirc.construct(processor=process)
            rdm = np.imag(newCirc.rdm.rdm) * hss

        #print('-- construct A: {}'.format(dt()-t0))
        t0 = dt()
        #rdm = newCirc.rdm.rdm-store.rdm.rdm
    elif propagate=='second':
        Psi1 = instruct(
                operator=operator,
                Nq=quantstore.Nq,
                quantstore=quantstore,
                propagate=True,
                HamiltonianOperator=store.H.qubit_operator,
                scaleH=hamiltonian_step_size,
                depth=depth,
                **kw
                )
        H2 = store.H.qubit_operator*(-1)
        Psi2 = instruct(
                operator=operator,
                Nq=quantstore.Nq,
                quantstore=quantstore,
                propagate=True,
                HamiltonianOperator=H2,
                scaleH=hamiltonian_step_size,
                depth=depth,
                **kw
                )
        if type(tomo)==type(None):
            Circ1 = StandardTomography(
                    quantstore,
                    verbose=verbose,
                    )
            Circ1.generate(real=False,imag=True)
            Circ2 = StandardTomography(
                    quantstore,
                    verbose=verbose,
                    )
            Circ2.generate(real=False,imag=True)
        else:
            Circ1 = StandardTomography(
                    quantstore,
                    preset=True,
                    Tomo=tomo,
                    verbose=verbose,
                    )
            Circ2 = StandardTomography(
                    quantstore,
                    preset=True,
                    Tomo=tomo,
                    verbose=verbose,
                    )
        Circ1.set(Psi1)
        Circ2.set(Psi2)
        if verbose:
            print('Running circuits...')
        Circ1.simulate(verbose=verbose)
        Circ2.simulate(verbose=verbose)
        if verbose:
            print('Constructing the RDMs...')
        Circ1.construct(processor=process)
        Circ2.construct(processor=process)
        #rdm = newCirc.rdm.rdm-store.rdm.rdm
        hss = (1/(2*hamiltonian_step_size))
        rdm = -np.imag(-Circ1.rdm.rdm+Circ2.rdm.rdm)*hss
    #print('Time to run A: {}'.format(dt()-t0))
    t0 = dt()
    if matrix:
        return rdm
    if verbose:
        print('Elements of S from quantum generation: ')
    newF = Operator()
    new = np.transpose(np.nonzero(rdm))
    for index in new:
        ind = tuple(index)
        val = rdm[ind]
        if abs(val)>S_min:
            l = len(ind)
            sop = l//2*'+'+l//2*'-'
            newF+= FermiString(
                    -val,
                    indices=list(ind),
                    ops=sop,
                    N=quantstore.dim,
                    )
    #print('Time to process A: {}'.format(dt()-t0))
    t0 = dt()
    #fullS = newF.transform(quantstore.transform)
    if verbose:
        print('S operator (pre-truncated)...')
        print('Fermionic S operator:')
        print(newF)
    #print('Time to transform A: {}'.format(dt()-t0))
    return newF

