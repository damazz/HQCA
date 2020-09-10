import numpy as np
import sys
from hqca.tools import *
from hqca.state_tomography import *

'''
/hqca/acse/_quant_S_acse.py

Will solve for the ACSE through the use of the time evolution operator

'''

def findSPairsQuantum(
        op_type,
        **kw
        ):
    if op_type=='fermionic':
        newS = _findFermionicSQuantum(**kw)
    elif op_type=='qubit':
        newS = _findQubitSQuantum(**kw)
    return newS

def _findFermionicSQuantum(
        operator=None,
        instruct=None,
        process=None,
        store=None,
        quantstore=None,
        verbose=False,
        separate=False,
        trotter_steps=1,
        qS_thresh_rel=0.1,
        qS_max=1e-10,
        qS_screen=0.1,
        hamiltonian_step_size=1.0,
        separate_hamiltonian=False,
        ordering='default',
        depth=1,
        commutative=True,
        tomo=None,
        piecewise=False,
        transform=None,
        **kw
        ):
    '''
    need to do following:
        3. find S from resulting matrix
    '''
    if verbose:
        print('Generating new S pairs with Hamiltonian step.')
    if separate_hamiltonian:
        try:
            #store.H._qubOp_sep
            rdm = np.zeros(store.rdm.rdm.shape)
            for opH in store.H._qubOp_sep:
                newPsi = instruct(
                        operator=operator,
                        Nq=quantstore.Nq,
                        quantstore=quantstore,
                        propagate=True,
                        HamiltonianOperator=opH,
                        scaleH=hamiltonian_step_size,
                        depth=depth,
                        **kw
                        )
                if type(tomo)==type(None):
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
                if verbose:
                    print('Running circuits...')
                newCirc.simulate(verbose=verbose)
                if verbose:
                    print('Constructing the RDMs...')
                newCirc.construct(processor=process)
                #rdm = newCirc.rdm.rdm-store.rdm.rdm
                rdm+= np.imag(newCirc.rdm.rdm)
                temp= np.transpose(np.nonzero(newCirc.rdm.rdm))
                hss = (1/hamiltonian_step_size)
                max_val = 0
                if verbose:
                    print('Contributions from H_i - ')
                    print('H_i: ')
                    print(opH)
                    print('S_i')
                    for inds in temp:
                        val = newCirc.rdm.rdm[tuple(inds)]*hss
                        if abs(val)>1e-5:
                            print(val,inds)
        except Exception as e:
            print(e)
            sys.exit('Error in Hamiltonian separation.')
    else:
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
        if verbose:
            print('Running circuits...')
        newCirc.simulate(verbose=verbose)
        if verbose:
            print('Constructing the RDMs...')
        newCirc.construct(processor=process)
        #rdm = newCirc.rdm.rdm-store.rdm.rdm
        rdm = np.imag(newCirc.rdm.rdm)
    new = np.transpose(np.nonzero(rdm))
    hss = (1/hamiltonian_step_size)
    max_val = 0
    for inds in new:
        ind = tuple(inds)
        v = abs(rdm[ind])*hss
        if v>max_val:
            max_val = v
    if verbose:
        print('Elements of S from quantum generation: ')
    newF = Operator()
    for index in new:
        ind = tuple(index)
        val = rdm[ind]*hss
        if abs(val)>qS_thresh_rel*max_val and abs(val)>qS_max:
            if quantstore.op_type=='fermionic':
                l = len(ind)
                sop = l//2*'+'+l//2*'-'
                newEl = FermiString(
                        -val,
                        indices=list(ind),
                        ops=sop,
                        N=quantstore.dim,
                        )
                newF+= newEl
    newS = newF.transform(quantstore.transform)
    if verbose:
        print('Fermionic S operator:')
        print(newF)
        print('Qubit S operator: ')
        print(newS)
    if commutative:
        newS.ca=True
    else:
        newS.ca=False
    return newS

def _findQubitSQuantum(
        operator,
        instruct,
        store,
        quantstore,
        verbose=False,
        separate=False,
        trotter_steps=1,
        qS_thresh_rel=0.1,
        qS_max=1e-10,
        qS_screen=0.1,
        hamiltonian_step_size=1.0,
        ordering='default',
        propagate_method='trotter',
        depth=1,
        commutative=True,
        tomo=None,
        process=None,
        ):
    sys.exit('Update qubits ACSE.')
    '''
    need to do following:
        1. prepare the appropriate Hailtonian circuit
        2. implement it
        3. find S from resulting matrix
    '''
    if verbose:
        print('Generating new S pairs with Hamiltonian step.')
    newPsi = instruct(
            operator=operator,
            Nq=quantstore.Nq,
            propagate=True,
            trotter_steps=trotter_steps,
            HamiltonianOperator=store.H.qubit_operator,
            scaleH=hamiltonian_step_size,
            depth=depth,
            propagate_method=propagate_method,
            )
    if type(tomo)==type(None):
        newCirc = StandardTomography(
                quantstore,
                verbose=verbose
                )
        newCirc.generate(real=True,imag=True)
    else:
        newCirc = StandardTomography(
                quantstore,
                preset=True,
                Tomo=tomo,
                verbose=verbose
                )
    newCirc.set(newPsi)
    if verbose:
        print('Running circuits...')
    newCirc.simulate(verbose=verbose)
    if verbose:
        print('Constructing the RDMs...')
    newCirc.construct(rdm=store.rdm)
    if store.H.real and store.H.imag:
        RDM = newCirc.rdm - store.rdm
    elif not store.H.imag and store.H.real:
        RDM = newCirc.rdm
    else:
        print(store.H.imag,store.H.real)
        sys.exit('Problem in H')
    if verbose:
        print('Current RDM')
        print(store.rdm.rdm)
    rdmRe = np.real(RDM.rdm)
    rdmIm = np.imag(RDM.rdm)
    if verbose:
        print('RDM:')
        print(newCirc.rdm.rdm)
        print(RDM.rdm)
    if quantstore.Nq==1 and verbose:
        print('Z1: {}'.format(rdmRe[0,0,0]))
        print('Z2: {}'.format(rdmRe[0,1,1]))
        print('X:  {}'.format(rdmRe[0,0,1]))
        print('Y:  {}'.format(rdmIm[0,0,1]))
        print('Real rdm...')
        print(rdmRe)
        print('Imaginary rdm...')
        print(rdmIm)
    newRe = np.transpose(
            np.nonzero(
                rdmRe
                )
            )
    newIm = np.transpose(
            np.nonzero(
                rdmIm
                )
            )
    newRho = np.transpose(
            np.nonzero(
                RDM.rdm
                )
            )
    hss = (1/hamiltonian_step_size)
    if quantstore.Nq==1:
        c = 2
    elif quantstore.Nq==2:
        c = 4
    max_val = 0
    for inds in newRe:
        ind = tuple(inds)
        v = abs(rdmRe[ind])*hss
        if v>max_val:
            max_val = v
    for inds in newIm:
        ind = tuple(inds)
        v = abs(rdmIm[ind])*hss*c
        if v>max_val:
            max_val = v
    print('Elements of S from quantum generation: ')
    newS = Operator()
    for index in newRho:
        ind = tuple(index)
        val = RDM.rdm[ind]*hss*c
        if abs(val)>qS_thresh_rel*max_val and abs(val)>qS_max:
            i = RDM.mapping[ind[0]]
            sq = RDM.sq_map[tuple(ind[1:])]
            newEl = QubitOperator(
                    val,
                    indices=i,
                    sqOp=sq,
                    add=True,
                    )
            newEl.generateOperators(Nq=quantstore.Nq,real=True,imag=True,
                        mapping=quantstore.mapping,
                        **quantstore._kw_mapping,
                        )
            newS+= newEl.formOperator()
    print(newS)
    if commutative:
        pass
    else:
        for i in newS.op:
            i.add=False
    #newS.reordering(method='hamiltonian',qubOpH=store.H.qubOp)
    return newS
