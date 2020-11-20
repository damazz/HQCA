import numpy as np
import sys
from hqca.tools import *
from hqca.state_tomography import *
import traceback

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
        trotter_steps=1,
        S_min=1e-10,
        hamiltonian_step_size=1.0,
        separate_hamiltonian=False,
        ordering='default',
        depth=1,
        parallel=False,
        commutative=True,
        tomo=None,
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
        '''
        going to try and run systems in parallele
        '''
        try:
            #print(store.H._qubOp_sep)
            rdm = np.zeros(store.rdm.rdm.shape)
            tomo_list = []
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
                if not parallel:
                    print('Running Hi...')
                    print(opH)
                    newCirc.simulate()
                    newCirc.construct(processor=process)
                    rdm+= np.imag(newCirc.rdm.rdm)
                    new = np.transpose(np.nonzero(newCirc.rdm.rdm))
                    hss = (1/hamiltonian_step_size)
                    if verbose:
                        print('Elements of S from quantum generation: ')
                        newF = Operator()
                        for index in new:
                            ind = tuple(index)
                            val = np.imag(newCirc.rdm.rdm[ind])*hss
                            if abs(val)>S_min:
                                l = len(ind)
                                sop = l//2*'+'+l//2*'-'
                                newF+= FermiString(
                                        -val,
                                        indices=list(ind),
                                        ops=sop,
                                        N=quantstore.dim,
                                        )
                        print(newF.transform(quantstore.transform))
            if parallel:
                print('Running circuits...')
                run_multiple(tomo_list,quantstore,verbose=verbose)
                for t,o in zip(tomo_list,store.H._qubOp_sep):
                    t.construct(processor=process)
                    rdm+= np.imag(t.rdm.rdm)
            hss = (1/hamiltonian_step_size)
        except Exception as e:
            traceback.print_exc(e)
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
    if verbose:
        print('Elements of S from quantum generation: ')
    newF = Operator()
    for index in new:
        ind = tuple(index)
        val = rdm[ind]*hss
        if abs(val)>S_min:
            if quantstore.op_type=='fermionic':
                l = len(ind)
                sop = l//2*'+'+l//2*'-'
                newF+= FermiString(
                        -val,
                        indices=list(ind),
                        ops=sop,
                        N=quantstore.dim,
                        )
    fullS = newF.transform(quantstore.transform)
    if verbose:
        print('S operator (pre-truncated)...')
        print('Fermionic S operator:')
        print(newF)
    return fullS

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
    print('Elements of S from quantum generation: ')
    newS = Operator()
    for index in newRho:
        ind = tuple(index)
        val = RDM.rdm[ind]*hss*c
        if abs(val)>=S_min:
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
