import numpy as np
import sys
from hqca.tools import *
from hqca.state_tomography import *

'''
/hqca/acse/FunctionsQACSE.py

Contains functions for performing ACSE calculations, with a focus on generating
the S matrix through time evolution of the Hamiltonian. 
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
        store=None,
        quantstore=None,
        verbose=False,
        separate=False,
        trotter_steps=1,
        qS_thresh_rel=0.1,
        qS_max=1e-10,
        qS_screen=0.1,
        hamiltonian_step_size=1.0,
        ordering='default',
        depth=1,
        tomo=None,
        ):
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
            HamiltonianOperator=store.H.qubit_operator,
            scaleH=hamiltonian_step_size,
            depth=depth
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
    newCirc.construct()
    rdm = np.imag(newCirc.rdm.rdm)
    new = np.transpose(np.nonzero(rdm))
    hss = (1/hamiltonian_step_size)
    max_val = 0
    for inds in new:
        ind = tuple(inds)
        v = abs(rdm[ind])*hss
        if v>max_val:
            max_val = v
    print('Elements of S from quantum generation: ')
    newS = []
    for index in new:
        ind = tuple(index)
        val = rdm[ind]*hss
        if abs(val)>qS_thresh_rel*max_val and abs(val)>qS_max:
            if quantstore.op_type=='fermionic':
                spin = ''
                for item in ind:
                    c = item in quantstore.alpha['active']
                    spin+= 'a'*c+(1-c)*'b'
                l = len(ind)
                sop = l//2*'+'+l//2*'-'
                newEl = FermionicOperator(
                        -val,
                        indices=list(ind),
                        sqOp=sop,
                        spin=spin
                        )
            if len(newS)==0:
                newS.append(newEl)
                print(newEl)
            else:
                add = True
                for o in newS:
                    if o.isSame(newEl) or o.isHermitian(newEl):
                        add = False
                        break
                if add:
                    newS.append(newEl)
                    if verbose:
                        st = 'S: {}: {}'.format(str(ind),val)
                        print(st)
    hold_type = [(op.opType=='de') for op in newS]
    if ordering=='default':
        new_S_ord_de_a = []
        new_S_ord_de_b = []
        new_S_ord_ne = []
        for i in range(len(hold_type)):
            if hold_type[i]:
                if abs(newS[i].qCo)>qS_screen*max_val:
                    new_S_ord_de_a.append(newS[i])
                else:
                    new_S_ord_de_b.append(newS[i])
        new_S_ord_de = new_S_ord_de_a+new_S_ord_de_b
        for i in range(len(hold_type)):
            if not hold_type[i]:
                new_S_ord_ne.append(newS[i])
        newS = new_S_ord_de[:]+new_S_ord_ne[:]
    elif ordering=='magnitude':
        done = False
        new_S_ord = []
        limit = np.copy(max_val)
        while not done:
            for i in range(len(hold_type)):
                if hold_type[i]:
                    if abs(newS[i].c)<=limit:
                        if abs(newS[i].c)>limit*0.1:
                            new_S_ord.append(newS[i])
            for i in range(len(hold_type)):
                if not hold_type[i]:
                    if abs(newS[i].c)<=limit:
                        if abs(newS[i].c)>limit*0.1:
                            new_S_ord.append(newS[i])
            if limit<max_val*qS_thresh_rel:
                done=True
            limit*=0.1
        newS = new_S_ord[:]
    newS = Operator(ops=newS,antihermitian=True)
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
        ):
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
    newCirc.construct()
    if store.H.real and store.H.imag:
        RDM = newCirc.rdm - store.rdm
    elif not store.H.imag and store.H.real:
        RDM = newCirc.rdm
    else:
        print(store.H.imag,store.H.real)
        sys.exit('Problem in H')
    if verbose:
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
            newEl.generateOperators(Nq=quantstore.Nq,real=True,imag=True)
            newS+= newEl.formOperator()
    print(newS)
    if commutative:
        pass
    else:
        for i in newS.op:
            i.add=False
    #newS.reordering(method='hamiltonian',qubOpH=store.H.qubOp)
    return newS
