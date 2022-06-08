import numpy as np
import sys 
np.set_printoptions(linewidth=200,suppress=False,precision=3)
from hqca.tools import *
from hqca.operators import *
from functools import reduce
from hqca.core import *
from copy import deepcopy as copy
import timeit
from hqca.processes import StandardProcess
from hqca.tomography import StandardTomography




def solvecACSE(
        acse,
        operator=None,
        S_min=1e-10,
        tomo=None,
        verbose=True,
        transform=None,
        matrix=False,
        norm='fro',
        **kw,
        ):
    '''
    Solve the ACSE, which traditionally has elements defined as:
    A^ik_jl = < [a_i+ a_k+ a_l a_j ,H] >

    Here, we instead let the residuals of A represent the gradient direciton,
    which properly is given as <[H,i+ k+ l j ]>

    Note that Eulers method adds a minus sign appropriately

    '''
    store = acse.store
    alp = store.alpha_mo['qubit']
    bet = store.beta_mo['qubit']
    if store.Ne_as<3 and tomo.p==2:
        D3  = RDM(
                order=3,
                alpha=alp,
                beta = bet,
                rdm = None,
                Ne=acse.store.No_as)
        #
        circ = acse._generate_circuit(
                op=operator,
                tomo=tomo,
                order=2,
                compact=False)
        D2 = circ.rdm
    elif tomo.p==3:
        circ = acse._generate_circuit(
                op=operator,
                tomo=tomo,
                order=3,
                compact=False)
        D3 = circ.rdm
        D2 = D3.reduce_order()
    else:
        raise ResidualError
    if verbose:
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('classical ACSE')
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('trace of the 3-RDM: {}'.format(D3.trace()))
        print('')
    keys = acse.rdme
    D2 = np.real(D2.rdm)
    D3 = np.real(D3.rdm)
    Na = len(alp)
    No = 2*Na
    S = []
    tS = []
    new = Operator()
    newF= Operator()
    max_val=0
    # 
    #
    A = np.zeros((No,No,No,No),dtype=np.complex_)
    #K1 = store.H.ints_1e
    #K2 = store.H.ints_2e
    #W = K2 - K2.transpose(0,1,3,2)
    W = (store.H.K2 - store.H.K2.transpose(0,1,3,2))
    so = alp+bet
    for i in so:
        for j in so:
            for k in so:
                for l in so:
                    A[i,k,j,l]-= np.dot(D2[:,:,j,l],W[:,:,i,k].T).trace()
                    A[i,k,j,l]+= np.dot(D2[:,:,i,k],W[:,:,j,l].T).trace()
                    for r in so:
                        A[i,k,j,l]+= np.dot(D3[:,r,k,j,l,:],W[:,r,i,:].T).trace()
                        A[i,k,j,l]-= np.dot(D3[:,r,i,j,l,:],W[:,r,k,:].T).trace()
                        A[i,k,j,l]-= np.dot(D3[i,k,:,r,:,j],W[:,l,r,:].T).trace()
                        A[i,k,j,l]+= np.dot(D3[i,k,:,r,:,l],W[:,j,r,:].T).trace()
    if matrix:
        newA = np.zeros(1*len(keys))
        for n,inds in enumerate(keys):
            i,k,l,j = inds[0],inds[1],inds[2],inds[3]
            newA[n]=A[i,k,l,j]
            #newA[n+len(keys)]=A[i,k,l,j]
            #newA[n+2*len(keys)]=A[i,k,l,j]
            #newA[n+3*len(keys)]=A[i,k,l,j]
        return -newA
    else:
        nz = np.nonzero(A)
        new = Operator()
        norm = 0
        for i,k,j,l in zip(nz[0],nz[1],nz[2],nz[3]):
            term = A[i,k,j,l]
            if abs(term)>=S_min:
                new+= FermiString(
                        coeff=-term,
                        indices=[i,k,l,j],
                        ops='++--',
                        N = acse.qs.dim,
                        )
                norm+= np.real(np.conj(term)*term)

        assert (np.sqrt(norm)-np.linalg.norm(A))<1e-8
        return new,0.5*np.linalg.norm(A)



