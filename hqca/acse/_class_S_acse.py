import numpy as np
import sys 
np.set_printoptions(linewidth=200,suppress=False,precision=3)
from hqca.tools import *
from hqca.operators import *
from functools import reduce
from copy import deepcopy as copy
import timeit
from hqca.processes import StandardProcess
from hqca.tomography import StandardTomography

'''
/hqca/acse/FunctionsACSE.py

Contains functions for performing ACSE calculations, particularly with a
classical solution of the S matrix.
'''



def solvecACSE(
        store,
        quantstore,
        S_min=1e-10,
        recon_approx='V',
        verbose=True,
        transform=None,
        matrix=False,
        keys=None,
        rdm=None,
        D3=None,
        norm='fro',
        operator=None,
        process=None,
        instruct=None,
        **kw,
        ):
    print(D3)
    if type(D3)==type(None) and store.rdm.p==2:
        store.rdm3 = store.rdm.reconstruct(approx=recon_approx)
        D3 = store.rdm3.rdm
        store.rdm.expand()
        D2 = store.rdm.rdm
    else:
        print('Measuring 3-RDM...')
        newPsi = instruct(
                operator=operator,
                Nq=quantstore.Nq,
                quantstore=quantstore,
                **kw
                )
        newCirc = StandardTomography(
                quantstore,
                preset=True,
                Tomo=D3,
                verbose=verbose,
                order=3,
                )
        newCirc.set(newPsi)
        newCirc.simulate(verbose=verbose)
        newCirc.construct(processor=process,compact=False)
        D3 = newCirc.rdm.rdm
        D2 = newCirc.rdm.reduce_order().rdm

    #if verbose:
    #    print('-- -- -- -- -- -- -- -- -- -- --')
    #    print('classical ACSE')
    #    print('-- -- -- -- -- -- -- -- -- -- --')
    #    print('trace of the 3-RDM: {}'.format(store.rdm3.trace()))
    #    print('')
    alp = store.alpha_mo['qubit']
    bet = store.beta_mo['qubit']
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
    W = 2*(store.H.K2 - store.H.K2.transpose(0,1,3,2))
    so = alp+bet
    for i in so:
        for j in so:
            for k in so:
                for l in so:
                    A[i,k,j,l]-= 0.5*np.dot(D2[:,:,j,l],W[:,:,i,k].T).trace()
                    A[i,k,j,l]+= 0.5*np.dot(D2[:,:,i,k],W[:,:,j,l].T).trace()
                    for r in so:
                        A[i,k,j,l]+= 0.5*np.dot(D3[:,r,k,j,l,:],W[:,r,i,:].T).trace()
                        A[i,k,j,l]-= 0.5*np.dot(D3[:,r,i,j,l,:],W[:,r,k,:].T).trace()
                        A[i,k,j,l]-= 0.5*np.dot(D3[i,k,:,r,:,j],W[:,l,r,:].T).trace()
                        A[i,k,j,l]+= 0.5*np.dot(D3[i,k,:,r,:,l],W[:,j,r,:].T).trace()
    if matrix:
        newA = np.zeros(len(keys))
        for n,inds in enumerate(keys):
            i,k,l,j = inds[0],inds[1],inds[2],inds[3]
            newA[n]=A[i,k,j,l]
        return -0.5*newA
    else:
        nz = np.nonzero(A)
        new = Operator()
        norm = 0
        for i,k,j,l in zip(nz[0],nz[1],nz[2],nz[3]):
            term = A[i,k,j,l]
            if abs(term)>=S_min:
                new+= FermiString(
                        coeff=-0.5*term,
                        indices=[i,k,l,j],
                        ops='++--',
                        N = quantstore.dim,
                        )
            norm+= np.real(np.conj(term)*term)
                
        #
        #fullS = new.transform(quantstore.transform)
        #if verbose:
        #    print('fermionic A operator:')
        #    print(new)
        #    print('')
        return new,np.sqrt(norm)



