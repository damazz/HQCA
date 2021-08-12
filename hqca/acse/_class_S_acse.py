import numpy as np
import sys 
np.set_printoptions(linewidth=200,suppress=False,precision=3)
from hqca.tools import *
from hqca.operators import *
from functools import reduce
from copy import deepcopy as copy
import timeit

'''
/hqca/acse/FunctionsACSE.py

Contains functions for performing ACSE calculations, particularly with a
classical solution of the S matrix.
'''



def findSPairs(
        store,
        quantstore,
        S_min=1e-10,
        recon_approx='V',
        verbose=True,
        transform=None,
        **kw,
        ):
    store.rdm3 = store.rdm.reconstruct(approx=recon_approx)
    D3 = store.rdm3.rdm
    store.rdm.expand()
    D2 = store.rdm.rdm
    if verbose:
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('classical ACSE')
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('trace of the 3-RDM: {}'.format(store.rdm3.trace()))
        print('')
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
    temp1 = np.zeros((No,No,No,No))
    #temp2 = np.zeros((N,N,N,N))
    #temp3 = np.zeros((N,N,N,N))
    #K1 = store.H.ints_1e
    #K2 = store.H.ints_2e
    #W = K2 - K2.transpose(0,1,3,2)
    W = 2*(store.H.K2 - store.H.K2.transpose(0,1,3,2))
    so = alp+bet
    for i in so:
        for j in so:
            for k in so:
                for l in so:
                    #if i==0 and j==1 and k==2 and l==3:
                    #    print(0.5*np.dot(D2[:,:,j,l],W[:,:,i,k].T).trace())
                    #    print( 0.5 * np.dot(D2[:, :, i, k], W[:, :, j, l].T).trace())
                    #    A1,A2 = D2[:,:,j,l], D2[:,:,i,k]
                    #    B1,B2 = W[:,:,i,k].T, W[:,:,j,l].T
                    #    for p in so:
                    #        for q in so:
                    #            if abs(D2[p,q,j,l])>1e-5:
                    #                print(p,q,l,j,D2[p,q,j,l],p,q,k,i, W[p,q,i,k])
                    #            if abs(D2[p, q, i, k]) > 1e-5:
                    #                print(p, q, k,i, D2[p, q, i, k], p,q,l,j,W[p, q, j, l])

                    #    print(D2[:,:,j,l])
                    #    print(D2[:,:,i,k])
                    #    print(W[:,:,i,k].T)
                    #    print(W[:,:,j,l].T)

                    #for p in so:
                    #    for r in so:
                    #        A[i,k,j,l]-= D2[p,r,j,l]*W[p,r,i,k]
                    #        A[i,k,j,l]+= D2[p,r,i,k]*W[p,r,j,l]
                    A[i,k,j,l]-= 0.5*np.dot(D2[:,:,j,l],W[:,:,i,k].T).trace()
                    A[i,k,j,l]+= 0.5*np.dot(D2[:,:,i,k],W[:,:,j,l].T).trace()
                    #
                    #A[i,k,j,l]-= np.dot(K1[:,i],D2[:,k,j,l].T)
                    #A[i,k,j,l]+= np.dot(K1[:,k],D2[:,i,j,l].T)
                    #A[i,k,j,l]-= np.dot(K1[:,l],D2[:,j,i,k].T)
                    #A[i,k,j,l]+= np.dot(K1[:,j],D2[:,l,i,k].T)
                    for r in so:
                        #if i == 0 and j == 1 and k == 2 and l == 3:
                        #    print(r,0.5*np.dot(D3[:,r,k,j,l,:],W[:,r,i,:].T).trace() )
                        #    print(r,0.5*np.dot(D3[:,r,k,j,l,:],W[:,r,i,:].T).trace())
                        #    print(r, 0.5*np.dot(D3[i,k,:,r,:,j],W[:,l,r,:].T).trace())
                        #    print(r,0.5*np.dot(D3[i,k,:,r,:,l],W[:,j,r,:].T).trace())
                        A[i,k,j,l]+= 0.5*np.dot(D3[:,r,k,j,l,:],W[:,r,i,:].T).trace()
                        A[i,k,j,l]-= 0.5*np.dot(D3[:,r,i,j,l,:],W[:,r,k,:].T).trace()
                        A[i,k,j,l]-= 0.5*np.dot(D3[i,k,:,r,:,j],W[:,l,r,:].T).trace()
                        A[i,k,j,l]+= 0.5*np.dot(D3[i,k,:,r,:,l],W[:,j,r,:].T).trace()
                        #A[i,k,j,l]+= 0.5*np.dot(D3[p,r,k,j,l,q],W[p,r,i,q].T).trace() 
                        #A[i,k,j,l]-= 0.5*np.dot(D3[p,r,i,j,l,q],W[p,r,k,q].T).trace()
                        #A[i,k,j,l]-= 0.5*np.dot(D3[i,k,p,r,q,j],W[:,l,r,:].T).trace()
                        #A[i,k,j,l]+= 0.5*np.dot(D3[i,k,p,r,q,l],W[:,j,r,:].T).trace()
                    #if abs(A[i,k,j,l])>1e-5:
                    #    print(A[i,k,j,l], i, k, l, j)
    nz = np.nonzero(A)
    new = Operator()
    for i,k,j,l in zip(nz[0],nz[1],nz[2],nz[3]):
        term = A[i,k,j,l]
        if abs(term)>=S_min:
            new+= FermiString(
                    coeff=-0.5*term,
                    indices=[i,k,l,j],
                    ops='++--',
                    N = quantstore.dim,
                    )
    #fullS = new.transform(quantstore.transform)
    if verbose:
        print('fermionic A operator:')
        print(new)
        print('')
    return new


