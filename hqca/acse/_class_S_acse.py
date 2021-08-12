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



def evaluate2S(i,k,l,j,Store):
    '''
    0.5 coefficient is for double counting
    ... though we dont double counts some stuff : ( huh. 
    '''
    k1,v2,v3 = 0,0,0
    orb = Store.alpha_mo['active']+Store.beta_mo['active']
    N = len(orb)
    for p in orb:
        c1 = int(p==i)
        c2 = int(p==k)
        c3 = int(p==l)
        c4 = int(p==j)
        if c1+c2+c3+c4>0:
            temp0a = +c1*Store.rdm.rdm[p,k,j,l]
            temp0a+= -c2*Store.rdm.rdm[p,i,j,l]
            temp0a+= +c3*Store.rdm.rdm[i,k,p,j]
            temp0a+= -c4*Store.rdm.rdm[i,k,p,l]
            k1+= temp0a*Store.H.ints_1e[p,p]
    for p in orb:
        for q in orb:
            if p==q:
                continue
            c1 = int(q==i)
            c2 = int(q==k)
            c3 = int(p==l)
            c4 = int(p==j)
            if c1+c2+c3+c4>0:
                temp0b = +c1*Store.rdm.rdm[p,k,j,l]
                temp0b+= -c2*Store.rdm.rdm[p,i,j,l]
                temp0b+= +c3*Store.rdm.rdm[i,k,q,j]
                temp0b+= -c4*Store.rdm.rdm[i,k,q,l]
                k1+= temp0b*Store.H.ints_1e[p,q]
    for p in orb:
        for r in orb:
            for s in orb:
                c2 = int(i==s)
                for q in orb:
                    c1 = int(i==q)
                    c3 = int(k==q)
                    c4 = int(k==s)
                    c5 = int(l==r)
                    c6 = int(l==p)
                    c7 = int(j==r)
                    c8 = int(j==p)
                    if c1+c2+c3+c4+c5+c6+c7+c8>0:
                        temp1 = -c1*Store.rdm3.rdm[p,r,k,j,l,s] #s,l,j
                        temp1+= +c2*Store.rdm3.rdm[p,r,k,j,l,q] #q ,l,j
                        temp1+= +c3*Store.rdm3.rdm[p,r,i,j,l,s] #slj
                        temp1+= -c4*Store.rdm3.rdm[p,r,i,j,l,q] #qlj
                        temp1+= +c5*Store.rdm3.rdm[i,k,p,q,s,j] #jsq
                        temp1+= -c6*Store.rdm3.rdm[i,k,r,q,s,j] #jsq 
                        temp1+= -c7*Store.rdm3.rdm[i,k,p,q,s,l] #lsq
                        temp1+= +c8*Store.rdm3.rdm[i,k,r,q,s,l] #lsq
                        temp1*= Store.H.ints_2e[p,r,q,s]
                        v3 += 0.5*temp1
                         
                        temp2 = (c1*c4-c2*c3)*Store.rdm.rdm[p,r,j,l]
                        temp2+= (c6*c7-c5*c8)*Store.rdm.rdm[i,k,q,s]
                        temp2*= Store.H.ints_2e[p,r,q,s]
                        v2 += 0.5*temp2
    return k1+v2+v3

def findSPairs_full(
        store,
        quantstore,
        S_min=1e-10,
        recon_approx='V',
        verbose=True,
        transform=None,
        **kw,
        ):
    '''
    '''
    store.rdm3 = store.rdm.reconstruct(approx=recon_approx)
    if verbose:
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('classical ACSE')
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('trace of the 3-RDM: {}'.format(store.rdm3.trace()))
        print('')
    alp = store.alpha_mo['active']
    bet = store.beta_mo['active']
    Na = len(alp)
    No = 2*Na
    S = []
    tS = []
    new = Operator()
    newF= Operator()
    max_val=0
    for p in alp+bet:
        for r in alp+bet:
            if p==r:
                continue
            i1 = (p==r)
            for s in alp+bet:
                i2,i3 = (s==p),(s==r)
                if i1+i2+i3==3:
                    continue
                for q in alp+bet:
                    i4,i5,i6 = (q==p),(q==r),(q==s)
                    if i1+i2+i3+i4+i5+i6>=3:
                        continue
                    if q==s:
                        continue
                    term  = evaluate2S(p,r,s,q,store)
                    if abs(term)>=S_min:
                        new+= FermiString(
                                coeff=term*0.5,
                                indices=[p,r,s,q],
                                ops='++--',
                                N = quantstore.dim,
                                )
    fullS = new.transform(quantstore.transform)
    #for op in new:
    #    if abs(op.c)>=abs(max_val):
    #        max_val = copy(op.c)
    if verbose:
        print('fermionic A operator:')
        print(new)
        print('')
    #if commutative:
    #    newS.ca=True
    #else:
    #    newS.ca=False
    return fullS

def find1Selements(
        store,
        i,j,
        S_min=1e-10,
           ):
    D2 = store.rdm.rdm
    val = 0
    W = 2*(store.H.K2 - store.H.K2.transpose(0,1,3,2))
    val += np.tensordot(D2[:,:,j,:],W[:,:,i,:])
    val -= np.tensordot(D2[:,:,i,:],W[:,:,j,:])
    return


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
                    if i==0 and j==1 and k==2 and l==3:
                        print(0.5*np.dot(D2[:,:,j,l],W[:,:,i,k].T).trace())
                        print( 0.5 * np.dot(D2[:, :, i, k], W[:, :, j, l].T).trace())
                        A1,A2 = D2[:,:,j,l], D2[:,:,i,k]
                        B1,B2 = W[:,:,i,k].T, W[:,:,j,l].T
                        for p in so:
                            for q in so:
                                if abs(D2[p,q,j,l])>1e-5:
                                    print(p,q,l,j,D2[p,q,j,l],p,q,k,i, W[p,q,i,k])
                                if abs(D2[p, q, i, k]) > 1e-5:
                                    print(p, q, k,i, D2[p, q, i, k], p,q,l,j,W[p, q, j, l])

                        print(D2[:,:,j,l])
                        print(D2[:,:,i,k])
                        print(W[:,:,i,k].T)
                        print(W[:,:,j,l].T)

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
                        if i == 0 and j == 1 and k == 2 and l == 3:
                            print(r,0.5*np.dot(D3[:,r,k,j,l,:],W[:,r,i,:].T).trace() )
                            print(r,0.5*np.dot(D3[:,r,k,j,l,:],W[:,r,i,:].T).trace())
                            print(r, 0.5*np.dot(D3[i,k,:,r,:,j],W[:,l,r,:].T).trace())
                            print(r,0.5*np.dot(D3[i,k,:,r,:,l],W[:,j,r,:].T).trace())
                        A[i,k,j,l]+= 0.5*np.dot(D3[:,r,k,j,l,:],W[:,r,i,:].T).trace()
                        A[i,k,j,l]-= 0.5*np.dot(D3[:,r,i,j,l,:],W[:,r,k,:].T).trace()
                        A[i,k,j,l]-= 0.5*np.dot(D3[i,k,:,r,:,j],W[:,l,r,:].T).trace()
                        A[i,k,j,l]+= 0.5*np.dot(D3[i,k,:,r,:,l],W[:,j,r,:].T).trace()
                        #A[i,k,j,l]+= 0.5*np.dot(D3[p,r,k,j,l,q],W[p,r,i,q].T).trace() 
                        #A[i,k,j,l]-= 0.5*np.dot(D3[p,r,i,j,l,q],W[p,r,k,q].T).trace()
                        #A[i,k,j,l]-= 0.5*np.dot(D3[i,k,p,r,q,j],W[:,l,r,:].T).trace()
                        #A[i,k,j,l]+= 0.5*np.dot(D3[i,k,p,r,q,l],W[:,j,r,:].T).trace()
                    if abs(A[i,k,j,l])>1e-5:
                        print(A[i,k,j,l], i, k, l, j)
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




def findSPairs_test(
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
    D2 = store.rdm.rdm
    if verbose:
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('classical ACSE')
        print('-- -- -- -- -- -- -- -- -- -- --')
        print('trace of the 3-RDM: {}'.format(store.rdm3.trace()))
        print('')
    alp = store.alpha_mo['active']
    bet = store.beta_mo['active']
    Na = len(alp)
    No = 2*Na
    S = []
    tS = []
    new = Operator()
    newF= Operator()
    max_val=0
    # 
    #  # #  #
    #
    A = np.zeros((No,No,No,No),dtype=np.complex_)
    temp1 = np.zeros((No,No,No,No))
    #temp2 = np.zeros((N,N,N,N))
    #temp3 = np.zeros((N,N,N,N))
    K1 = store.H.ints_1e
    K2 = store.H.ints_2e
    W = K2 - K2.transpose(0,1,3,2)

    so = alp+bet
    #
    def subroutine(setA,setB):
        M = np.zeros((No,No,No,No),dtype=np.complex_)
        for i in setA:
            for j in setA:
                for k in setB:
                    for l in setB:
                        M[i,k,j,l]-= 0.5*np.dot(D2[:,:,j,l],W[:,:,i,k].T).trace()
                        M[i,k,j,l]+= 0.5*np.dot(D2[:,:,i,k],W[:,:,j,l].T).trace()
                        #
                        M[i,k,j,l]-= np.dot(K1[:,i],D2[:,k,j,l].T)
                        M[i,k,j,l]+= np.dot(K1[:,k],D2[:,i,j,l].T)
                        M[i,k,j,l]-= np.dot(K1[:,l],D2[:,j,i,k].T)
                        M[i,k,j,l]+= np.dot(K1[:,j],D2[:,l,i,k].T)
                        for r in setA+setB:
                            M[i,k,j,l]+= 0.5*np.dot(D3[:,r,k,j,l,:],W[:,r,i,:].T).trace() 
                            M[i,k,j,l]-= 0.5*np.dot(D3[:,r,i,j,l,:],W[:,r,k,:].T).trace()
                            M[i,k,j,l]-= 0.5*np.dot(D3[i,k,:,r,:,j],W[:,l,r,:].T).trace()
                            M[i,k,j,l]+= 0.5*np.dot(D3[i,k,:,r,:,l],W[:,j,r,:].T).trace()
        return M
    A+= subroutine(alp+bet,alp+bet)
    #A+= subroutine(alp,bet)
    #A+= subroutine(bet,alp)
    #A+= subroutine(bet,bet)

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
    fullS = new.transform(quantstore.transform)
    if verbose:
        print('fermionic A operator:')
        print(new)
        print('')
    return fullS

