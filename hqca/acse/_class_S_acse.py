import numpy as np
import sys 
np.set_printoptions(linewidth=200,suppress=False,precision=3)
from hqca.tools import *
from functools import reduce

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
            temp0a = +c1*Store.rdm2.rdm[p,k,j,l]
            temp0a+= -c2*Store.rdm2.rdm[p,i,j,l]
            temp0a+= +c3*Store.rdm2.rdm[i,k,p,j]
            temp0a+= -c4*Store.rdm2.rdm[i,k,p,l]
            k1+= temp0a*Store.ints_1e[p,p]
    for p in orb:
        for q in orb:
            if p==q:
                continue
            c1 = int(q==i)
            c2 = int(q==k)
            c3 = int(p==l)
            c4 = int(p==j)
            if c1+c2+c3+c4>0:
                temp0b = +c1*Store.rdm2.rdm[p,k,j,l]
                temp0b+= -c2*Store.rdm2.rdm[p,i,j,l]
                temp0b+= +c3*Store.rdm2.rdm[i,k,q,j]
                temp0b+= -c4*Store.rdm2.rdm[i,k,q,l]
                k1+= temp0b*Store.ints_1e[p,q]
    for p in orb:
        for r in orb:
            for s in orb:
                for q in orb:
                    c1 = int(i==q)
                    c2 = int(i==s)
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
                        temp1*= Store.ints_2e[p,r,q,s]
                        v3 += 0.5*temp1
                        # 
                        temp2 = (c1*c4-c2*c3)*Store.rdm2.rdm[p,r,j,l]
                        temp2+= (c6*c7-c5*c8)*Store.rdm2.rdm[i,k,q,s]
                        temp2*= Store.ints_2e[p,r,q,s]
                        v2 += 0.5*temp2
    return k1,v2+v3

def findSPairs(
        store,
        recon_approx='V',
        **kw,
        ):
    '''
    '''
    if Store.Ne_tot>2:
        Store.rdm3 = Store.rdm2.reconstruct(approx=recon_approx)
        print('3-RDM Trace: {}'.format(Store.rdm3.trace()))
    alp = Store.alpha_mo['active']
    bet = Store.beta_mo['active']
    Na = len(alp)
    No = 2*Na
    S = []
    tS = []
    for i in alp:
        for k in alp:
            if i>=k:
                continue
            for l in alp:
                for j in alp:
                    if j>=l:
                        continue
                    if i*Na+k>=j*Na+l:
                        continue
                    Kt,Vt = evaluate2S(i,k,l,j,Store)
                    term = Kt+Vt
                    if abs(term)>1e-10:
                        newFermi = FermiOperator(
                                coeff=-term,
                                indices=[i,k,l,j],
                                sqOp='++--',
                                spin='aaaa',
                                add=True)
                        tS.append(newFermi)
    for i in bet:
        for k in bet:
            if i>=k:
                continue
            for l in bet:
                for j in bet:
                    if j>=l:
                        continue
                    if i*Na+k>=j*Na+l:
                        continue
                    Kt,Vt = evaluate2S(i,k,l,j,Store)
                    term = Kt+Vt
                    if abs(term)>1e-12:
                        newFermi = FermiOperator(
                                coeff=-term,
                                indices=[i,k,l,j],
                                sqOp='++--',
                                spin='bbbb')
                        tS.append(newFermi)
    for i in alp:
        for k in bet:
            for l in bet:
                for j in alp:
                    if i>j:
                        continue
                    elif i==j:
                        if k>=l:
                            continue
                    Kt,Vt = evaluate2S(i,k,l,j,Store)
                    term = Kt+Vt
                    if abs(term)>1e-12:
                        newFermi = FermiOperator(
                                coeff=-term,
                                indices=[i,k,l,j],
                                sqOp='++--',
                                spin='abba')
                        tS.append(newFermi)
    largest = 0
    for i in tS:
        if abs(i.c)>largest:
            largest = abs(i.c)
    for i in tS:
        if abs(i.c)>largest*0.1 and abs(i.c)>1e-10:
            S.append(i)
    hold_type = [(op.opType=='de') for op in S]
    S_ord = []
    for i in range(len(hold_type)):
        if hold_type[i]:
            S_ord.append(S[i])
    for i in range(len(hold_type)):
        if not hold_type[i]:
            S_ord.append(S[i])
    S = S_ord[:]
    print('Elements of S from classical ACSE: ')
    for item in S:
        print('S: {:.5f},{:.5f},{},{}'.format(
            np.real(item.c),np.real(item.qCo),item.qInd,item.qOp))
    return S


