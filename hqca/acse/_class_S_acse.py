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
                        temp1*= Store.H.ints_2e[p,r,q,s]
                        v3 += 0.5*temp1
                        # 
                        temp2 = (c1*c4-c2*c3)*Store.rdm.rdm[p,r,j,l]
                        temp2+= (c6*c7-c5*c8)*Store.rdm.rdm[i,k,q,s]
                        temp2*= Store.H.ints_2e[p,r,q,s]
                        v2 += 0.5*temp2
    return k1+v2+v3

def findSPairs(
        store,
        classS_thresh_rel=0.1,
        classS_max=1e-10,
        commutative=True,
        recon_approx='V',
        verbose=True,
        **kw,
        ):
    '''
    '''
    store.rdm3 = store.rdm.reconstruct(approx=recon_approx)
    if verbose:
        print('3-RDM Trace: {}'.format(store.rdm3.trace()))
    alp = store.alpha_mo['active']
    bet = store.beta_mo['active']
    Na = len(alp)
    No = 2*Na
    S = []
    tS = []
    newS = Operator()
    ferOp= Operator()
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
                    if abs(term)>=classS_max:
                        newOp = FermionicOperator(
                                coeff=term,
                                indices=[p,r,s,q],
                                sqOp='++--',
                                antisymmetric=True,
                                add=True
                                )
                        newOp.generateOperators(
                                2*store.H.No_tot,
                                real=True,imag=True,
                                mapping=store.H.mapping,
                                **store.H._kw_mapping,
                                )
                        ferOp+= newOp
                        newS+= newOp.formOperator()
    print('Elements of S from classical ACSE: ')
    newS.clean()
    if commutative:
        pass
    else:
        for i in newS.op:
            i.add=False
    print(ferOp)
    return newS


