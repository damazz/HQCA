import numpy as np
np.set_printoptions(linewidth=200)
from hqca.tools.EnergyFunctions import Storage
from hqca.quantum.QuantumFunctions import QuantumStorage
from hqca.tools.Fermi import FermiOperator
from hqca.tools import Functions as fx
from hqca.tools import Chem as chem
from hqca.tools.RDM import RDMs
from functools import reduce

'''
/hqca/acse/FunctionsACSE.py

Contains functions for performing ACSE calculations, particularly with a
classical solution of the S matrix.
'''

def findSPairs(Store):
    '''
    '''
    if Store.Ne_tot>2:
        Store.rdm3 = Store.rdm2.reconstruct()
        #print('3-RDM Trace: {}'.format(Store.rdm3.trace()))
    #print(Store.rdm3.reduce_order().reduce_order().rdm)
    alpha = Store.alpha_mo['active']
    beta = Store.beta_mo['active']
    S = []
    tS = []
    blocks = [
            [alpha,alpha,beta],
            [alpha,beta,beta],
            [alpha,beta,beta],
            [alpha,alpha,beta]
            ]
    block = ['aa','ab','bb']
    for ze in range(len(blocks[0])):
        for i in blocks[0][ze]:
            for k in blocks[1][ze]:
                for l in blocks[2][ze]:
                    for j in blocks[3][ze]:
                        if block[ze]=='ab':
                            if i>j or k>l:
                                continue
                            spin ='abba'
                        else:
                            if i>=k or k>l or j>=l:
                                continue
                            if block[ze]=='aa':
                                spin = 'aaaa'
                            else:
                                spin='bbbb'
                        Vt,Kt = 0,0
                        for p,r,q,s in Store.zipH2: 
                            # creation annihilation:
                            # iklj, prsq
                            # ei is, 1c2c,1a2a
                            # so, pr qs corresponds with properly, prsq
                            c1,c2 = int(i==q),int(i==s)
                            c3,c4 = int(k==q),int(k==s)
                            c5,c6 = int(j==p),int(l==p)
                            c7,c8 = int(j==r),int(l==r)
                            if c1+c2+c3+c4+c5+c6+c7+c8==0:
                                continue
                            t1 = -c1*Store.rdm3.rdm[k,p,r,j,l,s]
                            t1+= +c2*Store.rdm3.rdm[k,p,r,j,l,q]
                            t1+= +c3*Store.rdm3.rdm[i,p,r,j,l,s]
                            t1+= -c4*Store.rdm3.rdm[i,p,r,j,l,q]
                            t1+= +c5*Store.rdm3.rdm[i,k,r,l,q,s]
                            t1+= -c6*Store.rdm3.rdm[i,k,r,j,q,s]
                            t1+= -c7*Store.rdm3.rdm[i,k,p,l,q,s]
                            t1+= +c8*Store.rdm3.rdm[i,k,p,j,q,s]
                            t1+= (c1*c4-c2*c3)*Store.rdm2.rdm[p,r,j,l]
                            t1+= -(c5*c8-c7*c6)*Store.rdm2.rdm[i,k,q,s]
                            Vt = Vt + 0.5*Store.ints_2e[p,r,q,s]*t1
                        for p,q in Store.zipH1:
                            c1,c2 = int(i==q),int(k==q)
                            c3,c4 = int(j==p),int(l==p)
                            if c1+c2+c3+c4==0:
                                continue
                            t1 = -c1*Store.rdm2.rdm[k,p,j,l]
                            t1+=  c2*Store.rdm2.rdm[i,p,j,l]
                            t1+=  c3*Store.rdm2.rdm[i,k,l,q]
                            t1+= -c4*Store.rdm2.rdm[i,k,j,q]
                            Kt = Kt + Store.ints_1e[p,q]*t1
                        term = Kt+Vt
                        if abs(term)>1e-7:
                            newFermi = FermiOperator(
                                    coeff=term,
                                    indices=[i,k,l,j],
                                    sqOp='++--',
                                    spin=spin)
                            tS.append(newFermi)
    largest = 0
    for i in tS:
        if abs(i.c)>largest:
            largest = abs(i.c)
    for i in tS:
        if abs(i.c)>largest*0.05:
            S.append(i)
    return S


def findS0Pairs(Store):
    '''
    pass
    '''
    alpha = Store.alpha_mo['active']
    beta = Store.beta_mo['active']
    S = []
    blocks = [
            [alpha,alpha,beta],
            [alpha,beta,beta],
            [alpha,beta,beta],
            [alpha,alpha,beta]
            ]
    block = ['aa','ab','bb']
    h0 = Store.t
    h1 = 1-h0
    for ze in range(len(blocks[0])):
        for i in blocks[0][ze]:
            for k in blocks[1][ze]:
                for l in blocks[2][ze]:
                    for j in blocks[3][ze]:
                        if block[ze]=='ab':
                            if i>j or k>l:
                                continue
                            spin = 'abba'
                        else:
                            if i>=k or j>=l:
                                continue
                            if block[ze]=='aa':
                                spin = 'aaaa'
                            else:
                                spin='bbbb'
                        term = 0
                        for p,r,q,s in Store.zipH2:
                            # creation annihilation:
                            # iklj, prsq
                            # ei is, 1c2c,1a2a
                            # so, pr qs
                            c1,c2 = int(i==q),int(i==s)
                            c3,c4 = int(k==q),int(k==s)
                            c5,c6 = int(j==p),int(l==p)
                            c7,c8 = int(j==r),int(l==r)
                            if c1+c2+c3+c4+c5+c6+c7+c8==0:
                                continue
                            t1 = c1*Store.rdm3.rdm[k,r,p,j,l,s]
                            t1+= c3*Store.rdm3.rdm[i,p,r,j,l,s]
                            t1+= c5*Store.rdm3.rdm[i,k,r,l,q,s]
                            t1+= c6*Store.rdm3.rdm[i,k,r,j,s,q]
                            t2 = (c1*c4-c2*c3)*Store.rdm2.rdm[p,r,j,l]
                            t2+= -(c5*c8-c7*c6)*Store.rdm2.rdm[i,k,q,s]
                            temp2v = 1*(t1)+1*(t2)
                            temp2v*= Store.ints_2e[p,r,q,s]
                            term+= temp2v*h0
                        for p,q in Store.zipH1:
                            c1,c2 = int(i==q),int(k==q)
                            c3,c4 = int(j==p),int(l==p)
                            if c1+c2+c3+c4==0:
                                continue
                            t1 = -c1*Store.rdm2.rdm[k,p,j,l]
                            t2 =  c2*Store.rdm2.rdm[i,p,j,l]
                            t3 =  c3*Store.rdm2.rdm[i,k,l,q]
                            t4 = -c4*Store.rdm2.rdm[i,k,j,q]
                            temp1h = t1+t2+t3+t4
                            temp1h*= Store.ints_1e[p,q]
                            term+= temp1h*h0
                        for p,q in Store.zipF:
                            c1,c2 = int(i==q),int(k==q)
                            c3,c4 = int(j==p),int(l==p)
                            if c1+c2+c3+c4==0:
                                continue
                            t1 = -c1*Store.rdm2.rdm[k,p,j,l]
                            t2 =  c2*Store.rdm2.rdm[i,p,j,l]
                            t3 =  c3*Store.rdm2.rdm[i,k,l,q]
                            t4 = -c4*Store.rdm2.rdm[i,k,j,q]
                            temp1f = t1+t2+t3+t4
                            temp1f*= Store.F[p,q]
                            term+= temp1f*h1
                        if abs(term)>1e-10:
                            newFermi = FermiOperator(
                                    coeff=-term,
                                    indices=[i,k,l,j],
                                    sqOp='++--',
                                    spin=spin)
                            S.append(newFermi)
    return S


