import RDMFunctions as rdmf
import sys

alpha = {
        'active':[0,1,2,3,4,5],
        'inactive':[],
        'virtual':[]}
beta = {
        'active':[],
        'inactive':[],
        'virtual':[]}
import numpy as np
np.set_printoptions(precision=4,linewidth=200,suppress=True)
from random import random
def gen_rand_2e4o_wf(
        one='1010',
        two='1001',
        thr='0110',
        fou='0101'
        ):
    a = (random()-0.5)*2
    a2 = np.sqrt(1-a**2)
    b = (random()-0.5)*2*a2
    b2 = np.sqrt(1-a**2-b**2)
    c = (random()-0.5)*2*b2
    d = np.sqrt(1-a**2-b**2-c**2)
    wf = {
            one:a,
            two:b,
            thr:c,
            fou:d}
    return wf
def gen_rand_3e6o_wf(
        one='111000',
        two='110100',
        thr='101010',
        fou='100110',
        fiv='011001',
        six='010101',
        sev='001011',
        eig='000111'
        ):
    a = (random()-0.5)*2
    a2 = np.sqrt(1-a**2)
    b = (random()-0.5)*2*a2
    b2 = np.sqrt(1-a**2-b**2)
    c = (random()-0.5)*2*b2
    c2 = np.sqrt(1-a**2-b**2-c**2)
    d = (random()-0.5)*2*c2
    d2 = np.sqrt(1-a**2-b**2-c**2-d**2)
    e = (random()-0.5)*2*d2
    e2 = np.sqrt(1-a**2-b**2-c**2-d**2-e**2)
    f = (random()-0.5)*2*e2
    f2 = np.sqrt(1-a**2-b**2-c**2-d**2-e**2-f**2)
    g = (random()-0.5)*2*f2
    h = np.sqrt(1-a**2-b**2-c**2-d**2-e**2-f**2-g**2)
    wf = {
            one:a,
            two:b,
            thr:c,
            fou:d,
            fiv:e,
            six:f,
            sev:g,
            eig:h}
    return wf

def permute(wf,o1,o2):
    nwf = {}
    for k,v in wf.items():
        if o1>o2:
            o1,o2 = o2,o1
        t1 = k[o1]
        t2 = k[o2]
        nk = k[0:o1]+t2+k[o1+1:o2]+t1+k[o2+1:]
        nwf[nk]=v
    return nwf

def condense(mat):
    n = int((mat.shape[0]**(1/2)))
    l = np.arange(0,mat.shape[0])
    ind =[]
    for i in l[::-1]:
        a = np.nonzero(mat[i,:])
        try:
            if (not a[0]):
                mat = np.delete(mat,i,axis=0)
                mat = np.delete(mat,i,axis=1)
                ind.append('{}{}'.format((i)//(n),(i)%(n)))
        except ValueError:
            pass
    ind = ind[::-1]
    return mat,ind

wf = gen_rand_3e6o_wf()
for i in range(0,5):
    if i==0:
        pass
    elif i==1:
        wf = permute(wf,1,4)
    elif i==2:
        wf = permute(wf,2,3)
    elif i==3:
        wf = permute(wf,1,5)
        wf = permute(wf,0,2)
    elif i==4:
        wf = permute(wf,0,3)
        wf = permute(wf,4,5)
        wf = permute(wf,2,1)
    print(wf)

    rdm2 = rdmf.build_2rdm(
            wf,
            alpha,beta)
    rdm2 = np.reshape(rdm2,
            (36,36))
    rdm2,ax = condense(rdm2)
    #print(rdm2)
    print(ax)
    eigval,eigvec = np.linalg.eig(rdm2)
    print(np.sort(np.real(eigval))[::-1])
    rdm2,ax = condense(rdm2)

