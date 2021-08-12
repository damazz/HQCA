#
# rdmf.py 
# Deals with functions that involve or are pertaining to reduced density matrices 
#
import numpy as np
import numpy.linalg as LA
from numpy import conj as con
from numpy import complex_
from functools import reduce
from hqca.operators import *
from copy import deepcopy as copy

def contract(mat):
    size = len(mat.shape)
    L = mat.shape[0]
    if size==4:
        mat = np.reshape(
                mat,
                (
                    L**2,
                    L**2
                    )
                )
    return mat

def expand(mat):
    size = len(mat.shape)
    L = mat.shape[0]
    if size==2:
        mat = np.reshape(
                mat,
                (
                    int(np.sqrt(L)),
                    int(np.sqrt(L)),
                    int(np.sqrt(L)),
                    int(np.sqrt(L))
                    )
                )
    return mat

class Recursive:
    def __init__(self,
            depth='default',
            choices=[],
            ):
        if depth=='default':
            depth=len(choices)
        self.depth=depth
        self.total=[]
        self.choices = list(choices)

    def choose(self,choices='default',temp=[]):
        '''
        recursive function to give different choices? 
        '''
        if choices=='default':
            choices=self.choices
        done=True
        for c in choices:
            if not len(c)==0:
                done=False
                break
        if done:
            self.total.append(temp)
        else:
            for n,i in enumerate(choices):
                for m in reversed(range(len(i))):
                    nc = copy(choices)
                    j = nc[n].pop(m)
                    self.choose(nc,temp=temp[:]+[j])

    def simplify(self):
        '''

        '''
        new = []
        for i in self.total:
            r = ''.join(i)
            if r in new:
                pass
            else:
                new.append(r)
        self.total = new

    def permute(self,d='default',temp=[]):
        '''
        chooses d number of options from the choices, and returns 
        and ordered list of choices (01,02,03,12,13,23,etc.)
        '''
        if d=='default':
            d = self.depth
        if d==0:
            self.total.append(temp)
        else:
            for i in self.choices:
                if len(temp)==0:
                    self.permute(d-1,temp[:]+[i])
                elif i>temp[-1]:
                    self.permute(d-1,temp[:]+[i])

    def unordered_permute(self,d='default',temp=[],choices=[1],s=1):
        if d=='default':
            d = self.depth
        if d==0 and len(choices)==0:
            temp.append(s)
            self.total.append(temp)
        else:
            if len(temp)==0:
                for n,i in enumerate(self.choices):
                    s=(-1)**n
                    choices = self.choices.copy()
                    choices.pop(n)
                    self.unordered_permute(d-1,temp[:]+[i],choices,s)
                    temp=[]
            else:
                for n,j in enumerate(choices):
                    s*=(-1)**n
                    tempChoice = choices.copy()
                    tempChoice.pop(n)
                    self.unordered_permute(d-1,temp[:]+[j],tempChoice,s)

def get_Sz_mat(alpha,beta,s2s):
    '''
    Make sure that the molecular 
    '''
    norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
    norb+= len(beta['virtual']+beta['inactive']+beta['active'])
    alp = alpha['active']
    bet = beta['active']
    alpha = alpha['inactive']+alpha['active']
    beta = beta['inactive']+beta['active']
    sz = np.zeros((norb,norb),dtype=np.complex_)
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for pa in alp:
        sz[pa,pa]=0.5
    for pb in bet:
        sz[pb,pb]=-0.5
    return sz

def get_Sz2_mat(
        alpha,
        beta,
        s2s
        ):
    norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
    norb+= len(beta['virtual']+beta['inactive']+beta['active'])
    alp = alpha['active']
    bet = beta['active']
    alpha = alpha['inactive']+alpha['active']
    beta = beta['inactive']+beta['active']
    sz2_2 = np.zeros((norb,norb,norb,norb),dtype=np.complex_)
    sz2_1 = np.zeros((norb,norb),dtype=np.complex_)
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for pa in alp:
        pb = a2b[pa]
        for qb in bet:
            qa = b2a[qb]
            if pa==qa:
                sz2_2[pa,qa,pa,qa]+=0
            else:
                sz2_2[pa,qa,pa,qa]+=0.25
            if pb==qb:
                sz2_2[pb,qb,pb,qb]+=0
            else:
                sz2_2[pb,qb,pb,qb]+=0.25
            sz2_2[pb,qa,pb,qa]-=0.25
            sz2_2[pa,qb,pa,qb]-=0.25
    for pa in alp:
        sz2_1[pa,pa]+=0.25
    for pb in bet:
        sz2_1[pb,pb]+=0.25
    return sz2_1,sz2_2

def get_SpSm_mat(
        alpha,
        beta,
        s2s
        ):
    norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
    norb+= len(beta['virtual']+beta['inactive']+beta['active'])
    alp = alpha['active']
    bet = beta['active']

    alpha = alpha['inactive']+alpha['active']
    beta = beta['inactive']+beta['active']
    spsm_2 = np.zeros((norb,norb,norb,norb),dtype=complex_)
    spsm_1 = np.zeros((norb,norb),dtype=complex_)
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for pa in alp:
        pb = a2b[pa]
        for qb in bet:
            qa = b2a[qb]
            spsm_2[pa,qb,qa,pb]=-1
    for pa in alp:
        spsm_1[pa,pa]=1
    return spsm_1,spsm_2

def get_SmSp_mat(
        alpha,
        beta,
        s2s
        ):
    norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
    norb+= len(beta['virtual']+beta['inactive']+beta['active'])
    alp = alpha['active']
    bet = beta['active']
    alpha = alpha['inactive']+alpha['active']
    beta = beta['inactive']+beta['active']
    smsp = np.zeros((norb,norb,norb,norb),dtpye=complex_)
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for pa in alp:
        pb = a2b[pa]
        for qb in bet:
            qa = b2a[qb]
            smsp[pb,qa,pa,qb]=1
    return smsp


def S2(
        rdm2,
        rdm1,
        alpha,
        beta,
        s2s
        ):
    try: 
        alpha['active']
    except Exception:
        alpha = {'active':alpha,
                'inactive':[],
                'virtual':[]}
        beta = {'active':beta,
                'inactive':[],
                'virtual':[]}
    rdm2 = contract(rdm2)
    spm_1,spm_2  = get_SpSm_mat(
            alpha,
            beta,
            s2s)
    spm_2 = contract(spm_2)
    sz2_1,sz2_2 = get_Sz2_mat(
            alpha,
            beta,
            s2s)
    #print(sz2_1)
    sz2_2 = contract(sz2_2)
    #print(sz2_2)
    sz = get_Sz_mat(
            alpha,
            beta,
            s2s)
    s2pm_2 = reduce(np.dot, (spm_2,rdm2)).trace()
    s2pm_1 = reduce(np.dot, (spm_1,rdm1)).trace()
    s2z2_2= reduce(np.dot, (sz2_2,rdm2)).trace()
    s2z2_1= reduce(np.dot, (sz2_1,rdm1)).trace()
    s2z1= reduce(np.dot, (sz,rdm1)).trace()
    #print(s2pm_1,s2pm_2,s2z2_1,s2z2_2,s2z1)
    return s2pm_2+s2pm_1+s2z2_2+s2z2_1-s2z1

def S2_spatial(
        rdm2,
        rdm1,
        alpha,
        beta,
        s2s
        ):
    try: 
        alpha['active']
    except Exception:
        alpha = {'active':alpha,
                'inactive':[],
                'virtual':[]}
        beta = {'active':beta,
                'inactive':[],
                'virtual':[]}
    k = (alpha,beta,s2s)
    spm_1,spm_2  = get_SpSm_mat(*k)
    sz2_1,sz2_2 = get_Sz2_mat(*k)

    spm_1 = spin_to_spatial(spm_1,*k)
    spm_2 = spin_to_spatial(spm_2,*k)
    sz2_1 = spin_to_spatial(sz2_1,*k)
    sz2_2 = spin_to_spatial(sz2_2,*k)
    sz = spin_to_spatial(get_Sz_mat(*k),*k)


    rdm2 = contract(rdm2)
    spm_2 = contract(spm_2)
    sz2_2 = contract(sz2_2)
    #print('Spin!')
    #print(spm_1)
    #print(spm_2)
    #print(sz2_1)
    #print(sz2_2)
    print(sz)
    #print('------')

    s2pm_2 = reduce(np.dot, (spm_2,rdm2)).trace()
    s2pm_1 = reduce(np.dot, (spm_1,rdm1)).trace()
    s2z2_2= reduce(np.dot, (sz2_2,rdm2)).trace()
    s2z2_1= reduce(np.dot, (sz2_1,rdm1)).trace()
    s2z1= reduce(np.dot, (sz,rdm1)).trace()
    print(s2pm_1,s2pm_2,s2z2_1,s2z2_2,s2z1)
    return s2pm_2+s2pm_1+s2z2_2+s2z2_1-s2z1

def Sz_spatial(rdm1,alpha,beta,s2s):
    try: 
        alpha['active']
    except Exception:
        alpha = {'active':alpha,
                'inactive':[],
                'virtual':[]}
        beta = {'active':beta,
                'inactive':[],
                'virtual':[]}
    sz_mat = spin_to_spatial(
            get_Sz_mat(
                alpha,
                beta,
                s2s
                ),
            alpha,beta,s2s
            )
    sz = reduce(np.dot,
            (   sz_mat,
                rdm1
                )
        ).trace()
    return sz

def Sz(rdm1,alpha,beta,s2s):
    try: 
        alpha['active']
    except Exception:
        alpha = {'active':alpha,
                'inactive':[],
                'virtual':[]}
        beta = {'active':beta,
                'inactive':[],
                'virtual':[]}
    sz = reduce(np.dot,
            (
                get_Sz_mat(
                    alpha,
                    beta,
                    s2s
                    ),
                rdm1
                )
        ).trace()
    return sz



def factorial(p):
    if p==0:
        return 1
    else:
        return p*factorial(p-1)

def spin_to_spatial(
        rdm,
        alpha,
        beta,
        s2s
        ):
    '''
    Given a spin RDM (by default 2-RDM), provides a spin-traced or spatial RDM. 
    '''
    try:
        alpha = alpha['inactive']+alpha['active']+alpha['virtual']
        beta  =  beta['inactive']+ beta['active']+ beta['virtual']
    except:
        alpha = alpha
        beta = beta
    Nso = rdm.shape[0]
    No = Nso//2
    if len(rdm.shape)==4:
        # 2-RDM with indices (i k j l)
        nrdm = np.zeros((No,No,No,No),dtype=complex_)
        a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
        b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
        for i in alpha:
            for j in alpha:
                for k in alpha:
                    for l in alpha:
                        p,q,r,s = a2b[i],a2b[j],a2b[k],a2b[l]
                        temp = rdm[i,k,j,l] #aaaa
                        temp+= rdm[p,r,q,s] #bbbb
                        temp+= rdm[i,r,j,s] #abab
                        temp+= rdm[p,k,q,l] #baba
                        nrdm[i,k,j,l]=temp
    elif len(rdm.shape)==2:
        # 1-RDM
        nrdm = np.zeros((No,No),dtype=complex_)
        a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
        b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
        for i in alpha:
            for j in alpha:
                p,q = a2b[i],a2b[j]
                temp = rdm[i,j] #aa
                temp+= rdm[p,q] #bb
                nrdm[i,j]=temp

    return nrdm


def switch_alpha_beta(
        rdm2,
        alpha,
        beta
        ):
    Nso = rdm2.shape[0]
    alpha = alpha['inactive']+alpha['active']+alpha['virtual']
    beta  =  beta['inactive']+ beta['active']+ beta['virtual']
    nrdm2 = np.zeros(rdm2.shape,dtype=np.complex_)
    a2b = {alpha[i]:beta[i] for i in range(0,len(alpha))}
    b2a = {beta[i]:alpha[i] for i in range(0,len(beta ))}
    for i in alpha:
        for j in alpha:
            for k in alpha:
                for l in alpha:
                    p,q,r,s = a2b[i],a2b[j],a2b[k],a2b[l]
                    nrdm2[p,q,r,s]=rdm2[i,j,k,l]

    for i in beta:
        for j in beta:
            for k in beta:
                for l in beta:
                    p,q,r,s = b2a[i],b2a[j],b2a[k],b2a[l]
                    nrdm2[p,q,r,s]=rdm2[i,j,k,l]
    for i in alpha:
        for j in alpha:
            for k in beta:
                for l in beta:
                    p,q,r,s = a2b[i],a2b[j],b2a[k],b2a[l]
                    nrdm2[p,r,q,s]=rdm2[i,k,j,l]
                    nrdm2[r,p,q,s]=rdm2[k,i,j,l]
                    nrdm2[r,p,s,q]=rdm2[k,i,l,j]
                    nrdm2[p,r,s,q]=rdm2[i,k,l,j]
    return nrdm2

