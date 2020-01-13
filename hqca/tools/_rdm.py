import numpy as np
import traceback
import sys
import numpy.linalg as LA
from numpy import conj as con
from numpy import complex_
from functools import reduce
from hqca.tools import _rdmfunctions as rdmf


class RDM:
    '''
    RDM class which allows for construction from Hartree-Fock state,
    reconstruction, multiplication and addition. 
    '''
    def __init__(self,
            order=2,
            alpha=[],
            beta=[],
            state='hf',
            Ne=None,
            S=0,
            S2=0,
            verbose=0,
            rdm=None
            ):
        self.p = order
        self.r = len(alpha)+len(beta) # spin
        self.R = int(self.r/2)
        self.v = verbose
        self.alp = alpha
        self.bet = beta
        self.S=S
        self.S2=S2
        self.Ne=Ne
        self.s2s = {}
        for n,a in enumerate(alpha):
            self.s2s[a]=n
        for n,b in enumerate(beta):
            self.s2s[b]=n
        self.N_alp = int(Ne+S)//2
        self.N_bet = Ne-self.N_alp
        if state in ['hartree','hf','scf']:
            if self.v>0:
                print('Making Hartree-Fock {}-RDM'.format(self.p))
            if self.S==0:
                self._build_hf_singlet()
            elif self.S==1:
                self._build_hf_doublet()
        elif state in ['wf']:
            pass
        elif state in ['given','provided','spec']:
            try:
                self.rdm = rdm.copy()
            except Exception:
                self.rdm = rdm
        else:
            self.rdm = np.zeros(
                    tuple([self.r for i in range(2*self.p)]),dtype=np.complex_)

    def copy(self):
        nRDM = RDM(
                order=self.p,
                alpha=self.alp,
                beta=self.bet,
                state=None,
                S=self.S,
                Ne=self.Ne,
                )
        nRDM.rdm = self.rdm.copy()
        return nRDM

    def observable(self,H):
        self.contract()
        en = np.dot(H,self.rdm).trace()
        self.expand()
        return en

    def trace(self):
        try:
            self.rdm.trace()[0,0]
            self.switch()
            test=  self.rdm.trace()
            self.switch()
        except Exception as e:
            test = self.rdm.trace()
        return test

        # get trace of RDM

    def __add__(self,rdm):
        if type(rdm)==type(self):
            pass
        else:
            sys.exit('Wrong type specified.')
        c1, c2 = self.Ne==rdm.Ne,self.alp==rdm.alp
        c3, c4 = self.bet==rdm.bet,self.p==rdm.p
        if c1+c2+c3+c4<4:
            print('Checks: ')
            print('Ne: {}, alp: {}, bet: {}'.format(c1,c2,c3))
            traceback.print_exc()
            print('Error in adding.')
            sys.exit('You have RDMs for different systems apparently.')
        nRDM = RDM(
                order=self.p,
                alpha=self.alp,
                beta=self.bet,
                S=self.S,
                state=None,
                Ne=self.Ne,
                )
        self.expand()
        rdm.expand()
        nRDM.rdm = self.rdm+rdm.rdm
        return nRDM

    def reduce_order(self):
        nRDM = RDM(
                order=self.p-1,
                alpha=self.alp,
                beta=self.bet,
                state=None,
                Ne=self.Ne,
                S=self.S,
                )
        self.expand()
        if self.p==3:
            for i in range(0,self.r):
                for j in range(0,self.r):
                    for k in range(0,self.r):
                        for l in range(0,self.r):
                            i1 = tuple([i,j,k,l])
                            for x in range(0,self.r):
                                i2 = tuple([i,j,x,k,l,x])
                                nRDM.rdm[i1]+=self.rdm[i2]
        elif self.p==2:
            for i in range(0,self.r):
                for j in range(0,self.r):
                    i1 = tuple([i,j])
                    for x in range(0,self.r):
                        i2 = tuple([i,x,j,x])
                        nRDM.rdm[i1]+=self.rdm[i2]
        nRDM.rdm*=(1/(self.Ne-self.p+1))
        return nRDM

    def __sub__(self,rdm):
        if type(rdm)==type(self):
            pass
        else:
            sys.exit('Wrong type specified.')
        c1, c2 = self.Ne==rdm.Ne,self.alp==rdm.alp
        c3, c4 = self.bet==rdm.bet,self.p==rdm.p
        if c1+c2+c3+c4<4:
            print('Checks: ')
            print('Ne: {}, alp: {}, bet: {}'.format(c1,c2,c3))
            print(self.S,rdm.S)
            print('Error in subtracting.')
            traceback.print_exc()
            sys.exit('You have RDMs for different systems apparently.')
        nRDM = RDM(
                order=self.p,
                alpha=self.alp,
                beta=self.bet,
                state=None,
                Ne=self.Ne,
                )
        self.expand()
        nRDM.expand()
        nRDM.rdm = self.rdm-rdm.rdm
        return nRDM

    def __mul__(self,rdm):
        if type(rdm)==type(self):
            pass
        elif type(rdm) in [type(1),type(0),type(0.5)]:
            self.rdm = self.rdm*rdm
            return self
        self.expand()
        rdm.expand()
        c1, c2 = self.Ne==rdm.Ne,self.alp==rdm.alp
        c3, c4 = self.bet==rdm.bet,self.Ne==rdm.Ne
        if c1+c2+c3+c4<4:
            print('Checks: ')
            print(c1,c2,c3,c4)
            traceback.print_exc()
            print('Error in multiplication.')
            sys.exit('You have RDMs for different systems apparently.')
        nRDM = RDM(
                order=self.p+rdm.p,
                alpha=self.alp,
                beta=self.bet,
                state=None,
                Ne=self.Ne,
                S=self.S,
                )
        non1 = list((np.nonzero(self.rdm)))
        non2 = list((np.nonzero(rdm.rdm)))
        cr1 = np.asarray(non1[:len(non1)//2])
        cr2 = np.asarray(non2[:len(non2)//2])
        an1 = np.asarray(non1[len(non1)//2:])
        an2 = np.asarray(non2[len(non2)//2:])
        CreTerms = []
        AnnTerms = []
        for i in range(cr1.shape[1]):
            c = cr1[:,i]
            for j in range(cr2.shape[1]):
                s = cr2[:,j]
                dup=False
                for x in c:
                    if x in s:
                        dup=True
                newCre = list(np.concatenate((c,s)))
                for i in range(len(newCre)-1):
                    if newCre[i]>newCre[i+1]:
                        dup=True
                if newCre in CreTerms:
                    dup=True
                if dup:
                    continue
                CreTerms.append(newCre)
        for k in range(an1.shape[1]):
            a = an1[:,k]
            for j in range(an2.shape[1]):
                t = an2[:,j]
                dup=False
                for x in a:
                    if x in t:
                        dup=True
                newAnn=list(np.concatenate((a,t)))
                for l in range(len(newAnn)-1):
                    if newAnn[l]>newAnn[l+1]:
                        dup=True
                if newAnn in AnnTerms:
                    dup=True
                if dup:
                    continue
                AnnTerms.append(newAnn)
        for i in CreTerms:
            antiSymmCre = Recursive(choices=i)
            antiSymmCre.unordered_permute()
            for j in AnnTerms:
                antiSymmAnn = Recursive(choices=j)
                antiSymmAnn.unordered_permute()
                # need to calculate term
                sumTerms = 0
                for cre in antiSymmCre.total:
                    for ann in antiSymmAnn.total:
                        ind1 = tuple(cre[:self.p]+ann[:self.p])
                        ind2 = tuple(cre[self.p:-1]+ann[self.p:-1])
                        Term = (self.rdm[ind1]*rdm.rdm[ind2])
                        Term*= cre[-1]*ann[-1]
                        sumTerms+= Term
                #sumTerms*=(len(antiSymmCre.total))**-1
                sumTerms*=(len(antiSymmCre.total)*len(antiSymmAnn.total))**-(1)
                sumTerms*=((factorial(self.p+rdm.p))/(
                    factorial(self.p)*factorial(rdm.p)))
                for cre in antiSymmCre.total:
                    for ann in antiSymmAnn.total:
                        indN = tuple(cre[:-1]+ann[:-1])
                        sign = cre[-1]*ann[-1]
                        #print(indN,sign)
                        nRDM.rdm[indN]=sumTerms*sign
        return nRDM
        # wedge product

    def get_spin_properties(self):
        if self.p==3:
            pass
        elif self.p==2:
            rdm1 = self.reduce_order()
            self.sz = rdmf.Sz(
                    rdm1.rdm,
                    self.alp,
                    self.bet,
                    self.s2s)
            self.s2 = rdmf.S2(
                    self.rdm,
                    rdm1.rdm,
                    self.alp,
                    self.bet,
                    self.s2s)

    def get_overlap(self,rdm):

        c1, c2 = self.Ne==RDM.Ne,self.alp==RDM.alp
        c3, c4 = self.bet==RDM.bet,self.S==RDM.S
        if not (c1 and c2 and c3 and c4):
            print('Hrm.')

        t = self.Ne*(self.Ne-1)
        if self.Ne%2==0:
            coeff = 2*t*(1-self.Ne/(t*self.r**2))
        else:
            coeff = 2*t*(1-(self.Ne-1)/(t*self.r**2))
        self.contract()
        test = self.rdm-RDM.rdm

        eigs = np.sqrt(
                np.sum(
                np.abs(
                    np.linalg.eigvalsh(
                        np.dot(
                            test,test.T
                            )
                    )
                    )/coeff
                )
                )
        return np.real(eigs)

    def _build_hf_singlet(self):
        '''
        build a p-RDM that comes from a single determinant
        '''
        self.rdm = np.zeros(
                tuple([self.r for i in range(2*self.p)]),dtype=np.complex_)
        occAlp = self.alp[:self.Ne//2]
        occBet = self.bet[:self.Ne//2]
        Rec = Recursive(depth=self.p,choices=occAlp+occBet)
        Rec.permute()
        self.total=Rec.total
        for creInd in self.total:
            #annInd = creInd[::-1]
            annInd = creInd[:]
            annPerm = Recursive(choices=annInd)
            annPerm.unordered_permute()
            crePerm = Recursive(choices=creInd)
            crePerm.unordered_permute()
            for c in crePerm.total:
                for a in annPerm.total:
                    s = c[-1]*a[-1]
                    ind = tuple(c[:-1]+a[:-1])
                    self.rdm[ind]=s#*(1/factorial(self.p))

    def _build_hf_doublet(self):
        '''
        build a p-RDM that comes from a single determinant,
        but with S neq 0
        '''
        self.rdm = np.zeros(
                tuple([self.r for i in range(2*self.p)]),dtype=np.complex_)
        occAlp = self.alp[:int((self.Ne+self.S)//2)]
        occBet = self.bet[:int((self.Ne-self.S)//2)]
        Rec = Recursive(depth=self.p,choices=occAlp+occBet)
        Rec.permute()
        self.total=Rec.total
        for creInd in self.total:
            #annInd = creInd[::-1]
            annInd = creInd[:]
            annPerm = Recursive(choices=annInd)
            annPerm.unordered_permute()
            crePerm = Recursive(choices=creInd)
            crePerm.unordered_permute()
            for c in crePerm.total:
                for a in annPerm.total:
                    s = c[-1]*a[-1]
                    ind = tuple(c[:-1]+a[:-1])
                    self.rdm[ind]=s#*(1/factorial(self.p))

    def reconstruct(self,
            approx='V',
            method='cumulant',):
        if self.p==2 and self.Ne<3:
            nRDM = RDM(
                    order=self.p+1,
                    alpha=self.alp,
                    beta=self.bet,
                    state=None,
                    Ne=self.Ne,
                    S=self.S,
                    )
            return nRDM
        if not method=='cumulant':
            sys.exit('Can\'t perform non-cumulant reconstruction.')
        if self.p==2:
            self.expand()
            rdm1 = self.reduce_order()
        if approx in ['v','V','valdemoro','Valdemoro']:
            rdm3a = rdm1*rdm1*rdm1
            rdm2w = self-rdm1*rdm1
            rdm3b = (rdm2w*rdm1)*3
            rdm3 = rdm3a + rdm3b
            return rdm3

    def switch(self):
        size = len(self.rdm.shape)
        if size==2 and not self.p==1:
            self.expand()
        else:
            self.contract()
    
    def contract(self):
        size = len(self.rdm.shape)
        if not self.p==1:
            self.rdm = np.reshape(
                    self.rdm,
                    (
                        self.r**self.p,
                        self.r**self.p
                        )
                    )
    
    def expand(self):
        size = len(self.rdm.shape)
        if not self.p==1:
            self.rdm = np.reshape(
                    self.rdm,
                    (tuple([self.r for i in range(2*self.p)]))
                    )

class Recursive:
    def __init__(self,
            depth='default',
            choices=[]
            ):
        if depth=='default':
            depth=len(choices)
        self.depth=depth
        self.total=[]
        self.choices = list(choices)

    def permute(self,d='default',temp=[]):
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



def build_2rdm(
        wavefunction,
        alpha,beta,
        region='full',
        rdm2=None):
    '''
    Given a wavefunction, and the alpha beta orbitals, will construct the 2RDM
    for a system.
    Note, the output format is for a general two body operator, aT_i, aT_k, a_l, a_j
    i/j are electron 1, and k/l are electron 2
    '''
    def phase_2e_oper(I,K,L,J,det,low=0):
        # very general phase operator - works for ANY i,k,l,j
        # generates new occupation as well as the sign of the phase
        def new_det(i,k,l,j,det):
            # transform old determinant to new
            det=det[:j]+'0'+det[j+1:]
            det=det[:l]+'0'+det[l+1:]
            det=det[:k]+'1'+det[k+1:]
            det=det[:i]+'1'+det[i+1:]
            return det
        def delta(a,b):
            # delta function
            delta =0
            if a==b:
                delta = 1
            return delta
        def det_phase(det,place,start):
            # generate phase of a determinant based on number of occupied orbitals with index =1
            p = 1
            for i in range(start,place):
                if det[i]=='1':
                    p*=-1
            return p
        # conditionals for switching orbital orderings
        a1 = (L<=J)
        b1,b2 = (K<=L),(K<=J)
        c1,c2,c3 = (I<=K),(I<=L),(I<=J)
        eps1,eps2,eps3 = 1,1,1
        if a1%2==1: #i.e., if J<L
            eps1=-1
        if (b1+b2)%2==1 : # if K>L, K>J
            eps2=-1
        if (c1+c2+c3)%2==1: # if I>K,I>L,I>J
            eps3=-1
        t2 = 1-delta(L,J) # making sure L/J, I/K not same orbital
        t1 = 1-delta(I,K)
        t7 = eps1*eps2*eps3 
        d1 = delta(I,L) 
        d2 = delta(I,J)
        d3 = delta(K,L)
        d4 = delta(K,J)
        kI = int(det[I]) # occupation of orbitals
        kK = int(det[K])
        kL = int(det[L])
        kJ = int(det[J])
        pI = det_phase(det,I,start=low)
        pK = det_phase(det,K,start=low)
        pL = det_phase(det,L,start=low)
        pJ = det_phase(det,J,start=low)
        t6 = pJ*pL*pK*pI
        t5 = kL*kJ # if 0, then we have a problem
        t3 = d1+d2+1-kI # IL or IJ are the same, and I is already occupied - if I is occupied and not I=J, or I=L, then 0
        t4 = d3+d4+1-kK # same as above, for K and K/L, K/J
        Phase = t1*t2*t3*t4*t5*t6*t7
        ndet = new_det(I,K,L,J,det)
        return Phase,ndet
    #
    # First, alpha alpha 2-RDM, by selecting combinations only within alpha
    #
    wf = wavefunction
    if region=='full':
        norb = len(alpha['inactive']+alpha['virtual']+alpha['active'])
        norb+= len(beta['virtual']+beta['inactive']+beta['active'])
        alpha = alpha['inactive']+alpha['active']
        beta = beta['inactive']+beta['active']
        rdm2 = np.zeros((norb,norb,norb,norb),dtype=np.complex_)
    elif region=='active':
        ai = set(alpha['inactive'])
        bi = set(beta['inactive'])
        alpha = alpha['inactive']+alpha['active']
        beta = beta['inactive']+beta['active']
    for i in alpha:
        for k in alpha:
            if (i<k):
                for l in alpha:
                    for j in alpha:
                        if (l<j):
                            #low = min(i,l)
                            low = 0 
                            if region=='active':
                                if set((i,j,k,l)).issubset(ai):
                                    continue
                                rdm2[i,k,l,j]=0
                                rdm2[k,i,l,j]=0
                                rdm2[k,i,j,l]=0
                                rdm2[i,k,j,l]=0
                            for det1 in wf:
                                ph,check = phase_2e_oper(i,k,l,j,det1,low)
                                if ph==0:
                                    continue
                                else:
                                    if check in wf:
                                        rdm2[i,k,l,j]+=-1*ph*wf[det1]*con(wf[check])
                                        rdm2[k,i,j,l]+=-1*ph*wf[det1]*con(wf[check])
                                        rdm2[i,k,j,l]+=+1*ph*wf[det1]*con(wf[check])
                                        rdm2[k,i,l,j]+=+1*ph*wf[det1]*con(wf[check])

    for i in beta:
        for k in beta:
            if (i<k):
                for l in beta:
                    for j in beta:
                        if (l<j):
                            low = 0
                            if region=='active':
                                if set((i,j,k,l)).issubset(bi):
                                    continue
                                rdm2[i,k,l,j]=0
                                rdm2[k,i,l,j]=0
                                rdm2[k,i,j,l]=0
                                rdm2[i,k,j,l]=0
                            for det1 in wf:
                                ph,check = phase_2e_oper(i,k,l,j,det1,low)
                                if ph==0:
                                    continue
                                else:
                                    if check in wf:
                                        rdm2[i,k,l,j]+=-1*ph*wf[det1]*con(wf[check])
                                        rdm2[k,i,j,l]+=-1*ph*wf[det1]*con(wf[check])
                                        rdm2[i,k,j,l]+=+1*ph*wf[det1]*con(wf[check])
                                        rdm2[k,i,l,j]+=+1*ph*wf[det1]*con(wf[check])

    for i in alpha:
        for j in alpha:
            for k in beta:
                for l in beta:
                    low = 0
                    if region=='active':
                        if set((i,j,k,l)).issubset(ai.union(bi)):
                            continue
                        rdm2[i,k,j,l]=0
                        rdm2[i,k,l,j]=0
                        rdm2[k,i,l,j]=0
                        rdm2[k,i,j,l]=0
                    for det1 in wf:
                        ph,check = phase_2e_oper(i,k,l,j,det1,low)
                        if ph==0:
                            continue
                        else:
                            if check in wf:
                                rdm2[i,k,l,j]+=-1*ph*wf[det1]*con(wf[check])
                                rdm2[k,i,j,l]+=-1*ph*wf[det1]*con(wf[check])
                                rdm2[i,k,j,l]+=+1*ph*wf[det1]*con(wf[check])
                                rdm2[k,i,l,j]+=+1*ph*wf[det1]*con(wf[check])
    return rdm2

def factorial(p):
    if p==0:
        return 1
    else:
        return p*factorial(p-1)


