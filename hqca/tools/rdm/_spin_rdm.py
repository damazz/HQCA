import numpy as np
import traceback
import sys
import numpy.linalg as LA
import datetime
from numpy import conj as con
from numpy import complex_
from functools import reduce
from hqca.tools.rdm._functions import *
from copy import deepcopy as copy

class RDM:
    '''
    Spin RDM class which allows for construction from Hartree-Fock state,
    reconstruction, multiplication and addition.
    '''
    def __init__(self,
            order=2,
            alpha=[],
            beta=[],
            rdm='hf',
            verbose=0,
            Ne=2,
            **kw
            ):
        self.p = order
        self.r = len(alpha)+len(beta) # num spin orbitals, fermionic sites
        self.R = int(self.r/2)
        self.v = verbose
        self.alp = alpha
        self.bet = beta
        self.s2s = {}
        self.Ne = Ne
        for n,a in enumerate(alpha):
            self.s2s[a]=n
        for n,b in enumerate(beta):
            self.s2s[b]=n
        if isinstance(rdm,str):
            if rdm in ['hartree','hf','scf']:
                if self.v>0:
                    print('Making Hartree-Fock {}-RDM'.format(self.p))
                self._build_hartree_fock(**kw)
            elif rdm in ['alpha-beta']:
                self._generate_from_ab_block(**kw)
            elif rdm in ['pass']:
                pass
        elif isinstance(rdm,type(np.array([[0.0],[1.0]]))):
            self.rdm = rdm
        else:
            self.rdm = np.zeros(
                    tuple([self.r for i in range(2*self.p)]),dtype=np.complex_)

    def _build_hartree_fock(self,Sz=0,S2=0,**kw):
        self.N_alp = int((self.Ne+2*Sz)/2)
        self.N_bet = self.Ne-self.N_alp
        self.Sz = Sz
        self.S2 = S2
        self._build_hf_spin(**kw)


    def _generate_from_spatial(self,fragment=None):
        rdm = copy(fragment)
        # 
        sys.exit('Can not generate from alpha-beta block of non singlet state yet.')
        aa = rdm.rdm - rdm.rdm.transpose(1,0,2,3)
        new = np.zeros((self.R,self.R,self.R,self.R))
        for i in range(self.R):
            for j in range(self.R):
                for k in range(self.R):
                    for l in range(self.R):
                        new[i,k,j,l]+=rdm.rdm[i,k,j,l]
        self.rdm = new

    def _generate_from_ab_block(self,fragment=None):
        rdm = copy(fragment)
        #if not self.S2==0:
        #    sys.exit('Can not generate from alpha-beta block of non singlet state yet.')
        aa = rdm - rdm.transpose(1,0,2,3)
        new = np.zeros((self.r,self.r,self.r,self.r))
        for i in range(self.R):
            I =i+self.R
            for j in range(self.R):
                J = j+self.R
                for k in range(self.R):
                    K = self.R+k
                    for l in range(self.R):
                        L = l+self.R
                        new[i,k,j,l]+=aa[i,k,j,l]
                        new[I,K,J,L]+=aa[i,k,j,l]
                        # which means we also do ki jl? yeah. 

                        new[i,K,j,L]+=rdm[i,k,j,l]
                        new[I,k,J,l]+=rdm[i,k,j,l]

                        new[K,i,j,L]-=rdm[i,k,j,l]
                        new[k,I,J,l]-=rdm[i,k,j,l]
        self.rdm =  new

    def _get_ab_block(self):
        if self.p==2:
            ab_basis = []
            for i in self.alp:
                for j in self.bet:
                    ab_basis.append([i,j])
            N_ab = len(ab_basis)
            ab = np.zeros((N_ab,N_ab),dtype=np.complex_)
            for ni,I in enumerate(ab_basis):
                for nj,J in enumerate(ab_basis):
                    ind = tuple(I+J)
                    ab[ni,nj]+= self.rdm[ind]
        return ab

    def spatial_rdm(self):
        '''
        get the spin integrated 2-RDM

        note, ordering is preserved
        '''
        if self.p==2:
            new = np.zeros((self.R,self.R,self.R,self.R),dtype=np.complex_)
            for i in range(self.R):
                I = self.R+i
                for j in range(self.R):
                    J = self.R+j
                    for k in range(self.R):
                        K = self.R+k
                        for l in range(self.R):
                            L = self.R+l
                            new[i,k,j,l]+= self.rdm[i,K,j,L] #abba  
                            new[i,k,j,l]+= self.rdm[I,k,J,l] #baab
                            new[i,k,j,l]+= self.rdm[i,k,j,l] #aaaa
                            new[i,k,j,l]+= self.rdm[I,K,J,L] #bbbb
        elif self.p==1:
            new = np.zeros((self.R,self.R),dtype=np.complex_)
            for i in range(self.R):
                I = self.R+i
                for j in range(self.R):
                    J = self.R+j
                    new[i,j]+= self.rdm[I,J] #baab
                    new[i,j]+= self.rdm[i,j] #aaaa
        return new

    def symmetric_block(self):
        '''
        returns the alpha-alpha block (alpha beta alpha beta)
        '''
        new = np.zeros((self.R,self.R,self.R,self.R),dtype=np.complex_)
        for i in self.alp:
            I = i%self.R
            for j in self.alp:
                J = j%self.R
                for k in self.alp:
                    K = k%self.R
                    for l in self.alp:
                        L = l%self.R
                        new[I,K,J,L]+= self.rdm[i,k,j,l]
        return new

    def antisymmetric_block(self,spin=True):
        '''
        returns the alpha-beta blocks
        '''
        new = np.zeros((self.R,self.R,self.R,self.R),dtype=np.complex_)
        for i in self.alp:
            I = i%self.R
            for j in self.alp:
                J = j%self.R
                for k in self.bet:
                    K = k%self.R
                    for l in self.bet:
                        L = l%self.R
                        new[I,K,J,L]+= self.rdm[i,k,j,l]
        return new

    def transpose(self,*args,**kwargs):
        self.rdm.transpose(*args,**kwargs)
        return self

    def save(self,name='default',precision=8,dtype='rdm',spin=True):
        if name=='default':
            name = datetime.strftime(datetime.now(),'%m%d%y')
            name+= '-'
            name+= datetime.strftime(datetime.now(),'%H%M')
        if dtype=='rdm':
            with open(name+'.rdm','w') as fp:
                self.expand()
                if spin:
                    d = self.antisymmetric_block()
                else:
                    d = self.spatial_rdm()
                nz = np.nonzero(d)
                first = ''
                for i in self.alp:
                    first+='{} '.format(i)
                first+= '\n'
                fp.write(first)
                first = ''
                for i in self.bet:
                    first+='{} '.format(i)
                first+= '\n'
                fp.write(first)
                line = ''
                if self.p==2:
                    for i,j,k,l in zip(nz[0],nz[1],nz[2],nz[3]):
                        line+= '{} {} {} {} {:f} {:f}\n'.format(
                                i+1,j+1,k+1,l+1,
                                round(d[i,j,k,l].real,precision),
                                round(d[i,j,k,l].imag,precision),
                                )
                    fp.write(line[:-1])
        else:
            print(dtype)
            sys.exit('File type for save rdm not recognized')

    def analysis(self,verbose=True,print_rdms=False,split=False):
        # start with alpha alpha block
        # generate basis
        self.expand()
        if self.p==2:
            aa_basis = []
            ab_basis = []
            bb_basis = []
            for i in self.alp:
                for j in self.alp:
                    if j>i:
                        aa_basis.append([i,j])
            for i in self.alp:
                for j in self.bet:
                    ab_basis.append([i,j])
            for i in self.bet:
                for j in self.bet:
                    if j>i:
                        bb_basis.append([i,j])
            N_aa = len(aa_basis)
            N_ab = len(ab_basis)
            N_bb = len(bb_basis)
            aa = np.zeros((N_aa,N_aa),dtype=np.complex_)
            ab = np.zeros((N_ab,N_ab),dtype=np.complex_)
            bb = np.zeros((N_bb,N_bb),dtype=np.complex_)
            for ni,I in enumerate(aa_basis):
                for nj,J in enumerate(aa_basis):
                    ind = tuple(I+J)
                    aa[ni,nj]+= self.rdm[ind]
            for ni,I in enumerate(ab_basis):
                for nj,J in enumerate(ab_basis):
                    ind = tuple(I+J)
                    ab[ni,nj]+= self.rdm[ind]
            for ni,I in enumerate(bb_basis):
                for nj,J in enumerate(bb_basis):
                    ind = tuple(I+J)
                    bb[ni,nj]+= self.rdm[ind]
            aa_eig = np.linalg.eigvalsh(aa)
            ab_eig = np.linalg.eigvalsh(ab)
            bb_eig = np.linalg.eigvalsh(bb)
            print('Analysis of 2-RDM')
            print('---------------------------------')
            print('AA basis: ')
            print(aa_basis)
            print('alpha-alpha block: ')
            print(aa)
            print('eigenvalues: ')
            print(aa_eig)
            print('---------------------------------')
            print('BB basis: ')
            print(bb_basis)
            print('beta-beta block: ')
            print(bb)
            print('eigenvalues: ')
            print(bb_eig)
            print('---------------------------------')
            print('AB basis: ')
            print(ab_basis)
            print('alpha-beta block: ')
            if split:
                print('real: ')
                print(np.real(ab))
                print('imaginary')
                print(np.imag(ab))
            else:
                print(ab)
            print('eigenvalues: ')
            print(ab_eig)
            print('Analysis of 1-RDM and local properties')
            a1 = np.zeros((len(self.alp),len(self.alp)),dtype=np.complex_)
            b1 = np.zeros((len(self.alp),len(self.alp)),dtype=np.complex_)
            rdm1 = self.reduce_order()
            for ni,I in enumerate(self.alp):
                for nj,J in enumerate(self.alp):
                    ind = tuple([I,J])
                    a1[ni,nj]+= rdm1.rdm[ind]
            for ni,I in enumerate(self.bet):
                for nj,J in enumerate(self.bet):
                    ind = tuple([I,J])
                    b1[ni,nj]+= rdm1.rdm[ind]
            print('---------------------------------')
            print('alpha block: ')
            print(a1)
            print('eigenvalues: ')
            print(np.linalg.eigvalsh(a1))
            a_eig = np.linalg.eigvalsh(a1)
            print('---------------------------------')
            print('beta block: ')
            print(b1)
            print('eigenvalues: ')
            print(np.linalg.eigvalsh(b1))
            b_eig = np.linalg.eigvalsh(b1)
            return aa_eig,bb_eig,ab_eig,a_eig,b_eig
        elif self.p==1:
            print('Analysis of 1-RDM and local properties')
            a1 = np.zeros((len(self.alp),len(self.alp)),dtype=np.complex_)
            b1 = np.zeros((len(self.alp),len(self.alp)),dtype=np.complex_)
            for ni,I in enumerate(self.alp):
                for nj,J in enumerate(self.alp):
                    ind = tuple([I,J])
                    a1[ni,nj]+= self.rdm[ind]
            for ni,I in enumerate(self.bet):
                for nj,J in enumerate(self.bet):
                    ind = tuple([I,J])
                    b1[ni,nj]+= self.rdm[ind]
            print('---------------------------------')
            print('alpha block: ')
            print(a1)
            print('eigenvalues: ')
            print(np.linalg.eigvalsh(a1))
            print('---------------------------------')
            print('beta block: ')
            print(b1)
            print('eigenvalues: ')
            print(np.linalg.eigvalsh(b1))
            pass


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
            print('Trying to add a {} to a RDM, '.format(str(type(rdm))))
            sys.exit('wrong type specified.')
        c2 = self.alp==rdm.alp
        c3, c4 = self.bet==rdm.bet,self.p==rdm.p
        if c2+c3+c4<3:
            print('Checks: ')
            print('alp: {}, bet: {}'.format(c2,c3))
            traceback.print_exc()
            print('Error in adding.')
            sys.exit('You have RDMs for different systems apparently.')
        nRDM = RDM(
                order=self.p,
                alpha=self.alp,
                beta=self.bet,
                rdm=None,
                )
        self.expand()
        rdm.expand()
        nRDM.rdm = self.rdm+rdm.rdm
        return nRDM

    def reduce_order(self):
        #
        nRDM = RDM(
                order=self.p-1,
                alpha=self.alp,
                beta=self.bet,
                rdm=None,
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
        rdm.rdm*= -1
        return self+rdm

    def __mul__(self,rdm):
        if type(rdm)==type(self):
            pass
        elif  type(rdm) in [type(1),type(0),type(0.5)]:
            self.rdm = self.rdm*rdm
            return self
        self.expand()
        rdm.expand()
        c2 = self.alp==rdm.alp
        c3, c4 = self.bet==rdm.bet,self.Ne==rdm.Ne
        if c2+c3+c4<3:
            print('Checks: ')
            print(c2,c3,c4)
            traceback.print_exc()
            print('Error in multiplication.')
            sys.exit('You have RDMs for different systems apparently.')
        nRDM = RDM(
                order=self.p+rdm.p,
                alpha=self.alp,
                beta=self.bet,
                rdm=None,
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
            self.sz = Sz(
                    rdm1.rdm,
                    self.alp,
                    self.bet,
                    self.s2s)
            self.s2 = S2(
                    self.rdm,
                    rdm1.rdm,
                    self.alp,
                    self.bet,
                    self.s2s)

    def _build_hf_singlet(self,**kw):
        '''
        build a p-RDM that comes from a single determinant
        '''
        self.rdm = np.zeros(
                tuple([self.r for i in range(2*self.p)]),dtype=np.complex_)
        occAlp = self.alp[:self.N_alp]
        occAlp = [i for i in occAlp]
        occBet = self.bet[:self.N_bet]
        occBet = [i for i in occBet]
        Rec = Recursive(depth=self.p,choices=occAlp+occBet)
        Rec.permute()
        self.total=Rec.total
        for creInd in self.total:
            annInd = creInd[:]
            annPerm = Recursive(choices=annInd)
            annPerm.unordered_permute()
            crePerm = Recursive(choices=creInd)
            crePerm.unordered_permute()
            for c in crePerm.total:
                for a in annPerm.total:
                    s = c[-1]*a[-1]
                    ind = tuple(c[:-1]+a[:-1])
                    self.rdm[ind]=s

    def _build_hf_spin(self):
        '''
        build a p-RDM that comes from a single determinant,
        but with S = 1
        '''
        self.rdm = np.zeros(
                tuple([self.r for i in range(2*self.p)]),dtype=np.complex_)
        occAlp = self.alp[:int((self.Ne+2*self.Sz)//2)]
        occBet = self.bet[:int((self.Ne-2*self.Sz)//2)]
        Rec = Recursive(depth=self.p,choices=occAlp+occBet)
        Rec.permute()
        self.total=Rec.total
        for creInd in self.total:
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

    def nat_occ(self):
        if self.p==2:
            rdm1 = self.reduce_order()
            eigs = np.linalg.eigvalsh(rdm1.rdm)
        return eigs

    def cumulant(self):
        if self.p==2:
            rdm1 = self.reduce_order()
            rdm2 = rdm1*rdm1
        return self-rdm2

    def reconstruct(self,
            approx='V',
            method='cumulant',):
        if self.p==2 and self.Ne<3:
            nRDM = RDM(
                    order=self.p+1,
                    alpha=self.alp,
                    beta=self.bet,
                    rdm=None,
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
        elif self.p==1:
            rdm = self*self
            return rdm

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




