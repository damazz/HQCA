import numpy as np
np.set_printoptions(linewidth=200)
from hqca.tools.EnergyFunctions import Storage
from hqca.quantum.QuantumFunctions import QuantumStorage
from hqca.tools import Functions as fx
from hqca.tools.RDM import RDMs


class ACSEStorage(Storage):
    '''
    modified Storage object, more well suited for containing the ACSE related
    objets, such as the 2S matrix
    '''
    def __init__(self,**kwargs):
        Storage.__init__(self,**kwargs)
        Storage.gas(self)
        Storage.gsm(self)
        self.modified='ACSE'
        self.S2 = np.zeros((
                    2*self.No_tot,2*self.No_tot,
                    2*self.No_tot,2*self.No_tot)
                    )
        self.r = self.No_tot*2 # spin orbitals
        # if hartree fock
        self.rdm2 = RDMs(
                order=2,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='hf',
                Ne=self.Ne_as,
                Sz=0,S2=0)
        self.rdm3 = RDMs(
                order=3,
                alpha = self.alpha_mo['active'],
                beta  = self.beta_mo['active'],
                state='hf',
                Ne=self.Ne_as,
                Sz=0,S2=0)
        self.update_full_ints()
        self.ints_2e = fx.expand(self.ints_2e)
        self.nonH2 = np.nonzero(self.ints_2e)
        self.zipH2 = list(zip(
                self.nonH2[0],self.nonH2[1],
                self.nonH2[2],self.nonH2[3]))
        self.nonH1 = np.nonzero(self.ints_1e)
        self.zipH1 = list(zip(self.nonH1[0],self.nonH1[1]))
        self.active_rdm2e = list(np.nonzero(self.rdm2.rdm))
        self.active_rdm3e = list(np.nonzero(self.rdm3.rdm))

    def evaluate_energy(self):
        e_h1 = 0
        e_h2 = 0
        return e_h1 + e_h2 + self.E_ne

class ACSEQuantumStorage(QuantumStorage):
    '''
    modified QuantStorage object
    '''
    def __init__(self,
            method='acse',
            **kwargs
            ):
        QuantumStorage.__init__(self,
                method='acse',
                **kwargs)
        self.tomo_rdm='acse'

    def _adjust_ansatz(self,
            ):
        '''
        given self.S, generate an ansatz
        '''
        pass

def findSPairs(Store,QuantStore):
    '''
    '''
    pass
    alp_orb = Store.alpha_mo['active']
    bet_orb = Store.beta_mo['active']
    S = []
    alpha = QuantStore.alpha['active']
    beta = QuantStore.beta['active']
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
                            if i>=j or k>=l:
                                continue
                            spin = ['a','b','b','a']
                        else:
                            if i>=k or l>=j:
                                continue
                            if block[ze]=='aa':
                                spin = ['a','a','a','a']
                            else:
                                spin = ['b','b','b','b']
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
                            t2 = c3*Store.rdm3.rdm[i,p,r,j,l,s]
                            t3 = c5*Store.rdm3.rdm[i,k,r,l,q,s]
                            t4 = c6*Store.rdm3.rdm[i,k,r,j,s,q]
                            t5 = (c1*c4-c2*c3)*Store.rdm2.rdm[p,r,j,l]
                            t6 = (c5*c8-c7*c6)*Store.rdm2.rdm[i,k,q,s]
                            temp = 6*(t1+t2+t3+t4)+2*(t5+t6)
                            temp*= Store.ints_2e[p,r,q,s]
                            term+= temp
                        for p,q in Store.zipH1:
                            c1,c2 = int(i==q),int(k==q)
                            c3,c4 = int(j==p),int(l==p)
                            if c1+c2+c3+c4==0:
                                continue
                            t1 = -c1*Store.rdm2.rdm[k,p,j,l]
                            t2 =  c2*Store.rdm2.rdm[i,p,j,l]
                            t3 =  c3*Store.rdm2.rdm[i,k,l,q]
                            t4 = -c4*Store.rdm2.rdm[i,k,j,q]
                            temp = t1+t2+t3+t4
                            temp*= Store.ints_1e[p,q]
                            term+= temp
                        if abs(term)>1e-10:
                            S.append([i,k,l,j,term,['+','+','-','-'],spin])
    QuantStore.S = S
