import numpy as np
import traceback
import sys
import numpy.linalg as LA
import datetime
from numpy import conj as con
from numpy import complex_
from functools import reduce
from hqca.tools.rdm._functions import *
from hqca.tools.rdm._spin_rdm import *
from copy import deepcopy as copy


class SpatialRDM(RDM):
    def __init__(self,rdm='hf',**kw):
        if rdm=='given':
            kw['rdm']= 'pass'
            RDM.__init__(self,**kw)
            self.rdm = rdm
        else:
            kw['rdm']= rdm
            RDM.__init__(self,**kw)
            self.rdm = spin_to_spatial(self.rdm,
                    alpha=self.alp,
                    beta=self.bet,
                    s2s=self.s2s)

    def reduce_order(self):
        nRDM = SpatialRDM(
                order=self.p-1,
                alpha=self.alp,
                beta=self.bet,
                rdm=None,
                )
        self.expand()
        if self.p==2:
            for i in range(0,self.R):
                for j in range(0,self.R):
                    i1 = tuple([i,j])
                    for x in range(0,self.R):
                        i2 = tuple([i,x,j,x])
                        nRDM.rdm[i1]+=self.rdm[i2]
        nRDM.rdm*=(1/(self.Ne-self.p+1))
        return nRDM

    def get_spin_properties(self):
        if self.p==3:
            pass
        elif self.p==2:
            rdm1 = self.reduce_order()
            self.sz = Sz_spatial(
                    rdm1.rdm,
                    self.alp,
                    self.bet,
                    self.s2s)
            self.s2 = S2_spatial(
                    self.rdm,
                    rdm1.rdm,
                    self.alp,
                    self.bet,
                    self.s2s)
    

    def contract(self):
        size = len(self.rdm.shape)
        if not self.p==1:
            self.rdm = np.reshape(
                    self.rdm,
                    (
                        self.R**self.p,
                        self.R**self.p
                        )
                    )
    def expand(self):
        size = len(self.rdm.shape)
        if not self.p==1:
            self.rdm = np.reshape(
                    self.rdm,
                    (tuple([self.R for i in range(2*self.p)]))
                    )
            
