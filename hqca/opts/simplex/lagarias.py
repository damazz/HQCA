from hqca.opts.core import *
from hqca.opts.simplex.neldermead import *
import numpy as np
from math import pi
from random import random as r


class NelderMeadLagarias(NelderMead):
    def _contract_inner(self):
        self.C_x = self.M_x+self.gamma*(self.R_x-self.M_x)
        self.C_f = self.f(self.C_x)
        self.energy_calls+=1

    def _contract_outer(self):
        self.C_x = self.M_x-self.gamma*(self.R_x-self.M_x)
        self.C_f = self.f(self.C_x)
        self.energy_calls+=1
    
    def next_step(self):
        self._reflect()
        if self.R_f<=self.X_f: #note this is second worst
            if self.R_f>self.B_f: #reflected point not better than best
                self._update('reflect')
            else: # reflected points is best or better, so we extend it
                self._extend()
                if self.E_f<self.B_f:
                    self._update('extend')
                else:
                    if self.pr_o>1:
                        print('NM: Reflected point better than best.')
                        print(self.R_x)
                    self.simp_x[-1,:]=self.R_x
                    self.simp_f[-1]  =self.R_f
        else: #reflected point worse than second
            if self.R_f>self.W_f: #r worse than worst
                self._contract_inner()
                if self.C_f<self.W_f:
                    self._update('contract')
                else:
                    self._shrink()
            else: # between worst, second worst
                self._contract_outer()
                if self.C_f<self.R_f:
                    self._update('contract')
                else:
                    self._shrink()
        self.clean_up()


class AdaptiveNelderMead(NelderMeadLagarias):
    def initialize(self,start):
        nelder_mead_lagarias.initialize(self,start)
        self.adaptive_parameters()

    def adaptive_parameters(self):
        self.alpha = 1
        self.beta = 1+2/self.N
        self.gamma= 0.75 - 0.5/self.N
        self.delta = 1 - 1/self.N
