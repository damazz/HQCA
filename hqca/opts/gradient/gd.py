from hqca.opts.core import *
import numpy as np
from functools import reduce
from copy import deepcopy as copy
from math import pi

class GradientDescent(OptimizerInstance):
    def __init__(self,**kwargs):
        OptimizerInstance.__init__(self,**kwargs)
        OptimizerInstance._gradient_keywords(self,**kwargs)

    def initialize(self,start):
        OptimizerInstance.initialize(self,start)
        if self.verbose:
            print('Initializing the gradient-descent optimization class.')
            print('---------- ' )
        self.f0_x = np.asarray(start)+np.asarray(self.shift)
        self.f0_f = self.f(self.f0_x[:])
        self.energy_calls += 1
        self.g0_f = np.asarray(self.g(self.f0_x[:]))
        self.f1_x = self.f0_x - self.gamma*np.asarray(self.g0_f)
        self.f1_f = self.f(self.f1_x)
        if self.verbose:
            print('Initial f: {:.8f}'.format(self.f0_f))
        self.use = 0
        self.crit=1

    def next_step(self):
        self.s = (self.f1_x-self.f0_x).T
        self.g1_f = np.asarray(self.g(self.f1_x))
        self.y = (self.g1_f-self.g0_f).T
        self.f2_x = self.f1_x - self.gamma*np.asarray(self.g1_f)
        self.f2_f = self.f(self.f2_x)
        self.reassign()

    def reassign(self):
        self.f0_x = self.f1_x[:]
        self.f1_x = self.f2_x[:]
        self.f0_f = self.f1_f.copy()
        self.f1_f = self.f2_f.copy()
        self.g0_f = self.g1_f
        if self._conv_crit=='default':
            self.crit = np.sqrt(np.sum(np.square(self.g0_f)))
        self.best_f = self.f1_f.copy()
        self.best_x = self.f1_x[:]
        self.best_y = self.f1_x[:]

