from hqca.opts.core import *
import numpy as np
from functools import reduce
from copy import deepcopy as copy
from math import pi


class BacktrackingLineSearch(OptimizerInstance):
    '''
    Optimizer that attempts contracts the step size.
    
    See Nocedal & Wright, algorithm 3.1
    '''
    def __init__(self,
            *args,
            rho=0.75,
            c=0.5,
            alpha_bar=1,
            **kwargs
            ):
        '''
        Evaluate function and gradient values
        '''
        OptimizerInstance.__init__(self,*args,**kwargs)
        OptimizerInstance._gradient_keywords(self,**kwargs)
        # set x_0 
        self._rho = rho
        self._c = c
        self._alpha_bar =1
        self._conv_thresh = 0.0

    def initialize(self,start,p_k,f0=None,g0=None):
        def para(xs):
            # row vector to 
            return xs.tolist()[0]
        OptimizerInstance.initialize(self,start) #set potential shift, self.N
        #
        self.x0 = start
        print(start,p_k)
        if type(f0)==type(None):
            self.f0 = self.f(para(self.x0))
        else:
            self.f0 = f0
        if type(g0)==type(None):
            self.g0 = self.g(para(self.x0))
        else:
            self.g0 = g0
        self.p = p_k
        self.y = np.dot(self.g0,self.p.T)
        p = self.x0+self._alpha_bar * np.asarray(p_k)
        self.f1 = self.f(para(p))
        if self.f1 <= self.f0 + self._c*self._alpha_bar*self.y:
            self.crit=0 #
        else:
            self.crit =1 
            self._alpha_bar*= self._rho #shrink alpha by factor of rho

    def next_step(self):
        def para(xs):
            # row vector to 
            return xs.tolist()[0]
        p = self.x0+self._alpha_bar * np.asarray(self.p)
        self.f1 = self.f(para(p))
        if self.f1 <= self.f0 + self._c*self._alpha_bar*self.y:
            self.crit=0 #
        else:
            self.crit =1 
            self._alpha_bar*= self._rho #shrink alpha by factor of rho


class BisectingLineSearch(OptimizerInstance):
    def __init__(self,initial_left_bound=None,**kwargs):
        OptimizerInstance.__init__(self,**kwargs)
        OptimizerInstance._gradient_keywords(self,**kwargs)
        if type(initial_left_bound)==type(None):
            self.f_l = self.f([0])
        else:
            self.f_l = initial_left_bound

    def initialize(self,start):
        OptimizerInstance.initialize(self,start)
        # find approximate hessian
        self.x0 = np.asarray(start)
        # going to set left and right bounds
        f_l = copy(self.f_l)
        try:
            a_l = (0+self.shift)*self.unity
            a_r = (1+self.shift)*self.unity
        except Exception:
            a_l = 0
            a_r = 1
        print(a_l,a_r)
        bound=False
        while not bound:
            temp = self.x0*a_r
            f_r = self.f(temp)
            if f_r<f_l:
                a_l = copy(a_r)
                f_l = copy(f_r)
                a_r*=2
            else:
                bound=True
        self.a_r = a_r
        self.a_l = a_l
        self.f_r = f_r
        self.f_l = f_l

    def next_step(self):
        a_tmp = 0.5*(self.a_l+self.a_r)
        f_tmp = self.f(a_tmp*self.x0)
        if f_tmp<self.f_l:
            self.a_l = copy(a_tmp)
            self.f_l = copy(f_tmp)
        else:
            self.a_r = copy(a_tmp)
            self.f_r = copy(f_tmp)
        if self.f_l<self.f_r:
            self.best_f = self.f_l
            self.best_x = self.a_l
        else:
            self.best_f = self.f_r
            self.best_x = self.a_r
        self.crit = abs((self.a_l-self.a_r))


