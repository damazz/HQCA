from hqca.opts.core import *
from hqca.opts.simplex.neldermead import *
import numpy as np
from math import pi
from random import random as r


class NelderMeadNeverGrad(OptimizerInstance):
    '''
    Nelder mead with a twist! But really. Uses Nelder mead across a wide range
    and then will switch to cobyla or some better quadratic optimizer from
    nevergrad

    Probably doesnt work to be honest-  will check later
    '''
    def __init__(self,
            **kwargs):
        OptimizerInstance.__init__(self,**kwargs)
        OptimizerInstance._simplex_keywords(self,**kwargs)
        self.opt_macro = nelder_mead(
            conv_threshold=self.switch_thresh,
            **kwargs)
        if self.conv_threshold=='default':
            if self._conv_crit=='default':
                self._conv_thresh=0.001
            elif self._conv_crit=='energy':
                self._conv_thresh=0.0001
        else:
            self._conv_thresh=float(self.conv_threshold)
        self.macro = True
        self.micro = False
        self.kwargs = kwargs
        self.switch = switch_thresh

    def initialize(self,start):
        self.opt_macro.initialize(start)

    def next_step(self):
        if self.macro:
            self.opt_macro.next_step()
            self.best_x = self.opt_macro.best_x
            self.best_y = self.opt_macro.best_y
            self.best_f = self.opt_macro.best_f
            self.crit = self.opt_macro.crit
            self._check()
        elif self.micro:
            self.opt_micro.next_step()
            self.crit = self.opt_micro.crit
            self.best_x = self.opt_micro.best_x
            self.best_y = self.opt_micro.best_y
            self.best_f = self.opt_micro.best_f

    def _check(self):
        if self.macro:
            if self.opt_macro.crit<=self.switch:
                print('Switching to nevergrad optimization.')
                self.macro,self.micro = False,True
                self.kwargs['unity']=self.opt_macro.crit
                self.kwargs['shift']=self.opt_macro.best_x
                self.opt_micro = nevergradopt(
                        **self.kwargs)
                self.opt_micro.initialize(start=self.opt_macro.best_x)
                self.crit = self.opt_micro.crit
                self.best_x = self.opt_micro.best_x
                self.best_y = self.opt_micro.best_y
                self.best_f = self.opt_micro.best_f

