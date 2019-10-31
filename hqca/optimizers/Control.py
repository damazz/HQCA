from hqca.optimizers.Simplex import GeneralNelderMead,nelder_mead_lagarias
from hqca.optimizers.Simplex import adaptive_nelder_mead,nelder_mead_ng
from hqca.optimizers.GradientMethods import bfgs,gradient_descent,line_search
import traceback
try:
    from hqca.optimizers.NevergradOpt import nevergrad_opt
except ImportError:
    traceback.print_exc()
from hqca.optimizers.Stochastic import gradient_particle_swarm_optimizer
from hqca.optimizers.Stochastic import stochastic_gradient_descent 
from subprocess import CalledProcessError, check_output
from functools import partial
from numpy import copy
import timeit
import time
import sys

class Optimizer:
    '''
    Class optimizer which has optimizer run settings. Currently, available
    commands are:
        __init__
        initialize
        next_step
        check

    Optimizer class initializes an optimizers, and allows for utilization and
    control of the optimizer object. 
    '''
    def __init__(
            self,
            optimizer,
            pr_o=1,
            **kwargs
            ):
        '''
        Establish optimizer, and take in first parameters
        '''
        kwargs['pr_o']=pr_o
        self.method = optimizer
        methods = {
                'NM':GeneralNelderMead,
                'ls':line_search,
                'GD':gradient_descent,
                'sGD':stochastic_gradient_descent,
                'bfgs':bfgs,
                'gpso':gradient_particle_swarm_optimizer,
                'NM-ng':nelder_mead_ng,
                'gNM':GeneralNelderMead,
                'lNM':nelder_mead_lagarias,
                'aNMs':adaptive_nelder_mead,
                'sgo':stochastic_gradient_descent,
                'nevergrad':nevergrad_opt,
                }
        self.opt = methods[optimizer](**kwargs)
        # Selecting optimizers and setting parameters
        self.error = False
        self.pr_o = pr_o

    def initialize(self,start):
        try:
            self.opt.initialize(start)
            self.opt_done = False
        except CalledProcessError as e:
            traceback.print_exc()
            if b'IBMQXTimeOutError' in e.output:
                print('Timeout Error:')
                print(e.output)
                self.error = 'time'
            else:
                self.error = True
            self.opt_done=True
        except Exception as e:
            # Any other errors
            traceback.print_exc()
            self.error = True
            self.opt_done=True

    def next_step(self):
        '''
        Call for optimizer to take the next step, should involves
        appropriate number of calculations based on the optimizer method
        '''
        try:
            self.opt.next_step()
        except CalledProcessError as e:
            if b'IBMQXTimeOutError' in e.output:
                print('Timeout Error:')
                print(e.output)
                self.error = 'time'
            else:
                self.error = True
        except Exception as e:
            # Any other errors
            traceback.print_exc()
            self.error = True
            self.opt_done=True

    def check(self,
            cache=False
            ):
        '''
        check how criteria is 
        '''
        try:
            self.opt.crit
            if not cache:
                if self.opt.crit<=self.opt._conv_thresh:
                    self.opt_done=True
                    if self.pr_o>0:
                        print('Criteria met for convergence.')
                        print('----------')
                elif self.error in [True,'time']:
                    self.opt_done=True
                else:
                    self.opt_done=False
            else:
                cache.crit = self.opt.best_f
                if self.opt.crit<=self.opt._conv_thresh:
                    cache.done=True
                    if self.pr_o>0:
                        print('Criteria met for convergence.')
                        print('----------')
                elif self.error in [True,'time']:
                    cache.done=True
                    cache.err=True
                else:
                    cache.done=False
        except AttributeError as e:
            if self.error and not cache:
                self.opt_done=True
            elif self.error:
                cache.done=True
                cache.err=True
        except Exception as e:
            traceback.print_exc()
        if cache.done and self.method in ['nevergrad','NM-ng']:
            from nevergrad.optimization.recaster import _MessagingThread as mt
            import threading
            for t in threading.enumerate():
                if type(t)==type(mt(None)):
                    t.stop()
