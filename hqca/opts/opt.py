from hqca.opts.core import *
import traceback
from subprocess import CalledProcessError, check_output
from functools import partial
from numpy import copy
import timeit
import time
import sys
from hqca.opts.simplex import *
from hqca.opts.gradient import *
from hqca.opts.stochastic import *
try:
    from hqca.opts.nevergrad import *
except Exception:
    pass


class Cache:
    def __init__(self):
        self.use=True
        self.err=False
        self.msg=None
        self.iter=0
        self.done=False


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
            optimizer='generic',
            verbose=True,
            **kwargs
            ):
        '''
        Establish optimizer, and take in first parameters
        '''
        kwargs['verbose']=verbose
        self.method = optimizer
        simplex = ['nm','lnm']
        grad = ['gd','bfgs','ls']
        stochastic = ['sgd','gpso']
        if optimizer in simplex:
            methods = {
                'nm':NelderMead,
                'lnm':NelderMeadLagarias,
                }
        elif optimizer in grad:
            methods = {
                'ls':BacktrackingLineSearch,
                'gd':GradientDescent,
                'bfgs':BFGS,
                }
        elif optimizer in stochastic:
            methods = {
                'gpso':GradientParticleSwarmOptimizer,
                'sgd':StochasticGradientDescent,
                }
        elif optimizer in ['nevergrad','ng']:
            #methods = {
            #    'nevergrad':nevergrad_opt,
            #    }
            sys.exit('Nevergrad not implemented yet.')
        else:
            s = '{} not recognized, '.format(str(optimizer))
            s+= 'please select from: \n'
            s+= 'nm lnm ls gd bfgs gpso sgd ng'
            raise OptimizerNotImplemented(s)
        self.opt = methods[optimizer](**kwargs)
        # Selecting optimizers and setting parameters
        self.error = False
        self.verbose = verbose
        self.iter = 0

    def initialize(self,start,**kw):
        try:
            self.opt.initialize(start,**kw)
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
        self.iter+=1
        if self.verbose:
            print('    step: {:0}, f: {:.8f}, crit: {:.8f}'.format(
                self.iter,
                self.opt.best_f,
                self.opt.crit,
                )
                )
        
    def run(self):
        status = Cache()
        while not status.done:
            self.next_step()
            self.check(cache=status)
        if self.verbose:
            print('Final value: {:.8f}'.format(self.opt.best_f))
            print('Parameters: ')
            print(self.opt.best_x)

    def check(self,
            cache=False
            ):
        '''
        check how criteria is 
        '''
        try:
            self.opt.crit
            if not cache:
                if abs(self.opt.crit)<=abs(self.opt._conv_thresh):
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
                if abs(self.opt.crit)<=abs(self.opt._conv_thresh):
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
        #if cache.done and self.method in ['nevergrad','NM-ng']:
        #    from nevergrad.optimization.recaster import _MessagingThread as mt
        #    import threading
        #    for t in threading.enumerate():
        #        if type(t)==type(mt(None)):
        #            t.stop()

