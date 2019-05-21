import numpy as np
from numpy import copy
import sys
import traceback
import timeit
import time
from random import random as r
import random
from math import pi
from functools import reduce,partial


class Empty:
    def __init__(self):
        self.opt_done=True

def null_function():
    return 0


#
# Now, begin the various types of optimizers
#

class OptimizerInstance:
    '''
    Base OptimizerInstance for use by all Optimizers. Contains a default
    __init__ type command, that can be run for different types of optimizers.

    Please note unity is considered as the standard deviation, so that unity/2
    would be the distance extended in two directions. For instance, unity=2
    would map to the region [0,1] -> [0,pi]
    '''
    def __init__(self,
            function,
            gradient=None,
            pr_o=0,
            unity=2*pi,  # typically the bounds of the optimization
            conv_criteria='default',
            conv_threshold='default',
            shift=None,
            **kwargs):
        self.f = function
        self.g = gradient
        self.kw = kwargs
        self.pr_o = pr_o
        self.shift = shift 
        self.unity = unity
        self.energy_calls = 0
        self.conv_threshold = conv_threshold
        self._conv_crit = conv_criteria
        self._conv_thresh = conv_threshold

    def _simplex_keywords(self,
            initial_conditions='old',
            **kwargs,
            ):
        self.initial = initial_conditions


    def _gradient_keywords(self,
            gamma='default',
            **kwargs,
            ):
        if gamma=='default':
            self.gamma = 0.001
        else:
            self.gamma = float(gamma)

    def _swarm_keywords(self,
            max_velocity=0.5,
            slow_down=True,
            inertia=0.7,
            pso_iterations=10,
            particles=None,
            examples=None,
            func_eval=None,
            accel=[1,1],
            **kwargs
            ):
        self.v_max = max_velocity
        self.slow_down = slow_down
        self.Np = particles
        self.Ne = examples
        self.pso_iter = pso_iterations
        self.w = inertia
        self.a1,self.a2 = accel[0],accel[1]

    def _nevergrad_keywords(self,
            max_iter=100,
            nevergrad_opt='Cobyla',
            N_vectors=5,
            **kwargs):
        self.energy_calls=0
        self.vectors = []
        self.opt_name = nevergrad_opt
        self.max_iter = max_iter
        self.Nv = N_vectors


    def initialize(self,start):
        self.N = len(start)
        if type(self.shift)==type(None):
            self.shift = np.asarray([0.0]*self.N)
        else:
            self.shift = np.asarray(self.shift)

    def _check_MaxDist(self):
        pass

