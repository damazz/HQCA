import numpy as np
from numpy import copy
import sys
from subprocess import CalledProcessError, check_output
import traceback
import timeit
import time
import nevergrad
from nevergrad.optimization import optimizerlib,registry
from nevergrad.optimization.utils import Archive,Value
import threading
from random import random as r
import random
from math import pi
from functools import reduce,partial


class Empty:
    def __init__(self):
        self.opt_done=True

def null_function():
    return 0


class Optimizer:
    '''
    Class object which contains the various optimizers.

    Currently, we have a simple gradient descent program, and the Nelder
    Mead optimization method for simplexes. For use with determinant
    optimization.
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
        self.method=optimizer
        kwargs['pr_o']=pr_o
        # Selecting optimizers and setting parameters
        if self.method=='NM':
            self.opt = nelder_mead(
                    **kwargs)
        elif self.method=='GD':
            self.opt = gradient_descent(
                    **kwargs)
        elif self.method=='qNM':
            self.opt = quasi_nelder_mead(
                    **kwargs)
        elif self.method=='sGD':
            self.opt = stochastic_gradient_descent(
                    **kwargs)
        elif self.method=='nevergrad':
            self.opt = nevergradopt(
                    **kwargs)
        elif self.method=='bfgs':
            self.opt = bfgs(
                    **kwargs)
        elif self.method=='gpso':
            self.opt = gradient_particle_swarm_optimizer(
                    **kwargs)
        elif self.method=='NM-ng':
            self.opt = nelder_mead_ng(
                    **kwargs)
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

#
# Now, begin the various types of optimizers
#

class OptimizerInstance:
    def __init__(self,
            function=None,
            gradient=None,
            particles=None,
            examples=None,   # in some cases, the number of sub iterations
            conv_crit_type='default',
            conv_threshold='default',
            gamma='default',
            pr_o=0,
            unity=pi,  # typically the bounds of the optimization
            func_eval=True,
            shift=None,
            pso_iterations=10, #particle swarm
            inertia=0.7,
            accel=[1,1],
            max_velocity=0.5,
            slow_down=True,
            **kwargs):
        self.f = function
        self.g = gradient
        self.Np = particles
        self.Ne = examples
        self.pr_o = pr_o
        self.shift = shift
        self.pso_iter = pso_iterations
        self.a1,self.a2 = accel[0],accel[1]
        self.v_max = max_velocity
        self.slow_down = slow_down
        self.w = inertia
        self.unity = unity
        self.energy_calls = 0
        self.conv_threshold = conv_threshold
        if conv_threshold=='default':
            self._conv_thresh = 0.00001
        else:
            self._conv_thresh = float(conv_threshold)
        self.conv_crit_type = conv_crit_type
        if gamma=='default':
            self.gamma = 0.001
        else:
            self.gamma = float(gamma)
        self.ef = func_eval
        self.kwargs = kwargs

    def initialize(self,start):
        self.N = len(start)
        if type(self.shift)==type(None):
            self.shift = [0.0]*self.N


class gradient_particle_swarm_optimizer(OptimizerInstance):
    def initialize(self,start):
        self.v_max =self.unity*self.v_max
        OptimizerInstance.initialize(self,start)
        self.X = np.zeros((self.Np,self.N))
        self.V = np.zeros((self.Np,self.N))
        self._random_particle_position()
        self._random_particle_velocity()
        self.P = copy(self.X)
        self.F = np.zeros(self.Np)
        self.Pf= copy(self.F)
        for i in range(self.Np):
            self.F[i] = self.f(self.X[i,:])
        self.best = np.argsort(self.F)
        self.Gx = self.X[self.best[0],:]
        self.Gf = self.F[self.best[0]]
        self.best_x =copy(self.Gx)
        self.best_f =copy(self.Gf)
        self._update_criteria()
        self.pso=copy(self.pso_iter)

    def next_step(self):
        if self.pso>0:
            for i in range(self.Np):
                for d in range(self.N):
                    t1 = (r())*(self.P[i,d]-self.X[i,d])
                    t2 = (r())*(self.Gx[d]-self.X[i,d])
                    self.V[i,d] = self.w*self.V[i,d]+self.a1*t1+self.a2*t2
                    self.V[i,d]= np.sign(self.V[i,d])*min(
                            self.v_max,
                            abs(self.V[i,d])
                            )
                    self.X[i,d] = self.X[i,d]+self.V[i,d]
                self.F[i]=self.f(self.X[i,:])
                if self.F[i]<self.Pf[i]:
                    self.P[i,:]=copy(self.X[i,:])
                    self.Pf[i] = self.F[i]
                if self.F[i]<self.Gf:
                    self.Gx =copy(self.X[i,:])
                    self.Gf = self.F[i]
            self.best = np.argsort(self.F)
            self.pso-=1
        else:
            for i in range(min(self.Np,self.Ne)):
                # doing micro optimization
                self.sub = bfgs(
                        function=self.f,
                        gradient=self.g,
                        unity=self.unity)
                self.sub.initialize(self.X[self.best[i],:])
                self.sub_count = 0
                while self.sub.crit>1e-8:
                    self.sub.next_step()
                    if self.pr_o>1:
                        print('bfgs step: {:02}, crit: {:.8f}, f: {:.8f}'.format(
                            self.sub_count,
                            self.sub.best_f,
                            self.sub.crit
                            ))
                    self.sub_count+=1 
                self.X[i,:]=copy(self.sub.best_x)
                self.F[i]=self.sub.best_f
                if self.F[i]<self.Pf[i]:
                    self.P[i,:]=self.X[i,:]
                    self.Pf[i] = self.F[i]
                if self.F[i]<self.Gf:
                    self.Gx = copy(self.X[i,:])
                    self.Gf = self.F[i]
            self.best = np.argsort(self.F)
            self.pso=copy(self.pso_iter)
            if self.slow_down:
                self.a2 = (1-self.a2)*0.6 +self.a2
                self.a1 = self.a1*(0.6)
                self.w  = self.w*(0.9)
        self.best_x =copy(self.Gx)
        self.best_f =copy(self.Gf)
        self._update_criteria()

    def _update_criteria(self):
        self.vels_crit = np.zeros(self.Np)
        self.poss_crit = np.zeros(self.Np)
        for i in range(self.Np):
            self.vels_crit[i]=(np.sum(np.square(self.V[self.best[i],:])))
            d = self.X[self.best[i],:]-self.Gx[:]
            self.poss_crit[i]=(np.sum(np.square(d)))
        self.pos_crit=np.sqrt(np.average(self.poss_crit))
        self.vel_crit=np.sqrt(np.average(self.vels_crit))
        #print('Speed')
        #print(self.V)
        #print('Eval: ')
        #print(self.F)
        #print('Speed, average')
        #print(self.V)
        #print(self.vels_crit)
        if self.conv_crit_type=='default':
            self.crit=self.pos_crit
        else:
            self.crit=self.pos_crit
        #print('Pos: ',self.pos_crit)
        #print('Vel: ',self.vel_crit)
        if self.pr_o>2:
            print('Position, distance')
            print(self.X)



    def _random_particle_position(self):
        for i in range(1,self.Np):
            temp = np.zeros(self.N)
            for j in range(self.N):
                t = (random.random()*2-1)*self.unity
                temp[j]=t+self.shift[j]
            self.X[i,:] = temp[:]

    def _random_particle_velocity(self):
        for i in range(0,self.Np):
            temp = np.zeros(self.N)
            for j in range(self.N):
                t = (random.random()*2-1)*self.unity*random.random()
                temp[j]=copy(t-self.X[i,j])*0.1
            #self.V[i,:] = temp[:]



class stochastic_gradient_descent:
    def __init__(self,
            function,
            gradient,
            examples,
            conv_crit_type='default',
            conv_threshold='default',
            gamma='default',
            pr_o=0,
            unity=pi,
            func_eval=True,
            shift=None,
            **kwargs):
        self.f = function
        self.g = gradient
        self.Ne = examples
        self.pr_o = pr_o
        self.shift = shift
        self.unity = unity
        self.energy_calls = 0
        if conv_threshold=='default':
            self._conv_thresh = 0.00001
        else:
            self._conv_thresh = float(conv_threshold)
        self.conv_crit_type = conv_crit_type
        if gamma=='default':
            self.gamma = 0.001
        else:
            self.gamma = float(gamma)
        self.lp = self.gamma
        self.ef = func_eval
        self.kwargs = kwargs

    def _random_populate_parameters(self):
        for i in range(0,self.Ne):
            temp = np.zeros(self.Np)
            for j in range(self.Np):
                t = (random.random()*2-1)*self.unity
                temp[j]=t+self.shift[j]
            self.param[i,:] = temp[:]


    def initialize(self,start):
        OptimizerInstance.initialize(start)
        if self.pr_o>0:
            print('Initializing the stochastic gradient-descent optimization class.')
            print('---------- ' )
        self.param = np.zeros((self.Ne,self.N))
        self.data_eval  = np.zeros(self.Ne)
        self._random_populate_parameters()

        self.rand_list = random.sample(range(0,self.Ne),self.Ne)
        for i in self.rand_list:
            self.data_grad = np.asarray(self.g(self.param[i,:]))
            for j in range(self.Ne):
                self.param[j,:]=self.param[j,:]-self.data_grad*self.lp
        if self.ef:
            for i in range(self.Ne):
                self.data_eval[i]=self.f(self.param[i,:])
            self.reassign()


    def next_step(self):
        self.rand_list = random.sample(range(0,self.Ne),self.Ne)
        for i in self.rand_list:
            self.data_grad[:] = self.g(self.param[i,:])
            for j in range(self.Ne):
                self.param[j,:]=self.param[j,:]-self.data_grad*self.lp
        if self.ef:
            for i in range(self.Ne):
                self.data_eval[i]=self.f(self.param[i,:])
            self.reassign()

    def reassign(self):
        best = np.argsort(self.data_eval)
        self.best_x = self.param[best[0],:]
        self.best_y = self.param[best[0],:]
        self.best_f = (1/self.Ne)*np.sum(self.data_eval)

class gradient_descent(OptimizerInstance):
    def initialize(self,start):
        OptimizerInstance.initialize(self,start)
        if self.pr_o>0:
            print('Initializing the gradient-descent optimization class.')
            print('---------- ' )
        self.f0_x = np.asarray(start)+np.asarray(self.shift)
        self.f0_f = self.f(self.f0_x[:])
        self.energy_calls += 1
        self.g0_f = np.asarray(self.g(self.f0_x[:]))
        self.f1_x = self.f0_x - self.gamma*np.asarray(self.g0_f)*0.1
        self.f1_f = self.f(self.f1_x)
        if self.pr_o>0:
            print('Step:-01, Init. Energy: {:.8f} Hartrees'.format(self.f0_f))
        self.use = 0
        self.crit=1

    def next_step(self):
        self.s = (self.f1_x-self.f0_x).T
        self.g1_f = np.asarray(self.g(self.f1_x))
        self.y = (self.g1_f-self.g0_f).T
        #gam = np.dot(self.s.T,self.y)/np.dot(self.y.T,self.y)
        #gam = np.dot(self.s.T,self.s)/np.dot(self.s.T,self.y)
        '''
        if self.use:
            gam = gam2
            self.count+=1
            if self.count==200:
                self.use=0
                print('Switch to step based')
        else:
            gam = gam1
            self.count-=2
            if self.count==0:
                self.use=1
                print('Switch to gradient based')
            elif self.crit<0.000001:
                self.use=1
                self.count=0
                print('Switch to gradient based')
        '''
        self.f2_x = self.f1_x - self.gamma*np.asarray(self.g1_f)
        self.f2_f = self.f(self.f2_x)
        self.reassign()

    def reassign(self):
        self.f0_x = self.f1_x[:]
        self.f1_x = self.f2_x[:]
        self.f0_f = self.f1_f.copy()
        self.f1_f = self.f2_f.copy()
        self.g0_f = self.g1_f
        if self.conv_crit_type=='default':
            self.crit = np.sqrt(np.sum(np.square(self.g0_f)))
        self.best_f = self.f1_f.copy()
        self.best_x = self.f1_x[:]
        self.best_y = self.f1_x[:]

class nelder_mead_ng(OptimizerInstance):
    '''
    Nelder mead with a twist! But really. Uses Nelder mead across a wide range
    and then will switch to cobyla or some better quadratic optimizer from
    nevergrad
    '''
    def __init__(self,
            conv_threshold=None,
            conv_crit_type=None,
            switch_thresh=0.17,
            **kwargs):
        OptimizerInstance.__init__(self,
            conv_threshold=conv_threshold,
            conv_crit_type=conv_crit_type,
            **kwargs)
        self.simplex_scale=self.unity
        self.opt_macro = nelder_mead(
            conv_threshold=switch_thresh,
            conv_crit_type=conv_crit_type,
            **kwargs)
        if self.conv_threshold=='default':
            if self.conv_crit_type=='default':
                self._conv_thresh=0.001
            elif self.conv_crit_type=='energy':
                self._conv_thresh=0.0001
        else:
            self._conv_thresh=float(self.conv_threshold)
        self.macro = True
        self.micro = False
        self.kwargs = kwargs
        self.kwargs['conv_threshold']=conv_threshold
        self.kwargs['conv_crit_type']=conv_crit_type
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


class nelder_mead(OptimizerInstance):
    '''
    Nelder-Mead Optimizer! Uses the general dimension simplex method, so should
    be appropriate for arbitrary system size.
    '''
    def __init__(self,**kwargs):
        OptimizerInstance.__init__(self,**kwargs)
        self.simplex_scale=self.unity
        if self.conv_threshold=='default':
            if self.conv_crit_type=='default':
                self._conv_thresh=0.001
            elif self.conv_crit_type=='energy':
                self._conv_thresh=0.0001
        else:
            self._conv_thresh=float(self.conv_threshold)

    def initialize(self,start):
        OptimizerInstance.initialize(self,start)
        self.simp_x = np.zeros((self.N+1,self.N)) # column is dim coord - row is each point
        self.simp_f = np.zeros(self.N+1) # there are N para, and N+1 points in simplex
        self.simp_x[0,:] = start[:]
        for i in range(1,self.N+1):
            self.simp_x[i,:]=start[:]
            self.simp_x[i,i-1]+=1*self.simplex_scale
        for i in range(0,self.N+1):
            # here, we assign f - usually this will be energy
            self.simp_f[i] = self.f(self.simp_x[i,:])
            self.energy_calls+=1
        if self.pr_o>0:
            print('Step:-01, E:{:.8f} Hartrees'.format(self.simp_f[0]))
        self.order_points()
        self.calc_centroid()
        self.reassign()
        self.stuck = np.zeros((3,self.N))
        self.stuck_ind = 0
        # and now we have our ordered simplex! Begin!!

    def next_step(self):
        '''
        Carries out the next step to generate a new simplex. Each step contains
        various energy evaluations, so rarely will only be one evaluation.
        '''
        self.R_x = self.M_x+self.M_x-self.W_x
        if self.stuck_ind==0:
            self.stuck_ind = 1
            self.stuck[0,:]= self.R_x
        elif self.stuck_ind==1:
            self.stuck_ind = 2
            self.stuck[1,:]= self.R_x
        elif self.stuck_ind==2:
            self.stuck_ind=0
            self.stuck[2,:]= self.R_x
        self.N_stuck=0
        def check_stuck(self):
            v1 = self.stuck[0,:]
            v2 = self.stuck[1,:]
            v3 = self.stuck[2,:]
            diff = np.sqrt(np.sum(np.square(v1-v3)))
            if diff<1e-10:
                self.R_x = self.M_x+r()*(self.M_x-self.W_x)
                if self.pr_o>0:
                    print('Was stuck!')
                self.N_stuck+=1 
        check_stuck(self)
        self.R_f = self.f(self.R_x)
        self.energy_calls+=1
        if self.pr_o>1:
            print('NM: Reflection: {}'.format(self.R_x))
            print(self.R_f)

        if self.R_f<=self.X_f:
            if self.R_f>self.B_f: #reflected point not better than best
                if self.pr_o>1:
                    print('NM: Reflected point is soso.')
                self.simp_x[-1,:]=self.R_x
                self.simp_f[-1]  =self.R_f
            else: # reflected points is best or better
                self.E_x = self.R_x + self.R_x - self.M_x
                self.E_f = self.f(self.E_x)
                self.energy_calls+=1
                if self.E_f<self.B_f:
                    if self.pr_o>1:
                        print('NM: Extended point better than best.')
                        print(self.E_x)
                    self.simp_x[-1,:]=self.E_x
                    self.simp_f[-1]  =self.E_f
                else:
                    if self.pr_o>1:
                        print('NM: Reflected point better than best.')
                        print(self.R_x)
                    self.simp_x[-1,:]=self.R_x
                    self.simp_f[-1]  =self.R_f
        else:
            self.Cwm_x = self.W_x+0.5*(self.M_x-self.W_x)
            self.Crm_x = self.M_x+0.5*(self.R_x-self.M_x)
            self.Cwm_f = self.f(self.Cwm_x)
            self.Crm_f = self.f(self.Crm_x)
            self.energy_calls+=2
            if self.Crm_f<=self.Cwm_f:
                self.C_f = self.Crm_f
                self.C_x = self.Crm_x
            else:
                self.C_f = self.Cwm_f
                self.C_x = self.Cwm_x
            if self.C_f<self.W_f:
                if self.pr_o>1:
                    print('NM: Contracting the triangle.')
                    print(self.C_x)
                self.simp_x[-1,:]=self.C_x
                self.simp_f[-1]  =self.C_f
            else:
                for i in range(1,self.N+1):
                    self.simp_x[i,:]=self.B_x+0.5*(self.simp_x[i,:]-self.B_x)
                    self.simp_f[i]=self.f(self.simp_x[i,:])
                    self.energy_calls+=1
                if self.pr_o>1:
                    print('NM: Had to shrink..')
                    print(self.simp_x)

        self.order_points()
        self.calc_centroid()
        self.reassign()
        self.std_dev()
        self.best_f = self.B_f
        self.best_x = self.B_x
        self.best_y = self.B_x
        if self.conv_crit_type=='default':
            self.crit = self.sd_x
        elif self.conv_crit_type=='energy':
            self.crit = self.sd_f
        else:
            self.crit = self.sd_x
        if self.pr_o>1:
            print('Maximum distance from centroid: {}'.format(self.max))

    def reassign(self):
        self.W_x = self.simp_x[-1,:] #worst point
        self.W_f = self.simp_f[-1]
        self.B_x = self.simp_x[0,:] # best point
        self.B_f = self.simp_f[0]
        self.X_x = self.simp_x[-2,:] # second worst
        self.X_f = self.simp_f[-2]

    def std_dev(self):
        self.sd_f = np.std(self.simp_f)
        temp = np.zeros(self.N+1)
        for i in range(0,self.N+1):
            temp[i]=np.sqrt(np.sum(np.square(self.simp_x[i,:])))
        self.sd_x = np.std(temp)

    def calc_centroid(self):
        self.M_x = np.zeros(self.N)
        for i in range(0,self.N): #NOTE, we omit N+1, i.e. the worst point
            self.M_x = self.M_x + self.simp_x[i,:]
        self.M_x = self.M_x*(1/(self.N))
        self.max = 0
        for i in range(0,self.N+1):
            temp = np.sqrt(np.sum(np.square(self.simp_x[i,:]-self.M_x)))
            self.max = max(self.max,temp)

    def order_points(self):
        ind = np.arange(0,self.N+1)
        for i in range(0,self.N+1):
            temp = ind.copy()
            low_x = i
            low_f = self.simp_f[temp[i]]
            for j in range(0,self.N+1):
                if i<j:
                    if low_f>self.simp_f[ind[j]]: #then, swap
                        low_f = self.simp_f[j]
                        low_x= j
            temp[i] = ind[low_x] #swap the lowest
            temp[low_x] = ind[i]
            ind = temp.copy()
        new_f = np.zeros(self.N+1)
        new_x = np.zeros((self.N+1,self.N))
        for i in range(0,self.N+1):
            new_f[i] = self.simp_f[ind[i]]
            new_x[i,:] = self.simp_x[ind[i],:]
        self.simp_x = new_x
        self.simp_f = new_f

class bfgs(OptimizerInstance):

    def initialize(self,start):
        OptimizerInstance.initialize(self,start)
        # find approximate hessian
        self.x0 = np.asmatrix(start) # row vec? 
        self.g0 = np.asmatrix(self.g(np.asarray(self.x0)[0,:])) # row  vec


        #self.B0 = np.dot(self.g0.T,self.g0) #
        #print(self.B0)
        #self.B0i = np.linalg.inv(self.B0)
        self.B0 = np.identity(self.N)
        self.B0i = np.identity(self.N)
        self.p0 = -1*np.dot(self.B0i,self.g0.T).T # row vec
        self._line_search()
        self.s0 = self.p0*self.alp
        self.x1 = self.x0+self.s0
        self.y0 = np.asmatrix(self.g(np.asarray(self.x1)[0,:]))-self.g0
        Bn =  np.dot(self.y0.T,self.y0)
        Bd = (np.dot(self.y0,self.s0.T)[0,0])
        if abs(Bd)<1e-30:
            if Bd<0:
                Bd = -1e-30
            else:
                Bd = 1e-30
        Ba = Bn*(1/Bd)
        S = np.dot(self.s0.T,self.s0)
        d = reduce(np.dot, (self.s0,self.B0,self.s0.T))[0,0]
        if abs(d)<1e-30:
            if d<0:
                d = -1e-30
            else:
                d = 1e-30
        Bb = reduce(np.dot, (self.B0,S,self.B0.T))*(1/d)
        self.B1 = self.B0 + Ba - Bb
        syT = reduce(np.dot, (self.s0.T,self.y0))
        yTs = reduce(np.dot, (self.y0,self.s0.T))[0,0]
        if abs(yTs)<1e-30:
            if yTs<0:
                yTs = -1e-30
            else:
                yTs = 1e-30
        ysT = reduce(np.dot, (self.y0.T,self.s0))
        L = np.identity(self.N)-syT*(1/yTs)
        R = np.identity(self.N)-ysT*(1/yTs)
        self.B1i  = reduce(np.dot, (L,self.B0i,R))+S*(1/yTs)
        # reassign 
        self.x0 = self.x1.copy()
        self.g0 = np.asmatrix(self.g(np.asarray(self.x0)[0,:]))
        self.B0 = self.B1.copy()
        self.B0i = self.B1i.copy()
        self.best_x = self.x0.copy()
        self.best_f = self.f(np.asarray(self.x0)[0,:])
        if self.conv_crit_type=='default':
            self.crit = np.sqrt(np.sum(np.square(self.g0)))
        else:
            self.crit = np.sqrt(np.sum(np.square(self.g0)))
        self.stuck = np.zeros((3,self.N))
        self.stuck_ind = 0

    def next_step(self):
        if self.stuck_ind==0:
            self.stuck_ind = 1
            self.stuck[0,:]= self.x0
        elif self.stuck_ind==1:
            self.stuck_ind = 2
            self.stuck[1,:]= self.x0
        elif self.stuck_ind==2:
            self.stuck_ind=0
            self.stuck[2,:]= self.x0
        self.N_stuck=0
        def check_stuck(self):
            v1 = self.stuck[0,:]
            v2 = self.stuck[1,:]
            v3 = self.stuck[2,:]
            d13 = np.sqrt(np.sum(np.square(v1-v3)))
            if d13<1e-15:
                shrink = 0.5
                self.x0 = self.x0-(1-shrink)*self.s0
                if self.pr_o>0:
                    print('Was stuck!')
                self.N_stuck+=1
        check_stuck(self)
        self.p0 = -1*np.dot(self.B0i,self.g0.T).T # row vec
        # now, line search
        self._line_search()
        self.s0 = self.p0*self.alp
        self.x1 = self.x0+self.s0
        self.y0 = np.asmatrix(self.g(np.asarray(self.x1)[0,:]))-self.g0
        B_num = np.dot(self.y0.T,self.y0)
        B_den = (np.dot(self.y0,self.s0.T)[0,0])
        if abs(B_den)<1e-30:
            if B_den<0:
                B_den = -1e-30
            else:
                B_den = 1e-30
        Ba =  B_num*(1/B_den)
        S = np.dot(self.s0.T,self.s0)
        d = reduce(np.dot, (self.s0,self.B0,self.s0.T))[0,0]
        if abs(d)<=1e-30:
            if d<0:
                d = -1e-30
            else:
                d = 1e-30
        Bb = reduce(np.dot, (self.B0,S,self.B0.T))*(1/d)
        self.B1 = self.B0 + Ba - Bb
        syT = reduce(np.dot, (self.s0.T,self.y0))
        yTs = reduce(np.dot, (self.y0,self.s0.T))[0,0]
        if abs(yTs)<1e-30:
            if yTs<0:
                yTs = -1e-30
            else:
                yTs = 1e-30
        ysT = reduce(np.dot, (self.y0.T,self.s0))
        L = np.identity(self.N)-syT*(1/yTs)
        R = np.identity(self.N)-ysT*(1/yTs)
        self.B1i  = reduce(np.dot, (L,self.B0i,R))+S*(1/yTs)
        # reassign 
        self.x0 = copy(self.x1)
        self.g0 = np.asmatrix(self.g(np.asarray(self.x0)[0,:]))
        self.B0 = copy(self.B1)
        self.B0i = copy(self.B1i)
        self.best_x = copy(self.x0)
        self.best_f = self.f(np.asarray(self.x0)[0,:])
        if self.conv_crit_type=='default':
            self.crit = np.sqrt(np.sum(np.square(self.g0)))




    def _line_search(self):
        '''
        uses self.p0, and some others stuff
        '''
        bound = False
        f_l = self.f(np.asarray(self.x0)[0,:])
        a_l = 0
        a_r = 1
        while not bound:
            temp = self.x0+self.p0*a_r
            f_r = self.f(np.asarray(temp)[0,:])
            if f_r<f_l:
                a_l = copy(a_r)
                f_l = copy(f_r)
                a_r*=2
            else:
                bound=True
        while a_r-a_l-0.01>0:
            a_tmp = 0.5*(a_l+a_r)
            temp = self.x0+self.p0*a_tmp
            f_tmp = self.f(np.asarray(temp)[0,:])
            if f_tmp<f_l:
                a_l = copy(a_tmp)
                f_l = copy(f_tmp)
            else:
                a_r = copy(a_tmp)
                f_r = copy(f_tmp)
        self.alp = 0.5*(a_l+a_r)



class nevergradopt:
    def __init__(self,
            function,
            nevergrad_opt='Cobyla',
            conv_threshold='default',
            conv_crit_type='default',
            pr_o=0,
            max_iter=100,
            N_vectors=5,
            shift=None,
            use_radians=False,
            unity=pi, #always give in radians!
            **kwargs
            ):
        '''
        Need to get the optimizer
        '''
        self.kw = kwargs
        self.max_iter = max_iter
        self.f = function
        self.opt_name = nevergrad_opt
        self.energy_calls=0
        self.pr_o = pr_o
        self.Nv = N_vectors
        if conv_threshold=='default':
            self._conv_thresh = 0.00001
        else:
            self._conv_thresh = conv_threshold
        self.vectors = []
        self.opt_crit=conv_crit_type
        self.shift = shift
        self.use_radians=use_radians
        self.unity = unity
        print('###########')
        print(unity,shift)
        print('###########')
        if not self.use_radians:
            self.unity = self.unity*(180/pi)


    def check(self,initial=False):
        if self.opt_crit in ['iterations']:
            if self.energy_calls>=self.max_iter:
                self.crit=0
            else:
                self.crit=1
        elif self.opt_crit=='ImpAv':
            pass
        elif self.opt_crit in ['default','MaxDist']:
            if initial:
                self.vectors.sort(key=lambda x:x[0],reverse=False)
                self._update_MaxDist()
            else:
                dist = 0 
                for i in range(len(self.vectors[0][2])):
                    dist+=(self.vectors[0][2][i]-self.y[i])**2
                dist = dist**(1/2)
                comp2 = self.E<self.vectors[ 0][0]
                if not comp2:
                    for i in reversed(range(1,self.Nv)):
                        comp1 =dist<=self.vectors[i][3]
                        comp2 =dist>self.vectors[i-1][3]
                        if comp1 and comp2:
                            self.vectors.insert(
                                    i,
                                    [
                                        self.E,
                                        self.x,
                                        self.y.copy(),
                                        dist]
                                    )
                            del self.vectors[self.Nv]
                            break
                elif comp2:
                    self.vectors.insert(
                            0,
                            [
                                self.E,
                                self.x,
                                self.y.copy(),
                                0])
                    del self.vectors[self.Nv]
                self._update_MaxDist()
            self.best_f = self.vectors[0][0]
            self.best_x = self.vectors[0][1]
            self.best_y = self.vectors[0][2]
            self.crit = self.max_d

    def _update_MaxDist(self):
        self.max_d=0
        for n,v in enumerate(self.vectors):
            if n==0:
                self.vectors[0][3]=0
            else:
                dist = 0
                for i in range(len(self.vectors[0][2])):
                    dist+=(self.vectors[0][2][i]-v[2][i])**2
                dist = dist**(1/2)
                v[3]=dist
                if dist>=self.max_d:
                    self.max_d = dist
                    self.max_n = n

    def initialize(self,start):
        self.Np = len(start)
        self.temp_dat = []
        if type(self.shift)==type(None):
            self.shift = start
        self.opt = registry[self.opt_name](
                len(start),
                budget=self.max_iter
                )
        for i in range(0,self.Nv):
            x = self.opt.ask()
            y = np.asarray(x.args)[0]*self.unity+self.shift
            E = self.f(y)
            self.temp_dat.append([x.args,E])
            self.energy_calls+=1
            self.vectors.append(
                [
                    E,
                    x,
                    y,
                    0])
            self.opt.tell(x,E)
        self.x = x
        self.y = y
        self.E = E
        self.check(initial=True)


    def next_step(self):
        self.x = self.opt.ask()
        self.y = np.asarray(self.x.args)[0]*self.unity+self.shift
        self.E = self.f(self.y)
        self.opt.tell(self.x,self.E)
        self.temp_dat.append([self.x.args,self.E])
        self.check()
        self.energy_calls+=1 


    def save_opt(self):
        '''
        little function to try and convert an object and see if it will save
        properly with pickle.
        '''
        del self.opt,self.x

    def reload_opt(self):
        '''
        function to reload data from the temp_dat object 
        '''
        self.opt = registry[self.opt_name](
                self.Np,
                budget=self.max_iter
                )
        #try:
        #    for step in self.temp_dat:
        #        for i in range(step[1].count):
        #            x = self.opt.ask()
        #            print(x,step[0],step[1].mean)
        #            self.opt.tell(x,step[1].mean)
        #except Exception as e:
        #    traceback.print_exc()
        okay = True
        try:
            for item in self.temp_dat:
                x = self.opt.ask()
                print(x.args,item[0])
                self.opt.tell(x,item[1])
        except KeyError:
            print('huh')
            okay=False
            it+=1 


