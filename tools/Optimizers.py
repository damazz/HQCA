if __name__=='__main__':
    pass
else:
    #from hqca.tools.IBM_check import check, get_backend_object
    #from hqca.tools.QuantumFramework import wait_for_machine
    pass
import numpy as np
import sys
from subprocess import CalledProcessError, check_output
import traceback
import timeit
import time
import nevergrad
from nevergrad.optimization import optimizerlib,registry
from random import random as r
from functools import reduce,partial

#
#
# Various functions
#
#

class NotAvailableError(Exception):
    '''
    Means what it says. 
    '''

class Empty:
    def __init__(self):
        self.opt_done=True

def function_call(
        par,
        function=None,
        **kwargs
        ):
    tic = timeit.default_timer()
    E_t = function(par)
    toc = timeit.default_timer()
    if toc-tic > 1800:
        print('Really long run time. Not good.')
    return E_t

def gradient_call(
        par,energy='orbitals',
        **kwargs
        ):
    if energy=='orbitals':
        ddE = eno.orbital_energy_gradient_givens(
            par,**kwargs)
    elif energy=='test':
        ddE = g_x(par,**kwargs)
    elif energy=='qc':
        sys.exit('No analytical gradients on the quantum computer!')
    return ddE

#
#
# Main Optimizer class
#
# 

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
            function,
            pr_o=1,
            **kwargs
            ):
        '''
        Establish optimizer, and take in first parameters
        '''
        self.method=optimizer
        kwargs['pr_o']=pr_o
        kwargs['function']=function
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
        elif self.method=='nevergrad':
            self.opt = nevergradopt(
                    **kwargs)
        self.error = False
        self.pr_o = pr_o
        #
        # Error management for the IBM case
        #

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

#
#
# Now, begin the various types of optimizers
#
#

class gradient_descent:
    def __init__(self,
            n_par,
            conv_threshold='default',
            gamma='default',
            gradient='numerical',
            grad_dist=0.01,
            conv_crit_type='default',
            pr_o=0,
            **kwargs
            ):
        '''
        Note, this is called by the Optimizer class. Then, the optimizer will
        also call the initialize class.
        '''
        self.N = n_par
        self.energy_calls = 0
        if conv_threshold=='default':
            self._conv_thresh = 0.00001
        else:
            self._conv_thresh = float(conv_threshold)
        self.conv_crit_type = conv_crit_type
        self.grad=gradient
        if grad_dist=='default':
            self.dist=0.0001
        else:
            self.dist=float(grad_dist)
        self.gamma = gamma
        self.kwargs = kwargs

    def initialize(self,start):
        if self.pr_o>0:
            print('Initializing the gradient-descent optimization class.')
            print('---------- ' )
        if self.gamma=='default':
            gam = 0.00001
        self.f0_x = np.asarray(start[:])
        self.f0_f = self.f(self.f0_x)
        self.energy_calls += 1
        if self.grad=='numerical':
            self.g0_f = np.zeros(self.N)
            for i in range(0,self.N):
                temp = np.zeros(self.N)
                temp[i]=self.dist
                self.g0_f[i] = (
                        self.f(
                            self.f0_x+temp
                            )
                        -self.f(
                            self.f0_x-temp
                            )
                    )/(self.dist*2)
                self.energy_calls+= self.N
        elif self.grad=='analytical':
            self.g0_f = np.asarray(gradient_call(
                    self.f0_x,
                    **self.kwargs))
        self.energy_calls+= 1
        self.f1_x = self.f0_x - gam*np.asarray(self.g0_f)
        self.f1_f = self.f(self.f1_x)
        if self.pr_o>0:
            print('Step:-01, Init. Energy: {:.8f} Hartrees'.format(self.f0_f))
        self.use=1
        self.count=0
        self.use = 0
        self.count=50
        self.crit=1

    def numerical(self):
        self.s = (self.f1_x-self.f0_x).T
        self.g1_f = np.zeros(self.N)
        dist = min(self.dist,np.dot(self.s.T,self.s))
        for i in range(0,self.N):
            temp = np.zeros(self.N)
            temp[i]=dist
            self.g1_f[i] =(
                    self.f(
                        self.f1_x+temp
                        )
                    -self.f(
                        self.f1_x-temp
                        )
                )/(dist*2)
        self.energy_calls+= self.N*2

    def analytical(self):
        self.s = (self.f1_x-self.f0_x).T
        self.g1_f = np.asarray(gradient_call(
                self.f1_x,
                **self.kwargs))
        self.energy_calls+= 1


    def next_step(self):
        if self.grad=='numerical':
            self.numerical()
        elif self.grad=='analytical':
            self.analytical()

        self.y = (self.g1_f-self.g0_f).T

        if self.gamma=='default':
            gam2 = np.dot(self.s.T,self.y)/np.dot(self.y.T,self.y)
            gam1 = np.dot(self.s.T,self.s)/np.dot(self.s.T,self.y)
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
            self.f2_x = self.f1_x - gam*np.asarray(self.g1_f)
            self.f2_f = function_call(self.f2_x,**self.kwargs)
            self.reassign()



    def reassign(self):
        self.f0_x = self.f1_x[:]
        self.f1_x = self.f2_x[:]
        self.f0_f = self.f1_f
        self.f1_f = self.f2_f
        self.g0_f = self.g1_f[:]
        if self.conv_crit_type=='default':
            self.crit = np.sqrt(np.sum(np.square(self.g0_f)))
        self.best_f = self.f1_f
        self.best_x = self.f1_x


class nelder_mead:
    '''
    Nelder-Mead Optimizer! Uses the general dimension simplex method, so should
    be appropriate for arbitrary system size.
    '''
    def __init__(self,
            function,
            conv_threshold='default',
            conv_crit_type='default',
            simplex_scale=10,
            energy='classical',
            pr_o=0,
            **kwargs
            ):
        '''
        Begin the optmizer. Set parameters, etc.
        '''
        self.f = function
        if simplex_scale=='default':
            self.simplex_scale=5
        else:
            self.simplex_scale=simplex_scale
        kwargs['energy']=energy
        self.pr_o = pr_o
        self.conv_crit_type = conv_crit_type
        if conv_threshold=='default':
            if energy=='classical':
                if self.conv_crit_type=='default':
                    self._conv_thresh=0.001
                elif self.conv_crit_type=='energy':
                    self._conv_thresh=0.0001
            elif energy=='qc':
                if self.conv_crit_type=='default':
                    self._conv_thresh=0.5
                elif self.conv_crit_type=='energy':
                    self._conv_thresh=0.001
        else:
            self._conv_thresh=float(conv_threshold)
        self.energy_calls=0
        if self.pr_o>0:
            print('Initializing the Nelder-Mead optimization class.')
            print('---------- ' )
        self.kwargs =  kwargs

    def initialize(self,start):
        self.N = len(start)
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
            print('Step:-01, Init. Energy: {:.8f} Hartrees'.format(self.simp_f[0]))
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
        if self.conv_crit_type=='default':
            self.crit = self.sd_x
        elif self.conv_crit_type=='energy':
            self.crit = self.sd_f

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

class nevergradopt:
    def __init__(self,
            function,
            nevergrad_opt='Cobyla',
            conv_threshold='default',
            conv_crit_type='default',
            pr_o=0,
            max_iter=100,
            N_vectors=5,
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

    def check(self,initial=False):
        if self.opt_crit in ['default','iterations']:
            if self.energy_calls>=self.max_iter:
                self.crit=0
            else:
                self.crit=1
        elif self.opt_crit=='ImpAv':
            pass
        elif self.opt_crit=='MaxDist':
            self.vectors.sort(key=lambda x:x[0],reverse=False)
            if initial:
                pass
            else:
                comp1 = self.E<self.vectors[-1][0]
                comp2 = self.E<self.vectors[ 0][0]
                if comp1 and not comp2:
                    for i in reversed(range(1,self.Nv)):
                        comp1 =self.E<=self.vectors[i][0]
                        comp2 =self.E>self.vectors[i-1][0]
                        if comp1 and comp2:
                            self.vectors.insert(
                                    i,
                                    [
                                        self.E,
                                        self.x.copy()
                                        ,0]
                                    )
                            del self.vectors[self.Nv]
                            break
                elif comp2:
                    self.vectors.insert(
                            0,
                            [self.E,self.x.copy(),0])
                    del self.vectors[self.Nv]
            self.best_f = self.vectors[0][0]
            self.best_x = self.vectors[0][1]
            self._update_MaxDist()
            self.crit = self.max_d

    def _update_MaxDist(self):
        self.max_d=0
        for n,v in enumerate(self.vectors):
            if n==0:
                self.vectors[0][2]=0
            else:
                dist = 0 
                for i in range(len(self.vectors[0][1])):
                    dist+=(self.vectors[0][1][i]-v[1][i])**2
                dist = dist**(1/2)
                v[2]=dist
                if dist>=self.max_d:
                    self.max_d = dist
                    self.max_n = n

    def initialize(self,start):
        self.Np = len(start)
        self.opt = registry[self.opt_name](
                dimension=self.Np,
                budget=self.max_iter
                )
        for i in range(0,self.Nv):
            x = self.opt.ask()
            E = self.f(x)
            self.energy_calls+=1
            self.vectors.append(
                [
                    E,
                    x.copy(),
                    0])
            self.opt.tell(x,E)
        self.x = x.copy()
        self.E = E
        self.check(initial=True)


    def next_step(self):
        self.x = self.opt.ask()
        self.E = self.f(self.x)
        self.opt.tell(self.x,self.E)
        self.check()
        self.energy_calls+=1 





#
# test functions for optimizers: 
# very simple algebraic ones
#

def f_x(par,**kwargs):
    x = par[0]
    y = par[1]
    return x**4-3*(x**3)+2+y**2

def g_x(par,**kwargs):
    x = par[0]
    y = par[1]
    return [4*x**3 - 9*x**2,2*y]

#
#
#

'''
keys = {'energy':'test'}
keys['gradient']='analytical'
a = Optimizer('GD',[[6,1]],False,**keys)
#print(a.opt.simp_x)
iters = 0
while (a.opt_done==False and (iters<250)):
    a.next_step(**keys)
    print('Values after step {}: {}'.format(iters,a.opt.g0_f))
    a.check()
    iters+=1
'''
