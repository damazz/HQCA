import numpy as np
from math import pi
from random import random as r
from hqca.optimizers.BaseOptimizer import OptimizerInstance

class GeneralNelderMead(OptimizerInstance):
    '''
    Nelder-Mead Optimizer! Uses the general dimension simplex method, so should
    be appropriate for arbitrary system size.
    '''
    def __init__(self,**kwargs):
        OptimizerInstance.__init__(self,**kwargs)
        OptimizerInstance._simplex_keywords(self,**kwargs)
        self.set_parameters()

    def set_parameters(self,
            alpha=1,
            beta=2,
            gamma=0.5,
            delta=0.5):
        self.alpha=alpha
        self.beta = beta
        self.gamma= gamma
        self.delta= delta

    def initialize(self,start):
        '''
        Columns in simp_x are dimensions of problem, row has each point

        Note there are N+1 points in the simplex, and dimension N

        We generate simplex according to initial method, then evaluate function
        at each point

        '''
        OptimizerInstance.initialize(self,start)
        self.simp_x = np.zeros((self.N+1,self.N))
        self.simp_f = np.zeros(self.N+1)
        if self.initial=='old':
            for i in range(1,self.N+1):
                print(start,self.shift)
                self.simp_x[i,:]=start[:]+self.shift
                self.simp_x[i,i-1]+=0.99*self.unity
            self.simp_x[0,:] = start[:]+self.shift
        elif self.initial=='han':
            ch = min(max(max(start),self.unity*0.99),10)
            for i in range(1,self.N+1):
                self.simp_x[i,:]=start[:]+self.shift
                self.simp_x[i,i-1]+=ch
            t = np.ones(self.N)*ch*(1-np.sqrt(self.N+1))/self.N
            self.simp_x[0,:] = start[:]+t+self.shift
            print(t)
        elif self.initial=='varadhan':
            cs = max(np.sqrt(np.sum(np.square(start))),1)
            b1 = (cs/(self.N*np.sqrt(2)))*(np.sqrt(self.N+1)+self.N-1)
            b2 = (cs/(self.N*np.sqrt(2)))*(np.sqrt(self.N+1)-1)
        print(self.simp_x)
        for i in range(0,self.N+1):
            self.simp_f[i] = self.f(self.simp_x[i,:])
            self.energy_calls+=1
        if self.pr_o>0:
            print('Step:-01, F:{:.8f} '.format(self.simp_f[0]))
        self.order_points()
        self.calc_centroid()
        self.reassign()
        self.stuck = np.zeros((3,self.N))
        self.stuck_ind = 0

    def _check_stuck(self):
        v1 = self.stuck[0,:]
        v2 = self.stuck[1,:]
        v3 = self.stuck[2,:]
        diff = np.sqrt(np.sum(np.square(v1-v3)))
        if diff<1e-10:
            self.R_x = self.M_x+r()*(self.M_x-self.W_x)
            if self.pr_o>0:
                print('Was stuck!')
            self.N_stuck+=1 


    def _reflect(self):
        self.R_x = self.M_x+self.alpha*(self.M_x-self.W_x)
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
        self._check_stuck()
        self.R_f = self.f(self.R_x)
        self.energy_calls+=1
        if self.pr_o>1:
            print('NM: Reflection: {}'.format(self.R_x))
            print(self.R_f)

    def _update(self,target):
        if target=='reflect':
            if self.pr_o>1:
                print('NM: Reflected point is soso.')
            # replace worst point
            self.simp_x[-1,:]=self.R_x
            self.simp_f[-1]  =self.R_f
        elif target=='extend':
            if self.pr_o>1:
                print('NM: Extended point better than best.')
                print(self.E_x)
            self.simp_x[-1,:]=self.E_x
            self.simp_f[-1]  =self.E_f
        elif target=='contract':
            if self.pr_o>1:
                print('NM: Contracting the triangle.')
                print(self.C_x)
            self.simp_x[-1,:]=self.C_x
            self.simp_f[-1]  =self.C_f

    def _extend(self):
        self.E_x = self.R_x + self.beta*(self.R_x - self.M_x)
        self.E_f = self.f(self.E_x)
        self.energy_calls+=1

    def _contract(self):
        self.Cwm_x = self.W_x+self.gamma*(self.M_x-self.W_x)
        self.Crm_x = self.M_x+self.gamma*(self.R_x-self.M_x)
        self.Cwm_f = self.f(self.Cwm_x)
        self.Crm_f = self.f(self.Crm_x)
        self.energy_calls+=2
        if self.Crm_f<=self.Cwm_f:
            self.C_f = self.Crm_f
            self.C_x = self.Crm_x
        else:
            self.C_f = self.Cwm_f
            self.C_x = self.Cwm_x

    def _shrink(self):
        for i in range(1,self.N+1):
            self.simp_x[i,:]=self.B_x+self.delta*(self.simp_x[i,:]-self.B_x)
            self.simp_f[i]=self.f(self.simp_x[i,:])
            self.energy_calls+=1
        if self.pr_o>1:
            print('NM: Had to shrink..')
            print(self.simp_x)


    def next_step(self):
        '''
        Carries out the next step to generate a new simplex. Each step contains
        various energy evaluations, so rarely will only be one evaluation.

        W_x is worst point
        B_x is best point
        X_x is the second-worst point
        M_x is the centroid
        
        '''
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
        else: #reflected point worsed
            self._contract()
            if self.C_f<self.W_f:
                self._update('contract')
            else:
                self._shrink()
        self.clean_up()

    def clean_up(self):
        self.order_points()
        self.calc_centroid()
        self.reassign()
        self.check_criteria()

    def check_criteria(self):
        self.best_f = self.B_f
        self.best_x = self.B_x
        self.best_y = self.B_x
        self.sd_f = np.std(self.simp_f)
        temp = np.zeros(self.N+1)
        for i in range(0,self.N+1):
            temp[i]=np.sqrt(np.sum(np.square(self.simp_x[i,:])))
        self.sd_x = np.std(temp)
        if self._conv_crit=='default':
            self.crit = self.sd_x
        elif self._conv_crit=='energy':
            self.crit = self.sd_f
        else:
            self.crit = self.sd_x
        if self.pr_o>1:
            print('Maximum distance from centroid: {}'.format(self.max))

    def reassign(self):
        '''
        W_x is worst point
        B_x is best point
        X_x is the second-worst point
        M_x is the centroid
        '''
        self.W_x = self.simp_x[-1,:] #worst point
        self.W_f = self.simp_f[-1]
        self.B_x = self.simp_x[0,:] # best point
        self.B_f = self.simp_f[0]
        self.X_x = self.simp_x[-2,:] # second worst
        self.X_f = self.simp_f[-2]

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

class nelder_mead_lagarias(GeneralNelderMead):
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


class adaptive_nelder_mead(nelder_mead_lagarias):
    def initialize(self,start):
        nelder_mead_lagarias.initialize(self,start)
        self.adaptive_parameters()

    def adaptive_parameters(self):
        self.alpha = 1
        self.beta = 1+2/self.N
        self.gamma= 0.75 - 0.5/self.N
        self.delta = 1 - 1/self.N

class nelder_mead_ng(OptimizerInstance):
    '''
    Nelder mead with a twist! But really. Uses Nelder mead across a wide range
    and then will switch to cobyla or some better quadratic optimizer from
    nevergrad
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

