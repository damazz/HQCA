import numpy as np
from functools import reduce
from copy import deepcopy as copy
from math import pi
from hqca.optimizers.BaseOptimizer import OptimizerInstance

class gradient_descent(OptimizerInstance):
    def __init__(self,**kwargs):
        OptimizerInstance.__init__(self,**kwargs)
        OptimizerInstance._gradient_keywords(self,**kwargs)

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
        if self._conv_crit=='default':
            self.crit = np.sqrt(np.sum(np.square(self.g0_f)))
        self.best_f = self.f1_f.copy()
        self.best_x = self.f1_x[:]
        self.best_y = self.f1_x[:]

class bfgs(OptimizerInstance):
    def __init__(self,**kwargs):
        OptimizerInstance.__init__(self,**kwargs)
        OptimizerInstance._gradient_keywords(self,**kwargs)

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
        if self._conv_crit=='default':
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
        if self._conv_crit=='default':
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


class line_search(OptimizerInstance):
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
        a_l = 0
        a_r = 1
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


