from hqca.opts.core import *
import numpy as np
from functools import reduce
from copy import deepcopy as copy
from math import pi
from hqca.opts.gradient.linesearch import BacktrackingLineSearch

def para(xs):
    # row vector to 
    return xs.tolist()[0]

class BFGS(OptimizerInstance):
    '''
    See Nocedal & Wright, chapter 6, for more information on the 
    implementation of the BFGS algorithm.
    '''
    def __init__(self,**kwargs):
        OptimizerInstance.__init__(self,**kwargs)
        OptimizerInstance._gradient_keywords(self,**kwargs)

    def initialize(self,start):
        OptimizerInstance.initialize(self,start)
        # find approximate hessian
        self.x0 = np.asmatrix(start) # row vec?
        #self.g0 = np.asmatrix(self.g(np.asarray(self.x0)[0,:])) # row  vec
        self.g0 = np.asmatrix(self.g(para(self.x0)))
        if self.verbose:
            print('Step: -01 ')
            print('G0: ',self.g0)

        # set initial Hessian and inverse to identity
        self.B0 = np.identity(self.N)
        self.B0i = np.identity(self.N)
        self.p0 = -1*np.dot(self.B0i,self.g0.T).T # row vec
        # set initial search direction
        if self.verbose:
            print('Starting line search...')
        self._line_search()
        if self.verbose:
            print('LS: ',self.f_avg)
        self.s0 = self.p0*self.alp
        self.x1 = self.x0+self.s0
        self.g1 = np.asmatrix(self.g(para(self.x1)))
        self.y0 = self.g1 - self.g0 
        #self.y0 = np.asmatrix(self.g(para(self.x1)))-self.g0
        if self.verbose:
            print('Y: ',self.y0)
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
        #self.g0 = np.asmatrix(self.g(np.asarray(self.x0)[0,:]))
        self.g0 = self.g1.copy()
        if self.verbose:
            print('G: ',self.g0)
        self.B0 = self.B1.copy()
        self.B0i = self.B1i.copy()
        self.best_x = self.x0.copy()
        self.best_f = self.f(np.asarray(self.x0)[0,:])
        #
        self.crit = np.linalg.norm(self.g0)
        #
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
                if self.verbose:
                    print('Was stuck!')
                self.N_stuck+=1
        check_stuck(self)
        self.p0 = -1*np.dot(self.B0i,self.g0.T).T # row vec
        # now, line search
        self._line_search()
        if self.verbose:
            print('LS: ',self.f_avg)
        self.s0 = self.p0*self.alp
        self.x1 = self.x0+self.s0
        self.y0 = np.asmatrix(self.g(np.asarray(self.x1)[0,:]))-self.g0
        if self.verbose:
            print('Y: ',self.y0)
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
        if self.verbose:
            print('G: ',self.g0)
        self.B0 = copy(self.B1)
        self.B0i = copy(self.B1i)
        self.best_x = copy(self.x0)
        self.best_f = self.f(np.asarray(self.x0)[0,:])
        if self._conv_crit=='default':
            self.crit = np.sqrt(np.sum(np.square(self.g0)))

    def _line_search(self):
        '''
        algorithm 3.5,3.6 from Nocedal & Wright
        attempting to find alpha that satisfies Wolfe conditions
        '''
        self.f_evals = 0
        self.g_evals = 0
        c1,c2 = 0.6,0.9  #0 < c1 <  c2 < 1
        try:
            f_zed = self.best_f
        except AttributeError as e:
            f_zed = self.f(para(self.x0))
            self.f_evals+=1 

        p = self.p0
        x = self.x0
        g_zed = np.dot(self.g0,p.T)[0,0]

        def phi(alpha):
            self.f_evals +=1 
            a = para(x+alpha*p)
            return self.f(a)

        def dphi(alpha):
            a = para(x+alpha*p)
            self.g_evals +=1 
            return np.dot(self.g(a),p.T)[0,0]

        def zoom(alp_l,alp_h,f_l,f_h):
            # biset low and high
            done = False
            iters = 0
            while not done:
                #print(alp_l,alp_h)
                alp_j = 0.5*(alp_l+alp_h)
                f_j = phi(alp_j)
                if f_j > f_zed + c1*alp_j*g_zed or f_j >= f_l:
                    alp_h = alp_j
                    f_h = f_j
                else:
                    gj = dphi(alp_j)
                    if abs(gj)<= -c2* g_zed:
                        done = True
                        alp_star = alp_j
                    if gj*(alp_h-alp_l)>=0:
                        alp_h = alp_l
                    alp_l = alp_j
                    f_l = copy(f_j)
                iters+=1
                if iters>20:
                    done = True
                    raise OptimizerError
            return alp_star,f_j

        alp_0,alp_max = 0,5
        alp_1 = 1
        done = False
        iters = 1
        f0 = copy(f_zed) #actual alp=0, not alp=alp_0
        g0 = copy(g_zed) #same
        while not done:
            f1 = phi(alp_1)
            if f1>f_zed+c1*alp_1*g_zed or (f1>= f0 and iters>1):
                alp_star,f_star = zoom(alp_0,alp_1,f0,f1)
                done = True
                continue
            g1 = dphi(alp_1)
            if abs(g1)<= -c2*g_zed:
                alp_star = alp_1
                f_star = f1
                done = True
                continue
            if g1>= 0 :
                alp_star,f_star = zoom(alp_0,alp_1,f0,f1)
                done = True
                continue
            alp_0 = copy(alp_1)
            f0 = copy(f1)
            alp_1 = 0.5*(alp_1+alp_max) #bisect 
        
        self.alp = alp_star
        self.f_avg = f_star
        if self.verbose:
            print('f_calls = ({}),g_calls = ({}),alpha = {}'.format(self.f_evals,self.g_evals,alp_star))

    
    def _line_search_backtracking(self):
        '''
        uses backtracking linesearch
        '''
        f_evals = 0
        try:
            f = self.best_f
        except AttributeError as e:
            f = self.f(para(self.x0))
            f_evals+=1 

        c,rho,alpha = 0.5,0.75,1
        temp = self.x0+alpha*self.p0
        f1 = self.f(para(temp))
        f_evals+=1

        y = np.dot(self.g0,self.p0.T)[0,0]
        while not f1<= f+c*alpha*y:
            alpha*= rho
            temp = self.x0 + alpha*self.p0
            f1  = self.f(para(temp))
            f_evals+=1
        if self.verbose:
            print('f_calls = ({}),alpha = {}'.format(f_evals,alpha))
        self.alp = alpha
        self.f_avg = f1

    def _line_search_old(self):

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
        self.f_avg = 0.5*(f_l+f_r)
