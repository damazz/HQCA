import numpy as np
from numpy import copy
from functools import reduce
from math import pi
from nevergrad.optimization import optimizerlib,registry
from nevergrad.optimization.utils import Archive,Value
import threading
import nevergrad
from hqca.optimizers.BaseOptimizer import OptimizerInstance

class nevergrad_opt(OptimizerInstance):
    def __init__(self,**kwargs):
        '''
        Need to get the optimizer
        '''
        OptimizerInstance(self,**kwargs)
        Optimizer._gradient_keywords(self,**kwargs)

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
