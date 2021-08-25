import numpy as np
from hqca.opts.core import *

class StochasticGradientDescent(OptimizerInstance):
    def __init__(self,
            **kwargs):
        self.f = function
        self.g = gradient
        self.Ne = examples
        self.pr_o = pr_o
        self.shift = shift
        self.unity = unity
        self.energy_calls = 0
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





