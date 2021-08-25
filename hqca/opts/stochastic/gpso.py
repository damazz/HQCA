from hqca.opts.core import *
import numpy as np


class GradientParticleSwarmOptimizer(OptimizerInstance):
    def __init__(self,**kwargs):
        OptimizerInstance.__init__(self,**kwargs)
        OptimizerInstance._swarm_keywords(self,**kwargs)

    def initialize(self,start):
        OptimizerInstance.initialize(self,start)
        self.v_max =self.unity*self.v_max
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
        if self._conv_crit=='default':
            self.crit=self.pos_crit
        else:
            self.crit=self.pos_crit
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

