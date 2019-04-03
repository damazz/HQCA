import sys
from nevergrad.optimization import optimizerlib
from math import pi
import numpy as np
from nevergrad.optimization import registry
#for i in sorted(registry.keys()):
#    print(i)
#    opt = registry[i]
#    print(opt(dimension=2,budget=1000))
from hqca.tools import Optimizers as opt
np.set_printoptions(suppress=True,precision=4)




def beale2(p):
    x,y = p[0],p[1]
    a = (1.5- x + x*y)**2
    a+= (2.25- x + x*(y**2))**2
    a+= (2.625 - x + x*(y**3))**2
    return a

def grad(p):
    x,y =p[0],p[1]
    a = (2*x)*(y**6+y**4-2*y**3-y**2-2*y+3)
    a+= 5.25*y**3 + 4.5*y**2+3*y-12.75
    b = 6*x*x*(y**5+(2/3)*y**3-y**2-(1/3)*y-(1/3))
    b+= 6*x*(2.625*y**2+1.5*y+0.5)
    return [a,b]

def beale3(p):
    x,y,z = p[0],p[1],p[2]
    a = (1.5- x + x*y+z)**2
    a+= (2.25- x + x*(y**2)+z)**2
    a+= (2.625 - x + x*(y**3)+z)**2
    return a

def rosenbrock(p):
    n = len(p)
    y = 0
    for i in range(0,n-1):
        y+= 100*(p[i+1]-p[i]**2)**2+(1-p[i])**2
    return y

def grad_rosen(p):
    n = len(p)
    g = []
    t = 2*(200*p[0]**3-200*p[0]*p[1]+p[0]-1)
    g.append(t)
    for v in range(1,n-1):
        t = 2*(200*p[v]**3-200*p[v]*p[v+1]+p[v]-1)
        t+= 200*(p[v]-p[v-1]**2)
        g.append(t)
    t= 200*(p[-1]-p[-2]**2)
    g.append(t)
    return g

def rastrigin(p):
    d = len(p)
    f = 10*d
    for i in range(d):
        f+= p[i]**2-10*np.cos(2*np.pi*p[i])
    return f

def grad_rastrigin(p):
    d = len(p)
    g = []
    for i in range(d):
        g.append(2*p[i]+20*np.pi*np.sin(2*np.pi*p[i]))
    return g


def ackley(p,a=20,b=0.2,c=2*pi):
    p = np.asarray(p)
    f = -a*np.exp(-b*np.sqrt((1/2)*np.sum(np.square(p))))
    f+= -np.exp((1/2)*np.sum(np.cos(c*p)))
    f+= a + np.exp(1)
    return f

def grad_ackley(p):
    p  = np.asarray(p)
    ps = np.sqrt(np.sum(np.square(p)))
    print(ps)
    l = 0.2*np.sqrt(1/2)
    g = []
    t = (2.82843*np.exp(l*ps)*p[0])/ps
    t += np.pi*np.exp(0.5*np.sum(np.cos(2*pi*p)))*np.sin(2*pi*p[0])
    g.append(t)
    t = (2.82843*np.exp(l*ps)*p[1])/ps
    t += np.pi*np.exp(0.5*np.sum(np.cos(2*pi*p)))*np.sin(2*pi*p[1])
    g.append(t)
    return g


new = opt.Optimizer(
        #'sGD',
        'gpso',
        #'bfgs',
        #'GD',
        #function=ackley,
        #function=rosenbrock,
        #gradient=grad_rosen,
        #gradient=grad_ackley,
        function=rastrigin,
        gradient=grad_rastrigin,
        examples=3,
        particles=5,
        unity=5.12,
        pr_o=1,
        pso_iterations=1,
        gamma=0.001)
new.initialize([5,-5])
itera=0
while abs(new.opt.crit)>1e-10 and itera<50:
    new.next_step()
    itera+=1 
    #if i%100==0:
    print('f:',new.opt.best_f,'g:',new.opt.crit,'x:',new.opt.best_x)



#a = registry['OnePlusOne'](dimension=2,budget=500)
#for i in range(a.budget):
#    x = a.ask()
#    a.tell(x, beale2(x))
#    if i%50==0:
#        print(x,beale2(x))

class test:
    def __init__(self,d):
        if d==2:
            self.f = beale2
        elif d==3:
            self.f = beale3
        self.opt = registry['OnePlusOne'](dimension=d,budget=500)

    def go(self):
        for i in range(50):
            x = self.opt.ask()
            self.opt.tell(x,self.f(x))
            '''
            try:
                self.opt.tell(x,self.f(x))
            except IndexError:
                print('Error, here is the guess: ')
                print(x)
                print('Dimension should be '.format(self.x))
                break
            except Exception as e:
                print(e)
            '''
            if i%25==0:
                print(i,x,self.f(x))


