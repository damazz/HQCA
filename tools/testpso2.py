import sys
from nevergrad.optimization import optimizerlib
from math import pi
import numpy as np
from nevergrad.optimization import registry
#for i in sorted(registry.keys()):
#    print(i)
#    opt = registry[i]
#    print(opt(dimension=2,budget=1000))
np.set_printoptions(linewidth=200,suppress=True)
from hqca.tools import Optimizers as opt
import matplotlib.pyplot as plt
import pandas as pd
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation




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
    l = 0.2*np.sqrt(1/2)
    g = []
    t = (2.82843*np.exp(l*ps)*p[0])/ps
    t += np.pi*np.exp(0.5*np.sum(np.cos(2*pi*p)))*np.sin(2*pi*p[0])
    g.append(t)
    t = (2.82843*np.exp(l*ps)*p[1])/ps
    t += np.pi*np.exp(0.5*np.sum(np.cos(2*pi*p)))*np.sin(2*pi*p[1])
    g.append(t)
    return g


'''

    test optimization

'''

Np = 10
t = 0
time = 100
unit = 5
pso_steps = 10
func = rastrigin
grad = grad_rastrigin
new = opt.Optimizer(
        'gpso',
        function=func,
        gradient=grad,
        examples=1,
        inertia=0.6,
        max_velocity=0.5,
        slow_down=True,
        pso_iterations=pso_steps,
        particles=Np,
        unity=unit,
        accel=[1.0,1.0],
        pr_o=1,
        gamma=0.001)
new.initialize([5,-5])
data = np.empty((time,Np,3))
while t<time:
    new.next_step()
    data[t,:,0:2]=new.opt.X[:,:]
    data[t,:,2]=new.opt.F[:]
    t+=1
    print(new.opt.best_x,new.opt.best_f,new.opt.crit,new.opt.vel_crit)


def update_graph(num):
    dat = data[num,:,:]
    graph._offsets3d = (dat[:,0],dat[:,1],dat[:,2])
    title.set_text('test, steps={}'.format(num))

fig = plt.figure()
ax = p3.Axes3D(fig)

ax.set_xlim3d([np.amin(data[:,:,0]),np.amax(data[:,:,0])])
ax.set_xlabel('X')
ax.set_ylim3d([np.amin(data[:,:,1]),np.amax(data[:,:,1])])
ax.set_ylabel('Y')
ax.set_zlim3d([np.amin(data[:,:,2]),np.amax(data[:,:,2])])
ax.set_zlabel('f')
X = np.linspace(np.amin(data[:,:,0]),np.amax(data[:,:,0]),75)
Y = np.linspace(np.amin(data[:,:,1]),np.amax(data[:,:,1]),75)
Z = np.zeros((75,75))
for a,x in enumerate(X):
    for b,y in enumerate(Y):
        Z[a,b] = func([x,y])
        #Z[a,b] = rastrigin([x,y])
X,Y = np.meshgrid(X,Y,indexing='ij')
ax.plot_surface(X,Y,Z,alpha=0.2)
ax.scatter(data[0,:,0],data[0,:,1],data[0,:,2],color=(1,0,0))
graph = ax.scatter(data[0,:,0],data[0,:,1],data[0,:,2])
title = ax.set_title('test')
ani = animation.FuncAnimation(fig,update_graph,time-1,interval=250)
plt.show()


