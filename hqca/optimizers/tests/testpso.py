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

Np = 5
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
        pso_iterations=2,
        particles=Np,
        unity=5,
        accel=[0.1,0.1],
        pr_o=1,
        gamma=0.001)
new.initialize([5,-5])
it = 0
data =[]
length = 20
for i in range(Np):
    data.append(np.empty((3,length)))
#while abs(new.opt.crit)>1e-10 and it<length:
while it<length:
    new.next_step()
    for i in range(Np):
        print(new.opt.X[i,:])
        data[i][0:2,it]=new.opt.X[i,:]
        data[i][2,it]=new.opt.F[i]

    it+=1
    #if i%100==0:
    #print('f:',new.opt.best_f,'g:',new.opt.crit,'x:',new.opt.best_x)
tmin = np.amin(data[0],axis=1)
tmax = np.amax(data[0],axis=1)
print(data[0])
for i in range(1,Np):
    print(data[i])
    minx = np.amin(data[i],axis=1)
    maxx = np.amax(data[i],axis=1)
    for j in range(3):
        if minx[j]<tmin[j]:
            tmin[j]=minx[j]
        if maxx[j]>tmax[j]:
            tmax[j]=maxx[j]


fig = plt.figure()
ax = p3.Axes3D(fig)

lines = [ax.plot(dat[0,0:1],dat[1,0:1],dat[2,0:1])[0] for dat in data]

def update_lines(num,dataLines,lines):
    for line,data in zip(lines,dataLines):
        line.set_data(data[0:2,:num])
        line.set_3d_properties(data[2,:num])
    return lines
ax.set_xlim3d([tmin[0],tmax[0]])
ax.set_xlabel('X')
ax.set_ylim3d([tmin[1],tmax[1]])
ax.set_ylabel('Y')
ax.set_zlim3d([tmin[2],tmax[2]])
ax.set_zlabel('f')
X = np.linspace(tmin[0],tmax[0],75)
Y = np.linspace(tmin[1],tmax[1],75)
Z = np.zeros((75,75))
for a,x in enumerate(X):
    for b,y in enumerate(Y):
        #Z[a,b] = rosenbrock([x,y])
        Z[a,b] = rastrigin([x,y])

X,Y = np.meshgrid(X,Y,indexing='ij')
ax.plot_surface(X,Y,Z,alpha=0.2)
line_ani = animation.FuncAnimation(fig,update_lines,length,fargs=(data,lines),
        interval=750,blit=False)
#plt.show()
t = np.arange(0,length)
df = pd.DataFrame({'time':t,'x':

def update_graph(num):
    graph.__offset3d(data.x,data.y,data.z)

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


