import numpy as np
import sys
import pickle


N = 50


try:
    name = sys.argv[1]
except:
    name = 'e04_3p.out'
with open(name,'rb') as fp:
    data = pickle.load(fp)

N_data = len(data)
datagpc = np.zeros((N_data,3))
for i in range(0,N_data):
    datagpc[i,:]=data[i]['exp-ON'][3:]
    datagpc[i,:]=datagpc[i,::-1]
datagpc.tolist()



def distance(x,y,z,point_set,alpha):
    point = np.array([x,y,z])
    od = 0
    for val in point_set:
        val = np.asarray(val)
        nd = np.exp(-alpha*np.sqrt(np.sum(np.square(point-val))))
        if nd>od:
            od=nd
    return od

field = np.zeros((N,N,N))
alpha=25
axe = np.linspace(0.5,1,N)
i,j,k = 0,0,0
for x in axe:
    j = 0
    for y in axe:
        k = 0
        for z in axe:
            field[i,j,k] = distance(x,y,z,datagpc,alpha)
            k+=1
        j+=1
    i+=1 
    print('{}% Done!'.format(i*100/N))

print(field)

np.save('{}_n{}_a{}'.format(name.split('.')[0],N,alpha),field)
            
            


