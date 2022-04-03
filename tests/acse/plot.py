import matplotlib.pyplot as plt

import numpy as np

fig = plt.figure()
ax = fig.add_subplot()

t1_f = np.array([
    -1.28979351, 
    -1.25131050,
    -1.23096995,
    -1.22059149,
    -1.21535893,
    -1.21273293,
    -1.21141763])

t1_r = np.array([
    -1.12463845, 
    -1.16742945,
    -1.18886512,
    -1.19951850,
    -1.20481986,
    -1.20746307,
    -1.20878266])

t2_f = np.array([
        -1.414482412,
        -1.354231937,
        -1.289793512,
        -1.251310502,
        -1.230969951,
        -1.220591493,
        -1.215358932,
        ])

t2_r = np.array([
        -0.935349282,
        -1.044093014,
        -1.124638455,
        -1.167429449,
        -1.188865118,
        -1.199518496,
        -1.204819859,
        ])

epsilon = np.array([2**-i for i in range(1,8)])
print(epsilon)
f0 = -1.21010086545325
t1_g = -0.168638894363347
t2_g = -0.674555577453386

hess1 = 0.023490906

hess2 = 0.375877857

g1_f = np.divide((t1_f - f0),epsilon)
g1_r = np.divide((-t1_r + f0),epsilon)
g2_f = np.divide((t2_f - f0),epsilon)
g2_r = np.divide((-t2_r + f0),epsilon)
h1   = np.divide(g1_f-g1_r,2*epsilon)
h2   = np.divide(g2_f-g2_r,2*epsilon)

e1g = g1_f - t1_g
e1h = h1 - hess1

e2g = g2_f - t2_g
e2h = h2 - hess2

ax.scatter(np.log2(epsilon),np.log2(e1g),color='r',marker='x',label='t1g')
ax.scatter(np.log2(epsilon),np.log2(e1h),color='r',marker='o',label='t1h')
ax.scatter(np.log2(epsilon),np.log2(e2g),color='b',marker='x',label='t2g')
ax.scatter(np.log2(epsilon),np.log2(e2h),color='b',marker='o',label='t2h')
ax.plot([-i for i in range(10)],[-i for i in range(10)],color='k')


ax.legend()

plt.show()

