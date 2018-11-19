import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# plot the orbitals 
from .gpc import hold_data

np.set_printoptions(suppress=True,precision=5)
#ON = np.loadtxt(input('Input: '))
#ONc = ON.copy()
size = hold_data.shape
hold = np.zeros(size)

N = size[0]
# shape in other 
for i in range(0,N):
    hold_data.sort(axis=1)
    hold[i,0:]=hold_data[i,::-1]
    print(hold_data[i,:])
#print(hold)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = hold[:,0]
ys = hold[:,1]
zs = hold[:,2]
ax.zaxis.set_rotate_label(False) 
ax.xaxis.set_rotate_label(False) 
ax.yaxis.set_rotate_label(False) 
ax.set_xlabel('$\lambda$ 1',rotation='horizontal') 
ax.set_ylabel('$\lambda$ 2',rotation='horizontal')
ax.set_zlabel('$\lambda$ 3',rotation='horizontal')
ax.set_xlim(0.5,1)
ax.set_ylim(0.5,1)
ax.set_zlim(0.5,1)
ax.scatter(xs,ys,zs)

#ax = Axes3D(fig)
rt2 = 0.5 * np.sqrt(2)

verts1 = [[0.5,0.5,0.5],[1,1,1],[0.75,0.75,0.5]]
verts2 = [[1, 1, 1], [0.5,0.5, 0.5],[0.75,0.75,0.5]]
verts3 = [[1, 1, 1], [0.5, 0.5, 0.5], [1, 0.5, 0.5]]
verts4 = [[1,1,1],[0.75,0.75,0.5],[1,0.5,0.5]]


ax.add_collection3d(Poly3DCollection([verts1],facecolors=['g'],alpha=0.9))
ax.add_collection3d(Poly3DCollection([verts2],facecolors=['g'],alpha=0.9))
ax.add_collection3d(Poly3DCollection([verts3],facecolors=['g'],alpha=0.9))
ax.add_collection3d(Poly3DCollection([verts4],facecolors=['g'],alpha=0.9))
ax.zaxis.set_rotate_label(False) 
ax.xaxis.set_rotate_label(False) 
ax.yaxis.set_rotate_label(False) 
ax.set_xlabel('$\lambda$ 1',rotation='horizontal',labelpad=20) 
ax.set_ylabel('$\lambda$ 2',rotation='horizontal',labelpad=20)
ax.set_zlabel('$\lambda$ 3',rotation='horizontal',labelpad=8)
ax.set_xlim(0.5,1)
ax.set_ylim(0.5,1)
ax.set_zlim(0.5,1)




plt.show()
