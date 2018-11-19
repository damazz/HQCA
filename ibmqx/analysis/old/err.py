import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
print(matplotlib.__version__)
# plot the orbitals 
import sys
def gpc(set):
    #Borland Dennis Constraint
    # L6 + L5 >= L4
    # -> 1 - L1 + 1 - L2 >= 1 - L3
    # -> 1 + L3 >= L2 + L1
    set.sort() 
    set = set[::-1]
    if (set[0]+set[1])<=(1+set[2]):
        # GPC conditions are met
        met = True
    else:
        met = False
    return met 
         
np.set_printoptions(suppress=True,precision=5)

data = np.loadtxt('./compiled/'+sys.argv[1])
print(data)
size = data.shape
N = size[0]

hold = np.zeros((N,3))
# shape in other 

for i in range(0,N):
    hold[i,:]=data[i,6:9]

fig = plt.figure()

e1 = hold[:,0]
e2 = hold[:,1]
e3 = hold[:,2]

plt.hist(e1,normed=1,bins=40,label='q1')
plt.hist(e2,normed=1,bins=40,label='q2')
plt.hist(e3,normed=1,bins=40,label='q3')
plt.legend()
'''
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
xs = hold[:,0]
ys = hold[:,1]
zs = hold[:,2]
#colors = ['b' if gpc(hold[x,:])==1 else 'r' for x in Nlist]
ax.zaxis.set_rotate_label(False) 
ax.xaxis.set_rotate_label(False) 
ax.yaxis.set_rotate_label(False) 
ax.set_xlabel('$\lambda$ 1',rotation='horizontal') 
ax.set_ylabel('$\lambda$ 2',rotation='horizontal')
ax.set_zlabel('$\lambda$ 3',rotation='horizontal')
ax.set_xlim(0.5,1)
ax.set_ylim(0.5,1)
ax.set_zlim(0.5,1)
for xm, ym, zm in zip(xs,ys,zs):
    if (zm + 1) <= (ym + xm):
        col = 'r'
    else:
        col = 'b 
    ax.scatter(xm,ym,zm,c=col,edgecolors='k')

#ax.scatter(xs,ys,zs,c=colors)

#ax = Axes3D(fig)
rt2 = 0.5 * np.sqrt(2)

verts1 = [[0.5,0.5,0.5],[1,1,1],[0.75,0.75,0.5]]
verts2 = [[1, 1, 1], [0.5,0.5, 0.5],[0.75,0.75,0.5]]
verts3 = [[1, 1, 1], [0.5, 0.5, 0.5], [1, 0.5, 0.5]]
verts4 = [[1,1,1],[0.75,0.75,0.5],[1,0.5,0.5]]

face1= Poly3DCollection([verts1],linewidth=2,alpha=0.33)
face2= Poly3DCollection([verts2],linewidth=2,alpha=0.33)
face3= Poly3DCollection([verts3],linewidth=2,alpha=0.33)
face4= Poly3DCollection([verts4],linewidth=2,alpha=0.33)


aleph = 0.5
face1.set_facecolor((0,0.5,0.5,aleph))
face2.set_facecolor((0,0.5,0.5,aleph))
face3.set_facecolor((0,0.5,0.5,aleph))
face4.set_facecolor((0,0.5,0.5,aleph))
#face1.set_edgecolor((1,1,1,1))


ax.add_collection3d(face1)
ax.add_collection3d(face2)
ax.add_collection3d(face3)
ax.add_collection3d(face4)

#ax.add_collection3d(Poly3DCollection([verts1],facecolors=['g'],alpha=0.33))
#ax.add_collection3d(Poly3DCollection([verts2],facecolors=['g'],alpha=0.33))
#ax.add_collection3d(Poly3DCollection([verts3],facecolors=['g'],alpha=0.33))
#ax.add_collection3d(Poly3DCollection([verts4],facecolors=['g'],alpha=0.33))
ax.zaxis.set_rotate_label(False) 
ax.xaxis.set_rotate_label(False) 
ax.yaxis.set_rotate_label(False) 
ax.set_xlabel('$\lambda$ 1',rotation='horizontal',labelpad=20) 
ax.set_ylabel('$\lambda$ 2',rotation='horizontal',labelpad=20)
ax.set_zlabel('$\lambda$ 3',rotation='horizontal',labelpad=8)
ax.set_xlim(0.5,1)
ax.set_ylim(0.5,1)
ax.set_zlim(0.5,1)
'''



plt.show()
