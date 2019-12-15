from hqca.hamiltonian import *
from functools import partial
import matplotlib
import numpy as np
from hqca.instructions import *
from hqca.acse import *
from pyscf import gto
import sys
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
np.set_printoptions(precision=6)
mol = gto.Mole()

ham = SingleQubitHamiltonian(sq=True,
        p=-1,a=1+1j,c=1-1j,h=1)
        #p=-1,a=1,c=1,h=1)
Ins = partial(
        SingleQubitExponential,
        **{'simple':True}
        )

st = StorageACSE(ham,use_initial=False,second_quant=False,
        initial=[
            ]
        )
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        #backend='statevector_simulator',
        #backend='ibmq_qasm_simulator',
        #backend='qasm_simulator',
        #backend='ibmq_armonk',
        num_shots=8192,
        Nq=1,
        #provider='IBMQ'
        )
qs.set_error_correction(error_correction='measure',
        frequency=3)


acse = RunACSE(
        st,qs,Ins,
        method='euler',
        #method='newton',
        use_trust_region=True,
        #convergence_type='trust',
        #propagation='trotter', #use e^-ih1 e^-ih2
        #propagation='rotation', #use e^-ic n sigma
        #propagation='commute', # use 
        hamiltonian_step_size=0.25,
        verbose=True,
        quantS_thresh_rel=0.1,
        commutative_ansatz=False,
        trotter=1,
        ansatz_depth=1,
        max_iter=10,
        initial_trust_region=0.1,
        newton_step=-1,
        restrict_S_size=0.1,
        opt_thresh=1e-4,
        tr_taylor_criteria=1e-3,
        tr_objective_criteria=1e-3,
        )
acse.build(log_rdm=True)
acse.run()
x,y,z = [],[],[]
en = []
test = ham.matrix[0]
eigval,eigvec = np.linalg.eigh(test)
print(eigval)
x1 = eigvec[:,0]
x2 = eigvec[:,1]
rho1 = np.outer(x1,np.conj(x1))
rho2 = np.outer(x2,np.conj(x2))
for n,i in enumerate(acse.log_rdm):
    Z = np.real(i.rdm[0,0,0]-i.rdm[0,1,1])
    X = np.real(i.rdm[0,0,1]+i.rdm[0,1,0])
    Y = np.imag(i.rdm[0,0,1]-i.rdm[0,1,0])
    en.append(acse.log_E[n])
    x.append(X)
    y.append(Y)
    z.append(Z)

print('Logs.')
for a,b,c in zip(x,y,z):
    print('X: {}, Y: {}, Z: {}'.format(a,b,c))
rx1 = np.real(rho1[0,1]+rho1[1,0])
ry1 = np.imag(rho1[0,1]-rho1[1,0])
rz1 = np.real(rho1[0,0]-rho1[1,1])

rx2 = np.real(rho2[0,1]+rho2[1,0])
ry2 = np.imag(rho2[0,1]-rho2[1,0])
rz2 = np.real(rho2[0,0]-rho2[1,1])

def energy(x,y,z):
    return x+z

theta, phi = np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 25)
THETA, PHI = np.meshgrid(theta, phi)
xs = np.cos(THETA)*np.cos(PHI)
ys = np.cos(THETA)*np.sin(PHI)
zs = np.sin(THETA)
color_dimension = xs+zs # change to desired fourth dimension
minn, maxx = color_dimension.min(), color_dimension.max()
norm =matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.plot(
    x,y,z, #c=en,#cmap='coolwarm',
    linewidth=1, antialiased=False,
    alpha=0.75)
ax.scatter(
    x,y,z, #c=en,#cmap='coolwarm',
    c='k',s=1, antialiased=False,
    alpha=0.75)
ax.plot_surface(xs,ys,zs, rstride=1, cstride=1, facecolors=fcolors, vmin=minn,
        vmax=maxx, shade=False,alpha=0.25,linewidth=0.1)
ax.scatter(rx1,ry1,rz1,c='k')
ax.scatter(rx2,ry2,rz2,c='k')
ax.scatter(x[0],y[0],z[0],c='r')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')
plt.show()




