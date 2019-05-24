from hqca.sub.BaseRun import QuantumRun
from hqca.sub.VQA import RunNOFT,RunRDM
from hqca.sub.Circuit import Quantum
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class Scan:
    def __init__(self,
            theory='noft',
            scan_type='en',
            **kw):
        self.scanning = scan_type
        if theory=='noft':
            if scan_type=='en':
                self.run = RunNOFT(**kw)
            elif scan_type=='occ':
                self.run = Quantum(**kw)
        elif theory=='rdm':
            if scan_type=='occ':
                self.run = Quantum(**kw)

    def update_var(self,**kw):
        self.run.update_var(**kw)

    def set_print(self,**kw):
        self.run.set_print(**kw)

    def build(self):
        self.run.build()

    def scan(self,**kw):
        if self.scanning=='en':
            self._scan_energies(**kw)
        elif self.scanning in ['occ','sign']:
            self._scan_occ(**kw)

    def _scan_energies(self,
            shift, # starting
            index, # which variables
            lowers, # lower limits
            uppers, # upper limits
            steps, # number of steps 
            rdm=True
            ):
        if rdm==True:
            target='rdm'
        else:
            target='orb'
        if len(index)>3:
            print('Error too many variables.')
            sys.exit()
        if len(index)==1:
            X = np.linspace(lowers[0],uppers[0],steps[0])
            Y = np.zeros(steps[0])
            for n,i in enumerate(X):
                temp = shift.copy()
                temp[index[0]]+=i
                self.run.single(target,para=temp)
                Y[n] = self.run.E
                print('{:.1f}%'.format((n+1)*100/steps[0]))
            fig = plt.figure()
            Xp = X*(180/np.pi)
            ax = fig.add_subplot(111)
            ax.scatter(Xp,Y,linewidth=3)
        elif len(index)==2:
            para1 = np.linspace(lowers[0],uppers[0],steps[0])
            para2 = np.linspace(lowers[1],uppers[1],steps[1])
            X,Y = np.meshgrid(para1,para2,indexing='ij')
            Np = len(shift)
            Z = np.zeros((steps[0],steps[1]))
            for i,a in enumerate(para1):
                for j,b in enumerate(para2):
                    temp = shift.copy()
                    temp[index[0]]+=a
                    temp[index[1]]+=b
                    self.run.single(target,para=temp)
                    try:
                        Z[i,j] = self.run.E[0]
                    except IndexError:
                        Z[i,j] = self.run.E
                print('{:.1f}%'.format((i+1)*100/steps[0]))
            fig = plt.figure()
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            maps = ax1.plot_surface(X, Y, Z,
                    cmap=cm.coolwarm,
                    linewidth=1)
            for n,i in enumerate(Z):
                print('x,y:[{:+.4f},{:+.4f}],E:{:+.8f}'.format(
                        X[n,np.argmin(i)],
                        Y[n,np.argmin(i)],
                        Z[n,np.argmin(i)]))
        plt.show()

    def _scan_occ(self,
            shift, # starting 
            index, # which variables
            lowers,
            uppers,
            steps,
            **kw
            ):
        if len(index)>2:
            print('Error too many variables.')
            sys.exit()
        if len(index)==1:
            X = np.linspace(lowers[0],uppers[0],steps[0])
            Ya = []
            Yb = []
            S = []
            for n,i in enumerate(X):
                temp = shift.copy()
                temp[index[0]]+=i
                self.run.single(temp)
                Ya.append(self.run.noca)
                Yb.append(self.run.nocb)
                test = self.run.proc.holding
                stemp = []
                for k in test.keys():
                    stemp.append(test[k]['rdme']['fo'])
                    stemp.append(test[k]['rdme']['so'])
                    stemp.append(test[k]['rdme']['+-+-'])
                S.append(stemp)
                print('{:.1f}%'.format((n+1)*100/steps[0]))
            fig = plt.figure(figsize=(12,4))
            Xp = X*(180/np.pi)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            Ya = np.asarray(Ya)
            Yb = np.asarray(Yb)
            S = np.asarray(S)
            Ta = np.sum(Ya,axis=1)
            Tb = np.sum(Yb,axis=1)
            for i in range(Ya.shape[1]):
                ax1.scatter(Xp, Ya[:,i],label='nocc{}'.format(i))
                ax2.scatter(Xp, Yb[:,i],label='nocc{}'.format(i))
                print('Std dev for alpha, beta occ {}'.format(i))
                Av_a = np.average(Ya[:,i])
                Av_b = np.average(Yb[:,i])
                print(np.sqrt(np.sum(np.square(Ya[:,i]-Av_a))))
                print(np.sqrt(np.sum(np.square(Yb[:,i]-Av_b))))
            for i in range(S.shape[1]//3):
                ax3.scatter(Xp, S[:,3*i],label='fo-{}'.format(i))
                ax3.scatter(Xp, S[:,3*i+1],label='so-{}'.format(i))
                ax3.scatter(Xp, S[:,3*i+2],label='to-{}'.format(i))
            ax1.scatter(Xp,Ta,label='total')
            ax2.scatter(Xp,Tb,label='total')
            ax1.set_xlabel('alpha')
            ax2.set_xlabel('beta')
            ax3.set_xlabel('signs')
            ax3.legend()
        elif len(index)==2:
            para1 = np.linspace(lowers[0],uppers[0],steps[0])
            para2 = np.linspace(lowers[1],uppers[1],steps[1])
            X,Y = np.meshgrid(para1,para2,indexing='ij')
            Np = len(shift)
            Za = np.zeros((steps[0],steps[1],Np+1))
            Zb = np.zeros((steps[0],steps[1],Np+1))
            S = np.zeros((steps[0],steps[1],Np))
            for i,a in enumerate(para1):
                for j,b in enumerate(para2):
                    temp = shift.copy()
                    temp[index[0]]+=a
                    temp[index[1]]+=b
                    if self.run.QuantStore.qc:
                        self.run.single(para=temp)
                        Ya.append(self.run.noca)
                        Yb.append(self.run.nocb)
                        test = self.run.proc.holding
                        sign_key = self.run.QuantStore.tomo_approx
                        if sign_key=='full':
                            sign_key='++--'
                        for k in test.keys():
                            idx = int(test[k]['n'])
                            S[i,j,idx]=test[k]['rdme'][sign_key]
                    else:
                        self.run.single(target,para=temp)
                        wf = self.run.E
                        S[i,j,0]=wf['100100']*wf['010010']
                        S[i,j,1]=wf['001001']*wf['010010']
                print('{:.1f}%'.format((i+1)*100/steps[0]))
            fig = plt.figure()
            if self.scanning=='on':
                ax1 = fig.add_subplot(231,projection='3d')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax2 = fig.add_subplot(232,projection='3d')
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                ax3 = fig.add_subplot(233,projection='3d')
                ax3.set_xlabel('x')
                ax3.set_ylabel('y')
                ax4 = fig.add_subplot(234,projection='3d')
                ax4.set_xlabel('x')
                ax4.set_ylabel('y')
                ax5 = fig.add_subplot(235,projection='3d')
                ax5.set_xlabel('x')
                ax5.set_ylabel('y')
                ax6 = fig.add_subplot(236,projection='3d')
                ax6.set_xlabel('x')
                ax6.set_ylabel('y')
                ax1.plot_surface(X, Y,Za[:,:,0],
                  cmap=cm.coolwarm,
                  linewidth=1)
                ax2.plot_surface(X, Y,Za[:,:,1],
                  cmap=cm.coolwarm,
                  linewidth=1)
                ax3.plot_surface(X, Y,Za[:,:,2],
                  cmap=cm.coolwarm,
                  linewidth=1)
                ax4.plot_surface(X,Y,Zb[:,:,0],
                  cmap=cm.coolwarm,
                  linewidth=1)
                ax5.plot_surface(X,Y,Zb[:,:,1],
                  cmap=cm.coolwarm,
                  linewidth=1)
                ax6.plot_surface(X,Y,Zb[:,:,2],
                  cmap=cm.coolwarm,
                  linewidth=1)
                print(Za,Zb)
            elif self.scanning=='sign':
                print(X,Y,S)
                ax1 = fig.add_subplot(121,projection='3d')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax2 = fig.add_subplot(122,projection='3d')
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                sign1 = ax1.plot_surface(X,Y,S[:,:,0],
                            cmap=cm.coolwarm,
                            linewidth=2)
                sign2 = ax2.plot_surface(X,Y,S[:,:,1],
                            cmap=cm.coolwarm,
                            linewidth=2)
                print(S)
                print(Za)
                print(Zb)
            else:
                ax1 = fig.add_subplot(111,projection='3d')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                maps = ax1.plot_surface(X, Y, Z,
                        cmap=cm.coolwarm,
                        linewidth=1)
                for n,i in enumerate(Z):
                    print('x,y:[{:+.4f},{:+.4f}],E:{:+.8f}'.format(
                            X[n,np.argmin(i)],
                            Y[n,np.argmin(i)],
                            Z[n,np.argmin(i)]))
        plt.show()

