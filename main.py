'''
main.py 

Main program for executing the hybrid quantum classical optimizer. Consists of
several parts. 

'''
import os, sys
from importlib import reload
import nevergrad
from nevergrad.optimization import optimizerlib,registry
from nevergrad.optimization.utils import Archive,Value
import numpy as np
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from functools import reduce
from hqca.tools.QuantumFramework import add_to_config_log
import datetime
import sys
from pyscf import scf
np.set_printoptions(precision=3)
from hqca import sub
from hqca.tools import EnergyFunctions as enf
from hqca.tools.util import Errors as err
import pickle

version='0.1.1'

class sp:
    '''
    Class to perform a single point energy calculation
    '''
    def __init__(self,
            mol,
            theory,
            verbose=0,
            calc_E=False,
            restart=False):
        '''start function,
        assigns the chemical things and gets stuff going
        '''
        self.run_type = 'sp'
        self.theory=theory
        self.verbose=verbose
        if verbose:
            print('## ### hqca ### ###')
            print('  --- v{} ---'.format(version))
            print('')
            print('# Beginning a hybrid quantum classical algorithm.')
            print('# Run type: single point')
            if theory in ['rdm','RDM']:
                print('# Theory: RDM optimization')
        self._load_mol(mol,verbose,calc_E)
        if not restart==False:
            self._load_restart(restart)
        else:
            self._choose_theory()
        now = datetime.datetime.today()
        m,d,y = now.strftime('%m'), now.strftime('%d'), now.strftime('%y')
        H,M,S = now.strftime('%H'), now.strftime('%M'), now.strftime('%S')
        self.file = ''.join(('sp','-',theory,'_',m,d,y,'-',H,M))


    def _load_mol(self,
            mol,
            verbose,
            calc_E
            ):
        self.S = mol.intor('int1e_ovlp')
        self.T_1e = mol.intor('int1e_kin')
        self.V_1e = mol.intor('int1e_nuc')
        self.ints_1e = self.V_1e+self.T_1e
        self.Norb = self.S.shape[0]
        self.ints_2e = mol.intor('int2e')
        self.hf = scf.RHF(mol)
        self.hf.kernel()
        self.hf.analyze()
        self.C= self.hf.mo_coeff
        try:
            mol.as_Ne
        except Exception:
            mol.as_Ne=mol.nelec[0]+mol.nelec[1]
            mol.as_No= self.C.shape[0]
        store_kw = {
            'theory':self.theory,
            'Nels_tot':mol.nelec[0]+mol.nelec[1],
            'Norb_tot':self.C.shape[0], #spatial
            'Nels_as':mol.as_Ne,
            'Norb_as':mol.as_No,
            'moc_alpha':self.C,
            'moc_beta':self.C,
            'ints_1e_ao':self.ints_1e,
            'ints_2e_ao':self.ints_2e,
            'E_ne':mol.energy_nuc(),
            }
        if calc_E:
            from pyscf import mcscf
            mc = mcscf.CASCI(self.hf,mol.as_No,mol.as_Ne)
            mc.kernel()
            store_kw['e_fci']=mc.e_tot
            if verbose>1:
                print('')
                print('SCF energy: {:.8f} H'.format(self.hf.e_tot))
                print('FCI energy  : {:.8f} H'.format(mc.e_tot))
                print('')
                print('Det-alp, Det-bet,  CI coeff')
                obj = mc.fcisolver.large_ci(
                    mc.ci,
                    mol.as_No,
                    (1,1),
                    tol=0.01, 
                    return_strs=False
                    )
                for c,ia,ib in obj:
                    print('   {}      {}    {:+.12f}'.format(ia,ib,c))
        if self.verbose>1:
            print('#  Total e- count    : {}'.format(Nels_tot))
            print('#  Total orb count   : {}'.format(Norb_tot))
            print('#  Active e- count   : {}'.format(Nels_as))
            print('#  Active orb count  : {}'.format(Norb_as))
            print('#  Inactive orb count: {}'.format(self.Norb_ia))
        self.Store = enf.Storage(
            **store_kw)

    def _load_restart(self,name):
        try:
            with open(name, 'rb') as fp:
                self.run = pickle.load(fp)
        except FileNotFoundError:
            sys.exit()
        except Exception as e:
            sys.exit()
        for i in self.run.Run.keys():
            try:
                self.run.Run[i].opt.reload_opt()
            except Exception as e:
                traceback.print_exc()
                print(e)
                sys.exit()
        self.run._restart()

    def _save(self):
        for i in self.run.Run.keys():
            try:
                self.run.Run[i].opt.save_opt()
            except Exception as e:
                print(e)
                sys.exit()
        with open(self.file+'.run', 'wb') as fp:
            pickle.dump(
                    self.run,
                    fp,
                    pickle.HIGHEST_PROTOCOL
                    )
        print('Saved file as {}.run'.format(self.file))

    def _choose_theory(self):
        if self.theory in ['NOFT','noft']:
            self.run = sub.RunNOFT(self.Store)
        elif self.theory in ['rdm','RDM']:
            self.run = sub.RunRDM(self.Store)

    def update_var(self,
            **kw):
        self.run.update_var(**kw)

    def set_print(self,**kw):
        self.run.set_print(**kw)
    
    def build(self):
        self.run.build()

    def execute(self):
        try:
            self.run.go()
        except err.RunError:
            print('run error')
            self._save()

        self.result = self.run.Store

    def analysis(self):
        self.run.Store.opt_analysis()





class scan(sp):
    '''
    special class for performing scans or more specific analysis of the
    optimization or exploring the parameters in the optimization
    '''
    def update_rdm(self,para):
        self.run.single('rdm',para)
        #self.Store.update_rdm2()

    def update_full_ints(self,para):
        self.run.single('orb',para)
        self.Store.update_full_ints()

    def search_for_polytope(self):
        '''
        run to look for the polytope extrema
        '''
        self.run.single('orb',para)


    def scan(self,
            target,
            start,
            index,
            high,
            low,
            ns,
            prop='en',
            save=False):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        if len(index)>3:
            print('Error too many variables.')
            sys.exit()
        if len(index)==1:
            X = np.linspace(low[0],high[0],ns[0])
            if prop=='on':
                Ya = []
                Yb = []
                S = []
            else:
                Y = np.zeros(ns[0])
            for n,i in enumerate(X):
                temp = start.copy()
                temp[index[0]]=i
                self.run.single(target,temp,prop=prop)
                if prop=='en':
                    Y[n] = self.run.E
                elif prop=='on':
                    Ya.append(self.run.E[0])
                    Yb.append(self.run.E[1])
                    test = self.run.E[2].holding
                    stemp = []
                    for k in test.keys():
                        stemp.append(test[k]['rdme']['fo'])
                        stemp.append(test[k]['rdme']['so'])
                        stemp.append(test[k]['rdme']['+-+-'])
                    S.append(stemp)
                print('{:.1f}%'.format((n+1)*100/ns[0]))
            fig = plt.figure()
            Xp = X*(180/np.pi)
            if prop=='on':
                ax1 = fig.add_subplot(131)
                ax2 = fig.add_subplot(132)
                ax3 = fig.add_subplot(133)
                Ya = np.asarray(Ya)
                Yb = np.asarray(Yb)
                S = np.asarray(S)
                Ta = np.sum(Ya,axis=1)
                Tb = np.sum(Yb,axis=1)
                for i in range(Ya.shape[1]):
                    ax1.plot(Xp, Ya[:,i],label='nocc{}'.format(i))
                    ax2.plot(Xp, Yb[:,i],label='nocc{}'.format(i))
                    print('Std dev for alpha, beta occ {}'.format(i))
                    Av_a = np.average(Ya[:,i])
                    Av_b = np.average(Yb[:,i])
                    print(np.sqrt(np.sum(np.square(Ya[:,i]-Av_a))))
                    print(np.sqrt(np.sum(np.square(Yb[:,i]-Av_b))))
                for i in range(S.shape[1]//3):
                    ax3.plot(Xp, S[:,3*i],label='fo-{}'.format(i))
                    ax3.plot(Xp, S[:,3*i+1],label='so-{}'.format(i))
                    ax3.plot(Xp, S[:,3*i+2],label='to-{}'.format(i))
                ax1.plot(Xp,Ta,label='total')
                ax2.plot(Xp,Tb,label='total')
                ax1.set_xlabel('alpha')
                ax2.set_xlabel('beta')
                ax3.set_xlabel('signs')
                #ax1.legend()
                ax3.legend()
            else:
                ax = fig.add_subplot(111)
                ax.plot(Xp,Y,linewidth=3)
                # Plot the surface.
            if save==False:
                plt.show()
            else:
                plt.savefig(save,format='png')
        elif len(index)==2:
            para1 = np.linspace(low[0],high[0],ns[0])
            para2 = np.linspace(low[1],high[1],ns[1])
            X,Y = np.meshgrid(para1,para2,indexing='ij')
            Np = len(start)
            if prop in ['on','sign']:
                Za = np.zeros((ns[0],ns[1],Np+1))
                Zb = np.zeros((ns[0],ns[1],Np+1))
                S = np.zeros((ns[0],ns[1],Np))
            else:
                Z = np.zeros((ns[0],ns[1]))
            for i,a in enumerate(para1):
                for j,b in enumerate(para2):
                    temp = start.copy()
                    temp[index[0]]=a
                    temp[index[1]]=b
                    if prop in ['on','sign']:
                        if self.run.QuantStore.qc:
                            self.run.single(target,para=temp,prop='on')
                            Za[i,j,:]=self.run.E[0]
                            Zb[i,j,:]=self.run.E[1]
                            test = self.run.E[2].holding
                            sign_key = self.run.QuantStore.tomo_approx
                            if sign_key=='full':
                                sign_key='++--'
                            for k in test.keys():
                                idx = int(test[k]['n'])
                                S[i,j,idx]=test[k]['rdme'][sign_key]
                        else:
                            self.run.single(target,para=temp,prop='on')
                            wf = self.run.E
                            S[i,j,0]=wf['100100']*wf['010010']
                            S[i,j,1]=wf['001001']*wf['010010']
                    else:
                        self.run.single(target,para=temp,prop=prop)
                        try:
                            Z[i,j] = self.run.E[0]
                        except IndexError:
                            Z[i,j] = self.run.E
                print('{:.1f}%'.format((i+1)*100/ns[0]))
            fig = plt.figure()
            if prop=='on':
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
            elif prop=='sign':
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

