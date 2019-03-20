'''
main.py 

Main program for executing the hybrid quantum classical optimizer. Consists of
several parts. 

'''
import pickle
import os, sys
from importlib import reload
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

version='0.1.0'

class sp:
    '''
    Class to perform a single point energy calculation
    '''
    def __init__(self,
            mol,
            theory,
            pr_g=0,
            calc_E=False,
            restart=False):
        '''start function,
        assigns the chemical things and gets stuff going
        '''
        self.run_type = 'sp'
        self.theory=theory
        if pr_g>0:
            print('## ### hqca ### ###')
            print('  --- v{} ---'.format(version))
            print('')
            print('# Beginning a hybrid quantum classical algorithm.')
            print('# Run type: single point')
            if theory in ['rdm','RDM']:
                print('# Theory: RDM optimization')
        self._load_mol(mol,pr_g,calc_E)
        if restart:
            self._load_restart()
        else:
            self._choose_theory()

    def _load_mol(self,
            mol,
            pr_g,
            calc_E
            ):
        self.S = mol.intor('int1e_ovlp')
        self.T_1e = mol.intor('int1e_kin')
        self.V_1e = mol.intor('int1e_nuc')
        self.ints_1e = self.V_1e+self.T_1e
        self.Norb = self.S.shape[0]
        mol.verbose=0
        self.ints_2e = mol.intor('int2e')
        self.hf = scf.RHF(mol)
        self.hf.kernel()
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
            'pr_g':pr_g
            }
        if calc_E:
            from pyscf import mcscf
            mc = mcscf.CASCI(self.hf,mol.as_No,mol.as_Ne)
            mc.kernel()
            store_kw['e_fci']=mc.e_tot
            if pr_g>0:
                print('')
                print('# CASCI solution generated.')
                print('# Hartree energy: {:.8f} H'.format(self.hf.e_tot))
                print('# Total energy  : {:.8f} H'.format(mc.e_tot))
                print('# ')
                print('# Det-alp, Det-bet,  CI coeff')
                obj = mc.fcisolver.large_ci(
                    mc.ci,
                    mol.as_No,
                    (1,1),
                    tol=0.01, 
                    return_strs=False
                    )
                for c,ia,ib in obj:
                    print('#   {}      {}    {:+.12f}'.format(ia,ib,c))
                print('# ')
        self.Store = enf.Storage(
            **store_kw)

    def _load_restart(self):
        pass

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
        self.run.go()
        self.result = self.run.Store

    def analysis(self):
        import hqca.tools.RDMFunctions as rdmf
        from functools import reduce
        print('Calculating spin of system.')
        print('Rotating into the proper frame of reference.')
        rdm2 = self.result.rdm2
        rdm1 = rdmf.check_2rdm(rdm2,self.result.Nels_tot)
        self.Ci = np.linalg.inv(self.C)
        self.Ti_a = np.linalg.inv(self.result.T_alpha)
        self.Ti_b = np.linalg.inv(self.result.T_beta)
        Ni_a = reduce(
                np.dot, (
                    self.Ti_a,self.C)
                )
        Ni_b = reduce(
                np.dot, (
                    self.Ti_b,self.C)
                )
        rdm2_mo = rdmf.rotate_2rdm(
                rdm2,
                Ni_a.T,
                Ni_b.T,
                self.result.alpha_mo,
                self.result.beta_mo,
                spin2spac=self.result.s2s,
                region='full'
                )
        rdm1_mo = rdmf.check_2rdm(
                rdm2_mo,
                self.result.Nels_tot)
        print(np.real(rdm1_mo))
        sz = rdmf.Sz(
                rdm1_mo,
                self.result.alpha_mo,
                self.result.beta_mo,
                s2s=self.result.s2s)
        s2 = rdmf.S2(
                rdm2_mo,
                rdm1_mo,
                self.result.alpha_mo,
                self.result.beta_mo,
                s2s=self.result.s2s)
        print('Sz value: {:.6f}'.format(np.real(sz)))
        print('S2 value: {:.6f}'.format(np.real(s2)))
        print('Natural orbital wavefunction:')
        for k,v in self.result.wf.items():
            print(' |{}>: {}'.format(k,v))

class scan(sp):

    def update_rdm(self,para):
        self.run.single('rdm',para)
        self.Store.update_rdm2()

    def update_full_ints(self,para):
        self.run.single('orb',para)
        self.Store.update_full_ints()

    def scan(self,
            target,
            start,
            index,
            high,
            low,
            ns):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D
        if len(index)>3:
            print('Error too many variables.')
            sys.exit()
        if len(index)==1:
            X = np.linspace(low[0],high[0],ns[0])
            Y = np.zeros(ns[0])
            for n,i in enumerate(X):
                temp = start.copy()
                temp[index[0]]=i
                self.run.single(target,temp)
                Y[n] = self.run.E
                print('{:.1f}%'.format((n+1)*100/ns[0]))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            Xp = X*(180/np.pi)
            ax.plot(Xp, Y,linewidth=3)
            # Plot the surface.
            plt.show()
        elif len(index)==2:
            para1 = np.linspace(low[0],high[0],ns[0])
            para2 = np.linspace(low[1],high[1],ns[1])
            X,Y = np.meshgrid(para1,para2,indexing='ij')
            Z = np.zeros((ns[0],ns[1]))
            for i,a in enumerate(para1):
                for j,b in enumerate(para2):
                    temp = start.copy()
                    temp[index[0]]=a
                    temp[index[1]]=b
                    self.run.single(target,para=temp)
                    Z[i,j] = self.run.E
                print('{:.1f}%'.format((i+1)*100/ns[0]))
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            maps = ax.plot_surface(X, Y, Z,
                    cmap=cm.coolwarm,
                    linewidth=0)
            for n,i in enumerate(Z):
                print('x,y:[{:+.4f},{:+.4f}],E:{:+.8f}'.format(
                        X[n,np.argmin(i)],
                        Y[n,np.argmin(i)],
                        Z[n,np.argmin(i)]))
            plt.colorbar(maps)
            # Plot the surface.
            plt.show()

        elif len(index)==3:
            para1 = np.linspace(low[0],high[0],ns[0])
            para2 = np.linspace(low[1],high[1],ns[1])
            para3 = np.linspace(low[2],high[2],ns[2])
            X,Y = np.meshgrid(para1,para2,indexing='ij')
            for k,c in enumerate(para3):
                temp1 = start.copy()
                temp1[index[2]]=c
                Z = np.zeros((ns[0],ns[1]))
                for i,a in enumerate(para1):
                    for j,b in enumerate(para2):
                        temp = temp1.copy()
                        temp[index[0]]=a
                        temp[index[1]]=b
                        self.run.single(target,para=temp)
                        Z1[i,j] = self.run.E
                    print('{:.1f}%'.format((i+1)*100/ns[0]))
                fig = plt.figure()
                ax = fig.add_subplot(111,projection='3d')
                maps = ax.plot_surface(X, Y, Z,
                        cmap=cm.coolwarm,
                        linewidth=0)
                plt.colorbar(maps)
                # Plot the surface.
                plt.show()

