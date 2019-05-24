'''
./sub.py

Holds the subroutines for the optimizations.
'''
import pickle
import threading
import os, sys
from importlib import reload
import numpy as np
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from hqca.tools import Functions as fx
from hqca.optimizers.Control import Optimizer
from hqca.tools import RDMFunctions as rdmf
from hqca.tools import EnergyFunctions as enf
from hqca.sub.BaseRun import QuantumRun,Cache
from hqca.quantum import ErrorCorrection as ec
from hqca.quantum import Triangulation as tri
from hqca.quantum import QuantumFunctions as qf
from hqca.tools.util import Errors
from functools import reduce
import datetime
import sys
from hqca.tools import Preset as pre
np.set_printoptions(precision=3)


class RunNOFT(QuantumRun):
    '''
    Subroutine for a natural-orbital function theory approach with a quantum
    computing treatment of the natural orbitals.
    '''
    def __init__(self,
            mol,
            **kw
            ):
        QuantumRun.__init__(self,**kw)
        self.theory = 'noft'
        if mol==None:
            print('Did you forget to specify a mol object form pyscf?')
            print('Please try again.')
            sys.exit()
        QuantumRun._build_energy(self,mol,**kw)
        if self.Store.Ne_as==2:
            self.kw = pre.NOFT_2e()
        elif self.Store.Ne_as==3:
            self.kw = pre.NOFT_3e()
        else:
            print('Do not yet have support for more than 3 electron or less')
            print('than 2 electrons. Goodbye!')
            sys.exit()
        self.pr_g = self.kw['pr_g']
        self.kw_qc = self.kw['qc']
        self.kw_opt = self.kw['qc']['opt']
        self.kw_orb = self.kw['orb']
        self.kw_orb_opt = self.kw['orb']['opt']
        self.Store.pr_m = self.kw['pr_m']
        self.Store.pr_s = self.kw['pr_s']
        self.total=Cache()
        self.Run = {}

    def build(self,mol=None):
        QuantumRun._build_quantum(self)
        self.kw_orb_opt['function'] = enf.find_function(
                'noft',
                'orb',
                self.Store,
                self.QuantStore)
        grad_free = ['NM','nevergrad']
        if self.kw_opt['optimizer'] not in grad_free:
            self.kw_opt['gradient']=enf.find_function(
                    'noft',
                    'noft_grad',
                    self.Store,
                    self.QuantStore)
        if self.kw_orb_opt['optimizer'] in grad_free:
            pass
        else:
            self.kw_orb_opt['gradient']=enf.find_function(
                    'noft',
                    'orb_grad',
                    self.Store,
                    self.QuantStore)
        if not ('classical' in self.kw_qc['method']):
            self._pre()
        self.built=True
        qf.get_direct_stats(self.QuantStore)


    def update_var(self,**kw):
        QuantumRun.update_var(self,**kw)
        self.Store.pr_m = self.kw['pr_m']

    def _pre(self):
        try:
            if self.kw_qc['tri']:
                self.kw_qc['triangle']=tri.find_triangle(
                        Ntri=self.kw_qc['method_Ntri'],
                        **self.kw_qc)
        except KeyError:
            pass
        if self.kw_qc['error_correction'] and self.QuantStore.qc:
            ec_a,ec_b =ec.generate_error_polytope(
                    self.Store,
                    self.QuantStore)
            self.QuantStore.ec_a = ec_a
            self.QuantStore.ec_b = ec_b

    def run(self):
        if self.built:
            while not self.total.done:
                self._OptNO()
                self._OptOrb()
                self._check(self.kw['opt_thresh'],self.kw['max_iter'])

    def _restart(self):
        self.restart = True
    
    def _check(self,
            crit,
            max_iter
            ):
        diff = abs(self.main.crit-self.sub.crit)*1000
        if self.pr_g>0:
            print('Macro iteration: {}'.format(self.total.iter))
            print('E_noc: {}'.format(self.main.crit))
            print('E_nor: {}'.format(self.sub.crit))
            print('Diff in energies: {} mH '.format(diff))
        if self.main.err or self.sub.err:
            if self.main.err:
                self.total.done= True
                self.total.err = True
                print(self.main.msg)
            if self.sub.err:
                print(self.sub.msg)
        elif abs(self.main.crit-self.sub.crit)<crit:
            self.total.done=True
            self.Store.opt_analysis()
        elif self.total.iter>=max_iter:
            self.total.done=True
            self.total.err=True
        else:
            pass
        if self.total.done and not self.total.err==False:
            print('Got an error.')
            raise Exception
        self.total.crit = self.Store.energy_best
        self.main.crit=0
        self.sub.crit=0
        if self.pr_g>2:
            self.Store.opt_analysis()
        self.total.iter+=1

    def _set_opt_parameters(self):
        self.f = min(self.Store.F_alpha,self.Store.F_beta)
        #self.kw_opt['unity']=self.kw_opt['unity']*(1-self.f*0.5)
        #print('Scale factor: {}'.format(180*self.kw_opt['unity']/np.pi))

    def _OptNO(self):
        self.main=Cache()
        key = 'rdm{}'.format(self.total.iter)
        if not self.restart:
            if self.total.iter>0:
                self._set_opt_parameters()
            self.Run[key] = Optimizer(
                    **self.kw_opt
                    )
            self.Run[key].initialize(
                    self.QuantStore.parameters)
            if self.Run[key].error:
                print('##########')
                print('Encountered error in initialization.')
                print('##########')
                self.main.done=True
        self.main.done=False
        while not self.main.done:
            self.Run[key].next_step()
            if self.kw_opt['pr_o']>0:
                print('Step: {:02}, E: {:.8f} c: {:.8f}  '.format(
                    self.main.iter,
                    self.Run[key].opt.best_f,
                    self.Run[key].opt.crit)
                    )
            self.Run[key].check(self.main)
            if self.main.iter==self.kw_opt['max_iter']:
                self.main.error=False
                self.Run[key].opt_done=True
                self.main.done=True
            elif self.Run[key].opt_done:
                if self.Run[key].error:
                    print('Error in run.')
                    Store.opt_done=True
                continue
            self.main.iter+=1
        self.kw_opt['shift']=self.Run[key].opt.best_y.copy()
        self.Store.update_rdm2()

    def _OptOrb(self):
        self.sub=Cache()
        key = 'orb{}'.format(self.total.iter)
        self.sub.done=False
        self.para_orb = np.asarray([0.0]*self.Store.Np_orb)
        if self.kw['restart']==True:
            self._load()
        else:
            self.Run[key] = Optimizer(
                    **self.kw_orb_opt
                    )
            self.Run[key].initialize(self.para_orb)
        while not self.sub.done:
            self.Run[key].next_step()
            if self.kw['pr_s']>0 and self.sub.iter%1==0:
                print('Step: {:02}, E: {:.8f} c: {:.8f}'.format(
                    self.sub.iter,
                    self.Run[key].opt.best_f,
                    self.Run[key].opt.crit)
                    )
            self.Run[key].check(self.sub)
            if self.sub.iter==self.kw_orb_opt['max_iter']:
                self.sub.done=True
                self.sub.err=True
                self.sub.msg='Max iterations met.'
                self.Run[key].opt_done=True
            self.sub.iter+=1
        self.Store.update_full_ints()
    
    def single(self,target,para):
        if target=='rdm':
            self.E = self.kw_opt['function'](para)
        elif target=='orb':
            self.E = self.kw_orb_opt['function'](para)

    def _find_orb(self):
        self.main=Cache()
        self.main.done=True
        self._OptOrb()

    def find_occ(self,on=0,spin='alpha'):
        def f_test(para,**kw):
            self.single(para,**kw)
            if spin=='alpha':
                return -self.E[0][on]
            elif spin=='beta':
                return -self.E[1][on]
        self.Run = Optimizer(
                function=f_test,
                **self.kw_opt)
        self.Run.initialize(self.QuantStore.parameters)

    def go(self):
        self.run()
    def execute(self):
        self.run()


class RunRDM(QuantumRun):
    '''
    Subroutine for a RDM based approach. Orbital optimization is included in the
    wavefunction.
    '''
    def __init__(self,store,**k):
        QuantumRun.__init__(self)
        self.kw = pre.RDM()
        self.rc=Cache()
        self.theory = 'rdm'

    def run(self):
        if self.built:
            self._OptRDM()
            self._analyze()
        else:
            sys.exit('# Not built yet! Run build() before execute(). ')

    def build(self):
        QuantumRun._build_energy(self,mol)
        QuantumRun._build_quantum(self)
        self.Store.fund_npara_orb()
        self.built=True

    def _OptRDM(self):
        if self.kw['restart']==True:
            self._load()
        else:
            Run = Optimizer(
                    **self.kw_opt
                    )
            Run.initialize(self.QuantStore.parameters)
        if Run.error:
            print('##########')
            print('Encountered error in initialization.')
            print('##########')
            self.rc.done=True
        while not self.rc.done:
            Run.next_step()
            if self.kw_opt['pr_o']>0:
                print('Step: {:02}, E: {:.8f} Sigma: {:.8f}'.format(
                    self.rc.iter,
                    Run.opt.best_f,
                    Run.opt.crit)
                    )
            Run.check(self.rc)
            if self.pr_g>3:
                self.Store.opt_analysis()
            if self.rc.iter==self.kw_opt['max_iter'] and not(self.rc.done):
                self.rc.error=False
                Run.opt_done=True
                self.rc.done=True
            elif Run.opt_done:
                if Run.error:
                    print('Error in run.')
                    Store.opt_done=True
                continue
            self.rc.iter+=1
        self.Store.update_rdm2()

    def go(self):
        self.run()
    def execute(self):
        self.run()

