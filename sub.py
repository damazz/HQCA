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
from hqca.tools import Optimizers as opt
from hqca.tools import RDMFunctions as rdmf
from hqca.tools import EnergyFunctions as enf
from hqca.tools import ErrorCorrection as ec
from hqca.tools import Triangulation as tri
from hqca.tools import QuantumFunctions as qf
from hqca.tools.util import Errors
from functools import reduce
from hqca.tools.QuantumFramework import add_to_config_log
import datetime
import sys
import hqca.pre as pre
np.set_printoptions(precision=3)

class Cache:
    def __init__(self):
        self.use=True
        self.err=False
        self.msg=None
        self.iter=0
        self.done=False

class QuantumRun:
    '''
    Routine for handling different runs. 
    Has following main attributes. 
    '''
    def __init__(self,
            store,
            **kw
            ):
        self.Store=store
        self.kw_qc = self.kw['qc']
        self.kw_opt = self.kw_qc['opt']
        self.built=False
        self.restart = False

    def update_var(
            self,
            target='global',
            **args):
        if target=='global':
            for k,v in args.items():
                self.kw[k]=v
        elif target=='qc':
            for k,v in args.items():
                self.kw_qc[k] = v
        elif target=='opt':
            for k,v in args.items():
                self.kw_opt[k]=v
        elif target=='orb':
            for k,v in args.items():
                self.kw_orb[k]=v
        elif target=='orb_opt':
            for k,v in args.items():
                self.kw_orb_opt[k]=v

    def build(self):
        self.pr_g = self.kw['pr_g']
        self.Store.pr_m = self.kw['pr_m']
        if self.pr_g>1:
            print('')
            print('### #### ### ### ### ### ### ### ### ### ### ###')
            print('')
            print('# Initializing the optimization.')
            print('#')
            print('# Setting RDM parameters...')
        self.Store.gas()
        self.Store.gsm()
        self.kw_qc['Nels_as'] = self.Store.Nels_as
        self.kw_qc['Norb_as']=self.Store.Norb_as
        self.kw_qc['alpha_mos']=self.Store.alpha_mo
        self.kw_qc['beta_mos']=self.Store.beta_mo
        self.kw_qc['single_point']=self.Store.sp
        if self.pr_g>1:
            print('# ...done.')
            print('#')
            print('# Setting QC parameters...')
        self.QuantStore = qf.QuantumStorage(self.pr_g,**self.kw_qc)
        if self.pr_g>1:
            print('# ...done.')
            print('#')
            print('# Setting opt parameters...')
        self.Store.update_full_ints()
        self.kw_opt['function'] = enf.find_function(
                self.Store.theory,
                'qc',
                self.Store,
                self.QuantStore)
        if self.pr_g>1:
            if self.kw_opt['optimizer']=='nevergrad':
                print('#  optimizer  : {}'.format(self.kw_opt['nevergrad_opt']))
            else:
                print('#  optimizer  : {}'.format(self.kw_opt['optimizer']))
            print('#  max iter   : {}'.format(self.kw_opt['max_iter']))
            print('#  stop crit  : {}'.format(self.kw_opt['conv_crit_type']))
            print('#  crit thresh: {}'.format(self.kw_opt['conv_threshold']))
            print('# ...done.')
            print('# ')
            print('# Initialized successfully. Beginning optimization.')
            print('')
            print('### ### ### ### ### ### ### ### ### ### ### ###')
            print('')
        if self.kw_qc['info'] in ['calc']:
            qf.get_direct_stats(self.QuantStore)
        elif self.kw_qc['info'] in ['draw']:
            qf.get_direct_stats(self.QuantStore,extra='draw')
            sys.exit()
        elif self.kw_qc['info'] in ['count_only']:
            qf.get_direct_stats(self.QuantStore)
            sys.exit()
        elif self.kw_qc['info'] in ['compile','check','check_circuit','transpile']:
            qf.get_direct_stats(self.QuantStore,extra='compile')
            sys.exit()
        else:
            pass

    def set_print(self,level='default',
            record=False
            ):
        self.kw_qc['pr_e']=1
        self.kw_qc['pr_q']=1
        self.kw_opt['pr_o']=1
        self.kw['pr_m']=1
        self.kw['pr_g']=2
        self.kw['pr_s']=1
        if level=='terse':
            self.kw['pr_g']=1
            self.kw_opt['pr_o']=0
            self.kw['pr_m']=0
            self.kw['pr_s']=0
        elif level=='none':
            self.kw_qc['pr_q']=0
            self.kw_qc['pr_e']=0
            self.kw_opt['pr_o']=0
            self.kw['pr_m']=0
            self.kw['pr_g']=0
            self.kw['pr_s']=0
        elif level=='analysis':
            self.kw['pr_g']=4
        elif level=='diagnostic':
            self.kw_qc['pr_q']=9
            self.kw_opt['pr_o']=9
            self.kw['pr_m']=9
            self.kw['pr_g']=9
        elif level=='diagnostic_en':
            self.kw['pr_m']=4
        elif level=='diagnostic_qc':
            self.kw_qc['pr_q']=9
        elif level=='diagnostic_opt':
            self.kw_opt['pr_o']=9
        elif level=='diagnostic_orb':
            self.kw['pr_s']=9
            self.kw_opt['pr_o']=9

class RunRDM(QuantumRun):
    '''
    Subroutine for a RDM based approach. Orbital optimization is included in the
    wavefunction.
    '''
    def __init__(self,store,**k):
        self.kw = pre.RDM()
        self.rc=Cache()
        QuantumRun.__init(store,**kw)
        self.Store.sp='rdm'

    def go(self):
        if self.built:
            self._OptRDM()
            self._analyze()
        else:
            sys.exit('# Not built yet! Run build() before execute(). ')

    def build(self):
        QuantumRun.build(self)
        self.Store.fund_npara_orb()
        self.built=True

    def _OptRDM(self):
        if self.kw['restart']==True:
            self._load()
        else:
            Run = opt.Optimizer(
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

    def _load(self):
        pass

    def _analyze(self,
            ):
        if self.rc.err:
            self.rc.done= True
            print(self.rc.msg)
            print(self.rc.msg)
        if self.pr_g>0:
            print('done!')


class RunNOFT(QuantumRun):
    '''
    Subroutine for a natural-orbital function theory approach with a quantum
    computing treatment of the natural orbitals.
    '''
    def __init__(self,
            store,
            **kw
            ):
        if store.kw['Nels_as']==2:
            self.kw = pre.NOFT_2e()
        elif store.kw['Nels_as']==3:
            self.kw = pre.NOFT_3e()
        else:
            sys.exit()
        QuantumRun.__init__(self,store,**kw)
        self.Store.sp='noft'
        self.kw_orb = self.kw['orb']
        self.kw_orb_opt = self.kw_orb['opt']
        self.total=Cache()
        self.Run = {}

    def build(self):
        self.Store.pr_s = self.kw['pr_s']
        QuantumRun.build(self)
        self.Store.find_npara_orb()
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
        self.unity = self.kw_opt['unity']
        self.shift = None
        if self.pr_g>1:
            if self.kw_orb_opt['optimizer']=='nevergrad':
                print('#  orb opt    : {}'.format(self.kw_orb_opt['nevergrad_opt']))
            else:
                print('#  orb opt    : {}'.format(self.kw_orb_opt['optimizer']))
            print('#  max iter   : {}'.format(self.kw_orb_opt['max_iter']))
            print('#  stop crit  : {}'.format(self.kw_orb_opt['conv_crit_type']))
            print('#  crit thresh: {}'.format(self.kw_orb_opt['conv_threshold']))
            print('# ...done.')
            print('# ')
            print('# Initialized successfully. Beginning optimization.')
            print('')
            print('### ### ### ### ### ### ### ### ### ### ### ###')
            print('')
        self.built=True
        if 'classical' in self.kw_qc['method']:
            pass
        else:
            self._pre()

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

    def go(self):
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
        self.kw_opt['unity']=self.unity*(1-self.f*0.95)
        #print('Scale factor: {}'.format(180*self.kw_opt['unity']/np.pi))

    def _OptNO(self):
        self.main=Cache()
        key = 'rdm{}'.format(self.total.iter)
        if not self.restart:
            if self.total.iter>0:
                self._set_opt_parameters()
            self.Run[key] = opt.Optimizer(
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
            self.Run[key] = opt.Optimizer(
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

    def single(self,target,para,prop='en'):
        if target=='rdm':
            if prop=='on':
                self.QuantStore.random='on'
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
        self.Run = opt.Optimizer(
                function=f_test,
                **self.kw_opt)
        self.Run.initialize(self.QuantStore.parameters)
            



