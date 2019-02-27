'''
./sub.py

Holds the subroutines for the optimizations.

'''
import pickle
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
from hqca.tools import Triangulation as tri
from hqca.tools import QuantumFunctions as qf
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


class RunRDM:
    '''
    Subroutine for a RDM based approach. Orbital optimization is included in the
    wavefunction.
    '''
    def __init__(self,
            store,
            **kw
            ):
        self.Store=store
        self.Store.sp='rdm'
        self.kw = pre.RDM()
        self.kw_qc = self.kw['qc']
        self.kw_opt = self.kw['opt']
        self.rc=Cache()
        self.built=False


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

    def build(self):
        self.pr_g = self.Store.pr_g
        self.Store.pr_m = self.kw['pr_m']
        if self.pr_g>0:
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
        if self.pr_g>0:
            print('# ...done.')
            print('#')
            print('# Setting QC parameters...')

        self.QuantStore = qf.QuantumStorage(self.pr_g,**self.kw_qc)
        if self.pr_g>0:
            print('# ...done.')
            print('#')
            print('# Setting opt parameters...')
        self.Store.update_full_ints()

        self.kw_opt['function'] = enf.find_function(
                'rdm',
                'main',
                self.Store,
                self.QuantStore)
        if self.pr_g>0:
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
        if self.kw_qc['pr_q']>1:
            qf.get_direct_stats(self.QuantStore)
        self.built=True

    def go(self):
        if self.built:
            self._OptRDM()
            self._analyze()
        else:
            sys.exit('# Not built yet! Run build() before execute(). ')

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
                print('Step: {:02}, Total Energy: {:.8f} Sigma: {:.8f}'.format(
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

    def set_print(self,level='default',
            record=False
            ):
        self.kw_qc['pr_q']=1
        self.kw_opt['pr_o']=1
        self.kw['pr_m']=1
        self.kw['pr_g']=2
        if level=='stats':
            self.kw_qc['pr_q']=2
        elif level=='min':
            self.kw['pr_g']=1
        elif level=='none':
            self.kw_qc['pr_q']=0
            self.kw_opt['pr_o']=0
            self.kw['pr_m']=0
            self.kw['pr_g']=0
        elif level=='analysis':
            self.kw['pr_g']=4
        elif level=='diagnostic_full':
            self.kw_qc['pr_q']=9
            self.kw_opt['pr_o']=9
            self.kw['pr_m']=9
            self.kw['pr_g']=9
        elif level=='diagnostic_en':
            self.kw['pr_g']=4
        elif level=='diagnostic_qc':
            self.kw_qc['pr_q']=9
        elif level=='diagnostic_opt':
            self.kw_opt['pr_o']=9

class RunNOFT:
    '''
    Subroutine for a natural-orbital function theory approach
    stretch
    '''
    def __init__(self,
            store
            ):
        self.store=store
        self.store.sp='noft'
        self.store.gas()
        self.store.gsm()
        self.store.gip()
        self.store.update_full_ints()
        self.kw=pre.NOFT()
        self.total=Cache()
        self.kw_main={}
        self.kw_sub={}
        self.load=False
        self.kw_main = self.kw['main']
        self.kw_sub = self.kw['sub']
        self.kw_main['store']=self.store
        self.kw_sub['store']=self.store
        self.Np = enf.rotation_parameter_generation(
                store.alpha_mo,
                region=self.kw['sub']['region'],
                output='Npara'
                )
        main_func = enf.find_function('noft','main')
        self.kw_main['function']=main_func
        sub_func = enf.find_function('noft','sub')
        self.kw_sub['function']=sub_func
        mapping = fx.get_mapping(self.kw_main['wf_mapping'])
        self.kw_main['wf_mapping']=mapping
        self.pr_g = self.kw['pr_g']
        self.store.kw['spin_mapping']=self.kw_sub['spin_mapping']

    def _get_triangle(self):
        if self.kw_main['tri']:
            self.kw_main['triangle']=tri.find_triangle(
                    Ntri=self.kw_main['method_Ntri'],
                    **self.kw_main)


    def update_var(
            self,
            main=False,
            sub=False,
            **args):
        if (main and (not sub)):
            for k,v in args.items():
                self.kw_main[k]=v
            mapping = fx.get_mapping(self.kw_main['wf_mapping'])
            if mapping==None:
                pass
            else:
                self.kw_main['wf_mapping']=mapping
        elif ((not main) and sub):
            for k,v in args.items():
                self.kw_sub[k]=v
            self.store.kw['spin_mapping']=self.kw_sub['spin_mapping']
        elif not (main and sub):
            for k,v in args.items():
                self.kw[k]=v
            self.pr_g = self.kw['pr_g']


    def go(self):
        para_orb = []
        para_orb.append([])
        if self.kw_sub['spin_mapping']=='restricted':
            for i in range(0,self.Np):
                para_orb[0].append(0)
        elif self.kw_sub['spin_mapping']=='unrestricted':
            for i in range(0,2*self.Np):
                para_orb[0].append(0)
        self.para = para_orb
        if 'classical' in self.kw_main['method']:
            pass
        else:
            self._get_triangle()
        while not self.total.done:
            self._OptNO()
            self._OptOrb()
            self._check(self.kw['opt_thresh'],self.kw['max_iter'])


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
            self.total.err = True
            self.total.done= True
            print(self.main.msg)
            print(self.sub.msg)
        elif abs(self.main.crit-self.sub.crit)<crit:
            self.total.done=True
        elif self.total.iter>=max_iter:
            self.total.done=True
            self.total.err=True
        else:
            pass
        if self.total.err:
            print('Got an error.')
        self.main.crit=0
        self.sub.crit=0
        if self.pr_g>2:
            self.store.opt_analysis()
        self.total.iter+=1



    def _OptNO(self):
        self.main=Cache()
        if self.load==True:
            pass
        else:
            Run = opt.Optimizer(
                    parameters=[self.kw_main['store'].parameters],
                    **self.kw_main
                    )
            para = []
        if Run.error:
            print('##########')
            print('Encountered error in initialization.')
            print('##########')
            self.main.done=True
        self.main.done=False
        while not self.main.done:
            Run.next_step(**self.kw_main)
            if self.kw_main['pr_o']>0:
                print('Step: {:02}, Total Energy: {:.8f} Sigma: {:.8f}  '.format(
                    self.main.iter,
                    Run.opt.best_f,
                    Run.opt.crit)
                    )
                    #if self.kw_main['optimizer']=='NM':
                    #    print(Run.opt.B_x)
            Run.check(self.main)
            if self.main.iter==self.kw_main['max_iter'] and not(self.main.done):
                self.main.error=False
                Run.opt_done=True
                self.main.done=True
            elif Run.opt_done:
                if Run.error:
                    print('Error in run.')
                    Store.opt_done=True
                continue
            self.main.iter+=1
        self.store.update_rdm2()

    def _OptOrb(self):
        self.sub=Cache()
        self.sub.done=False
        if self.load==True:
            pass
        else:
            Run = opt.Optimizer(
                    parameters=self.para,
                    **self.kw_sub
                    )
        while not self.sub.done:
            Run.next_step(**self.kw_sub)
            if self.kw_sub['pr_o']>0:
                print('Step: {:02}, Total Energy: {:.8f} Sigma: {:.8f}  '.format(
                    self.sub.iter,
                    Run.opt.best_f,
                    Run.opt.crit)
                    )
            Run.check(self.sub)
            if self.sub.iter==self.kw_sub['max_iter'] and not self.main.done:
                self.sub.done=True
                self.sub.err=True
                Run.opt_done=True
            self.sub.iter+=1
        self.store.update_full_ints()








