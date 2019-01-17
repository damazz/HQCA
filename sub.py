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
        self.store=store
        self.store.sp='rdm'
        self.kw = pre.RDM()
        self.rc=Cache()
        self.kw_qc = self.kw['qc']
        self.kw_qc['store']=self.store
        self.pr_g = self.kw['pr_g']
        qc_func = enf.find_function('rdm','main')
        self.kw_qc['function']=qc_func
        if self.pr_g>0:
            print('Starting an optimization of the RDM.')
        self.store.kw['entangled_pairs']=self.kw_qc['entangled_pairs']
        self.store.kw['spin_mapping']=self.kw_qc['spin_mapping']



    def update_var(
            self,
            qc=False,
            **args):
        if not qc:
            for k,v in args.items():
                self.kw[k]=v
        elif qc:
            for k,v in args.items():
                self.kw_qc[k] = v
            self.store.kw['entangled_pairs']=self.kw_qc['entangled_pairs']
            self.store.kw['spin_mapping']=self.kw_qc['spin_mapping']

    def go(self):
        self.store.gas()
        self.store.gsm()
        self.store.gip()
        self.store.update_full_ints()
        self._OptRDM()
        self._analyze()

    def _OptRDM(self,
            **kw
            ):
        if self.kw['restart']==True:
            self._load()
        else:
            Run = opt.Optimizer(
                    parameters=[self.kw_qc['store'].parameters],
                    **self.kw_qc
                    )
        if Run.error:
            print('##########')
            print('Encountered error in initialization.')
            print('##########')
            self.rc.done=True
        while not self.rc.done:
            Run.next_step(**self.kw_qc)
            print('Step: {:02}, Total Energy: {:.8f} Sigma: {:.8f}  '.format(
                self.rc.iter,
                Run.opt.best_f,
                Run.opt.crit)
                )
            Run.check(self.rc)
            if self.rc.iter==self.kw_qc['max_iter'] and not(self.rc.done):
                self.rc.error=False
                Run.opt_done=True
                self.rc.done=True
            elif Run.opt_done:
                if Run.error:
                    print('Error in run.')
                    Store.opt_done=True
                continue
            self.rc.iter+=1
        self.store.update_rdm2()

    def _load(self):
        pass

    def _analyze(self,
            ):
        if self.rc.err:
            self.rc.done= True
            print(self.rc.msg)
            print(self.rc.msg)
        print('done!')





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
            #filename = pre.filename
            #with open(pre.filename+'.run.tmp', 'wb') as fp:
            #    pickle.dump(
            #            [Run,Store,keys,orb_keys],
            #            fp,
            #            pickle.HIGHEST_PROTOCOL
            #            )
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








