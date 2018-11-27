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
            store):
        self.store=store
        self.kw = pre.RDM()
        self.kw_qc = self.kw['qc']
        self.pr = self.kw['prolix']
        if self.pr:
            print('Starting an optimization of the RDM.')




    def update_var(self,args):
        for k,v in args.items():
            self.kw[k]=v
        self.kw_qc = self.kw['qc']

    def go():





class RunNOFT:
    '''
    Subroutine for a natural-orbital function theory approach
    Methods - 
    stretch

    '''
    def __init__(self,
            store
            ):
        self.store=store
        self.kw=pre.NOFT()
        self.total=Cache()
        self.kw_main={}
        self.kw_sub={}
        self.load=False
        self.kw_main = self.kw['main']
        self.kw_sub = self.kw['lsub']
        self.kw_main['store']=self.store
        self.kw_sub['store']=self.store
        self.Np = enf.rotation_parameter_generation(
                store.alpha_mo,
                region=self.kw['sub']['region'],
                output='Npara'
                )
        para_orb = []
        para_orb.append([])
        for i in range(0,2*self.Np):
           para_orb[0].append(0)
        self.para = para_orb
        main_func = enf.find_function('noft','main')
        self.kw_main['function']=main_func
        sub_func = enf.find_function('noft','sub')
        self.kw_sub['function']=sub_func
        self._get_triangle()
        mapping = fx.get_mapping(self.kw_main['wf_mapping'])
        self.kw_main['wf_mapping']=mapping

    def _get_triangle(self):
        if self.kw_main['tri']:
            self.kw_main['triangle']=tri.find_triangle(
                    Ntri=self.kw_main['method_Ntri'],
                    **self.kw_main)


    def update_var(self,args):
        for k,v in args.items():
            self.kw[k]=v

    def go(self):
        while not self.total.done:
            self._OptNO()
            self._OptOrb()
            self._check(self.kw['opt_thresh'],self.kw['max_iter'])


    def _check(self,
            crit,
            max_iter
            ):
        diff = abs(self.main.crit-self.sub.crit)*1000
        print(self.main.crit)
        print(self.sub.crit)
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








