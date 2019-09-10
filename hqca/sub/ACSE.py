import pickle
import threading
import os, sys
from importlib import reload
import numpy as np
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from hqca.tools import Functions as fx
from hqca.tools import RDMFunctions as rdmf
from hqca.sub.BaseRun import QuantumRun,Cache
from hqca.tools import FunctionsACSE as acse
from hqca.quantum import ErrorCorrection as ec
from hqca.quantum import RDMTomography as tomo
from hqca.quantum import QuantumFunctions as qf
from hqca.quantum import NoiseSimulator as ns
from hqca.tools.util import Errors
from functools import reduce
import datetime
import sys
from hqca.tools import Preset as pre
np.set_printoptions(precision=3,suppress=True)


class RunACSE(QuantumRun):
    '''
    '''
    def __init__(self,
            mol,
            theory='qACSE',
            **kw
            ):
        self.theory=theory
        QuantumRun.__init__(self,**kw)
        kw['mol']=mol
        self.Store = acse.ACSEStorage(**kw)
        if mol==None:
            print('Did you forget to specify a mol object form pyscf?')
            print('Please try again.')
            sys.exit()
        else:
            pass
        self.kw = pre.qACSE()
        self.pr_g = self.kw['pr_g']
        self.kw_qc = self.kw['qc']
        self.total=Cache()

    def build(self):
        QuantumRun._build_quantum(self)
        self._find_2S_matrix()
        self.built=True

    def update_var(self,**kw):
        QuantumRun.update_var(self,**kw)
        self.Store.pr_m = self.kw['pr_m']

    def _pre(self):
        '''
        calcualte S matrix
        '''
        pass

    def _find_2S_matrix(self):
        acse.findSPairs(self.Store,self.QuantStore)
    
    def _build_2S_ansatz(self):
        if self.theory=='acse':
            self.tomo = tomo.Tomography(self.QuantStore)
            self.tomo.build_circuit()
        else:
            print(self.theory)

    def _apply_2S(self):
        self.tomo.run_circuits()
        self.tomo._build_rdms()


    def execute(self):
        self.run()

    def run(self):
        if self.built:
            while not self.total.done:
                self._run_qACSE()
                self.check()

    def _run_qACSE(self):
        '''
        run tomography
        '''
        self._build_2S_ansatz()
        self._apply_2S()
        self._find_2S_matrix()

    def check(self):
        # need to find energy
        self.total.iter+=1 
        if self.total.iter==1:
            self.total.done=True




