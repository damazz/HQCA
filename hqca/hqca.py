'''
main.py 

Main program for executing the hybrid quantum classical optimizer. Consists of
several parts. 

'''
import os, sys
from importlib import reload
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from functools import reduce
import sys
from hqca.sub import VQA,Scan,Circuit
import pickle

version='0.1.2'

def sp(theory,**kwargs):
    '''
    will return a single point energy calculation
    '''
    if theory=='noft':
        return VQA.RunNOFT('noft',**kwargs)
    elif theory=='rdm':
        return VQA.RunRDM('rdm',**kwargs)

class scan:
    '''
    special class for performing scans or more specific analysis of the
    optimization or exploring the parameters in the optimization

    might need to get storage, but maybe not! heh heh we have a good partition
    now
    '''
    def update_rdm(self,para):
        self.run.single('rdm',para)
        #self.Store.update_rdm2()

    def update_full_ints(self,para):
        self.run.single('orb',para)
        self.Store.update_full_ints()


def circuit(theory,**kwargs):
    '''
    basically, method to
    '''
    kwargs['theory']=theory
    return Circuit.Quantum(**kwargs)


