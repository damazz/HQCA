'''
main.py 

Main program for executing the hybrid quantum classical optimizer. Consists of
several parts. 

'''
from abc import ABC, abstractmethod
import os, sys
from importlib import reload
import traceback
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from functools import reduce
import sys
from hqca.sub import VQA,Scan,Circuit,ACSE
import pickle

version='0.2.0'

def sp(theory=None,
        ansatz=None,
        hamiltonian=None,
        **kwargs):
    '''
    will return a single point energy calculation given a particular solver,
    ansatz, and hamiltonian. 

    solvers are,
    '''
    if solver in ['vqa','VQA']:
        return VQA.RunVQA(**kwargs)
    elif solver in ['acse','ACSE']:
        return ACSE.RunACSE(**kwargs)

def scan(**kwargs):
    return Scan.Scan(**kwargs)

def circuit(theory,**kwargs):
    '''
    basically, method to
    '''
    kwargs['theory']=theory
    return Circuit.Quantum(**kwargs)


