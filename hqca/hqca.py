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
        return VQA.RunNOFT(**kwargs)
    elif theory=='rdm':
        return VQA.RunRDM(**kwargs)

def scan(**kwargs):
    return Scan.Scan(**kwargs)


def circuit(theory,**kwargs):
    '''
    basically, method to
    '''
    kwargs['theory']=theory
    return Circuit.Quantum(**kwargs)


