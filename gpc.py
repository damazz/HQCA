'''
gpc.py

Test module to see if we can reach a non higuchi state using the 1-rdm
tomographytomography
'''
import sys
from math import pi
from tools import RDMFunctions as rdmf
from tools import Functions as fx
from tools.QuantumFramework import evaluate
import numpy
numpy.set_printoptions(precision=5,suppress=True)

keys = {
        'connect':False,
        'verbose':True,
        'backend':'local_qasm_simulator',
        'algorithm':'3qtest',
        'order':'default',
        'num_shots':2048,
        'split_runs':False,
        'tomography':'2rdm'
        }
from random import randint
for i in range(0,1):
    #para = [randint(0,45) for j in range(0,12)]
    para=[10,15,20,30,45,5]
    #para+=para
    #print(para)
    keys['para']=para
    data = evaluate(**keys)
    try:
        print(data.rdm1)
    except:
        pass
    try:
        print(data.rdm1c)
    except:
        pass
    #on = numpy.asarray(on)
    #on.sort()
    #print(on)
    #print(rdm1)
