from qiskit import register, available_backends, get_backend
from qiskit import execute
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit
from math import pi


#README
'''
Please note, if an algorithm is updated or added, it should be added in 3
places, or completely specified via input. 

First, there are two places in the QuantumAlgorthims module:
1.  GenerateCircuit class: circuit is actually written down
2.  algorithm_tomography dictionary: need to update standard parameters that 
    other module will try to fetch.

Finally, in the function module:
1.  counts_to_1rdm function: update the standard trace order (note, this should
    be updated), or just include the algorithm in the function 

'''
tf_ibm_qx2 = {'01':True,'02':True, '12':True, '10':False,'20':False, '21':False}
tf_ibm_qx4 = {'01':False,'02':False, '12':False, '10':True,'20':True, '21':True}
# note, qx4 is 'raven', or 'tenerife'

def read_qasm(input_qasm):
    pass


class GenerateCircuit:
    '''
    Class for initializing quantum algorithms. Has the quantum circuit
    attribute 
    '''
    def __init__(
            self,
            para,
            algorithm,
            order='default',
            _name=False,
            verbose=False,
            **kwargs
            ):
        #try:
        #    print('You might need to clean this kwarg up:')
        #    print(kwargs)
        #except Exception:
        #    pass
        self.qa = algorithm
        self.Nq = algorithm_tomography[self.qa]['Nq']
        self.q = QuantumRegister(self.Nq,name='q')
        self.c = ClassicalRegister(self.Nq,name='c')
        if _name==False:
            self.qc = QuantumCircuit(self.q,self.c)
        else:
            self.qc = QuantumCircuit(self.q,self.c,name=_name)
        self.p  = []
        for i in para:
            self.p.append(i*pi/90)
        self.apply_algorithm(order)

    def apply_algorithm(self,order='default'):
        ''' 
        List of algorithms for use with quantum computer. Reminder to read
        the top of this module before adding or removing algorithms.
        '''
        if self.qa=='affine_2p_curved_tenerife':
            if order=='default':
                o = [0,2,1,0]
            else:
                o = [int(i) for i in order]
            self.qc.ry(self.p[0],self.q[o[0]])
            self.qc.cx(self.q[o[0]],self.q[o[1]])
            self.qc.ry(self.p[1],self.q[o[2]])
            self.qc.cx(self.q[o[2]],self.q[o[3]])
        elif self.qa=='affine_2p_flat_tenerife':
            if order=='default':
                o = [2,1,1,0]
            else:
                o = [int(i) for i in order]
            self.qc.ry(self.p[0],self.q[o[0]])
            self.qc.cx(self.q[o[0]],self.q[o[1]])
            self.qc.ry(self.p[1],self.q[o[2]])
            self.qc.cx(self.q[o[2]],self.q[o[3]])
        elif self.qa=='affine_2p_flatfish_tenerife':
            if order=='default':
                o = [0,2,2,1]
            else:
                o = [int(i) for i in order]
            self.qc.ry(self.p[0],self.q[o[0]])
            self.qc.ry(self.p[1],self.q[o[1]])
            self.qc.cx(self.q[o[0]],self.q[o[1]])
            self.qc.cx(self.q[o[2]],self.q[o[3]])
        elif self.qa=='4qtest':
            if order=='default':
                o = [0,1,0,2,0,3,1,2,1,3,2,3]
            else:
                o = [int(i) for i in order]
            self.qc.ry(self.p[0],self.q[o[0]])
            self.qc.ry(self.p[1],self.q[o[1]])
            self.qc.cx(self.q[o[0]],self.q[o[1]])
            self.qc.ry(self.p[2],self.q[o[2]])
            self.qc.ry(self.p[3],self.q[o[3]])
            self.qc.cx(self.q[o[2]],self.q[o[3]])
            self.qc.ry(self.p[4],self.q[o[4]])
            self.qc.ry(self.p[5],self.q[o[5]])
            self.qc.cx(self.q[o[4]],self.q[o[5]])
            self.qc.ry(self.p[6],self.q[o[6]])
            self.qc.ry(self.p[7],self.q[o[7]])
            self.qc.cx(self.q[o[6]],self.q[o[7]])
            self.qc.ry(self.p[8],self.q[o[8]])
            self.qc.ry(self.p[9],self.q[o[9]])
            self.qc.cx(self.q[o[8]],self.q[o[9]])
            self.qc.ry(self.p[10],self.q[o[10]])
            self.qc.ry(self.p[11],self.q[o[11]])
            self.qc.cx(self.q[o[10]],self.q[o[11]])
        elif self.qa=='3qtest':
            if order=='default':
                o = [0,2,0,1,2,1]
            else:
                o = [int(i) for i in order]
            self.qc.ry(self.p[0],self.q[o[0]])
            self.qc.ry(self.p[1],self.q[o[1]])
            self.qc.cx(self.q[o[0]],self.q[o[1]])
            self.qc.ry(self.p[2],self.q[o[2]])
            self.qc.ry(self.p[3],self.q[o[3]])
            self.qc.cx(self.q[o[2]],self.q[o[3]])
            self.qc.ry(self.p[4],self.q[o[4]])
            self.qc.ry(self.p[5],self.q[o[5]])
            self.qc.cx(self.q[o[4]],self.q[o[5]])
        else:
            pass

algorithm_tomography = {
        'affine_2p_curved_tenerife':{
            'tomo':'d1rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2]
            },
        'affine_2p_flat_tenerife':{
            'tomo':'1rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2]
            },
        'affine_2p_flatfish_tenerife':{
            'tomo':'1rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2]
            },
        '4qtest':{
            'tomo':'2rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2,3]
            },
        '3qtest':{
            'tomo':'2rdm',
            'Nq':5,
            'qb_to_orb':[0,1,2]
            }
        }

