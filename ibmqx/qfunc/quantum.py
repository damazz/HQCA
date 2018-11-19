from qiskit import QuantumProgram, QuantumCircuit
import numpy as np
try:
    import rdm
except ImportError:
    from . import rdm
from . import rand
import numpy.linalg as LA
import traceback
import json
import pickle
import pprint
from qiskit.tools.visualization import plot_histogram,plot_state
from qiskit.tools.qi.qi import state_fidelity,concurrence
from qiskit.tools.qi.qi import purity,outer,partial_trace
from qiskit.tools.qcvv import tomography as tomo
import datetime
from qiskit import QISKitError
import timeit

np.set_printoptions(precision=4,linewidth=200)

#
# Connectivity for two qubit gate connectivity for the IBM machines. 
#

tf_ibm_qx2 = {'01':True,'02':True, '12':True, '10':False,'20':False, '21':False}
tf_ibm_qx4 = {'01':False,'02':False, '12':False, '10':True,'20':True, '21':True}

#
# Classes for quantum computation
#


class IBMQXTimeOutError(Exception):
    '''
    Error for timeout in IBM process. Need to end gracefully.
    '''

class SQP:
    '''
     Simple Quantum Programs (Circuit Algorithms)
    
     Class which allows for easy constructions of different types of quantum
     circuits. The operations which obtain desired information for a typical
     IBM circuit are input into this class. 
    ''' 

    def __init__(self,backend_input='local_qasm_simulator'):
        self.qp = QuantumProgram()
        self.backend=backend_input

    def set_api(self,API_token,API_config):
            self.qp.set_api(API_token,API_config)

    def determine_CNOT(self):
        ''' Determines CNOT ordering dependent on the backend.
        '''
        self.d1 = str(self.qorder[0])+str(self.qorder[1])
        self.d2 = str(self.qorder[2])+str(self.qorder[3])
        self.d3 = str(self.qorder[4])+str(self.qorder[5])
        if self.backend=='ibmqx2':
            self.c1 = tf_ibm_qx2[self.d1]
            self.c2 = tf_ibm_qx2[self.d2]
            self.c3 = tf_ibm_qx2[self.d3]
        elif self.backend=='ibmqx4':
            self.c1 = tf_ibm_qx4[self.d1]
            self.c2 = tf_ibm_qx4[self.d2]
            self.c3 = tf_ibm_qx4[self.d3]
        else:
            self.c1 = True
            self.c2 = True
            self.c3 = True

    def create_off_diagonal(self,size=3):
        '''
         Creates off-diagonal circuit. Adds it to the quantum program. 
        '''
        self.qr_err  = self.qp.create_quantum_register("err_qr", size)
        self.cr_err  = self.qp.create_classical_register("err_cr",size)
        self.err     = self.qp.create_circuit('err',[self.qr_err],[self.cr_err])
        #print('Successfully initiated error circuit.')

    def ry6p(self,specify_circuit,parameters,qorder,radians=False):
        ''' 
         Generic 6 parameter algorithm, mainly for simulator. 
        '''
        self.qr_main = self.qp.create_quantum_register("main_qr",3)
        self.cr_main = self.qp.create_classical_register("main_cr",3)
        self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
        if qorder=='default':
            self.qorder = [0,1,1,2,2,0]
        else:
            self.qorder = [int(i) for i in qorder]
        self.determine_CNOT()
        #print(self.c1,self.c2,self.c3)
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        # start with transformations
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[1]])
        if self.c1:
            self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]]) 
        else:
            self.circ.h(self.qreg[self.qorder[0]])
            self.circ.h(self.qreg[self.qorder[1]])
            self.circ.cx(self.qreg[self.qorder[1]],self.qreg[self.qorder[0]])
            self.circ.h(self.qreg[self.qorder[0]])
            self.circ.h(self.qreg[self.qorder[1]])

        self.circ.ry(2*para[2],self.qreg[self.qorder[2]])
        self.circ.ry(2*para[3],self.qreg[self.qorder[3]])
        if self.c2:
            self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]]) 
        else:
            self.circ.h(self.qreg[self.qorder[2]])
            self.circ.h(self.qreg[self.qorder[3]])
            self.circ.cx(self.qreg[self.qorder[3]],self.qreg[self.qorder[2]]) 
            self.circ.h(self.qreg[self.qorder[2]])
            self.circ.h(self.qreg[self.qorder[3]])

        self.circ.ry(2*para[4], self.qreg[self.qorder[4]])
        self.circ.ry(2*para[5], self.qreg[self.qorder[5]])
        if self.c3:
            self.circ.cx(self.qreg[self.qorder[4]],self.qreg[self.qorder[5]]) 
        else:
            self.circ.h(self.qreg[self.qorder[4]])
            self.circ.h(self.qreg[self.qorder[5]])
            self.circ.cx(self.qreg[self.qorder[5]],self.qreg[self.qorder[4]]) 
            self.circ.h(self.qreg[self.qorder[4]])
            self.circ.h(self.qreg[self.qorder[5]])
        if specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[0])
            self.circ.ry(-np.pi/2,self.qreg[1])
            self.circ.ry(np.pi/2,self.qreg[2])

    def ry3p(self,specify_circuit,parameters,qorder,radians=False):
        '''
        # Generic 3 parameter algorithm, mainly for simulator. 
        '''
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",3)
            self.cr_main = self.qp.create_classical_register("main_cr",3)
            self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        #self.qr_main = self.qp.create_quantum_register("main_qr",3)
        #self.cr_main = self.qp.create_classical_register("main_cr",3)
        #self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
        if qorder=='default':
            self.qorder = [0,2,0,1,2,1]
        else:
            self.qorder = [int(i) for i in qorder]
        self.determine_CNOT()
        #print(self.c1,self.c2,self.c3)
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        #print(para)
        #if specify_circuit=='main':
        #    self.circ=self.main
        #    self.qreg=self.qr_main
        #elif specify_circuit=='err':
        #    self.circ=self.err
        #    self.qreg=self.qr_err
        # start with transformations
        #print(para[0],para[1],para[2])
        #print(self.qorder)
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]]) 
        self.circ.ry(2*para[1],self.qreg[self.qorder[2]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        self.circ.ry(2*para[2], self.qreg[self.qorder[4]])
        self.circ.cx(self.qreg[self.qorder[4]],self.qreg[self.qorder[5]])

        if specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[0])
            self.circ.ry(-np.pi/2,self.qreg[1])
            self.circ.ry(np.pi/2,self.qreg[2])

    def ry4p_err_det_sparrow(self,parameters,qorder,radians=False):
        '''
         A diagonal circuit with built in error detection in the form of 3 CNOT gates
         designed for the IBM sparrow machine
        '''
        self.qr_main = self.qp.create_quantum_register("main_qr",5)
        self.cr_main = self.qp.create_classical_register("main_cr",5)
        self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
        if qorder=='default':
            self.qorder = [3,2,4,2]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters 
        self.circ=self.main
        self.qreg=self.qr_main
        self.circ.ry(2*para[0],self.qreg[3])
        self.circ.cx(self.qreg[3],self.qreg[2])
        self.circ.ry(2*para[1],self.qreg[4])
        self.circ.cx(self.qreg[4],self.qreg[2])
        # should be a swap on 2 and 1 
        self.circ.cx(self.qreg[1],self.qreg[2])
        self.circ.h(self.qreg[2])
        self.circ.h(self.qreg[1])
        self.circ.cx(self.qreg[1],self.qreg[2])
        self.circ.h(self.qreg[2])
        self.circ.h(self.qreg[1])
        self.circ.cx(self.qreg[3],self.qreg[2])
        self.circ.cx(self.qreg[4],self.qreg[2])

    def ry6p_sparrow(self,specify_circuit,parameters,qorder,radians=False):
        #
        # Circuit designed for the IBM Sparrow device, has 6 parameters. 
        #
        if qorder=='default':
            self.qorder = [3,2,3,4,4,2]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",5)
            self.cr_main = self.qp.create_classical_register("main_cr",5)
            self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[1]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        self.circ.ry(2*para[2],self.qreg[self.qorder[2]])
        self.circ.ry(2*para[3],self.qreg[self.qorder[3]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        self.circ.ry(2*para[4], self.qreg[self.qorder[4]])
        self.circ.ry(2*para[5], self.qreg[self.qorder[5]])
        self.circ.cx(self.qreg[self.qorder[4]],self.qreg[self.qorder[5]])
        if specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[2])
            self.circ.ry(-np.pi/2,self.qreg[3])
            self.circ.ry(np.pi/2,self.qreg[4])

    def ry3p_sparrow(self,specify_circuit,parameters,qorder,radians=False):
        #
        # Circuit designed for the IBM Sparrow device, has 3 parameters. 
        # Generates a diagonal 1-RDM, and so mainly measures the off-diagonal.
        # 
        if qorder=='default':
            self.qorder = [3,4,3,2,4,2]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",5)
            self.cr_main = self.qp.create_classical_register("main_cr",5)
            self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        # start with transformations
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[2]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        self.circ.ry(2*para[2], self.qreg[self.qorder[4]])
        self.circ.cx(self.qreg[self.qorder[4]],self.qreg[self.qorder[5]])
        if specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[2])
            self.circ.ry(-np.pi/2,self.qreg[3])
            self.circ.ry(np.pi/2,self.qreg[4])

    def ry2p_sparrow(self,specify_circuit,parameters,qorder,radians=False):
        #
        # Circuit designed for the IBM Sparrow device, ibmqx2, and has 2 parameters. 
        # Generates a diagonal 1-RDM, and so mainly measures the off-diagonal.
        # 
        if qorder=='default':
            self.qorder = [3,2,4,2]
        elif qorder=='alt':
            self.qorder = [1,2,0,1]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",5)
            self.cr_main = self.qp.create_classical_register("main_cr",5)
            self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[2]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        if specify_circuit=='err' and qorder=='alt':
            self.circ.ry(np.pi/2,self.qreg[0])
            self.circ.ry(-np.pi/2,self.qreg[1])
            self.circ.ry(np.pi/2,self.qreg[2])
        elif specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[2])
            self.circ.ry(-np.pi/2,self.qreg[3])
            self.circ.ry(np.pi/2,self.qreg[4])

    def ry2p_raven(self,specify_circuit,parameters,qorder,radians=False):
        #
        # Circuit designed for the IBM Raven device, ibmqx4, and has 2 parameters. 
        # Generates a diagonal 1-RDM, and so mainly measures the off-diagonal.
        # 
        if qorder=='default':
            self.qorder = [1,0,2,1]
        elif qorder=='alt': # worse error in current calibration
            self.qorder = [3,2,4,2]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",5)
            self.cr_main = self.qp.create_classical_register("main_cr",5)
            self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[2]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        if specify_circuit=='err' and qorder=='alt':
            self.circ.ry(np.pi/2,self.qreg[2])
            self.circ.ry(-np.pi/2,self.qreg[3])
            self.circ.ry(np.pi/2,self.qreg[4])
        elif specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[0])
            self.circ.ry(-np.pi/2,self.qreg[1])
            self.circ.ry(np.pi/2,self.qreg[2])


    def ry2p(self,specify_circuit,parameters,qorder,radians=False):
        #
        # 
        if qorder=='default':
            self.qorder = [0,1,2,1]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",3)
            self.cr_main = self.qp.create_classical_register("main_cr",3)
            self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err

        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[2]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        if specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[0])
            self.circ.ry(-np.pi/2,self.qreg[1])
            self.circ.ry(np.pi/2,self.qreg[2])

    def ry4p(self,specify_circuit,parameters,qorder,radians=False):
        if qorder=='default':
            self.qorder = [3,4,3,2,4,2]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",5)
            self.cr_main = self.qp.create_classical_register("main_cr",5)
            self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        # start with transformations
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[1]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        self.circ.ry(2*para[2],self.qreg[self.qorder[2]])
        self.circ.ry(2*para[3],self.qreg[self.qorder[3]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        if specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[2])
            self.circ.ry(-np.pi/2,self.qreg[3])
            self.circ.ry(np.pi/2,self.qreg[4])

    def ry1p_edge_sparrow(self,specify_circuit,parameters,qorder,radians=False):
        if qorder=='default':
            self.qorder = [1,2]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",3)
            self.cr_main = self.qp.create_classical_register("main_cr",3)
            self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        # start with transformations
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        if specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[0])
            self.circ.ry(-np.pi/2,self.qreg[1])
            self.circ.ry(np.pi/2,self.qreg[2])

    def ry2p_face1_sparrow(self,specify_circuit,parameters,qorder,radians=False):
        if qorder=='default':
            self.qorder = [0,2,0,1]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",3)
            self.cr_main = self.qp.create_classical_register("main_cr",3)
            self.main    = self.qp.create_circuit('main',[self.qr_main],[self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        # start with transformations
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[2]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        if specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[0])
            self.circ.ry(-np.pi/2,self.qreg[1])
            self.circ.ry(np.pi/2,self.qreg[2])

    def ry2p_face2_sparrow(self,specify_circuit,parameters,qorder,radians=False):
        if qorder=='default':
            self.qorder = [1,2,0,1]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",3)
            self.cr_main = self.qp.create_classical_register("main_cr",3)
            self.main    = self.qp.create_circuit(
                    'main',
                    [self.qr_main],
                    [self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        # start with transformations
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[2]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        if specify_circuit=='err':
            self.circ.ry(np.pi/2,self.qreg[0])
            self.circ.ry(-np.pi/2,self.qreg[1])
            self.circ.ry(np.pi/2,self.qreg[2])

    def four_qubit_test(
            self,
            parameters,
            qorder,
            specifiy_circuit='local_qasm_simulator',
            radians=False
            ):
        if qorder=='default':
            self.qorder=[0,1,1,2,2,3,3,0]
        else:
            self.qorder = [int(i) for i in qorder]
        if radians==False:
            para = []
            for par in parameters:
                para.append(par*np.pi/180)
        else:
            para = parameters
        if specify_circuit=='main':
            self.qr_main = self.qp.create_quantum_register("main_qr",4)
            self.cr_main = self.qp.create_classical_register("main_cr",4)
            self.main    = self.qp.create_circuit(
                    'main',
                    [self.qr_main],
                    [self.cr_main])
            self.circ=self.main
            self.qreg=self.qr_main
        elif specify_circuit=='err':
            self.circ=self.err
            self.qreg=self.qr_err
        # start with transformations
        self.circ.ry(2*para[0],self.qreg[self.qorder[0]])
        self.circ.ry(2*para[1],self.qreg[self.qorder[1]])
        self.circ.cx(self.qreg[self.qorder[0]],self.qreg[self.qorder[1]])
        self.circ.ry(2*para[2],self.qreg[self.qorder[2]])
        self.circ.ry(2*para[3],self.qreg[self.qorder[3]])
        self.circ.cx(self.qreg[self.qorder[2]],self.qreg[self.qorder[3]])
        self.circ.ry(2*para[4],self.qreg[self.qorder[4]])
        self.circ.ry(2*para[5],self.qreg[self.qorder[5]])
        self.circ.cx(self.qreg[self.qorder[4]],self.qreg[self.qorder[5]])
        self.circ.ry(2*para[6],self.qreg[self.qorder[6]])
        self.circ.ry(2*para[7],self.qreg[self.qorder[7]])
        self.circ.cx(self.qreg[self.qorder[6]],self.qreg[self.qorder[7]])
        if specify_circuit=='err':
            self.circ.ry(-np.pi/2,self.qreg[0])
            self.circ.ry(np.pi/2,self.qreg[1])
            self.circ.ry(-np.pi/2,self.qreg[2])
            self.circ.ry(np.pi/2,self.qreg[3])

    def measure(self,specify_circ, measurements):
        # 
        # Begins the measurement procedure. 
        #
        if specify_circ=='main':
            self.circ=self.main
            self.qreg=self.qr_main
            self.creg=self.cr_main
        elif specify_circ=='err':
            self.circ=self.err
            self.qreg=self.qr_err
            self.creg=self.cr_err
        for m in measurements:
            self.circ.measure(self.qreg[m],self.creg[m])


    def execute(
            self,
            num_shots,
            nmeasure,
            verbose=False
            ):
        try:
            self.err
            self.circuits = ['main','err']
        except:
            self.circuits = ['main']
        if self.backend=='local_unitary_simulator':
            pass
        else:
            for circ in self.circuits:
                self.measure(circ,nmeasure)
        self.qobj = self.qp.compile(
                self.circuits,shots=num_shots,
                backend=self.backend)
        self.id = self.qobj['id']
        self.results = self.qp.run(
                self.qobj,wait=5,
                timeout=3600,silent=not(verbose))
        return self.results,self.circuits

    def calc_ideal(self):
        self.unit_results = self.qp.execute(
                'main',backend='local_unitary_simulator')
        self.unit_transformation = self.unit_results.get_data(
                'main')['unitary']
        try:
            self.unit_results_err = self.qp.execute(
                    'err',backend='local_unitary_simulator')
            self.unit_transformation_err = self.unit_results_err.get_data(
                    'err')['unitary']
        except:
            pass
        self.unit_state = self.unit_transformation[:,0]
    
    def get_ideal(self):
        try:
            self.unit_results
        except:
            self.calc_ideal()
        try:
            return [self.unit_transformation,self.unit_transformation_err]
        except:
            return [self.unit_transformation]


    def tomography(self,num_shots=1024,rdms=True,
            pfidelity=False,ppurity=False):
        # gets the rho, or density matrix from the state
        self.calc_ideal()
        #self.main.ry(2*np.pi,self.qr_main[1])
        self.tom_circ = tomo.build_state_tomography_circuits(
                self.qp,'main',[0,1,2],
                self.qr_main,self.cr_main)
        self.tomo_results = self.qp.execute(
                self.tom_circ,shots=num_shots,
                backend=self.backend,
                wait=5,timeout=3600,silent=True) ## tag for silent Silent
        self.data = tomo.state_tomography_data(
                self.tomo_results,'main',[0,1,2])
        self.rho = tomo.fit_tomography_data(self.data)
        self.purity = purity(self.rho)
        self.fidelity = state_fidelity(self.rho,self.unit_state)
        if pfidelity:
            print('State fidelity: ', self.fidelity)
        if ppurity:
            print('State purity: ', self.purity)
        self.rdm_1 = partial_trace(self.rho,[1,2])
        self.rdm_2 = partial_trace(self.rho,[0,2])
        self.rdm_3 = partial_trace(self.rho,[0,1])
        if rdms:
            return self.rho, [self.rdm_1,self.rdm_2,self.rdm_3]
        else:
            return self.rho

class Quant_Algorithms:
    def __init__(self):
        #print('Starting a quantum algorithm.')
        pass


    def run(self,algorithm,parameters,
            qorder,nqubits,num_shots,
            API_token,API_config,
            backend_input='local_qasm_simulator',
            connect=True,verbose=False):
        self.circuit = SQP(backend_input)
        if connect:
            self.circuit.set_api(API_token,API_config)
        self.num_shots = num_shots
        self.n_qb = nqubits
        self.backend = backend_input
        self.verbose=verbose
        if algorithm=='ry6p':
            self.circuit.create_off_diagonal()
            self.circuit.ry6p('main',parameters,qorder)
            self.circuit.ry8p('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [0,1,2]
        elif algorithm=='ry3p':
            self.circuit.create_off_diagonal(size=3)
            self.circuit.ry3p('main',parameters,qorder)
            self.circuit.ry3p('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [0,1,2]
        elif algorithm=='ry4p_err_det_sparrow':
            self.circuit.ry4p_err_det_sparrow(parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [0,1,2,3,4]
        elif algorithm=='ry6p_sparrow':
            self.circuit.create_off_diagonal(size=5)
            self.circuit.ry6p_sparrow('main',parameters,qorder)
            self.circuit.ry6p_sparrow('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [2,3,4]
        elif algorithm=='ry3p_sparrow':
            self.circuit.create_off_diagonal(size=5)
            self.circuit.ry3p_sparrow('main',parameters,qorder)
            self.circuit.ry3p_sparrow('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [2,3,4]
        elif algorithm=='ry4p':
            self.circuit.create_off_diagonal(size=5)
            self.circuit.ry4p('main',parameters,qorder)
            self.circuit.ry4p('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [2,3,4]
        elif algorithm=='ry2p':
            #self.circuit.create_off_diagonal(size=3)
            self.circuit.ry2p('main',parameters,qorder)
            #self.circuit.ry2p('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [0,1,2]
        elif algorithm=='ry2p_sparrow_diag':
            self.circuit.ry2p_sparrow('main',parameters,qorder)
            self.qorder = self.circuit.qorder
            if qorder=='default':
                self.nmeasure = [2,3,4]
            elif qorder=='alt':
                self.nmeasure = [0,1,2]
        elif algorithm=='ry2p_sparrow':
            self.circuit.create_off_diagonal(size=5)
            self.circuit.ry2p_sparrow('main',parameters,qorder)
            self.circuit.ry2p_sparrow('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            if qorder=='default':
                self.nmeasure = [2,3,4]
            elif qorder=='alt':
                self.nmeasure = [0,1,2]
        elif algorithm=='ry2p_raven_diag':
            self.circuit.ry2p_raven('main',parameters,qorder)
            self.qorder = self.circuit.qorder
            if qorder=='alt':
                self.nmeasure = [2,3,4]
            elif qorder=='default':
                self.nmeasure = [0,1,2]
        elif algorithm=='ry2p_raven':
            self.circuit.create_off_diagonal(size=5)
            self.circuit.ry2p_raven('main',parameters,qorder)
            self.circuit.ry2p_raven('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            if qorder=='alt':
                self.nmeasure = [2,3,4]
            elif qorder=='default':
                self.nmeasure = [0,1,2]
        elif algorithm=='ry1p_edge_sparrow':
            self.circuit.create_off_diagonal(size=3)
            self.circuit.ry1p_edge_sparrow('main',parameters,qorder)
            self.circuit.ry1p_edge_sparrow('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [0,1,2]
        elif algorithm=='ry2p_face1_sparrow':
            self.circuit.create_off_diagonal(size=3)
            self.circuit.ry2p_face1_sparrow('main',parameters,qorder)
            self.circuit.ry2p_face1_sparrow('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [0,1,2]
        elif algorithm=='ry2p_face2_sparrow':
            self.circuit.create_off_diagonal(size=3)
            self.circuit.ry2p_face2_sparrow('main',parameters,qorder)
            self.circuit.ry2p_face2_sparrow('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [0,1,2]
        elif algorithm=='four_qubit_test':
            self.circuit.create_off_diagonal(size=4)
            self.circuit.ry2p_face2_sparrow('main',parameters,qorder)
            self.circuit.ry2p_face2_sparrow('err',parameters,qorder)
            self.qorder = self.circuit.qorder
            self.nmeasure = [0,1,2,3]


    def start(self,tomography=False):
        try:
            if tomography==True:
                self.rho,  self.rdms = self.circuit.tomography(
                        self.num_shots)
            elif tomography=='both':
                self.rho, self.rdms = self.circuit.tomography(
                        self.num_shots)
                self.results, self.circs = self.circuit.execute(
                        self.num_shots,
                        nmeasure=self.nmeasure,
                        verbose=self.verbose)
            elif tomography==False:
                self.results, self.circs = self.circuit.execute(
                        self.num_shots,
                        nmeasure=self.nmeasure,
                        verbose=self.verbose)
        except Exception:
            traceback.print_exc()


class Run_Types:
    #
    # Class of objects which will generate runs, when given the run type 
    # and a list of parameters. For a given run type, 
    #
    #
    #
    def __init__(self,run_type,parameters):
        #print('Generating run types.')
        self.para = parameters
        self.run_type = run_type
        self.runs = []

    def span_range(self):
        self.hold_para = []
        self.num_par = 1
        for item in self.para:
            par_range = np.arange(item[0],item[1]+item[2]*0.5,item[2]).tolist()
            self.hold_para.append(par_range)
            self.num_par*= len(par_range)
        self.runs = rand.recursive(self.hold_para,0,[],self.num_par,[])

    def user_specify(self):
        for item in self.para:
            self.runs.append(item)

    def determine(self):
        #print('Determining the run type...')
        if self.run_type=='single':
            self.runs.append(self.para[0])
        elif self.run_type=='range':
            #print('Generateing a range of parameters')
            self.span_range()
        elif self.run_type=='user':
            self.user_specify()
        return self.runs


# Begin executing functions


def combine_dict(one,two):
    for key,val in two.items():
        try:
            one[key] = int(one[key])+int(val)
        except:
            one[key] = int(val)
    return one

def Run(run_type,
        algorithm,
        parameters,
        qorders,
        n_qubits,
        num_shots,
        API_token,API_config,
        name,
        backend='local_qasm_simulator',
        tomography=False,
        combine=True,
        ibm_connect=True,
        verbose=False):
    print('Beginning of run.')
    '''
     Function to execute the result. Needs the following inputs:
       -run_type:
       -algorithm:
       -parameters:
       -qorders:
       -n_qubits:
       -num_shots:
       -API_token:
       -API_config:
       -name:
       -backend:

     Other optional inputs are:
       -tomography:
       -combine: whether or not to combine the results of several different runs

    First, the parameters and run type define the circuits which are
    constructed. Then, each run is stored as a dictionary type which is
    accessible regardless of the backend type. Raw data can also be accessed in
    this format.


    '''
    run_list = Run_Types(run_type,parameters).determine()
    runs = []
    if combine==False:
        n_runs=1
    elif combine==True:
        n_runs = max(num_shots//1024,1)
        num_shots=1024
    for run in run_list:
        # Each unique run is stored in the array as a dict object
        # with a lots of informations
        run_item = {}
        run_item['algorithm'] = algorithm
        run_item['backend']=backend
        run_item['combined']=combine
        run_item['parameters']=run
        run_item['data']=[]
        run_item['id'] = 'null'
        for n in range(0,n_runs):
            # The data key leads to an array, where each run is 
            # stored as a dict object
            temp = {}
            temp['shots']=num_shots
            temp['run']=n
            circuit_run = Quant_Algorithms()
            tic = timeit.default_timer()
            circuit_run.run(
                    algorithm,run,qorders,
                    n_qubits,num_shots,API_token,
                    API_config,backend,
                    connect=ibm_connect,verbose=verbose)
            try: 
                circuit_run.start(tomography)
            except Exception as e:
                print('Got a QISKit error!')
                traceback.print_exc()
                toc = timeit.default_timer()
                if toc-tic > 3600:
                    raise IBMQXTimeOutError
                else:
                    raise QISKitError
            temp['status'] = circuit_run.results.__str__()
            temp['qobj']=circuit_run.circuit.qobj
            if (temp['status']=='DONE' 
                    or temp['status']=='Done' 
                    or temp['status']=='COMPLETED'):
                for circ in circuit_run.circs:
                    if verbose:
                        print('Circuit counts for {}:'.format(circ))
                        print(circuit_run.results.get_data(circ))
                    if backend=='local_unitary_simulator':
                        test = circuit_run.results.get_data(circ)['unitary']
                        print(test)
                        temp_res = rdm.U_to_counts(test,9)
                    else:
                        temp_res = circuit_run.results.get_data(circ)['counts']
                    temp['{}-counts'.format(circ)] = temp_res
            else:
                pass
            run_item['data'].append(temp)
        for circ in circuit_run.circs:
            run_item['total-counts-{}'.format(circ)]={}
            for n in range(0,n_runs):
                run_item['total-counts-{}'.format(circ)] = combine_dict(run_item['data'][n]['{}-counts'.format(circ)],run_item['total-counts-{}'.format(circ)])
        run_item['order']=circuit_run.qorder
        if verbose:
            for circ in circuit_run.circs:
                print('Results for circuit: {}\n{}'.format(circ,run_item['total-counts-{}'.format(circ)]))
        #
        # Formatting the log file...not really important, as the ID can't 
        # actually be obtained from IBM without explicitly calling the API 
        # or something    
        #
        try:
            jobs_loc = './ibmqx/jobs.txt'
            test_jobs = open(jobs_loc)
        except FileNotFoundError:
            jobs_loc = './jobs.txt'
        with open(jobs_loc,'r') as fp:
            data = fp.readlines()
        lastline = data[-1]
        last = lastline.split(' ')
        try:
            ind = int(last[0])
        except:
            ind = -1
        ind += 1
        ''''
        with open(jobs_loc,'a+') as fp:
            fp.write(
            '{} {} {} {} {} \n'.format(
                str(ind).zfill(5),
                run_item['id'],
                run_item['data'][-1]['status'],
                name,
                datetime.datetime.now().strftime('%m%d%y')
                )
            )
        '''
        #
        #
        # 
        runs.append(run_item)
    print('End of run')
    return runs

