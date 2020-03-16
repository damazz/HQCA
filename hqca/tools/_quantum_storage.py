'''
Contains QuantumStorage and other useful quantum functions. Basically,
QuantumStorage is a class which is needed for any quantum calculation on the QC.
Generates important information to that end.

'''

import numpy as np
np.set_printoptions(suppress=True,precision=4)
import pickle
import sys
from math import pi
from hqca.core import *
try:
    from qiskit import Aer,IBMQ
    from qiskit.providers.aer import noise
except Exception:
    pass
from qiskit import QuantumRegister,QuantumCircuit,ClassicalRegister
from qiskit import execute
from qiskit.ignis.mitigation.measurement import(
        complete_meas_cal,
        tensored_meas_cal,
        TensoredMeasFitter,
        CompleteMeasFitter,
        MeasurementFilter)

class KeyDict(dict):
    def __missing__(self,key):
        return key

class QuantumStorage:
    '''
    Object for storing information relevant to the quantum optimization. In
    particular, should generate the mapping between the quantum and molecular
    2RDMs.

    '''
    def __init__(self,
            verbose=True,
            **kwargs
            ):
        self.check = 0
        self.verbose = verbose
        pass

    def set_algorithm(self,
            Storage,
            depth=1,
            rdm_to_qubit='default',
            **kwargs,
            ):
        self.depth = depth
        self.check+=1
        self.p = Storage.p
        if Storage.H.model in ['molecular','molecule','mol']:
            self.alpha= Storage.alpha_mo
            self.beta = Storage.beta_mo
            self.Ne = Storage.Ne_as
            self.op_type = 'fermionic'
            if rdm_to_qubit=='default':
                alp = [i for i in range(Storage.No_as)]
                bet = [i+Storage.No_as for i in range(Storage.No_as)]
            elif rdm_to_qubit=='alternating':
                alp = [2*i for i in range(Storage.No_as)]
                bet = [2*i+1 for i in range(Storage.No_as)]
            self.a2b = {
                    i:j for i,j in zip(self.alpha['active'],self.beta['active'])
                        }
            self.mapping = Storage.H.mapping
            self._kw_mapping = Storage.H._kw_mapping
            self.groups = [
                    alp,
                    bet
                    ]
            self.initial = []
            for i in range(Storage.H.Ne_alp):
                self.initial.append(alp[i])
            for i in range(Storage.H.Ne_bet):
                self.initial.append(bet[i])
            self.Ne_alp = Storage.H.Ne_alp
            self.Ne_bet = Storage.H.Ne_bet
        elif Storage.H.model in ['sq','tq']:
            self.op_type = 'qubit'
            self.initial = []
            self.mapping = 'qubit'
            self._kw_mapping = {}
        self.use_meas_filter=False
        self.post = False
        self.process = False

    def set_backend(self,
            Nq=4,
            Nq_backend=20,
            Nq_ancilla=0,
            backend='qasm_simulator',
            num_shots=1024,
            backend_coupling_layout=None,
            backend_initial_layout=None,
            backend_file=None,
            noise=False,
            noise_model_location=None,
            noise_gate_times=None,
            transpile='default',
            transpiler_keywords={},
            provider='Aer',
            **kwargs):
        if self.check==0:
            sys.exit('Have not set up the quantumstorage algorithm yet.')
        self.transpile=transpile
        if self.transpile in [True,'default']:
            self.transpile='default'
        elif self.transpile in ['test']:
            pass
        else:
            print('Transpilation scheme not recognized.')
            print(self.transpile)
            sys.exit()
        self.transpiler_keywords = transpiler_keywords
        self.be_initial = backend_initial_layout
        self.be_coupling = backend_coupling_layout
        self.be_file = backend_file
        self.Nq = Nq  # active qubits
        if Nq_backend is not None:
            self.Nq_be = Nq_backend
        else:
            self.Nq_be = self.Nq
        self.Nq_anc = Nq_ancilla
        self.Nq_tot = self.Nq_anc+self.Nq
        self.Ns = num_shots
        self.provider = provider
        if self.provider=='IBMQ':
            try:
                prov = IBMQ.load_account()
            except AttributeError:
                pass
        else:
            prov = Aer
        self.backend=backend
        self.beo = prov.get_backend(backend)
        self.use_noise=False
        if self.verbose:
            print('# Summary of quantum parameters:')
            print('#  backend   : {}'.format(self.backend))
            print('#  num shots : {}'.format(self.Ns))
            print('#  num qubit : {}'.format(self.Nq))
            print('#  provider  : {}'.format(provider))
            print('#  transpile  : {}'.format(self.transpile))
            print('#######')

    def set_noise_model(self,
            custom=False,
            **kw
            ):
        self.use_noise = True
        if custom:
            self._set_custom_noise_model(**kw)
        else:
            self._get_noise_model(**kw)


    def set_error_mitigation(self,
            mitigation=False,
            **kwargs):
        if mitigation=='stabilizer':
            self._set_stabilizers(**kwargs)

    def set_error_correction(self,
            error_correction=False,
            **kwargs
            ):
        '''
        Note there are three types of error correction.
            (1) Post correction (hyperplane, symmetry)
                ec_type = 'post'
            (2) Correction in a entangler, in circuit
                ec_type = 'ent'
            (3) Syndrome, in circuit
                ec_type = 'syndrome'
        Honestly, probably better to just manually specify them...
        '''
        if error_correction=='measure':
            self.use_meas_filter = True
            self._get_measurement_filter(initial=True,**kwargs)
        elif error_correction=='symmetry':
            self._set_symmetries(**kwargs)

    def _set_symmetries(self,symmetries):
        self.post=True
        self._symm = []
        for s in symmetries:
            self._symm.append(s)

    def _set_stabilizers(self):
        self.process =True
        # need to see if we can set it for...parity check? 
        # etc. 
        pass

    def __set_ec_post_correction(self,
            symm_verify=False,
            symmetries=[],
            hyperplane=False,
            error_shift=None,
            vertices=None,
            **kwargs):
        '''
        Used for hyper plane set-up, as well as verifying symmetries in the
        wavefunction.
        '''
        self.post = True

        self.hyperplane=hyperplane
        self.symm_verify = symm_verify
        self.symmetries = symmetries
        self.error_shift=error_shift
        self.hyperplane_custom=False
        if hyperplane==True:
            self._get_hyper_para()
        elif hyperplane=='custom':
            self.hyperplane_custom=True
            self._get_hyper_para()
            self.hyper_alp = np.asarray(vertices[0])
            self.hyper_bet = np.asarray(vertices[1])


    def _get_hyper_para(self,expand=False):
        if self.method=='carlson-keller':
            arc = [2*np.arccos(1/np.sqrt(i)) for i in range(1,self.Nq+1)]
            self.ec_para = []
            self.ec_Ns = 1
            self.Nv = self.Nq//2
            self.ec_Nv = self.Nv
            self.ec_vert = np.zeros((self.Nv,self.Nv))
            for i in range(0,self.Nv):
                temp = []
                for j in range(0,i+1):
                    self.ec_vert[j,i]=1/(i+1)
                    temp.append(-arc[j])
                for k in range(i+1,self.Nv):
                    temp.insert(0,0)
                temp = temp[::-1]
                del temp[-1]
                self.ec_para.append(temp)
            self.ec_para = [self.ec_para]
        else:
            print('Error in function quantum/QuantumFunctions/_get_hyper_para')
            sys.exit('Unsupported method!')

    def _get_measurement_filter(self,
            initial=False,
            frequency=3,
            **kw
            ):
        if initial:
            self.freq = frequency
            self.n = 0 
        if self.n==0:
            print('Reculating measurement filter')
            qubit_list= [i for i in range(self.Nq_tot)]
            cal_circuits,state_labels = complete_meas_cal(
                    qubit_list,
                    QuantumRegister(self.Nq_tot),
                    ClassicalRegister(self.Nq_tot)
                    )
            if self.use_noise:
                job = execute(
                        cal_circuits,
                        backend=self.beo,
                        backend_options=self._noisy_be_options,
                        noise_model=self.noise_model,
                        )
            else:
                job = execute(cal_circuits,
                        backend=self.beo,
                        shots=self.Ns,
                        initial_layout=self.be_initial)
            cal_results = job.result()
            meas_fitter = CompleteMeasFitter(
                    cal_results,
                    state_labels)
            meas_filter = meas_fitter.filter
            #self.meas_fitter = meas_fitter
            self._meas_filter = meas_filter
            self.n = np.copy(self.freq)-1
            print(meas_filter.cal_matrix)
        else:
            self.n-=1

    @property
    def meas_filter(self):
        self._get_measurement_filter()
        return self._meas_filter

    @meas_filter.setter
    def meas_filter(self,b):
        self._meas_filter = b

    def _set_custom_noise_model(self,
            noise_model=None,
            noise_options={}):
        self.noise_model=noise_model
        self._noisy_be_options = {
                'noise_model':self.noise_model,
                'basis_gates':self.noise_model.basis_gates
                }
        self.noise_model.coupling_map = self.beo.configuration().coupling_map


    def _get_noise_model(self,
            times=None,
            saved=False):
        if (not saved) or (saved is None):
            backend = self.beo
            properties = self.beo.properties()
            self.coupling=backend.configuration
        else:
            try:
                with open(saved,'rb') as fp:
                    data = pickle.load(fp)
            except FileNotFoundError:
                print('Wrong one :(')
            properties = data['properties']
            self._be_coupling = data['config'].coupling_map
        self._be_properties = properties
        noise_model = noise.NoiseModel()
        noise_model.from_backend(properties)
        #if times is not None:
        #    noise_model = noise.device.basic_device_noise_model(
        #        properties,times)
        #else:
        #    noise_model = noise.device.basic_device_noise_model(
        #        properties)
        noise_model.coupling_map = self._be_coupling
        self.noise_model = noise_model
        self._noisy_be_options = {
                'noise_model':self.noise_model,
                'basis_gates':self.noise_model.basis_gates
                }


def print_qasm(circuit):
    filename = input('Save qasm as: ')
    print('Great!')
    print(circuit.qasm())
    print(circuit.count_ops())
    with open('{}.qasm'.format(filename),'w') as fp:
        fp.write(circuit.qasm())

def get_direct_stats(QuantStore):
    '''
    function to preview and visualize circuits, can use one of the test examples
    as a good template

    'draw' - draw simple circuit
    'tomo' - draw circuit with tomography
    'build'- construct circuit with transpiler
    'qasm' - get qasm from transpiler
    'calc' - calculate counts
    'stats'- do some more compilcated statistics
    '''
    from hqca.quantum.QuantumFramework import build_circuits
    from qiskit.tools.monitor import backend_overview
    from qiskit.compiler import transpile
    hold_para = QuantStore.parameters.copy()
    QuantStore.parameters=[1]*QuantStore.Np
    qcirc,qcirc_list = build_circuits(QuantStore)
    extra = QuantStore.info
    print('')
    if extra=='draw':
        print(qcirc[0])
        print(qcirc[0].count_ops())
    elif extra=='tomo':
        for c,n in zip(qcirc,qcirc_list):
            print('Circuit name: {}'.format(n))
            print(c)
            print(c.count_ops())
    elif extra=='calc':
        print('Gate counts:')
        print(qcirc[0].count_ops())
    elif extra=='ibm':
        be = IBMQ.get_backend(QuantStore.backend)
        print(be.configuration().coupling_map)
        for c,n in zip(qcirc,qcirc_list):
            print('Circuit name: {}'.format(n))
            qt = transpile(c,be,
                    initial_layout=QuantStore.be_initial,
                    **QuantStore.transpiler_keywords
                    )
            print(qt)
            print(qt.count_ops())
    elif extra in ['qasm','build','stats']:
        if extra=='stats':
            print('Initial gate counts: ')
            print(qcirc[0].count_ops())
        be = Aer.get_backend('qasm_simulator')
        print('Getting coupling.')
        if QuantStore.be_file in [None,False]:
            if QuantStore.be_coupling in [None,False]:
                if QuantStore.backend=='qasm_simulator':
                    coupling=None
                else:
                    IBMQ.load_account()
                    backend_overview()
                    beo = IBMQ.get_backend(QuantStore.backend)
                    coupling = beo.configuration().coupling_map
            else:
                coupling = QuantStore.be_coupling
        else:
            try:
                coupling = NoiseSimulator.get_coupling_map(
                        device=QuantStore.backend,
                        saved=QuantStore.be_file
                        )
            except Exception as e:
                print(e)
                sys.exit()
        print('Transpiling...')
        if QuantStore.transpile=='default':
            qt = transpile(qcirc[0],
                    backend=self.beo,
                    coupling_map=coupling,
                    initial_layout=QuantStore.be_initial,
                    **QuantStore.transpiler_keywords
                    )
        else:
            pass
        if extra=='qasm':
            print_qasm(qt)
        elif extra=='build':
            print(qt)
            print(qt.count_ops())
        elif extra=='stats':
            print('Gates after compilation/transpilation')
            print(qt.count_ops())
    QuantStore.parameters=hold_para


