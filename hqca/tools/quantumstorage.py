'''

Contains QuantumStorage and other useful quantum functions. Basically,
QuantumStorage is a class which is needed for any quantum calculation on the QC.
Generates important information to that end.
'''
import numpy as np
np.set_printoptions(suppress=True,precision=4)
import sys
from math import pi
from hqca.core import *
from qiskit import Aer,IBMQ

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
            **wargs
            ):
        self.check = 0
        pass

    def set_algorithm(self,
            Storage,
            depth=1,
            rdm_to_qubit='default',
            **kwargs
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
            self.groups = [
                    alp,
                    bet
                    ]
            self.initial = []
            for i in range(Storage.H.Ne_alp):
                self.initial.append(alp[i])
            for i in range(Storage.H.Ne_bet):
                self.initial.append(bet[i])
            self.mapping = Storage.H.mapping
        elif Storage.H.model in ['sq','tq']:
            self.op_type = 'qubit'
            self.initial = []

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
            prov = IBMQ.load_account()
        else:
            prov = Aer
        self.backend=backend
        self.beo = prov.get_backend(backend)
        self.use_noise = noise
        self.noise_gate_times=noise_gate_times
        if self.use_noise:
            self.noise_model = NoiseSimulator.get_noise_model(
                    device=backend,
                    times=noise_gate_times,
                    saved=noise_model_location)
        self.noise_model_loc = noise_model_location
        self.kwargs=kwargs

    def set_error_correction(self,
            pr_e=0,
            ec_pre=False,
            ec_pre_kw={},
            ec_syndrome=False,
            ec_syndrome_kw={},
            ec_post=False,
            ec_post_kw={},
            ec_comp_ent=False, #composite entangler
            ec_comp_ent_kw={},
            ec_custom=False,
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
        self.pr_e = pr_e
        self.ec_syndrome=ec_syndrome
        self.ec_post=ec_post
        self.ec_pre = ec_pre
        self.ec_comp_ent=ec_comp_ent
        self.ancilla = [] # used ancilla
        self.ancilla_list = [i+self.Nq for i in range(self.Nq_anc)] # potential
        if self.ec_pre:
            self.__set_ec_pre_filters(**ec_pre_kw)
        if self.ec_post:
            self.__set_ec_post_correction(**ec_post_kw)
        if self.ec_syndrome:
            self.__set_ec_syndrome(**ec_syndrome_kw)
        if self.ec_comp_ent:
            self.__set_ec_composite_entangler(**ec_comp_ent_kw)

    def __set_ec_pre_filters(self,
            filter_measurements=False,
            **kw
            ):
        self.filter_meas = filter_measurements

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

    def __set_ec_syndrome(self,
            apply_syndromes={},
            **kwargs):
        try:
            self.ancilla_sign
        except Exception:
            self.ancilla_sign=[]
        '''
        Included in the circuit, checks for certain types of errors, at
        different locations in the overall circuit.

        Includes circuits designed to check for symmetries of the system. Most
        can simply be added at the end. Might use ancilla qubits, maybe not.

        locations, where to check for then
        '''
        self.syndromes = apply_syndromes
        ind = 0
        for synd,locations in self.syndromes.items():
            for item in locations:
                try:
                    item['ancilla']=self.ancilla_list[ind:ind+item['N_anc']]
                except Exception as e:
                    print(e)
                    sys.exit('Huh.')
                self.ancilla += self.ancilla_list[ind:ind+item['N_anc']]
                if item['use']=='sign':
                    self.ancilla_sign.append(item['ancilla'])
                ind+= item['N_anc']

    def __set_ec_composite_entangler(self,
            ec_replace_pair='default',
            ec_replace_quad='default',
            **kwargs):
        '''
        Sets up entangling gates that some useful error correction in them.
        Will almost always use ancilla qubits, but also draws from different
        circuits.

        Specifying the entangling gates will be a little different in
        most cases.
        '''
        try:
            self.ancilla_sign
        except Exception:
            self.ancilla_sign=[]
        if ec_replace_pair=='default':
            entry = {
                    'replace':False,
                    'N_anc':0,
                    'circ':None,
                    'kw':{},
                    }
            self.ec_replace_pair=[entry]*len(self.pair_list)
        elif len(ec_replace_pair)<len(self.pair_list):
            sys.exit('Wrong number of gates for composite pair entanglers..')
        else:
            self.ec_replace_pair = ec_replace_pair

        if ec_replace_quad=='default':
            entry = {
                    'replace':False,
                    'N_an':0,
                    'circ':None,
                    'use':'None',
                    'kw':{},
                    }
            self.ec_replace_quad=[entry]*len(self.quad_list)
        elif len(ec_replace_quad)<len(self.quad_list):
            sys.exit('Wrong number of gates for composite quad entanglers..')
        else:
            self.ec_replace_quad = ec_replace_quad
        ind = 0
        for item in self.ec_replace_quad:
            item['ancilla']=self.ancilla_list[ind:ind+item['N_anc']]
            self.ancilla += self.ancilla_list[ind:ind+item['N_anc']]
            if item['use']=='sign':
                self.ancilla_sign.append(item['ancilla'])
            ind+= item['N_anc']


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
    from qiskit import Aer,IBMQ,execute
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
                    backend=be,
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



local_qubit_tomo_pairs = {
        2:[
            ['01']],
        3:[
            ['01'],['02'],['12']],
        4:[
            ['01','23'],
            ['12','03'],
            ['02','13']],
        5:[
            ['01','23'],['04','12'],
            ['02','34'],['13','24'],
            ['03','14']],
        6:[
            ['01','23','45'],
            ['02','14','35'],
            ['13','04','25'],
            ['03','24','15'],
            ['12','34','05']],
        7:[
            ['01','23','46'],
            ['02','13','56'],
            ['03','14','25'],
            ['04','15','26'],
            ['05','16','34'],
            ['06','24','35'],
            ['12','36','45']],
        8:[
            ['01','24','35','67'],
            ['02','14','36','57'],
            ['03','15','26','47'],
            ['04','12','56','37'],
            ['05','13','27','46'],
            ['06','17','23','45'],
            ['07','16','25','34']],
        12:[
            ['0-1','2-3','4-5','6-7','8-9','10-11'],
            ['0-2','1-3','2-5'],
            ['0-3','1-4','2-6'],
            ['0-4','1-5','2-7'],
            ['0-5','1-6','2-4'],
            ['0-6','1-7','2-8'],
            ['0-7','1-8','2-9'],
            ['0-8','1-9','2-10'],
            ['0-9','1-10','2-11'],
            ['0-10','1-11'],
            ['0-11','1-2']]
        }
nonlocal_qubit_tomo_pairs_part = {
        0:[[]],
        1:[[]],
        2:[
            ['01']],
        3:[
            ['01'],['02'],['12']],
        4:[ 
            ['01','23'],
            ['02'],
            ['03'],
            ['12'],
            ['13']],
        5:[
            ['01','23'],['02','34'],
            ['24'],['03'],['13'],
            ['04'],['14'],['12']
            ],
        6:[
            ['05'],['04'],['15'],['14'],['12','34'],
            ['03','45'],['01','25'],['02','35'],
            ['13'],['24'],['23']
            ],
        8:[
            ['07'],['06'],['17'],
            ['05','67'],['16'],
            ['01','27'],['04','57'],
            ['02','37'],['15'],['26'],
            ['03','47'],['12','34','56'],
            ['23','46'],['13','45'],
            ['14'],['25'],['36'],['24'],['35']
            ],
        }

# in terms of efficiency...
# 4 has 1 fewer (5 vs 6) but 2 more (5 v 3)
# 5 has 2 fewer (8 v 10) but 3 more (8 v 5)
# 6 has 4 fewer (11 v 15) but 6 more (11 v 5)
# 8 has 9 fewer (19 v 28) but 12 more (19 v 7)

diag = {i:[''] for i in range(2,9)}

nonlocal_qubit_tomo_pairs_full = {
        i:[
            ['{}{}'.format(
                b,a)]
                for a in range(i) for b in range(a)
            ] for i in range(2,9)}


