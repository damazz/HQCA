'''
tools/QuantumFunctions

Contains QuantumStorage and other useful quantum functions. Basically,
QuantumStorage is a class which is needed for any quantum calculation on the QC.
Generates important information to that end.
'''
import numpy as np
np.set_printoptions(suppress=True,precision=4)
import sys
from math import pi
from hqca.quantum import NoiseSimulator
import hqca.quantum.algorithms._ECC as ecc
from qiskit import IBMQ,Aer
class KeyDict(dict):
    def __missing__(self,key):
        return key

class QuantumStorage:
    '''
    Object for storing information relevant to the quantum optimization. In
    particular, should generate the mapping between the quantum and molecular
    2RDMs.

    Also should assign the difference parameters of the quantum algorithm.
    '''
    def __init__(self,
            pr_g=0,
            qc=True,
            opt=None,
            info='calc',
            method='variational',
            **kwargs
            ):
        self.opt_kw = opt
        self.kwargs = kwargs
        self.pr_g =pr_g
        self.method = method
        self.qc = qc
        self.info = info
        if qc:
            self._set_up_backend(**self.kwargs)
            self._set_up_algorithm(**self.kwargs)
            self._set_up_error_correction(**self.kwargs)
        else:
            # need to check to make sure method is compatible
            classically_supported = ['borland-dennis','carlson-keller']
            if self.method in classically_supported:
                pass
            else:
                sys.exit('Trying to run a non-supported classical algorithm.')
        self.active_qb = self.alpha_qb+self.beta_qb
        self.print_summary()

    def _set_up_algorithm(self,
            pr_q=0,
            Ne_as=None,
            No_as=None,
            alpha_mos=None,
            beta_mos=None,
            Sz=0,
            theory='noft',
            fermion_mapping='jordan-wigner',
            ansatz='default',
            spin_mapping='default',
            entangled_pairs='d',
            entangler_p='Ry_cN',
            entangler_q='UCC2c',
            depth=1,
            compact_algorithm=None,
            tomo_basis='no',
            tomo_rdm='1rdm',
            tomo_extra=False,
            tomo_approx=None,
            entangler_kw=None,
            **kwargs
            ):
        self.pr_q = pr_q
        self.theory=theory
        self.spin_mapping = spin_mapping
        self.Sz=Sz
        self.ansatz = ansatz
        self.No = No_as # note, spatial orbitals
        self.alpha = alpha_mos
        self.beta = beta_mos
        self.Ne = Ne_as # active space
        if spin_mapping in ['default','alternating']:
            self.Ne_alp = int(0.5*Ne_as+Sz)
        else:
            self.Ne_alp = int(Ne_as)
        self.Ne_bet = self.Ne-self.Ne_alp
        self.tomo_bas = tomo_basis
        self.tomo_rdm = tomo_rdm
        self.tomo_ext = tomo_extra
        self.tomo_approx = tomo_approx
        self.ent_pairs= entangled_pairs
        self.ent_circ_p = entangler_p
        self.ent_circ_q = entangler_q
        self.ent_kw = entangler_kw
        self.depth = depth
        self.algorithm = compact_algorithm
        self.fermion_mapping = fermion_mapping
        if self.fermion_mapping=='jordan-wigner':
            self._map_rdm_jw()
            if self.method in ['variational','vqe']:
                self._get_ent_pairs_jw()
                self._gip()
                if self.tomo_ext in ['sign_2e',
                        'sign_2e_pauli',
                        'sign_2e_from_ancilla'
                        ]:
                    self._get_2e_no()
        self.kwargs=kwargs

    def _set_up_backend(self,
            Nq=4,
            Nq_backend=20,
            Nq_ancilla=0,
            backend='qasm_simulator',
            backend_coupling_layout=None,
            backend_initial_layout=None,
            backend_file=None,
            noise=False,
            noise_model_location=None,
            noise_gate_times=None,
            transpile='default',
            transpiler_keywords={},
            num_shots=1024,
            provider='Aer',
            **kwargs):
        self.alpha_qb = [] # in the backend basis
        self.beta_qb  = [] # backend set of qubits
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
        self.backend = backend
        self.Ns = num_shots
        self.provider = provider
        if self.provider=='IBMQ':
            IBMQ.load_accounts()
        self.use_noise = noise
        self.noise_gate_times=noise_gate_times
        if self.use_noise:
            self.noise_model = NoiseSimulator.get_noise_model(
                    device=backend,
                    times=noise_gate_times,
                    saved=noise_model_location)
        self.noise_model_loc = noise_model_location
        self.kwargs=kwargs

    def _set_up_error_correction(self,
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


    def _map_rdm_jw(self):
        '''
        note, if you change the spin mapping, it also effects the quantum
        circuit and the type of double excitation that you need
        '''
        self.qubit_to_rdm = {}
        if self.spin_mapping=='default':
            for n,alp in enumerate(self.alpha['active']):
                self.qubit_to_rdm[n]=alp
                self.alpha_qb.append(n)
            for n,bet in enumerate(self.beta['active']):
                m = n+self.No
                self.qubit_to_rdm[m]=bet
                self.beta_qb.append(m)
        elif self.spin_mapping=='spin-free':
            for n,alp in enumerate(self.alpha['active']):
                self.qubit_to_rdm[n]=alp
                self.alpha_qb.append(n)
            for n,bet in enumerate(self.beta['active']):
                m = n+self.No
                self.qubit_to_rdm[m]=bet
                self.alpha_qb.append(m)
        elif self.spin_mapping=='alternating':
            i = 0
            self.alpha_qb,self.beta_qb = [],[]
            self.qubit_to_rdm = {}
            a,b = 0,len(self.alpha['active'])
            for i in range(len(self.alpha['active']+self.beta['active'])):
                if i%2==0:
                    self.alpha_qb.append(i)
                    self.qubit_to_rdm[i]=a
                    a+=1 
                else:
                    self.beta_qb.append(i)
                    self.qubit_to_rdm[i]=b
                    b+=1 
        self.rdm_to_qubit = {v:k for k,v in self.qubit_to_rdm.items()}

    def _get_ent_pairs_jw(self):
        '''
        using the jordan-wigner mapping, generate entangling pairs based on the
        proper ansatz. for instance, we have ucc ansatz, with singles and double
        excitations, but we also have simple double excitations in the NO basis 
        that work for certain cases

        possible ansatz:
            -ucc
            -natural orbitals
            -nat-orb-no (reduced form)

        Note, the proper functioning of this function should take entanglement
        in the RDM basis, preparing quad list and pair list.

        Then, it takes this function and arranges it according to qc format,
        producing qc_quad_list and qc_pair_list. These are ordered and have the
        proper operator for (below). 

        Also, wyou can include the type of operator that is being applied for
        the given configuration. For isntance, a transition form a1->a2, b1->b2
        that is applied as i<j<k<l, would be a -+-+ operator, NOT ++--
        '''
        self.pair_list = []
        self.quad_list = []
        if self.ansatz=='ucc': 
            # UPDATE
            if self.ent_pairs in ['sd','d']:
                # generate double excitations
                for l in range(0,len(self.alpha_qb)):
                    for k in range(0,l):
                        for j in range(0,k):
                            for i in range(0,j):
                                self.quad_list.append([i,j,k,l])
                for l in range(self.No,self.No+len(self.beta_qb)):
                    for k in range(self.No,l):
                        for j in range(0,len(self.alpha_qb)):
                            for i in range(0,j):
                                self.quad_list.append([i,j,k,l])
                for l in range(self.No,self.No+len(self.beta_qb)):
                    for k in range(self.No,l):
                        for j in range(self.No,k):
                            for i in range(self.No,j):
                                self.quad_list.append([i,j,k,l])
            if self.ent_pairs in ['s','sd']:
                for j in range(0,len(self.alpha_qb)):
                    for i in range(0,j):
                        self.pair_list.append([i,j])
                for j in range(self.No,self.No+len(self.beta_qb)):
                    for i in range(self.No,j):
                        self.pair_list.append([i,j])
        elif self.ansatz=='natural-orbitals':
            if self.ent_pairs in ['sd','d']:
                # generate double excitations
                for l in range(self.No,self.No+len(self.beta_qb)):
                    for k in range(self.No,l):
                        i,j = k%self.No, l%self.No
                        self.quad_list.append([i,j,k,l])
        elif self.ansatz=='nat-orb-no':
            # this is okay, important part is you are using len of beta_qb,
            # and not the actula qb's 
            if self.ent_pairs in ['sd','d']:
                for k in range(self.No,self.No+len(self.beta_qb)-1):
                    i= k%self.No
                    self.quad_list.append([i,i+1,k,k+1,'-+-+','aabb'])
        # now, order them 
        self.qc_quad_list = []
        for quad in self.quad_list:
            qd = []
            for i in range(len(quad)-2):
                qd.append(self.rdm_to_qubit[quad[i]])
            try:
                sign = list(quad[4])
                spin = list(quad[5])
            except IndexError:
                sign = ['-','+','-','+']
                spin = ['a','a','b','b']
            sort = False
            while not sort:
                i,j,k,l = qd[0],qd[1],qd[2],qd[3]
                sort = True
                if i>j:
                    qd[1],qd[0]=qd[0],qd[1]
                    sign[0],sign[1]=sign[1],sign[0]
                    spin[0],spin[1]=spin[1],spin[0]
                    sort = False
                if j>k:
                    qd[1],qd[2]=qd[2],qd[1]
                    sign[2],sign[1]=sign[1],sign[2]
                    spin[2],spin[1]=spin[1],spin[2]
                    sort = False
                if k>l:
                    sign[3],sign[2]=sign[2],sign[3]
                    spin[2],spin[3]=spin[3],spin[2]
                    qd[3],qd[2]=qd[2],qd[3]
                    sort = False
            qd.append(''.join(sign))
            qd.append(''.join(spin))
            self.qc_quad_list.append(qd)
        print('Excitations on the quantum computer: ')
        print(self.qc_quad_list)

    def _get_2e_no(self):
        '''
        set the tomography elements for the elements

        note, these are also in the RDM basis. these have to get transferred to
        other ones
        '''
        self.tomo_quad = []
        temp = iter(zip(
                self.alpha['active'][:-1],
                self.alpha['active'][1:],
                self.beta['active'][:-1],
                self.beta['active'][1:]))
        for a,b,c,d in temp:
            self.tomo_quad.append([a,b,c,d])
        self.qc_tomo_quad = []
        for quad in self.tomo_quad:
            qd = []
            for i in quad:
                qd.append(self.rdm_to_qubit[i])
            try:
                sign = list(quad[4])
                spin = list(quad[5])
            except IndexError:
                sign = ['-','+','-','+']
                spin = ['a','a','b','b']
            sort = False
            while not sort:
                i,j,k,l = qd[0],qd[1],qd[2],qd[3]
                sort = True
                if i>j:
                    qd[1],qd[0]=qd[0],qd[1]
                    sign[0],sign[1]=sign[1],sign[0]
                    spin[0],spin[1]=spin[1],spin[0]
                    sort = False
                if j>k:
                    qd[1],qd[2]=qd[2],qd[1]
                    sign[2],sign[1]=sign[1],sign[2]
                    spin[2],spin[1]=spin[1],spin[2]
                    sort = False
                if k>l:
                    sign[3],sign[2]=sign[2],sign[3]
                    spin[2],spin[3]=spin[3],spin[2]
                    qd[3],qd[2]=qd[2],qd[3]
                    sort = False
            qd.append(''.join(sign))
            qd.append(''.join(spin))
            self.qc_tomo_quad.append(qd)
        print('Tomography on the quantum computer: ')
        print(self.qc_tomo_quad)

    def print_summary(self):
        if self.pr_g>1:
            print('# Number of active qubits: {}'.format(self.Nq))
            print('# Number of ancilla qubits: {}'.format(len(self.ancilla)))
            print('# Number of backend qubits: {}'.format(self.Nq_tot))
            print('# Summary of quantum parameters:')
            print('#  backend   : {}'.format(self.backend))
            print('#  num shots : {}'.format(self.Ns))
            print('#  num qubit : {}'.format(self.Nq))
            print('#  num e-    : {}'.format(self.Ne))
            print('#  provider  : {}'.format(self.prov))
            print('#  tomo type : {}'.format(tomo_rdm))
            print('#  tomo basis: {}'.format(tomo_basis))
            print('#  tomo extra: {}'.format(tomo_extra))
            print('#  transpile  : {}'.format(self.transpile))
            print('# Summary of quantum algorithm:')
            print('#  spin orbs : {}'.format(self.No))
            print('#  fermi map : {}'.format(self.fermion_mapping))
            print('#  ansatz    : {}'.format(self.ansatz))
            print('#  method    : {}'.format(self.method))
            print('#  algorithm : {}'.format(self.algorithm))
            print('#  spin map  : {}'.format(self.spin_mapping))
            print('#  ent scheme: {}'.format(self.ent_pairs))
            print('#  ent pairs : {}'.format(self.ent_circ_p))
            print('#  ent quads : {}'.format(self.ent_circ_q))
            print('#  circ depth: {}'.format(self.depth))
            print('#  init type : {}'.format(self.init))
            print('#  qubit entangled pairs:')
            p = '#  '
            for n,pair in enumerate(self.pair_list):
                p+='{} '.format(pair)
                if n%4==0 and n>0:
                    print(p)
                    p='#  '
            print(p)
            p = '#  '
            print('#  rdm entangled quadruplets:')
            for n,quad in enumerate(self.quad_list):
                p+='{} '.format(quad)
                if n%2==0 and n>0:
                    print(p)
                    p='#  '
            print(p)
            p = '#  '
            print('#  qubit entangled quadruplets:')
            for n,quad in enumerate(self.qc_quad_list):
                p+='{} '.format(quad)
                if n%2==0 and n>0:
                    print(p)
                    p='#  '
            print(p)
            print('# ')
            print('Unused kwargs:')
            for k,v in self.kwargs.items():
                print(k,v)

    def _gip(self):
        '''
        'Get Initial Parameters (GIP) function.
        '''
        if self.theory=='noft':
            self.parameters=[0]*(self.No-1)
            self.Np = self.No-1
        elif self.theory=='rdm':
            self.c_ent_p=1
            self.c_ent_q=1
            if self.ent_circ_p=='Uent1_cN':
                self.c_ent_p=2
            elif self.ent_circ_p=='phase':
                self.c_ent_p=2
            if self.ent_circ_q=='UCC2':
                self.c_ent_q=3
            self.Np = self.depth*len(self.pair_list)*self.c_ent_p
            self.Np+= self.depth*len(self.quad_list)*self.c_ent_q
            self.parameters=[0.0]*self.Np
        if self.pr_g>1:
            print('Number of initial parameters: {}'.format(self.Np))

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
                    IBMQ.load_accounts()
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


