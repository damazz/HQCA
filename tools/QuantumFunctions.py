'''
tools/QuantumFunctions


'''
import numpy as np
import sys
from math import pi
from hqca.tools import NoiseSimulator
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
            pr_g,
            Nqb,
            Nels_as,
            Norb_as,
            backend,
            num_shots,
            tomo_basis,
            tomo_rdm,
            tomo_extra,
            provider,
            entangled_pairs,
            entangler_p,
            entangler_q,
            error_correction,
            Sz,
            depth,
            alpha_mos,
            beta_mos,
            single_point,
            theory='noft',
            Nqb_backend=None,
            fermion_mapping='jordan-wigner',
            load_triangle=False,
            qc=True,
            ansatz='default',
            spin_mapping='default',
            method='variational',
            backend_configuration=None,
            noise_model_loc=None, #specify file name for noise_model 
            noise=False,
            transpile=False,
            info=None,
            initialize='default',
            algorithm=None,
            pr_q=0,
            pr_e=1,
            circuit_times=None,
            use_radians=False,
            opt=None
            ):
        '''
        method;
            variational
            trotter?
        variational;
            default
            ucc
        entangler
            Ry_cN
            Rx_cN
            Rz_cN
            trott
        spin_mapping
            default
            spin_free
            spatial? not sure if necessary
        '''
        self.opt_kw = opt
        self.theory= theory
        self.pr_g =pr_g
        self.use_radians=use_radians
        self.Ns = num_shots
        self.Nq = Nqb  # active qubits
        if Nqb_backend is not None:
            self.Nq_tot = Nqb_backend
        else:
            self.Nq_tot = self.Nq
        self.Ne = Nels_as # active space
        if spin_mapping=='default':
            self.Ne_alp = int(0.5*Nels_as+Sz)
        else:
            self.Ne_alp = int(Nels_as)
        self.No = Norb_as # note, spatial orbitals
        if self.No>self.Nq: # wrong!
            print('Error in the mapping. Too many orbitals per qubits.')
        self.backend = backend
        self.Ns = num_shots
        self.prov = provider
        self.tomo_bas = tomo_basis
        self.tomo_rdm = tomo_rdm
        self.tomo_ext = tomo_extra
        self.provider = provider
        self.ent_pairs= entangled_pairs
        self.ent_circ_p = entangler_p
        self.ent_circ_q = entangler_q
        self.depth = depth
        self.init = initialize
        self.algorithm = algorithm
        self.method = method
        self.ec=error_correction
        self.sp = single_point
        self.alpha = alpha_mos
        self.beta = beta_mos
        if not qc:
            # need to check to make sure method is compatible
            classically_supported = ['borland-dennis','carlson-keller']
            if self.method in classically_supported:
                pass
            else:
                sys.exit('Trying to run a non-supported classical algorithm.')
        self.qc = qc
        self.pr_q = pr_q
        self.pr_e = pr_e
        self.ansatz = ansatz
        self.spin_mapping = spin_mapping
        self.bec = backend_configuration
        self.qubit_to_backend = backend_configuration
        if self.qubit_to_backend is None:
            self.qubit_to_backend=KeyDict()
            self.backend_to_qubit=KeyDict()
        else:
            #self.backend_to_qubit={v:k for k,v in self.qubit_to_backend.items()}
            self.qubit_to_backend=KeyDict()
            self.backend_to_qubit=KeyDict()
        self.transpile = transpile
        self.use_noise = noise
        self.info=info
        self.fermion_mapping = fermion_mapping
        self.alpha_qb = [] # in the backend basis
        self.beta_qb  = [] # backend set of qubits
        if self.fermion_mapping=='jordan-wigner':
            self._map_rdm_jw()
            self._get_ent_pairs_jw()
            self._gip()
            if self.tomo_ext=='sign_2e':
                self._get_2e_no()
        if self.ec=='hyperplane':
            self._get_hyper_para()
        if self.use_noise:
            self.noise_model = NoiseSimulator.get_noise_model(
                    device=backend,
                    times=circuit_times,
                    saved=noise_model_loc)
        if self.pr_g>1:
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
            print('#  qubit entangled quadruplets:')
            for n,quad in enumerate(self.quad_list):
                p+='{} '.format(quad)
                if n%2==0 and n>0:
                    print(p)
                    p='#  '
            print(p)
            print('# ')

    def _get_hyper_para(self):
        if self.method=='carlson-keller':
            if self.Nq==8:
                ma = np.arccos(1/np.sqrt(3))
                self.ec_Ns = 1  # num surfaces
                self.ec_Nv = 4  # num dimensions 
                self.ec_vert = np.matrix([
                    [1/1,1/2,1/3,1/4],
                    [0/1,1/2,1/3,1/4],
                    [0/1,0/2,1/3,1/4],
                    [0/1,0/2,0/3,1/4]])
                self.ec_para = [
                        [
                            [0,0,0],
                            [pi/2,0,0],
                            [2*ma,pi/2,0],
                            [2*pi/3,2*ma,pi/2]
                            ]
                        ]
            elif self.Nq==4:
                self.ec_Nv = 2
                self.ec_Ns = 1
                self.ec_vert = np.matrix([
                    [1/1,1/2],
                    [0/1,1/2]])
                self.ec_para = [
                        [
                            [0],
                            [pi/2]
                            ]
                        ]
            elif self.Nq==12:
                arc = [
                        np.arccos(1/np.sqrt(2)),
                        np.arccos(1/np.sqrt(3)),
                        np.arccos(1/np.sqrt(4)),
                        np.arccos(1/np.sqrt(5)),
                        np.arccos(1/np.sqrt(6))]
                self.ec_Nv = 6
                self.ec_vert = np.matrix([
                    [1/1,1/2,1/3,1/4],
                    [0/1,1/2,1/3,1/4],
                    [0/1,0/2,1/3,1/4],
                    [0/1,0/2,0/3,1/4]])

    def _map_rdm_jw(self):
        self.qubit_to_rdm = {} #
        self.backend_to_rdm = {}
        if self.spin_mapping=='default':
            for n,alp in enumerate(self.alpha['active']):
                self.qubit_to_rdm[n]=alp
                self.alpha_qb.append(self.qubit_to_backend[n])
                self.backend_to_rdm[self.qubit_to_backend[n]]=alp
            for n,bet in enumerate(self.beta['active']):
                m = n+self.No
                self.qubit_to_rdm[m]=bet
                self.beta_qb.append(self.qubit_to_backend[m])
                self.backend_to_rdm[self.qubit_to_backend[m]]=bet
        elif self.spin_mapping=='spin-free':
            for n,alp in enumerate(self.alpha['active']):
                self.qubit_to_rdm[n]=alp
                self.alpha_qb.append(self.qubit_to_backend[n])
                self.backend_to_rdm[self.qubit_to_backend[n]]=alp
            for n,bet in enumerate(self.beta['active']):
                m = n+self.No
                self.qubit_to_rdm[m]=bet
                self.alpha_qb.append(self.qubit_to_backend[m])
                self.backend_to_rdm[self.qubit_to_backend[m]]=bet
        self.rdm_to_backend = {v:k for k,v in self.backend_to_rdm.items()}


    def _get_ent_pairs_jw(self):
        # ansatz also includes orders of excitations, etc. 
        self.pair_list = []
        self.quad_list = []
        if self.ansatz=='ucc':
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
        elif self.ansatz=='default':
            if self.ent_pairs in ['sd','d']:
                for l in range(0,len(self.alpha_qb)):
                    for k in range(0,l):
                        for j in range(0,k):
                            for i in range(0,j):
                                self.quad_list.append([i,j,k,l])
                for l in range(0,len(self.alpha_qb)):
                   for k in range(0,l):
                        for j in range(0,len(self.beta_qb)):
                            for i in range(0,j):
                                self.quad_list.append([i,j,k,l])
                for l in range(0,len(self.beta_qb)):
                    for k in range(0,l):
                        for j in range(0,k):
                            for i in range(0,j):
                                self.quad_list.append([i,j,k,l])
            if self.ent_pairs in ['s','sd']:
                for j in range(0,len(self.alpha_qb)):
                    for i in range(0,j):
                        self.pair_list.append([i,j])
                for j in range(0,len(self.beta_qb)):
                    for i in range(0,j):
                        self.pair_list.append([i,j])
            elif self.ent_pairs=='scheme1_Tavernelli':
                for j in range(1,len(self.alpha_qb)):
                    self.pair_list.append([j-1,j])
                for j in range(1,len(self.beta_qb)):
                    self.pair_list.append([j-1,j])
        elif self.ansatz=='natural-orbitals':
            if self.ent_pairs in ['sd','d']:
                # generate double excitations
                for l in range(self.No,self.No+len(self.beta_qb)):
                    for k in range(self.No,l):
                        i,j = k%self.No, l%self.No
                        self.quad_list.append([i,j,k,l])
        elif self.ansatz=='nat-orb-no':
            if self.ent_pairs in ['sd','d']:
                for k in range(self.No,self.No+len(self.beta_qb)-1):
                    i= k%self.No
                    self.quad_list.append([i,i+1,k,k+1])

    def _get_2e_no(self):
        self.tomo_quad = []
        temp = iter(zip(
                self.alpha['active'][:-1],
                self.alpha['active'][1:],
                self.beta['active'][:-1],
                self.beta['active'][1:]))
        for a,b,c,d in temp:
            self.tomo_quad.append([a,b,c,d])

    def _gip(self):
        '''
        'Get Initial Parameters (GIP) function.
        '''
        if self.sp=='noft':
            self.parameters=[0]*(self.No-1)
            self.Np = self.No-1
        elif self.sp=='rdm':
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

def get_direct_stats(QuantStore,extra=False):
    from hqca.tools import QuantumAlgorithms
    QuantStore.parameters=[1]*QuantStore.Np
    test = QuantumAlgorithms.GenerateDirectCircuit(QuantStore)
    if QuantStore.pr_g>1:
        print('# Getting circuit parameters...')
        print('#')
        print('# Gate counts:')
        print('#  N one qubit gates: {}'.format(test.sg))
        print('#  N two qubit gates: {}'.format(test.cg))
        print('# ...done.')
        print('# ')
    if extra=='draw':
        try:
            print(test.qc)
            test.qc.measure(test.q,test.c)
            test.qc.draw()
        except Exception as e:
            print(e)
    elif extra=='tomo':
        print('Trying to draw.')
        try:
            print(test.qc)
            test.qc.draw()
        except Exception as e:
            print(e)
    elif extra=='compile':
        from qiskit import Aer,IBMQ,execute
        from qiskit.mapper import Layout
        from qiskit.transpiler import transpile
        from qiskit.compiler import assemble_circuits
        from qiskit.tools.monitor import backend_overview
        import qiskit
        test.qc.measure(test.q,test.c)
        be = Aer.get_backend('qasm_simulator')
        IBMQ.load_accounts()
        ibm= IBMQ.get_backend('ibmqx4')
        coupling = ibm.configuration().coupling_map
        print(coupling)
        # going to try transpiler? yeah. 
        print(QuantStore.bec)
        print(test.qc)
        backend_overview()
        #layout = Layout()
        if QuantStore.bec is not None:
            layout = []
            for i in QuantStore.bec:
                if i is not None:
                    layout.append((test.q,i))
                else:
                    layout.append(None)
            #for n,i in enumerate(QuantStore.bec):
            #    if i is not None:
            #        layout[n] = (test.q,i)
            #    else:
            #        layout[n] = None
            print(layout)
            print('')
            print('space')
        else:
            layout = None
        qt = transpile(test.qc,
                backend=be,
                coupling_map=coupling,
                initial_layout=layout
                )
        #qo = be.run(qt)
        result = execute(qt,be).result()
        print(qt)
        print(result.get_counts())
        sys.exit()
    QuantStore.parameters=[0]*QuantStore.Np



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


