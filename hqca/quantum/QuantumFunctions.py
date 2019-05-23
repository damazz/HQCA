'''
tools/QuantumFunctions


'''
import numpy as np
import sys
from math import pi
from hqca.quantum import NoiseSimulator
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
            method='variational',
            **kwargs
            ):
        self.opt_kw = opt
        self.pr_g =pr_g
        self.method = method
        self.qc = qc
        if qc:
            self._set_up_backend(**kwargs)
            self._set_up_algorithm(**kwargs)
            self._set_up_error_correction(**kwargs)
        else:
            # need to check to make sure method is compatible
            classically_supported = ['borland-dennis','carlson-keller']
            if self.method in classically_supported:
                pass
            else:
                sys.exit('Trying to run a non-supported classical algorithm.')

    def _set_up_algorithm(self,
            pr_q=0,
            Nels_as=None,  
            Norb_as=None,
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
            **kwargs
            ):
        self.theory=theory
        self.spin_mapping = spin_mapping
        self.ansatz = ansatz
        self.No = Norb_as # note, spatial orbitals
        self.alpha = alpha_mos
        self.beta = beta_mos
        self.Ne = Nels_as # active space
        if spin_mapping in ['default','alternating']:
            self.Ne_alp = int(0.5*Nels_as+Sz)
        else:
            self.Ne_alp = int(Nels_as)
        self.tomo_bas = tomo_basis
        self.tomo_rdm = tomo_rdm
        self.tomo_ext = tomo_extra
        self.tomo_approx = tomo_approx
        self.ent_pairs= entangled_pairs
        self.ent_circ_p = entangler_p
        self.ent_circ_q = entangler_q
        self.depth = depth
        self.algorithm = compact_algorithm
        self.fermion_mapping = fermion_mapping
        if self.fermion_mapping=='jordan-wigner':
            self._map_rdm_jw()
            self._get_ent_pairs_jw()
            self._gip()
            if self.tomo_ext in ['sign_2e','sign_2e_pauli']:
                self._get_2e_no()

    def _set_up_backend(self,
            Nq=10,
            Nq_backend=20,
            backend='qasm_simulator',
            backend_coupling_layout=None,
            backend_initial_layout=None,
            noise=True,
            noise_model_location=None,
            circuit_times=None,
            transpile=False,
            num_shots=1024,
            provider='Aer',
            **kwargs):
        self.alpha_qb = [] # in the backend basis
        self.beta_qb  = [] # backend set of qubits
        self.be_initial = backend_initial_layout
        self.be_coupling = backend_coupling_layout
        self.Nq = Nq  # active qubits
        if Nq_backend is not None:
            self.Nq_tot = Nq_backend
        else:
            self.Nq_tot = self.Nq
        self.backend = backend
        self.Ns = num_shots
        self.provider = provider
        self.use_noise = noise
        if self.use_noise:
            self.noise_model = NoiseSimulator.get_noise_model(
                    device=backend,
                    times=circuit_times,
                    saved=noise_model_location)
        self.noise_model_loc = noise_model_location

    def _set_up_error_correction(self,
            pr_e=0,
            Nq_ancilla=0,
            ec=False,
            ec_method='parity',
            ec_ent_list ='default',
            **kwargs
            ):
        '''
        Note there are three types of error correction.
        (1) Post correction
        (2) Correction in a entangler, in circuit
        (3) Syndrome, in circuit
        '''
        self.Nq_ancilla = Nq_ancilla
        self.ec = ec
        self.ec_method = ec_method
        self.ec_type = None
        if self.ec_method=='hyperplane':
            self._get_hyper_para()
            self.ec_type='p'
        elif self.ec_method=='hyperplane+':
            self._get_hyper_para(expand=True)
            self.ec_type='p'
        else:
            self.ancilla_qb = []
            for i in range(self.Nq,self.Nq+Nq_ancilla):
                print(self.Nq)
                print(i)
                if i<=self.Nq_tot:
                    # basically your available ancilla
                    self.ancilla_qb.append(i)
        print(self.ancilla_qb)
        if self.ec_method=='parity':
            self.ec_circ = 'parity'
            self.ec_type = 's'
            self.ec_keys = []
            if ec_ent_list=='default':
                ec_ent_list = [1]*self.N_ent
            if len(ec_ent_list)<self.N_ent:
                print('## QuantumFunctions ##')
                print('Not enough specified ec functions for a parity check.')
                sys.exit('Goodbye!')
            else:
                self.ec_ent_list = ec_ent_list
            for item in ec_ent_list:
                if item==1:
                    self.ec_keys.append(1-(-1**self.Ne))
            self.ec_keys = zip(self.ec_keys,self.ancilla_qb)
        elif self.ec=='multi':
            pass


    def _get_hyper_para(self,expand=False):
        if self.method=='carlson-keller':
            if not expand:
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
                        temp.append(arc[j])
                    for k in range(i+1,self.Nv):
                        temp.insert(0,0)
                    temp = temp[::-1]
                    del temp[-1]
                    self.ec_para.append(temp)
                self.ec_para = [self.ec_para]
                print(self.ec_para)
            elif expand:
                arc = [2*np.arccos(1/np.sqrt(i)) for i in range(1,self.Nq+1)]
                self.ec_para = []
                self.ec_Ns = 1
                self.Nv = self.Nq//2
                self.ec_Nv = self.Nv
                self.ec_vert = np.zeros((self.Nv,self.Nv))
                self.mid = np.zeros(self.Nv)
                for i in range(0,self.Nv):
                    for j in range(i,self.Nv):
                        self.mid[i]+=1/(j+1)
                self.mid = self.mid/self.Nv
                diff = np.zeros((self.Nv,self.Nv))
                for i in range(0,self.Nv):
                    temp = []
                    for j in range(0,i+1):
                        self.ec_vert[j,i]=1/(i+1)
                        temp.append(arc[j])
                    for k in range(i+1,self.Nv):
                        temp.insert(0,0)
                    temp = temp[::-1]
                    del temp[-1]
                    self.ec_para.append(temp)
                for i in range(0,self.Nv):
                    diff[:,i] = self.ec_vert[:,i]-self.mid
                    self.ec_vert[:,i] = self.ec_vert[:,i]+0.1*diff[:,i]
                self.ec_para = [self.ec_para]

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
                    print(k)
                    i= k%self.No
                    self.quad_list.append([i,i+1,k,k+1,'-+-+','aabb'])
                    print(i,i+1,k,k+1)
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
        self.N_ent = len(self.quad_list)+len(self.pair_list)

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
            for i in range(len(quad)):
                qd.append(self.rdm_to_qubit[quad[i]])
            sort = False
            while not sort:
                i,j,k,l = qd[0],qd[1],qd[2],qd[3]
                sort = True
                if i>j:
                    qd[1],qd[0]=qd[0],qd[1]
                    sort = False
                if j>k:
                    qd[1],qd[2]=qd[2],qd[1]
                    sort = False
                if k>l:
                    qd[3],qd[2]=qd[2],qd[3]
                    sort = False
            self.qc_tomo_quad.append(qd)

    def print_summary(self):
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

def get_direct_stats(QuantStore,extra=False):
    '''
    function to do a few statistics on the proposed circuit,
    namely it can:
        do simple circuit gate counts
        draw the pre- and post- compiled/transpiled circuits

    '''
    from hqca.quantum import BuildCircuit
    QuantStore.parameters=[1]*QuantStore.Np
    test = BuildCircuit.GenerateDirectCircuit(QuantStore)
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
    elif extra in ['compile','qasm']:
        def print_qasm(circuit):
            print(circuit.qasm())
            print(circuit.count_ops())
            with open('h3_qasm.txt','w') as fp:
                fp.write(circuit.qasm())
        from qiskit import Aer,IBMQ,execute
        from qiskit.mapper import Layout
        from qiskit.transpiler import transpile
        from qiskit.compiler import assemble_circuits
        from qiskit.tools.monitor import backend_overview
        import qiskit
        IBMQ.load_accounts()
        test.qc.measure(test.q,test.c)
        be = Aer.get_backend('qasm_simulator')
        if extra=='compile':
            print(test.qc)
        else:
            import os 
        #layout = Layout()
        if QuantStore.be_coupling in [None,False]:
            IBMQ.load_accounts()
            backend_overview()
            beo = IBMQ.get_backend(QuantStore.backend)
            coupling = beo.configuration().coupling_map
        else:
            try:
                coupling = NoiseSimulator.get_coupling_map(
                        device=QuantStore.backend,
                        saved=QuantStore.noise_model_loc
                        )
            except Exception as e:
                print(e)
                sys.exit()
        if QuantStore.bec is not None:

            layout = []
            for i in QuantStore.bec:
                layout.append(i)
            '''
            for i in QuantStore.bec:
                if i is not None:
                    layout.append((test.q,i))
                else:
                    layout.append(None)
            '''
        else:
            layout = None
        qt = transpile(test.qc,
                backend=be,
                coupling_map=coupling,
                initial_layout=layout
                )
        #qo = be.run(qt)
        #result = execute(qt,be).result()
        print_qasm(qt)
        print(qt)

        #print(result.get_counts())
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


