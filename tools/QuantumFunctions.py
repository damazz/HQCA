'''
tools/QuantumAlgorithms
test module to handle all of the nitty gritty of the quantum algorithm...in
particular, should handle:
    - mappings
    - entanglement schemes
    - circuit design, etc. 
    - .....maybe connecting alot of stuff

for computation on the computer, we need to know....alpha,beta orbital sfor the
RDM functions? yes and no. maybe not. unrestricted form will help. 
but...qc is only concerned with mapping out a RDM. that is accomplsiehd through
algorithms and tomography. but we still need to handle the mappings.....
so we have 

'''
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
            tri,
            load_triangle,
            Sz,
            depth,
            alpha_mos,
            beta_mos,
            single_point,
            fermion_mapping='jordan-wigner',
            backend_configuration=None,
            variational='default',
            spin_mapping='default',
            method='variational',
            compiler=None,
            initialize='default',
            algorithm=None,
            pr_q=0,
            pr_t=0,
            use_radians=False
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
        self.use_radians=use_radians
        self.Ns = num_shots
        self.Nq = Nqb  # active qubits
        self.Ne = Nels_as # active space
        if spin_mapping=='default':
            self.Ne_alp = int(0.5*Nels_as+Sz)
        else:
            self.Ne_alp = int(Nels_as)
        self.No = Norb_as # note, spatial orbitals
        if self.No>self.Ne:
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
        self.tri=tri,
        self.sp = single_point
        self.alpha = alpha_mos
        self.beta = beta_mos
        self.pr_q = pr_q
        self.variational = variational
        self.spin_mapping = spin_mapping
        self.qubit_to_backend = backend_configuration
        if self.qubit_to_backend==None:
            self.qubit_to_backend=KeyDict()
            self.backend_to_qubit=KeyDict()
        else:
            self.backend_to_qubit={v:k for k,v in self.qubit_to_backend.items()}
        self.compiler = compiler
        self.fermion_mapping = fermion_mapping
        self.alpha_qb = [] # in the backend basis
        self.beta_qb  = [] # backend set of qubits
        if self.fermion_mapping=='jordan-wigner':
            self._map_rdm_jw()
            self._get_ent_pairs_jw()
            self._gip()



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
        self.pair_list = []
        self.quad_list = []
        if self.variational=='ucc':
            if self.ent_pairs in ['s','sd']:
                for j in range(0,len(self.alpha_qb)):
                    for i in range(0,j+1):
                        self.pair_list.append([i,j])
                for j in range(0,len(self.beta_qb)):
                    for i in range(0,j+1):
                        self.pair_list.append([i,j])
            if self.ent_pairs=='sd':
                for l in range(0,len(self.alpha_qb)):
                    for k in range(0,l+1):
                        for j in range(0,k+1):
                            for i in range(0,j+1):
                                self.quad_list.append([i,j,k,l])
                for l in range(0,len(self.alpha_qb)):
                    for k in range(0,l+1):
                        for j in range(0,len(self.beta_qb)):
                            for i in range(0,j+1):
                                self.quad_list.append([i,j,k,l])
                for l in range(0,len(self.beta_qb)):
                    for k in range(0,l+1):
                        for j in range(0,k+1):
                            for i in range(0,j+1):
                                self.quad_list.append([i,j,k,l])
        elif self.variational=='default':
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
            if self.ent_pairs=='sd':
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

    def _gip(self):
        '''
        'Get Initial Parameters (GIP) function.
        '''
        if self.sp=='noft':
            self.parameters=[0,0]
        elif self.sp=='rdm':
            self.c_ent_p=1
            self.c_ent_q=1
            if self.ent_circ_p=='Uent1_cN':
                self.c_ent_p=2
            self.Np = self.depth*len(self.pair_list)*self.c_ent_p
            self.Np+= self.depth*len(self.quad_list)*self.c_ent_q
            self.parameters=[0]*self.Np

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

nonlocal_qubit_tomo_pairs_full = {
        i:[
            ['{}{}'.format(
                b,a)]
                for a in range(i) for b in range(a)
            ] for i in range(2,9)}


