"""
The core for the ACSE portion of the hqca module. Contains the RunACSE class,
which focuses on a quantum generation of the 2-RDM, with classical and quantum
generation of the A matrix.
"""

import pickle
import warnings

from hqca.acse._ansatz_S import *
from hqca.acse._check_acse import check_routine
from hqca.acse._class_A_acse import *
from hqca.acse._bfgs_acse import _bfgs_step
from hqca.acse._conjugate_acse import _conjugate_gradient_step
from hqca.acse._euler_acse import _euler_step
from hqca.acse._mitigation import *
from hqca.acse._newton_acse import _newton_step
from hqca.acse._quant_A_acse import *
from hqca.processes import IterativeUnitarySimulator
from hqca.acse._qubit_A import *
from hqca.acse._tools_acse import Log
from hqca.core import *
import scipy.sparse as sparse
warnings.simplefilter(action='ignore', category=FutureWarning)


class RunACSE(QuantumRun):
    """Generates a anti-hermitian Contracted Schrodinger Equation run object. 

    The ACSE CQE attempts to find a solution to the anti-hermitian component 
    of the contracted Schrodinger equation. Principly, we do this by finding
    a solution of the A matrix, and then obtaining a new wavefunction. 

    Attributes:
        en (float): current energy
        norm (float): current norm of the 2A matrix 
        var (float): current variance
        psi (Ansatz): entire ansatz
        S (list): given residual vector at a particular iteration
        qs (QuantumStorage): relevant quantumstorage
        store (Storage): relevant storage
        ins (insions): relevant instructions
        process (None or StandardProcess): relevant process
    """

    def __init__(self, storage, quantstore, instructions, **kw):
        super().__init__(**kw)
        self.store = storage
        self.qs = quantstore
        self.ins = instructions
        self._update_acse_kw(**kw)

    def _update_acse_kw(self,
                        method='euler',
                        update='quantum',
                        opt_thresh=1e-8,
                        max_iter=100,
                        expiH_approximation='first',
                        S_thresh_rel=0.1,
                        S_min=1e-10,
                        S_num_terms=None,
                        convergence_type='default',
                        hamiltonian_step_size=0.1,
                        epsilon=1.0,
                        separate_hamiltonian=None,
                        trunc_method='delta',
                        trunc_include=False,
                        verbose=True,
                        tomo_A=None,
                        tomo_psi=None,
                        statistics=False,
                        processor=None,
                        max_depth=None,
                        transform_psi =None,
                        A_norm=None, #numpy keyword corresponding to numpy.linalg.norm
                        output=0,
                        **kw):
        '''
        Updates the ACSE keywords. 
        '''
        self._output = output
        if update in ['quantum', 'Q', 'q', 'qso', 'qfo']:
            self.acse_update = 'q'
        elif update in ['class', 'classical', 'c', 'C']:
            self.acse_update = 'c'
        elif update in ['para', 'p']:
            self.acse_update = 'p'
        elif update in ['user','u']:
            self.acse_update = 'u'
        elif update in ['hybrid','h']:
            self.acse_update = 'h'
        else:
            raise QuantumRunError
        if not method in [
                'newton',
                'euler','bfgs','lbfgs',
                'cg',
                ]:
            raise QuantumRunError('Specified method not valid. Update acse_kw: \'method\'')
        if not A_norm in [2,'inf',None]:
            raise QuantumRunError('Specified norm not valied. Update acse_kw: \' A_norm\'')

        self.process = processor
        self.verbose = verbose
        self.stats = statistics
        self.acse_method = method
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.crit = opt_thresh
        self.psi,self.S = None,None
        self.hamiltonian_step_size = hamiltonian_step_size
        self.sep_hamiltonian = separate_hamiltonian
        self.S_expiH_approx = expiH_approximation
        self.S_thresh_rel = S_thresh_rel
        self.S_min = S_min
        self.A_norm = A_norm
        self.S_num_terms = S_num_terms
        self.trunc_method = trunc_method
        self.trunc_include = trunc_include
        self.epsilon = epsilon
        self._conv_type = convergence_type
        self.tomo_A = tomo_A
        self.tomo_psi = tomo_psi
        self.transform_psi = transform_psi
        assert not type(tomo_psi)==type(None), 'Need to specify tomography of psi!'
        assert not type(tomo_A)==type(None), 'Need to specify tomography of the A matrix!'
        self.rho = None
        self._A_as_matrix = False
        if type(self.tomo_psi) == type(None):
            self.tomo_preset = False
        else:
            self.tomo_preset = True
        if self.verbose:
            print('\n\n')
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('      --  ACSE KEYWORDS --      ')
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('algorithm')
            print('-- -- -- --')
            print('ACSE method: {}'.format(method))
            print('ACSE update: {}'.format(update))
            print('max iterations: {}'.format(max_iter))
            print('max depth: {}'.format(max_depth))
            print('convergence type: {}'.format(convergence_type))
            print('convergence threshold: {}'.format(self.crit))

            print('-- -- -- --')
            print('ACSE solution')
            print('-- -- -- --')
            if self.acse_update == 'q':
                print('hamiltonian delta: {}'.format(hamiltonian_step_size))
            print('S rel threshold: {}'.format(S_thresh_rel))
            print('S max threshold: {}'.format(S_min))
            print('-- -- -- --')
            print('ansatz')
            print('-- -- -- --')
            print('S epsilon: {}'.format(self.epsilon))
            print('-- -- -- --')
            print('optimization')
            print('-- -- -- --')
        self._optimizer = None
        self._opt_thresh = None
        if self.acse_method == 'newton':
            kw = self._update_acse_newton(**kw)
        elif self.acse_method in ['line',]:
            kw = self._update_acse_opt(**kw)
        elif self.acse_method in ['bfgs','lbfgs']:
            kw = self._update_acse_bfgs(**kw)
        elif self.acse_method in ['cg']:
            kw = self._update_acse_cg(**kw)
        kw = self._update_experimental(**kw)
        kw = self._update_ACSE(**kw)
        if len(kw)>0:
            print('Unused or improper keywords: ')
            for k in kw:
                print(k)
        self.grad = 0

    def _update_ACSE(self,
            D3=None,**kw):
        self.tomo_D3 = D3
        return kw

    def _update_experimental(self,
                             split_ansatz=False,
                             split_threshold=1.0,
                             **kw
                             ):
        self.split_ansatz = split_ansatz
        self.split_threshold = split_threshold
        return kw

    def _update_acse_cg(self,
            cg_update='PR+',
            cg_reset_beta=False,
            **kw):
        self._A_as_matrix = True
        self._log_psi = [] #used for p-depth tracking
        self.log_A = []
        self.log_p = []
        if cg_update in ['FR','HS','PR','PR+','HZ']:
            self._cg_update = cg_update
            print('CG Update: {}'.format(cg_update))
        else:
            raise QuantumRunError('Unspecified update.')

        return kw

    def _update_acse_bfgs(self,
                          optimizer_threshold=0.01,
                          bfgs_limited=3,
                          bfgs_update='',
                          bfgs_restart=False,
                          **kw):
        if self.acse_method=='lbfgs':
            print('L-BFGS with {} steps... '.format(bfgs_limited))
        self._A_as_matrix = True
        self._limited = bfgs_limited
        self._opt_thresh = optimizer_threshold
        self._bfgs_restart = bfgs_restart
        self._update_step = None
        self._log_psi = [] #used for p-depth tracking
        self.log_A = []
        self.log_p = []
        return kw

    def _update_acse_newton(self,
                            use_trust_region=False,
                            newton_step=+2,
                            initial_trust_region=np.pi / 2,
                            tr_taylor_criteria=1e-10,
                            tr_objective_criteria=1e-10,
                            tr_gamma_inc=2,
                            tr_gamma_dec=0.5,
                            tr_nu_accept=0.9,
                            tr_nu_reject=0.1,
                            **kw
                            ):
        self.use_trust_region = use_trust_region
        self.d = newton_step  # for estimating derivative
        self.tr_ts_crit = tr_taylor_criteria
        self.tr_obj_crit = tr_objective_criteria
        self.tr_Del = initial_trust_region  # trust region
        self.tr_gi = tr_gamma_inc
        self.tr_gd = tr_gamma_dec
        self.tr_nv = tr_nu_accept  # very good?
        self.tr_ns = tr_nu_reject  # shrink
        self.epsilon = self.epsilon*-1
        if self.verbose:
            print('newton step: {}'.format(newton_step))
            print('newton trust region: {}'.format(use_trust_region))
            print('trust region: {:.6f}'.format(initial_trust_region))
        self.tr_taylor = 1
        self.tr_object = 1
        return kw


    def _generate_circuit(self, op=None,
            tomo=None,
            order=2,
            compact=False,
            ins_kwargs={},
            initial=False):
        if type(op)==type(None):
            op = self.psi
        if type(tomo)==type(None):
            tomo = self.tomo_psi
        if isinstance(tomo,type(QubitTomography())):
            circ = QubitTomography(
                quantstore=self.qs,
                preset=self.tomo_preset,
                tomo=tomo,
                verbose=self.verbose,
            )
        elif isinstance(tomo,type(StandardTomography())):
            circ = StandardTomography(
                quantstore=self.qs,
                preset=self.tomo_preset,
                tomo=tomo,
                verbose=self.verbose,
            )
        if self.qs.be_type=='rho':
            if len(op)>0 and self.psi.p==0:
                op = op[-1]
            elif len(op)>0 and self.psi.p>0:
                pass

            op = op.transform(self.transform_psi)
            ins = self.ins(
                operator=op,
                Nq=self.qs.Nq,
                quantstore=self.qs,
                order=order,
                **ins_kwargs)
            sim = IterativeUnitarySimulator()
            sim.set(
                    quantstore=self.qs,
                    initial=initial,
                    instruct=ins,
                    )
            sim.simulate(
                    tomo=circ,
                    rho_i=self.rho)
        else:
            op = op.transform(self.transform_psi)
            ins = self.ins(
                operator=op,
                Nq=self.qs.Nq,
                quantstore=self.qs,
                order=order,
                **ins_kwargs)
            circ.set(ins)
            circ.simulate()
            circ.rho = None
        circ.construct(processor=self.process,compact=compact)
        return circ

    @property
    def S(self):
        try: 
            return self.psi
        except AttributeError:
            return self.S

    @S.setter
    def S(self,a):
        self.psi = a

    def build(self):
        if self.verbose:
            print('\n\n')
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('building the ACSE run')
            print('-- -- -- -- -- -- -- -- -- -- --')
        if self.store.use_initial:
            try:
                self.psi = copy(self.psi)
                en = np.real(self.store.evaluate(self.store.rdm))
            except Exception as e:
                print(e)
                self.qs = copy(self.store.psi)
                circ = self._generate_circuit(self.psi)
                self.store.rdm = circ.rdm
                en = np.real(self.store.evaluate(circ.rdm))
            self.e_k = np.real(en)
            self.ei = np.real(en)
            if self.verbose:
                print('Initial energy: {:.8f}'.format(self.ei))
            if self.verbose:
                print('S: ')
                print(self.psi)
                print('Initial density matrix.')
                self.store.rdm.contract()
                print(np.real(self.store.rdm.rdm))
        else:
            self.psi = copy(self.store.psi)
            self.e_k = self.store.e0
            self.ei = self.store.ei
            if self.verbose:
                print('taking energy from storage')
                print('initial energy: {:.8f}'.format(np.real(self.e_k)))
        if self.qs.be_type=='rho':
            print('Attempting a iterative solution of the ACSE -  ')
            print('Checking if run is compatible...')
            #assert self.psi.closed==1
            print('Continuing...')
            self.rho = np.zeros(
                    (2**self.qs.Nq_tot,2**self.qs.Nq_tot),
                        dtype=np.complex_)
            self.rho[0,0] = 1
            circ = self._generate_circuit(op=self.psi,initial=True)
            self.rho = copy(circ.rho)
            self.store.update(circ.rdm)
        self._psi_old = copy(self.psi)
        self.best,self.grad = self.e_k,0
        self.best_avg = self.e_k

        self.e0 = self.e_k
        self.total = Cache()
        self.accept_previous_step = True
        #
        self.rdme = self.tomo_psi.rdme_keys[:]
        self.rdme_mapping = {}
        for i in range(len(self.rdme)):
            str_form = ' '.join([str(k) for k in self.rdme[i]])
            self.rdme_mapping[str_form]=i
        if self.acse_method=='bfgs':
            dim = len(self.rdme)
            self.Bi_k0 = np.identity(1*dim)
        if self.acse_method=='lbfgs':
            self._lbfgs_yk = []
            self._lbfgs_rk = []
            self._lbfgs_r = []
            self._lbfgs_sk = []
        #
        # update norms 
        #
        if type(self.transform_psi)==type(None):
            self.transform_psi = copy(self.store.H._transform)
        self._get_A()
        if self.acse_method in ['bfgs','cg','lbfgs',]:
            self.p = -np.asmatrix([self.A]).T
        self.log_E = [self.e_k]
        self.log_norm = [self.norm]


        if self.verbose:
            print('||A||: {:.10f}'.format(np.real(self.norm)))
            print('-- -- -- -- -- -- -- -- -- -- --')
        # run checks
        check_routine(self)
        self.built = True
        if self._output:
            print('Step {:02}, E: {:.12f}, S: {:.12f}'.format(
                self.total.iter,
                np.real(self.e_k),
                np.real(self.norm)))

    def _get_A(self):
            # 
        if self.acse_update == 'q':
            if type(self.sep_hamiltonian)==type(None):
                H = self.store.H.qubit_operator
            else:
                H = self.sep_hamiltonian
            A_sq = solveqACSE(
                self,
                H=H,
                operator=self.psi,
                S_min=self.S_min,
                hamiltonian_step_size=self.hamiltonian_step_size,
                expiH_approx=self.S_expiH_approx,
                verbose=self.verbose,
                tomo=self.tomo_A,
                matrix=self._A_as_matrix,
                norm=self.A_norm,
                )
        elif self.acse_update == 'c':
            if not self.accept_previous_step:
                if self.verbose:
                    print('Rejecting previous step. No recalculation of A.')
                return
            A_sq = solvecACSE(
                self,
                S_min=self.S_min,
                verbose=self.verbose,
                matrix=self._A_as_matrix,
                keys=self.rdme,
                norm=self.A_norm,
                tomo=self.tomo_A,
                operator=self.psi,
            )
        elif self.acse_update == 'p':
            # TODO: need to update
            if type(self.sep_hamiltonian)==type(None):
                H = self.store.H.qubit_operator
            else:
                H = self.sep_hamiltonian
            A_sq = solvepACSE(
                self,
                H=H,
                operator=self.psi,
                S_min=self.S_min,
                hamiltonian_step_size=self.hamiltonian_step_size,
                verbose=self.verbose,
                tomo=self.tomo_A,
                matrix=self._A_as_matrix,
                norm=self.A_norm,
                )
        else:
            raise QuantumRunError
        if self._A_as_matrix:
            # used in BFGS implementation
            self.norm = np.linalg.norm(A_sq,ord=self.A_norm)
            # factor of sqrt(4) accounts for quadruple counting of symmetries
            self.A = A_sq
        else:
            ## ## TODO: need to redo 
            # we use the actual 2A matrix, no need to renormalize
            A_sq, norm = A_sq[0],A_sq[1]
            print('norm...',norm)
            self.norm = norm
            if self._output==2:
                print('Particle operator')
                print(A_sq)
            if self.split_ansatz:
                raise QuantumRunError('Split ansatz needs update.')
                max_v, norm = 0, 0
                new = Operator()
                for op in A_sq:
                    if abs(op.c) >= abs(max_v):
                        max_v = copy(op.c)
                for op in A_sq:
                    if abs(op.c) >= abs(self.S_thresh_rel * max_v):
                        new += op
                if self.acse_update in ['c','q']:
                    A = A_sq.transform(self.qs.transform)
                    A = A_sq
                elif self.acse_update in ['p']:
                    A = A_sq.transform(self.qs.qubit_transform)
                #
                inc = Operator()
                exc = Operator()
                for n in A:
                    added = False
                    for m in reversed(range(self.psi.get_lim(), 0)):
                        # now, we check if in previous ansatz
                        ai = self.psi[m]
                        for o in ai:
                            if n == o:
                                inc += n
                                added = True
                                # 
                    if not added:
                        exc += n
                ninc = Operator()
                nexc = Operator()
                max_inc, max_exc = 0,0
                if self.verbose:
                    print(A)
                for op in inc:
                    if abs(op.c) >= abs(max_inc):
                        max_inc = copy(op.c)
                for op in inc:
                    if abs(op.c) >= abs(self.S_thresh_rel * max_inc):
                        ninc += op
                for op in exc:
                    if abs(op.c) >= abs(max_exc):
                        max_exc = copy(op.c)
                for op in exc:
                    if abs(op.c) >= abs(self.S_thresh_rel * max_exc):
                        nexc += op
                #
                if self.verbose:
                    print('--------------')
                    print('Included in previous ansatz: ')
                    print(ninc)
                    print('New exterior terms: ')
                    print(nexc)
                    max_val = 0
                if ninc.norm() == 0 or nexc.norm() == 0:
                    new = Operator()
                    for op in A:
                        norm += op.norm()**2
                        if abs(op.c) >= abs(max_val):
                            max_val = copy(op.c)
                    for op in A:
                        if abs(op.c) >= abs(self.S_thresh_rel * max_val):
                            new += op
                    self.A = copy(new)
                elif exc.norm() / inc.norm() > self.split_threshold:
                    print('Exc > Inc * thresh')
                    print('Added terms:')
                    # 
                    self.A = copy(nexc)
                else:
                    print('Exc < Inc * thresh')
                    self.A = copy(ninc)
                norm = 0
            else:
                #
                max_val, norm = 0, 0
                new = Operator()
                for op in A_sq:
                    norm += op.norm()**2
                    if abs(op.c) >= abs(max_val):
                        max_val = copy(op.c)
                for op in A_sq:
                    if abs(op.c) >= abs(self.S_thresh_rel * max_val):
                        new += op
                t0 = dt()
                #self.A = new.transform(self.transform_psi)
                self.A = new
                pauli = new.transform(self.transform_psi)

                if self.verbose:
                    print('A operator (fermi)')
                    print(new)
                    print('A operator (pauli')
                    print(pauli)

    def _run_acse(self):
        '''
        Function to the run the ACSE algorithm

        Note, the algorithm is configured to optimize the energy, and then
        calculate the residual of the ACSE.
        '''
        if self.verbose:
            print('\n\n')
        check_mitigation(self)
        try:
            self.built
        except AttributeError:
            sys.exit('Not built! Run acse.build()')
        if self.acse_method in ['NR', 'newton']:
            _newton_step(self)
            self._get_A()
        elif self.acse_method in ['default', 'em', 'EM', 'euler']:
            _euler_step(self)
            self._get_A()
        elif self.acse_method in ['line', 'opt']:
            _opt_step(self)
            self._get_A()
        elif self.acse_method in ['bfgs']:
            _bfgs_step(self)
        elif self.acse_method in ['lbfgs']:
            _bfgs_step(self,limited=True)
        elif self.acse_method in ['cg']:
            _conjugate_gradient_step(self)
        else:
            raise QuantumRunError('Incorrect acse_method.')
        # 
        if self.qs.be_type=='rho':
            if self.psi.p==0:
                self.rho  = copy(self.circ.rho)
            elif self.psi.p>0:
                if len(self.psi)>self.psi.p:
                    #print('Psi increased! Updating rho')
                    self.circ = self._generate_circuit(self.psi.A[0])
                    self.rho = copy(self.circ.rho)
                    self._psi_old+= self.psi.A.pop(0)
                    #del self.psi.A[0]





            # check to see if len(psi) increased

    def _particle_number(self, rdm):
        return rdm.trace()

    def next_step(self):
        if self.built:
            self._run_acse()
            self._check()
            if self.verbose:
                #print('E,init: {:+.12f} U'.format(np.real(self.ei)))
                print('E,iter: {:+.12f} U'.format(np.real(self.best)))
                try:
                    diff = 1000 * (self.best - self.store.H.ef)
                    print('E, fin: {:+.12f} U'.format(self.store.H.ef))
                    print('E,diff: {:.12f} mU'.format(diff))
                except KeyError:
                    pass
                except AttributeError:
                    pass

    def reset(self,full=False):
        if not full:
            self.store.use_initial=True
            self.build()
        else:
            self.store.use_initial=False
            self.build()

    def run(self):
        """
        Note, run for any ACSE has the generic steps:
            - find the S matrix,
            - build the S ansatz
            - evaluate the ansatz, or D, evaluate energy
        """
        if self.built:
            while not self.total.done:
                self._run_acse()
                self._check()
            print('')
            #print('E init: {:+.12f} U'.format(np.real(self.ei)))
            print('E run : {:+.12f} U'.format(np.real(self.best)))
            try:
                diff = 1000 * (self.best - self.store.H.ef)
                print('E goal: {:+.12f} U'.format(self.store.H.ef))
                print('Energy difference from goal: {:.12f} mU'.format(diff))
            except KeyError:
                pass
            except AttributeError:
                pass

    def _check(self):
        '''
        Internal check on the energy as well as norm of the S matrix
        '''
        en = self.store.evaluate(self.store.rdm)
        if self.total.iter == 0:
            self.best = copy(self.e_k)
        self.total.iter += 1
        if self.total.iter == self.max_iter:
            print('Max number of iterations met. Ending optimization.')
            self.total.done = True
        elif len(self.psi) == self.max_depth:
            if copy(self.psi) + copy(self.A) > self.max_depth:
                print('Max ansatz depth reached. Ending optimization.')
                self.total.done = True

        # updating logs...
        self.log_E.append(np.real(en))
        self.log_norm.append(np.real(self.norm))
        #
        i = 1
        temp_std_En = []
        temp_std_S = []
        while i <= min(3, self.total.iter+1):
            temp_std_En.append(self.log_E[-i])
            temp_std_S.append(self.log_norm[-i])
            i += 1
        avg_En = np.real(np.average(np.asarray(temp_std_En)))
        avg_S = np.real(np.average(np.asarray(temp_std_S)))
        std_En = np.real(np.std(np.asarray(temp_std_En)))
        std_S = np.real(np.std(np.asarray(temp_std_S)))
        if self.verbose:
            #self.store.analysis()
            print('')
            print('---------------------------------------------')
            print('Step {:02}, E: {:.12f}, S: {:.12f}'.format(
                self.total.iter,
                np.real(en),
                np.real(self.norm)))
            print('-- -- -- -- --')
            print('Steps: {} -> {}'.format(max(0,self.total.iter-2),self.total.iter))
            print('<E>: {:+.8f} +/- {:.8f}'.format(avg_En,std_En))
            print('<S>: {:+.8f} +/- {:.8f}'.format(avg_S,std_S))

        if avg_En <= en:
            # increasing energy? 
            print('Average energy increasing!')
            print(self.log_E)
            print(temp_std_En,avg_En,en)
            try:
                self.increasing+=1 
            except Exception as e:
                self.increasing = 1
            if self.increasing>=3:
                self.total.done = True

            #self.total.done=True
        if self._output >= 1:
            print('Step {:02}, E: {:.12f}, S: {:.12f}'.format(
                self.total.iter,
                np.real(en),
                np.real(self.norm)))
        if en < self.best:
            self.best = np.real(en)
        #
        if self._conv_type in ['trust']:
            if not self.verbose and self._output > 0:
                print('Taylor: {:.10f}, Objective: {:.10f}'.format(
                    self.tr_taylor.real, self.tr_object.real))
            if abs(self.tr_taylor) <= max(self.tr_ts_crit,self.crit):
                self.total.done = True
                if self.verbose:
                    print('optimization status 0: criteria met in taylor series model.')
                    print('...ending optimization')
            elif abs(self.tr_object) <= max(self.tr_obj_crit,self.crit):
                self.total.done = True
                print('Criteria met in objective function.')
                print('Ending optimization.')
        elif self._conv_type in ['S-norm', 'norm']:
            if self.norm < self.crit:
                self.total.done = True
        else:
            raise QuantumRunError('Convergence type not specified.')
        self.e_k = copy(en)
        if self.verbose:
            print('---------------------------------------------')
        self.e0=self.e_k

    def save(self,
             name,
             description=None
             ):
        try:
            self.log_A
        except AttributeError:
            sys.exit('Forgot to turn logging on!')
        data = {
            'H': self.store.H.matrix,
            'run_config': {
                'method': self.acse_method,
                'verbose': self.verbose,
                'S_thresh_rel': self.S_thresh_rel,
                'S_min': self.S_min,
                'S_num_terms': self.S_num_terms,
                'update': self.acse_update,
                'opt_threshold': self.crit,
                'max_depth': self.max_depth,
                'H size': self.hamiltonian_step_size,
                'separate hamiltonian': self.sep_hamiltonian,
                'convergence type': self._conv_type,
                'optimizer': self._optimizer,
                'optimizer_threshold': self._opt_thresh,
            },
            'quantum_storage': {
                'backend': self.qs.backend,
                'provider': self.qs.provider,
                'number of qubits': self.qs.Nq,
                'number of shots': self.qs.Ns,
                'stabilizers': self.qs.method,
            },
            'description': description,
        }
        try:
            data['log-Gamma'] = self.log_Gamma
        except AttributeError as e:
            pass
        with open(name + '.log', 'wb') as fp:
            pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)

    def check_variance(self,H):
        try:
            self.rho
        except AttributeError as e:
            return None
        E = np.dot(self.rho,H).trace()
        v1 = np.dot(self.rho,np.dot(H,H)).trace()
        print('    -> Var: {}'.format(-E**2+v1))
        return np.real(v1-E**2)





