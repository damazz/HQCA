"""
The core for the ACSE portion of the hqca module. Contains the RunACSE class,
which focuses on a quantum generation of the 2-RDM, with classical and quantum
generation of the A matrix.
"""

import pickle
import warnings

from hqca.acse._ansatz_S import *
from hqca.acse._check_acse import check_routine
from hqca.acse._class_S_acse import *
from hqca.acse._bfgs_acse import _bfgs_step
from hqca.acse._conjugate_acse import _conjugate_gradient_step
from hqca.acse._euler_acse import _euler_step
from hqca.acse._mitigation import *
from hqca.acse._newton_acse import _newton_step
from hqca.acse._opt_acse import _opt_step
from hqca.acse._quant_S_acse import *
from hqca.acse._user_A import *
from hqca.acse._qubit_A import *
from hqca.acse._tools_qacse import Log
from hqca.core import *
import scipy.sparse as sparse
warnings.simplefilter(action='ignore', category=FutureWarning)


class RunACSE(QuantumRun):
    """
    Quantum ACSE method.
    """

    def __init__(self, storage, quantstore, instructions, **kw):
        super().__init__(**kw)
        self.Store = storage
        self.QuantStore = quantstore
        self.Instruct = instructions
        self._update_acse_kw(**kw)

    def _update_acse_kw(self,
                        method='newton',
                        update='quantum',
                        opt_thresh=1e-8,
                        max_iter=100,
                        expiH_approximation='first',
                        S_thresh_rel=0.1,
                        S_min=1e-10,
                        S_num_terms=None,
                        convergence_type='default',
                        hamiltonian_step_size=0.1,
                        restrict_S_size=1.0,
                        separate_hamiltonian=None,
                        truncation=['abs'],
                        verbose=True,
                        tomo_S=None,
                        tomo_Psi=None,
                        tomo_Sq=None,
                        statistics=False,
                        processor=None,
                        max_depth=None,
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
                'opt', 'newton',
                'euler', 'line','bfgs',
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
        self.hamiltonian_step_size = hamiltonian_step_size
        self.sep_hamiltonian = separate_hamiltonian
        self.S_expiH_approx = expiH_approximation
        self.S_thresh_rel = S_thresh_rel
        self.S_min = S_min
        self.A_norm = A_norm
        self.S_num_terms = S_num_terms
        self.S_trunc = truncation
        self.delta = restrict_S_size
        self.epsilon = restrict_S_size
        self._conv_type = convergence_type
        self.tomo_S = tomo_S
        self.tomo_Sq = tomo_Sq
        self.tomo_Psi = tomo_Psi
        self._A_as_matrix = False
        if type(self.tomo_Psi) == type(None):
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
            print('S epsilon: {}'.format(self.delta))
            print('-- -- -- --')
            print('optimization')
            print('-- -- -- --')
        self._optimizer = None
        self._opt_thresh = None
        if self.acse_method == 'newton':
            kw = self._update_acse_newton(**kw)
        elif self.acse_method in ['line', 'opt']:
            kw = self._update_acse_opt(**kw)
        elif self.acse_method in ['bfgs']:
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

    def _update_acse_opt(self,
                         optimizer='nm',
                         optimizer_threshold='default',
                         **kw,
                         ):
        if self.verbose:
            print('optimizer : {}'.format(optimizer))
            print('optimizer threshold: {}'.format(optimizer_threshold))
        self._optimizer = optimizer
        self._opt_thresh = optimizer_threshold
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
            cg_update='FP',
            cg_reset_beta=False,
            **kw):
        self._A_as_matrix = True
        self._log_p = []
        if cg_update in ['FR','HS','PRP','HZ']:
            self._cg_update = cg_update

        return kw

    def _update_acse_bfgs(self,
                          optimizer_threshold=0.01,
                          bfgs_limited=False,
                          bfgs_update='',
                          **kw):
        self._A_as_matrix = True
        self._limited = bfgs_limited
        self._opt_thresh = optimizer_threshold
        self._update_step = None
        self._log_p = []
        return kw

    def _update_acse_newton(self,
                            use_trust_region=False,
                            newton_step=-1,
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
        if self.verbose:
            print('newton step: {}'.format(newton_step))
            print('newton trust region: {}'.format(use_trust_region))
            print('trust region: {:.6f}'.format(initial_trust_region))
        self.tr_taylor = 1
        self.tr_object = 1
        return kw


    def _generate_real_circuit(self, op):
        #if isinstance(op, type(Ansatz())):
        #    op = op.op_form()
        #else:
        #    raise QuantumRunError('Problem with input to generate real circuit.')
        ins = self.Instruct(
            operator=op,
            Nq=self.QuantStore.Nq,
            quantstore=self.QuantStore)
        circ = StandardTomography(
            QuantStore=self.QuantStore,
            preset=self.tomo_preset,
            Tomo=self.tomo_Psi,
            verbose=self.verbose,
        )
        if not self.tomo_preset:
            circ.generate(real=self.Store.H.real,imag=self.Store.H.imag)
        circ.set(ins)
        circ.simulate()
        circ.construct(processor=self.process)
        return circ

    def build(self):
        if self.verbose:
            print('\n\n')
            print('-- -- -- -- -- -- -- -- -- -- --')
            print('building the ACSE run')
            print('-- -- -- -- -- -- -- -- -- -- --')
        if self.Store.use_initial:
            try:
                self.S = copy(self.S)
                en = np.real(self.Store.evaluate(self.Store.rdm))
            except Exception as e:
                print(e)
                self.QuantStore = copy(self.Store.S)
                circ = self._generate_real_circuit(self.S)
                self.Store.rdm = circ.rdm
                en = np.real(self.Store.evaluate(circ.rdm))
            self.e_k = np.real(en)
            self.ei = np.real(en)
            if self.verbose:
                print('Initial energy: {:.8f}'.format(self.ei))
            if self.verbose:
                print('S: ')
                print(self.S)
                print('Initial density matrix.')
                self.Store.rdm.contract()
                print(np.real(self.Store.rdm.rdm))
        else:
            self.S = copy(self.Store.S)
            self.e_k = self.Store.e0
            self.ei = self.Store.ei
            if self.verbose:
                print('taking energy from storage')
                print('initial energy: {:.8f}'.format(np.real(self.e_k)))
        self.best,self.grad = self.e_k,0
        self.best_avg = self.e_k

        self.e0 = self.e_k
        self.total = Cache()
        self.accept_previous_step = True
        #
        self.rdme = None
        if self.acse_method=='bfgs':
            self.rdme = self.tomo_S.rdme_keys[:]
            if not self._limited:
                if self.acse_update=='u':
                    dim = len(self.rdme)
                    self.B_k0 = sparse.identity(dim)*self.epsilon**2
                    self.Bi_k0 = sparse.identity(dim)*self.epsilon**2
                else:
                    dim = len(self.rdme)
                    self.B_k0 = np.identity(dim)#*self.epsilon**2
                    self.Bi_k0 = np.identity(dim)#/self.epsilon**2
            else:
                # using limited bfgs method
                self._lbfgs_sk = []
                self._lbfgs_yk = []
                self._lbfgs_rk = []
                self._lbfgs_r = []
        if self.acse_method=='cg':
            self.rdme = self.tomo_S.rdme_keys
        #
        # update norms 
        #
        self._get_S()
        if self.acse_method in ['bfgs','cg']:
            self.A = self.A*self.epsilon
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

    def _get_S(self):
            # 
        if self.acse_update == 'q':
            if type(self.sep_hamiltonian)==type(None):
                H = self.Store.H.qubit_operator
            else:
                H = self.sep_hamiltonian
            A_sq = solveqACSE(
                H=H,
                operator=self.S,
                process=self.process,
                instruct=self.Instruct,
                store=self.Store,
                quantstore=self.QuantStore,
                S_min=self.S_min,
                hamiltonian_step_size=self.hamiltonian_step_size,
                expiH_approx=self.S_expiH_approx,
                verbose=self.verbose,
                tomo=self.tomo_S,
                matrix=self._A_as_matrix,
                norm=self.A_norm,
                )
        elif self.acse_update == 'c':
            if not self.accept_previous_step:
                if self.verbose:
                    print('Rejecting previous step. No recalculation of A.')
                return
            #
            A_sq = solvecACSE(
                self.Store,
                self.QuantStore,
                S_min=self.S_min,
                verbose=self.verbose,
                matrix=self._A_as_matrix,
                keys=self.rdme,
                norm=self.A_norm,
                D3=self.tomo_D3,
                operator=self.S,
                process=self.process,
                instruct=self.Instruct,
            )
        elif self.acse_update == 'p':
            # TODO: need to update
            if type(self.sep_hamiltonian)==type(None):
                H = self.Store.H.qubit_operator
            else:
                H = self.sep_hamiltonian
            A_sq = solvepACSE(
                H=H,
                operator=self.S,
                process=self.process,
                instruct=self.Instruct,
                store=self.Store,
                quantstore=self.QuantStore,
                S_min=self.S_min,
                hamiltonian_step_size=self.hamiltonian_step_size,
                expiH_approx=self.S_expiH_approx,
                verbose=self.verbose,
                tomo=self.tomo_S,
                matrix=self._A_as_matrix,
                norm=self.A_norm,
                )
        elif self.acse_update =='u': #user specified
            A_sq = findUserA(
                operator=self.S.op_form(),
                process=self.process,
                instruct=self.Instruct,
                store=self.Store,
                quantstore=self.QuantStore,
                hamiltonian_step_size=self.hamiltonian_step_size,
                verbose=self.verbose,
                tomo=self.tomo_S,
                matrix=self._A_as_matrix,
            )
        else:
            raise QuantumRunError
        if self._A_as_matrix:
            # used in BFGS implementation
            self.norm = np.linalg.norm(A_sq,ord=self.A_norm)
            # factor of sqrt(8) accounts for 
            self.A = A_sq
        else:
            A_sq, norm = A_sq[0],A_sq[1]
            self.norm = norm/np.sqrt(4)
            if self._output==2:
                print('Fermi operator')
                print(A_sq)
            if self.split_ansatz:
                max_v, norm = 0, 0
                new = Operator()
                for op in A_sq:
                    if abs(op.c) >= abs(max_v):
                        max_v = copy(op.c)
                for op in A_sq:
                    if abs(op.c) >= abs(self.S_thresh_rel * max_v):
                        new += op
                if self.acse_update in ['c','q']:
                    A = A_sq.transform(self.QuantStore.transform)
                    #A = new.transform(self.QuantStore.transform)
                elif self.acse_update in ['p']:
                    A = A_sq.transform(self.QuantStore.qubit_transform)
                    #A = new.transform(self.QuantStore.qubit_transform)
                #
                inc = Operator()
                exc = Operator()
                for n in A:
                    added = False
                    for m in reversed(range(self.S.get_lim(), 0)):
                        # now, we check if in previous ansatz
                        ai = self.S[m]
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
                if self.acse_update in ['c','q']:
                    self.A = new.transform(self.QuantStore.transform)
                elif self.acse_update in ['p']:
                    self.A = new.transform(self.QuantStore.qubit_transform)
                norm = 0
                #for op in self.A:
                #    norm += op.norm()**2
                #self.norm = norm ** (0.5)
                # check if operator is split #
                #   #
                if self.verbose:
                    print('A operator (fermi)')
                    print(new)
                    print('A operator (pauli')
                    print(self.A)
                #    print('Norm: {}'.format(self.norm))
                #    print('-- -- -- -- -- -- -- -- -- -- --')

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
            self._get_S()
        elif self.acse_method in ['default', 'em', 'EM', 'euler']:
            _euler_step(self)
            self._get_S()
        elif self.acse_method in ['line', 'opt']:
            _opt_step(self)
            self._get_S()
        elif self.acse_method in ['bfgs']:
            _bfgs_step(self)
        elif self.acse_method in ['cg']:
            _conjugate_gradient_step(self)
        else:
            raise QuantumRunError('Incorrect acse_method.')
        # self._check_norm(self.A)
        # check if ansatz will change length


    def _opt_acse_function(self, parameter, newS=None, verbose=False):
        testS = copy(newS)
        currS = copy(self.S)
        for f in testS:
            f.c *= parameter[0]
        temp = currS + testS
        tCirc = self._generate_real_circuit(temp)
        en = np.real(self.Store.evaluate(tCirc.rdm))
        self._opt_log.append(tCirc)
        self._opt_en.append(en)
        #

        return en

    def _test_acse_function(self, parameter, newS=None, verbose=False):
        testS = copy(newS)
        currS = copy(self.S)
        for f in testS:
            f.c *= parameter[0]
        temp = currS + testS
        tCirc = self._generate_real_circuit(temp)
        en = np.real(self.Store.evaluate(tCirc.rdm))
        self.circ = tCirc
        return en, tCirc.rdm

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
                    diff = 1000 * (self.best - self.Store.H.ef)
                    print('E, fin: {:+.12f} U'.format(self.Store.H.ef))
                    print('E,diff: {:.12f} mU'.format(diff))
                except KeyError:
                    pass
                except AttributeError:
                    pass

    def reset(self,full=False):
        if not full:
            self.Store.use_initial=True
            self.build()
        else:
            self.Store.use_initial=False
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
            print('E init: {:+.12f} U'.format(np.real(self.ei)))
            print('E run : {:+.12f} U'.format(np.real(self.best)))
            try:
                diff = 1000 * (self.best - self.Store.H.ef)
                print('E goal: {:+.12f} U'.format(self.Store.H.ef))
                print('Energy difference from goal: {:.12f} mU'.format(diff))
            except KeyError:
                pass
            except AttributeError:
                pass

    def _check(self):
        '''
        Internal check on the energy as well as norm of the S matrix
        '''
        en = self.Store.evaluate(self.Store.rdm)
        if self.total.iter == 0:
            self.best = copy(self.e_k)
        self.total.iter += 1
        if self.total.iter == self.max_iter:
            print('Max number of iterations met. Ending optimization.')
            self.total.done = True
        elif len(self.S) == self.max_depth:
            if copy(self.S) + copy(self.A) > self.max_depth:
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
            #self.Store.analysis()
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
            'H': self.Store.H.matrix,
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
                'backend': self.QuantStore.backend,
                'provider': self.QuantStore.provider,
                'number of qubits': self.QuantStore.Nq,
                'number of shots': self.QuantStore.Ns,
                'stabilizers': self.QuantStore.method,
            },
            'description': description,
        }
        try:
            data['log-Gamma'] = self.log_Gamma
        except AttributeError as e:
            pass
        with open(name + '.log', 'wb') as fp:
            pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)
