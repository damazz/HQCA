from hqca.core import *
from hqca.tools import *
from hqca.vqe._store_vqe import *
from hqca.tomography import *
from hqca.vqe._ucc import *
from hqca.vqe._pair import *
from copy import deepcopy as copy
from hqca.operators import *
from hqca.acse._class_S_acse import *
from hqca.acse._quant_S_acse import *
from hqca.acse._qubit_A import *
import sympy as sy


class Cache:
    def __init__(self):
        self.use=True
        self.err=False
        self.msg=None
        self.iter=0
        self.done=False

class RunADAPTVQE(QuantumRun):
    def __init__(self,
            Storage,
            Optimizer,
            QuantStore,
            Instructions,
            **kw,
            ):
        self.Store = Storage
        self.Opt = Optimizer
        self.QuantStore = QuantStore
        self.Instruct = Instructions
        self._update_vqe_kw(**kw)

    def _update_adapt_kw(self,
            recycle=True,**kw):
        self.recycle = recycle
        return kw

    def _update_vqe_kw(self,
            update_vqe='numerical',
            update_acse = 'quantum',
            pool='rdm',
            initial='hf',
            opt_thresh=1e-8,
            max_iter=50,
            trotter=1,
            ansatz_depth=1,
            tomo_Psi=None,
            verbose=True,
            gradient=False,
            kw_opt={},
            **kw):
        if update_vqe in ['numerical']:
            self.update_vqe = 'n' #update VQE gradient
        else:
            raise QuantumRunError('Not supported method of VQE gradients.')
        if update_acse in ['quantum', 'Q', 'q', 'qso', 'qfo']:
            self.update_acse = 'q'
        elif update_acse in ['class', 'classical', 'c', 'C']:
            self.update_acse = 'c'
        elif update_acse in ['para', 'p']:
            self.update_acse = 'p'
        else:
            raise QuantumRunError

        if pool in ['rdm','ucc']:
            self.operator_pool = pool
        else:
            raise QuantumRunError
        self.operator_pool = pool
        self.max_iter=max_iter
        self.use_gradient = gradient
        self.sep_hamiltonian = False
        self.crit = opt_thresh
        self.tomo_Psi = tomo_Psi
        self.depth = ansatz_depth
        self.verbose= verbose
        if type(self.tomo_Psi)==type(None):
            self.tomo_preset=False
        else:
            self.tomo_preset=True
        self.kw_opt=kw_opt
        kw = self._update_acse_kw(**kw)
        kw = self._update_adapt_kw(**kw)
        if len(kw)>0:
            print('Unused or improper keywords: ')
            for k in kw:
                print(k)

    def _update_acse_kw(self,
                        expiH_approximation='first',
                        hamiltonian_step_size=0.1,
                        separate_hamiltonian=None,
                        verbose=True,
                        tomo_S=None,
                        statistics=False,
                        processor=None,
                        **kw):
        '''
        Updates the ACSE keywords. 
        '''
        self.process = processor
        self.verbose = verbose
        self.stats = statistics
        self.hamiltonian_step_size = hamiltonian_step_size
        self.sep_hamiltonian = separate_hamiltonian
        self.tomo_S = tomo_S
        self._A_as_matrix = True
        return kw

    def __test_vqe_function(self,para):
        psi = self.S.assign_variables(para,T=self.QuantStore.transform,N=self.QuantStore.dim,
                fermi=self.update_acse in ['q','c'])
        #
        ins = self.Instruct(psi,
                self.QuantStore.Nq,
                depth=self.depth,
                )
        #
        circ = StandardTomography(
                self.QuantStore,
                preset=self.tomo_preset,
                Tomo=self.tomo_Psi,
                verbose=False,
                )
        #
        if not self.tomo_preset:
            circ.generate(real=self.Store.H.real)
        circ.set(ins)
        circ.simulate()
        circ.construct()
        en = np.real(self.Store.evaluate(circ.rdm))
        if en<self.best:
            self.Store.update(circ.rdm)
            self.best = copy(en)
        return en

    def __test_vqe_gradient(self,para,forward=True,diff=0.0000001):
        gradient = []
        if forward:
            e_0 = self.__test_vqe_function(para)
            for i in range(len(para)):
                tpara = []
                for j in range(len(para)):
                    c = para[j]
                    if i==j:
                        c+=diff
                    tpara.append(c)
                e_p = self.__test_vqe_function(tpara)
                gradient.append((e_p-e_0)/(diff))
        else:
            for i in range(len(para)):
                tpara = []
                for j in range(len(para)):
                    c = para[j]
                    if i==j:
                        c-=diff
                    tpara.append(c)
                e_m = self.__test_vqe_function(tpara)
                tpara = []
                for j in range(len(para)):
                    c = para[j]
                    if i==j:
                        c+=diff
                    tpara.append(c)
                e_p = self.__test_vqe_function(tpara)
                gradient.append((e_p-e_m)/(2*diff))
        return np.asarray([gradient])


    def build(self,**kw):
        print('Building....')
        try:
            if self.built:
                raise QuantumRunError('No need to rebuild what is built.')
        except Exception:
            pass
        self.total = Cache()
        self.micro = Cache()

        #
        # if criteria ...
        #

        print('Starting optimizer...')
        self.S = ADAPTAnsatz(closed=True)
        self.para=  []
        self._get_ACSE_residuals()
        self._ADAPT()
        self.best = 0
        if self.use_gradient:
            self.mOpt = self.Opt(
                function=self.__test_vqe_function,
                gradient=self.__test_vqe_gradient,
                **self.kw_opt)
        else:
            self.mOpt = self.Opt(
                function=self.__test_vqe_function,
                **self.kw_opt)
        #
        print('Initializing optimizer....')
        #
        self.mOpt.initialize(self.para)
        self.ei = self.mOpt.opt.best_f
        self.e0 = self.mOpt.opt.best_f
        self.log_E = [self.ei]
        self.log_oE = []
        self.log_S = [self.norm]
        self.log_oN = []

        self.built=True
        print('Done')
        # here....we should generate the ansatz


    def _ADAPT(self):
        #
        # given A matrix, find largest corresponding elements and add to ansatz
        if self.recycle:
            new_A = self.A[np.argmax(np.abs(self.A))]
            new_key = self.tomo_S.rdme_keys[np.argmax(np.abs(self.A))]
            if self.verbose:
                if self.update_acse in ['c','q']:
                    print('-- -- --')
                    print('ADAPT additional term: ')
                    new = Operator()
                    new+= FermiString(
                            coeff=1,indices=new_key,
                            N=self.QuantStore.dim,
                            ops='++--')
                    new-= FermiString(
                            coeff=+1,indices=new_key[::-1],
                            N=self.QuantStore.dim,
                            ops='++--')
                    print(new)
                else:
                    print('-- -- --')
                    print('ADAPT additional term: ')
                    new = Operator()
                    new+= QubitString(
                            coeff=1,indices=new_key,
                            N=self.QuantStore.dim,
                            ops='++--')
                    new-= QubitString(
                            coeff=+1,indices=new_key[::-1],
                            N=self.QuantStore.dim,
                            ops='++--')
                    print(new)
                print(new.transform(self.QuantStore.transform))
                print('-- -- --')
            self.S.add_term(
                    indices=new_key,
                    ind_key = np.argmax(np.abs(self.A)),
                    )
            temp = np.zeros(len(self.para)+1,dtype=np.complex_)
            temp[:-1] = self.para[:]
            self.para = temp[:]
        else:
            pass


    def _run_adapt(self):
        try:
            self.built
        except AttributeError:
            sys.exit('Run not built. Run vqe.build()')
        log_e = [self.mOpt.opt.best_f]
        log_n = [self.mOpt.opt.crit]
        while not self.micro.done:
            self.mOpt.next_step()
            self.mOpt.check(self.micro)
            log_e.append(self.mOpt.opt.best_f)
            log_n.append(self.mOpt.opt.crit)
        self.log_oE.append(log_e)
        self.log_E.append(self.mOpt.opt.best_f)
        self.log_oN.append(log_n)
        self.para = np.asarray(self.mOpt.opt.best_x)[0].tolist()
        self.micro = Cache()
        self._get_ACSE_residuals()
        self._ADAPT()
        self.log_S.append(self.norm)
        #if self.verbose:
        #    self.Store.rdm.analysis()

    def _check(self):
        en = self.mOpt.opt.best_f
        self.best = en
        self.total.iter += 1
        if self.total.iter==self.max_iter:
            self.total.done=True
        elif self.norm<self.crit:
            self.total.done=True
        if self.verbose:
            #self.Store.analysis()
            print('---------------------------------------------')
            print('Step {:02}, E: {:.12f}, ||A||: {:.12f}'.format(
                self.total.iter,
                np.real(en),
                np.real(self.norm)))
        # now, reset optimizer
        if self.use_gradient:
            self.mOpt = self.Opt(
                function=self.__test_vqe_function,
                gradient=self.__test_vqe_gradient,
                **self.kw_opt)
        else:
            self.mOpt = self.Opt(
                function=self.__test_vqe_function,
                **self.kw_opt)
        self.mOpt.initialize(self.para)
    
    def next_step(self):
        if self.built:
            self._run_adapt()
            self._check()
            if self.verbose:
                print('E,init: {:+.12f} U'.format(np.real(self.ei)))
                print('E, run: {:+.12f} U'.format(np.real(self.best)))
                try:
                    diff = 1000 * (self.best - self.Store.H.ef)
                    print('E, fin: {:+.12f} U'.format(self.Store.H.ef))
                    print('E, dif: {:.12f} mU'.format(diff))
                except KeyError:
                    pass
                except AttributeError:
                    pass

    def run(self,**kw):
        if self.built:
            while not self.total.done:
                self._run_adapt()
                self._check()
            print('E, init: {:+.12f} U'.format(np.real(self.ei)))
            print('E, run: {:+.12f} U'.format(np.real(self.best)))
            try:
                diff = 1000*(self.best-self.Store.H.ef)
                print('E, fin: {:+.12f} U'.format(self.Store.H.ef))
                print('Energy difference from goal: {:.12f} mU'.format(diff))
            except KeyError:
                pass
            except AttributeError:
                pass

    def _get_ACSE_residuals(self):
        if self.update_acse == 'q':
            if type(self.sep_hamiltonian)==type(None):
                H = self.Store.H.qubit_operator
            else:
                H = self.sep_hamiltonian
            A_sq = solveqACSE(
                H=H,
                operator=self.S.assign_variables(self.para,
                    T=self.QuantStore.transform,
                    N=self.QuantStore.dim,
                    ),      
                process=self.process,
                instruct=self.Instruct,
                store=self.Store,
                quantstore=self.QuantStore,
                hamiltonian_step_size=self.hamiltonian_step_size,
                verbose=self.verbose,
                tomo=self.tomo_S,
                matrix=self._A_as_matrix,
                )
        elif self.update_acse == 'c':
            if not self.accept_previous_step:
                if self.verbose:
                    print('Rejecting previous step. No recalculation of A.')
                return
            A_sq = findSPairs(
                self.Store,
                self.QuantStore,
                S_min=self.S_min,
                verbose=self.verbose,
            )
        elif self.update_acse == 'p':
            # TODO: need to update
            if type(self.sep_hamiltonian)==type(None):
                H = self.Store.H.qubit_operator
            else:
                H = self.sep_hamiltonian
            A_sq = solvepACSE(
                H=H,
                operator=self.S.assign_variables(self.para,
                    T=self.QuantStore.transform,
                    N=self.QuantStore.dim,
                    fermi=False,
                    ),
                process=self.process,
                instruct=self.Instruct,
                store=self.Store,
                quantstore=self.QuantStore,
                hamiltonian_step_size=self.hamiltonian_step_size,
                verbose=self.verbose,
                tomo=self.tomo_S,
                matrix=self._A_as_matrix,
            )
        elif self.update_acse =='u': #user specified
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
            self.norm = np.linalg.norm(A_sq)
            # factor of sqrt(8) accounts for 
            self.A = A_sq
        else:
            A_sq, norm = A_sq[0],A_sq[1]
