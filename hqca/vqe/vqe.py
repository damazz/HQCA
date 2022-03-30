from hqca.core import *
from hqca.tools import *
from hqca.vqe._store_vqe import *
from hqca.tomography import *
from hqca.vqe._ucc import *
from hqca.vqe._pair import *
from copy import deepcopy as copy
from hqca.operators import *

class Cache:
    def __init__(self):
        self.use=True
        self.err=False
        self.msg=None
        self.iter=0
        self.done=False
    

class RunVQE(QuantumRun):
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

    def _update_vqe_kw(self,
            #
            method='noft',
            ansatz='ucc',
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
        self.ansatz=ansatz
        self.vqe_method=method
        self.max_iter=50
        self.use_gradient = gradient
        self.crit = opt_thresh
        self.tomo = tomo_Psi
        self.depth = ansatz_depth
        self.verbose= verbose
        if type(self.tomo)==type(None):
            self.tomo_preset=False
        else:
            self.tomo_preset=True
        self.kw_opt=kw_opt

    def build(self,**kw):
        print('Building....')
        try:
            if self.built:
                sys.exit('Building VQE again?')
        except Exception:
            pass
        self.total = Cache()
        # if criteria ...
        print('Getting UCC...')
        if self.ansatz=='ucc':
            self.T= getUCCAnsatz(self.QuantStore,verbose=self.verbose)

            self.para= []
            if self.Store.initial in ['hf','hartree-fock']:
                for i in range(len(self.T.T)):
                    self.para.append(0)
            print('VQE has {} variables'.format(len(self.T.T)))
        print('Starting optimizer...')
        self.best = 0
        if self.use_gradient:
            self.Opt = self.Opt(
                function=self.__test_vqe_function,
                gradient=self.__test_vqe_gradient,
                **self.kw_opt)
        else:
            self.Opt = self.Opt(
                function=self.__test_vqe_function,
                **self.kw_opt)
        print('Initializing optimizer....')
        self.Opt.initialize(self.para)
        self.ei = self.Opt.opt.best_f
        self.e0 = self.Opt.opt.best_f
        self.log_E = [self.ei]
        self.log_S = [self.Opt.opt.crit]

        self.built=True
        # here....we should generate the ansatz

    def __test_vqe_function(self,para):
        psi = self.T.assign(para,self.QuantStore.transform,
                N=self.QuantStore.dim)
        ins = self.Instruct(psi,
                self.QuantStore.Nq,
                depth=self.depth,
                )
        circ = StandardTomography(
                self.QuantStore,
                preset=self.tomo_preset,
                Tomo=self.tomo,
                verbose=False,
                )
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

    def next_step(self):
        if self.built:
            self._run_vqe()
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

    def _run_vqe(self):
        try:
            self.built
        except AttributeError:
            raise QuantumRunError('Run VQE.build()')
        self.Opt.next_step()
        self.para = np.asarray(self.Opt.opt.best_x)[0]
        print('para',self.para)
        self.log_E.append(self.Opt.opt.best_f)
        self.log_S.append(self.Opt.opt.crit)
        #if self.verbose:
        #    self.Store.rdm.analysis()

    def _check(self):
        en = self.Opt.opt.best_f
        self.best = en
        self.Opt.check(self.total)
        if self.total.iter==self.max_iter:
            self.total.done=True
        elif self.Opt.opt.crit<self.crit:
            self.total.done=True
        if self.verbose:
            self.Store.analysis()

    def run(self,**kw):
        if self.built:
            while not self.total.done:
                self._run_vqe()
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


