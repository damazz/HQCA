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
            method='noft',
            ansatz='ucc',
            initial='hf',
            opt_thresh=1e-8,
            max_iter=50,
            trotter=1,
            ansatz_depth=1,
            tomography=None,
            verbose=True,
            gradient=False,
            kw_opt={},
            **kw):
        self.ansatz=ansatz
        self.vqe_method=method
        self.max_iter=50
        self.use_gradient = gradient
        self.crit = opt_thresh
        self.tomo = tomography
        self.depth = ansatz_depth
        self.verbose= verbose
        if type(self.tomo)==type(None):
            self.tomo_preset=False
        else:
            self.tomo_preset=True
        self.kw_opt=kw_opt

    def __test_vqe_function(self,para):
        psi = self.T.evaluate(para)
        ins = self.Instruct(psi,
                self.QuantStore.Nq,
                depth=self.depth,
                )
        circ = StandardTomography(
                self.QuantStore,
                preset=self.tomo_preset,
                Tomo=self.tomo,
                verbose=False,
                #erbose=self.verbose)
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

    def __test_vqe_gradient(self,para,diff=0.01):
        gradient = []
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
        return gradient

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
                for i in range(len(self.T.xi)):
                    self.para.append(0)
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

        self.built=True
        print('Done')
        # here....we should generate the ansatz

    def _run_vqe(self):
        try:
            self.built
        except AttributeError:
            sys.exit('Run not built. Run vqe.build()')
        self.Opt.next_step()
        self.para = np.asarray(self.Opt.opt.best_x)[0]
        print(self.para)
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


