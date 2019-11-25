from hqca.tools import EnergyFunctions as enf
from hqca.quantum import QuantumFunctions as qf
from hqca.quantum import NoiseSimulator as ns
import datetime
import sys

class Cache:
    def __init__(self):
        self.use=True
        self.err=False
        self.msg=None
        self.iter=0
        self.done=False

class QuantumRun:
    '''
    Handles different runs which call the quantum functions package
    '''
    def __init__(self,**kw):
        self.built = False
        self.restart = False
        self.Store=None
        now = datetime.datetime.today()
        m,d,y = now.strftime('%m'), now.strftime('%d'), now.strftime('%y')
        H,M,S = now.strftime('%H'), now.strftime('%M'), now.strftime('%S')
        self.file = ''.join(('sp','-',m,d,y,'-',H,M))

    def update_var(
            self,
            target='global',
            **args):
        if target=='global':
            for k,v in args.items():
                self.kw[k]=v
                self.pr_g = self.kw['pr_g']
        elif target=='qc':
            for k,v in args.items():
                self.kw_qc[k] = v
        elif target=='opt':
            for k,v in args.items():
                self.kw_opt[k]=v
        elif target=='orb':
            for k,v in args.items():
                self.kw_orb[k]=v
        elif target=='orb_opt':
            for k,v in args.items():
                self.kw_orb_opt[k]=v
        elif target=='store':
            for k,v in args.items():
                self.kw_store[k]=v


    def _build_energy(self,mol,**kw):
        self.Store = enf.Storage(mol,**kw)
        self.Store.gas()
        self.Store.gsm()
        self.Store.update_full_ints()
        self.Store.find_npara_orb()

    def _build_quantum(self):
        self.kw_qc['theory']=self.theory
        if not self.Store==None:
            self.kw_qc['Ne_as'] = self.Store.Ne_as
            self.kw_qc['No_as']=self.Store.No_as
            self.kw_qc['Ne_alp']=self.Store.Ne_alp
            self.kw_qc['Ne_bet']=self.Store.Ne_bet
            self.kw_qc['alpha_mos']=self.Store.alpha_mo
            self.kw_qc['beta_mos']=self.Store.beta_mo
        else:
            try:
                self.kw_qc['Ne_as']
            except KeyError:
                print('It is okay to not use energy, but you can\'t')
                print('forget to specify the following parameters!')
                print('  (1) Ne_as,   (2) No_as,')
                print('  (3) alpha_mos, (4) beta_mos')
                sys.exit()
        self.QuantStore = qf.QuantumStorage(**self.kw_qc)
        if self.QuantStore.method=='variational':
            self.kw_opt['function'] = enf.find_function(
                    self.theory,
                    'qc',
                    self.Store,
                    self.QuantStore)

    def set_print(self,level='default',
            record=False
            ):
        self.kw_qc['pr_e']=1
        self.kw_qc['pr_q']=1
        self.kw_opt['pr_o']=1
        self.kw['pr_m']=1
        self.kw['pr_g']=2
        self.kw['pr_s']=1
        if level=='terse':
            self.kw['pr_g']=1
            self.kw_opt['pr_o']=0
            self.kw['pr_m']=0
            self.kw['pr_s']=0
        elif level=='none':
            self.kw_qc['pr_q']=0
            self.kw_qc['pr_e']=0
            self.kw_opt['pr_o']=0
            self.kw['pr_m']=0
            self.kw['pr_g']=0
            self.kw['pr_s']=0
        elif level=='analysis':
            self.kw['pr_g']=4
        elif level=='diagnostic':
            self.kw_qc['pr_q']=9
            self.kw_opt['pr_o']=9
            self.kw['pr_m']=9
            self.kw['pr_g']=9
        elif level=='diagnostic_en':
            self.kw['pr_m']=4
        elif level=='diagnostic_qc':
            self.kw_qc['pr_q']=9
        elif level=='diagnostic_opt':
            self.kw_opt['pr_o']=9
        elif level=='diagnostic_orb':
            self.kw['pr_s']=9
            self.kw_opt['pr_o']=9

