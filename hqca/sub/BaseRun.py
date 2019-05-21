from hqca.tools import EnergyFunction as enf


class QuantumRun:
    '''
    Handles different runs which call the quantum functions package
    '''
    def __init__(self,
            **kwargs
            ):
        self.built=False
        self.restart = False
        now = datetime.datetime.today()
        m,d,y = now.strftime('%m'), now.strftime('%d'), now.strftime('%y')
        H,M,S = now.strftime('%H'), now.strftime('%M'), now.strftime('%S')
        self.file = ''.join(('sp','-',theory,'_',m,d,y,'-',H,M))

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

    def _build_quantum(self):
        self.

    def build(self):
        self.pr_g = self.kw['pr_g']
        self.Store.pr_m = self.kw['pr_m']
        if self.pr_g>1:
            print('')
            print('### #### ### ### ### ### ### ### ### ### ### ###')
            print('')
            print('# Initializing the optimization.')
            print('#')
            print('# Setting RDM parameters...')
        self.Store.gas()
        self.Store.gsm()
        self.kw_qc['Nels_as'] = self.Store.Nels_as
        self.kw_qc['Norb_as']=self.Store.Norb_as
        self.kw_qc['alpha_mos']=self.Store.alpha_mo
        self.kw_qc['beta_mos']=self.Store.beta_mo
        self.kw_qc['single_point']=self.Store.sp
        if self.pr_g>1:
            print('# ...done.')
            print('#')
            print('# Setting QC parameters...')
        self.QuantStore = qf.QuantumStorage(self.pr_g,**self.kw_qc)
        if self.pr_g>1:
            print('# ...done.')
            print('#')
            print('# Setting opt parameters...')
        self.Store.update_full_ints()
        self.kw_opt['function'] = enf.find_function(
                self.Store.theory,
                'qc',
                self.Store,
                self.QuantStore)
        if self.pr_g>1:
            if self.kw_opt['optimizer']=='nevergrad':
                print('#  optimizer  : {}'.format(self.kw_opt['nevergrad_opt']))
            else:
                print('#  optimizer  : {}'.format(self.kw_opt['optimizer']))
            print('#  max iter   : {}'.format(self.kw_opt['max_iter']))
            print('#  stop crit  : {}'.format(self.kw_opt['conv_crit_type']))
            print('#  crit thresh: {}'.format(self.kw_opt['conv_threshold']))
            print('# ...done.')
            print('# ')
            print('# Initialized successfully. Beginning optimization.')
            print('')
            print('### ### ### ### ### ### ### ### ### ### ### ###')
            print('')
        if self.kw_qc['info'] in ['calc']:
            qf.get_direct_stats(self.QuantStore)
        elif self.kw_qc['info'] in ['draw']:
            qf.get_direct_stats(self.QuantStore,extra='draw')
            sys.exit()
        elif self.kw_qc['info'] in ['count_only']:
            qf.get_direct_stats(self.QuantStore)
            sys.exit()
        elif self.kw_qc['info'] in ['compile','check','check_circuit','transpile']:
            qf.get_direct_stats(self.QuantStore,extra='compile')
            sys.exit()
        else:
            #print('Getting circuit information for keyword: {}'.format(
            #    self.kw_qc['info'])
            #    )
            qf.get_direct_stats(self.QuantStore,extra=self.kw_qc['info'])
            #sys.exit()

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
