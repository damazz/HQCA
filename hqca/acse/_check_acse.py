from hqca.core import *
import warnings

def check_routine(acse):
    #
    # 
    if 'shift' in acse.qs.method:
        if not acse.store.psi.closed in [1,-1]:
            print('For shift method, the ansatz should have an accessible depth of 0 or 1')
            raise KeywordError
        if not acse.split_ansatz:
            print('Need to use a split ansatz for shift method. Otherwise it is not well defined.')
            raise KeywordError
    if acse.acse_method=='newton' and not acse._conv_type=='trust':
        warnings.warn('Did you mean to not use the trust region convergence criteria?')

    if acse.acse_method not in ['newton']:
        msg = 'Only 1-D line search uses the trust region criteria...'
        assert not acse._conv_type=='trust',msg


