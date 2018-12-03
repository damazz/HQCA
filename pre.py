# pre.py
#
# Reads the configuration file, which has been set by the main program, and then
# read in all of the parameters. Then, if any parameters have not been assigned,
# critical ones which must be assigned as set to default values. 
# 
#

def NOFT():
    var = {
        'max_iter':25,
        'restart':False,
        'wait':True,
        'prolix':True,
        'chem_orb':'HF',
        'opt_thresh':0.00001,
        'main':{
            'qc_backend':'qasm_simulator',
            'qc_num_shots':4096,
            'qc_provider':'Aer',
            'qa_fermion':'compact',
            'tri':True,
            'method_Ntri':3,
            'load_triangle':False,
            'algorithm':'affine_2p_curved_tenerife',
            'wf_mapping':'zeta',
            'method':'stretch',
            'optimizer':'NM',
            'tomo_basis':'no',
            'tomo_extra':None,
            'tomo_rdm':'1rdm',
            'opt_thresh':0.01,
            'opt_crit':'default',
            'max_iter':100,
            'gd_gradient':'numerical',
            'gd_gradient_distance':0.01,
            'nm_simplex':10,
            'verbose':False},
        'sub':{
            'method':'givens',
            'optimizer':'NM',
            'opt_thresh':0.00001,
            'opt_crit':'default',
            'region':'active_space',
            'max_iter':1000,
            'gd_gradient':'numerical',
            'gd_gradient_distance':0.01,
            'nm_simplex':5,
            'verbose':True}
        }
    return var

def RDM():
    var = {
        'max_iter':25,
        'restart':False,
        'wait':True,
        'prolix':True,
        'chem_orb':'HF',
        'opt_thresh':0.00001,
        'qc':{
            'qc_backend':'qasm_simulator',
            'qc_num_shots':2048,
            'qc_tomography':'1RDM',
            'qc_verbose':True,
            'qc_provider':'Aer',
            'qa_fermion':'direct',
            'tomo_basis':'bch',
            'tomo_rdm':'1rdm',
            'tomo_extra':False,
            'entangled_pairs':'full',
            'entangler':'Ry_cN', #Ry with a constant N
            'tri':False,
            'Nqb':4,
            #'method_Ntri':3,
            'load_triangle':False,
            'optimizer':'NM',
            'opt_thresh':0.01,
            'opt_crit':'default',
            'max_iter':100,
            #'gd_gradient':'numerical',
            #'gd_gradient_distance':0.01,
            'simplex_scale':45,
            'verbose':True}
        }
    return var




