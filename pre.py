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
        'pr_g':1,
        'chem_orb':'HF',
        'opt_thresh':0.000001,
        #'opt_thresh':0.0001,
        'main':{
            'qc_backend':'qasm_simulator',
            'pr_m':0,
            'pr_o':0,
            'pr_q':0,
            'qc_num_shots':4096,
            'qc_provider':'Aer',
            'qa_fermion':'compact',
            'tri':True,
            'method_Ntri':3,
            'load_triangle':False,
            'algorithm':'affine_2p_curved_tenerife',
            'wf_mapping':'zeta',
            'method':'classical-default',
            'optimizer':'NM',
            'tomo_basis':'no',
            'tomo_extra':None,
            'tomo_rdm':'1rdm',
            'opt_thresh':0.0001,
            'opt_crit':'default',
            'max_iter':250,
            'gd_gradient':'numerical',
            'gd_gradient_distance':0.01,
            'simplex_scale':40,
            'verbose':False},
        'sub':{
            'method':'givens',
            'optimizer':'NM',
            'opt_thresh':0.0000001,
            'spin_mapping':'unrestricted',
            #'opt_thresh':0.0001,
            'opt_crit':'default',
            'pr_m':0,
            'pr_o':0,
            'region':'active_space',
            'max_iter':1000,
            'gd_gradient':'numerical',
            'gd_gradient_distance':0.01,
            'simplex_scale':90,
            'verbose':True}
        }
    return var

def RDM():
    var = {
        'max_iter':25,
        'restart':False,
        'wait':True,
        'pr_g':1,
        'chem_orb':'HF',
        'qc':{
            'max_iter':100,
            'verbose':True,
            #
            'qc_backend':'qasm_simulator',
            'qc_num_shots':2048,
            'qc_tomography':'1RDM',
            'qc_verbose':True,
            'qc_provider':'Aer',
            'Nqb':4,
            'qa_fermion':'direct',
            'tomo_basis':'hada',
            'tomo_rdm':'1rdm',
            'tomo_extra':False,
            'spin_mapping':'default',
            #'default' - alpha,beta spin treated seperately
            #'spin-free' -no assignment of alpha beta, i.e. no beta orbitals
            #'spatial' - alpha/beta parameters are the same.
            'entangled_pairs':'full', #
            'entangler':'Ry_cN', #Ry with a constant N
            'tri':False,
            'pr_t':1,
            'print_run':False, #optimizer? 
            'pr_o':1,
            'pr_q':0,
            'pr_m':0,
            'load_triangle':False,
            'optimizer':'NM',
            'opt_thresh':0.01,
            'opt_crit':'default',
            'gradient':'numerical',
            'grad_dist':10,
            'simplex_scale':45
            }
        }
    return var

