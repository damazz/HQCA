from math import pi
# pre.py
#
# Reads the configuration file, which has been set by the main program, and then
# read in all of the parameters. Then, if any parameters have not been assigned,
# critical ones which must be assigned as set to default values. 
# 
#

def NOFT_2e():
    var = {
        'max_iter':25,
        'restart':False,
        'wait':True,
        'pr_g':2,
        'pr_m':0,
        'pr_s':1,
        'chem_orb':'HF',
        'opt_thresh':0.0001,
        'qc':{
            'backend':'qasm_simulator',
            'num_shots':2048,
            'provider':'Aer',
            'fermion_mapping':'jordan-wigner',
            'backend_configuration':None,
            'ansatz':'natural-orbitals',
            'method':'carlson-keller',
            'compiler':None,
            'initialize':'default',
            'Nqb':4,
            'tomo_basis':'hada+imag',
            'tomo_rdm':'1rdm',
            'tomo_extra':False,
            'spin_mapping':'default',
            'entangled_pairs':'d', #
            'entangler_p':'Ry_cN', #Ry with a constant N
            'entangler_q':'UCC2c', #Ry with a constant N
            'pr_e':2,
            'pr_q':0,
            'error_correction':False,
            'Sz':0.0,
            'depth':1,
            'opt':{
                'pr_o':1,
                'shift':None,
                'N_vectors':5,
                'use_radians':True,
                'unity':pi/2,
                'max_iter':100,
                'optimizer':'NM',
                #'conv_threshold':'default',
                'conv_threshold':0.1,
                'conv_crit_type':'default',
                'gradient':'numerical',
                'grad_dist':10,
                'simplex_scale':45
                }
            },
        'orb':{
            'method':'givens',
            'spin_mapping':'unrestricted',
            'opt':{
                'unity':pi,
                'shift':None,
                'use_radians':True,
                'optimizer':'nevergrad',
                'nevergrad_opt':'PSO',
                'pr_o':0,
                'N_vectors':10,
                'max_iter':1000,
                'conv_threshold':1e-8,
                'conv_crit_type':'MaxDist',
                'simplex_scale':90},
            'pr_m':0,
            'region':'active_space'}
        }
    return var

def NOFT():
    var = {
        'max_iter':25,
        'restart':False,
        'wait':True,
        'pr_g':1,
        'chem_orb':'HF',
        #'opt_thresh':0.000000001,
        'opt_thresh':0.0001,
        'qc':{
            'qc_backend':'qasm_simulator',
            'pr_m':0,
            'pr_o':0,
            'pr_q':0,
            'qc_num_shots':4096,
            'qc_provider':'Aer',
            'qa_fermion':'compact',
            'tri':False,
            'method_Ntri':3,
            'load_triangle':False,
            'algorithm':'affine_2p_curved_tenerife',
            'wf_mapping':'zeta',
            'method':'classical-default',
            'opt':{
                'opt_thresh':0.0000001,
                'opt_crit':'default',
                'max_iter':250,
                'gd_gradient':'numerical',
                'gd_gradient_distance':0.01,
                'simplex_scale':40,
                'optimizer':'NM'},
            'tomo_basis':'no',
            'tomo_extra':None,
            'tomo_rdm':'1rdm'},
        'orb':{
            'method':'givens',
            'optimizer':'NM',
            'opt_thresh':1e-10,
            'spin_mapping':'unrestricted',
            'opt':{
                'opt_thresh':0.0001,
                'opt_crit':'default',
                'pr_o':0,
                'max_iter':1000,
                'gd_gradient':'numerical',
                'gd_gradient_distance':0.01,
                'simplex_scale':90},
            'pr_m':0,
            'region':'active_space'}
        }
    return var

def RDM():
    var = {
        'max_iter':25,
        'restart':False,
        'wait':True,
        'chem_orb':'HF',
        'pr_m':0,
        'pr_g':0,
        'qc':{
            'backend':'qasm_simulator',
            'num_shots':2048,
            'provider':'Aer',
            'fermion_mapping':'jordan-wigner',
            'backend_configuration':None,
            'ansatz':'default',
            'method':'variational',
            'compiler':None,
            'initialize':'default',
            'Nqb':4,
            'tomo_basis':'hada',
            'tomo_rdm':'1rdm',
            'tomo_extra':False,
            'spin_mapping':'default',
            'entangled_pairs':'s', #
            'entangler_p':'Ry_cN', #Ry with a constant N
            'entangler_q':'Ry_cN', #Ry with a constant N
            'pr_e':1,
            'pr_q':0,
            'load_triangle':False,
            'tri':False,
            'Sz':0.0,
            'depth':1,
            'opt':{
                'pr_o':1,
                'max_iter':100,
                'optimizer':'NM',
                #'opt_thresh':0.01,
                #'opt_crit':'default',
                'conv_threshold':'default',
                'conv_crit_type':'default',
                'gradient':'numerical',
                'grad_dist':10,
                'simplex_scale':45
                }
            }
        }
    return var

