'''
tools/Preset.py

provides preset configurations for most....not sure if they are okay, or if they
are really needed. But, good to have a default configuration. 

'''
from math import pi
 
def qACSE():
    var = {
        'pr_m':0,
        'pr_g':2,
        'pr_s':1,
        'qc':{
            'backend':'qasm_simulator',
            'num_shots':2048,
            'info':'calc',
            'method':'acse',
            'provider':'Aer',
            'fermion_mapping':'jordan-wigner',
            'backend_configuration':None,
            'ansatz':'',
            'transpile':None,
            'initialize':'default',
            'Nq':4,
            'noise_model_loc':None,
            'noise':False,
            'tomo_basis':'no',
            'tomo_rdm':'acse',
            'spin_mapping':'default',
            'pr_e':0, # error correction
            'pr_q':0,
            'depth':1
            },
        'acse':{
            'opt_thresh':0.00001,
            'opt_criteria':'default',
            'max_iter':100,
            'trotter':1,
            'ansatz_depth':1,
            'pr_a':1,
            'reconstruct':'default',
            }
        }
    return var


def NOFT_2e():
    var = {
        'max_iter':25,
        'pr_m':0,
        'pr_g':2,
        'pr_s':1,
        'opt_thresh':0.00001,
        'qc':{
            'backend':'qasm_simulator',
            'num_shots':2048,
            'info':'calc',
            'provider':'Aer',
            'fermion_mapping':'jordan-wigner',
            'backend_configuration':None,
            'ansatz':'natural-orbitals',
            'method':'carlson-keller',
            'transpile':None,
            'initialize':'default',
            'Nq':4,
            'noise_model_loc':None,
            'noise':False,
            'tomo_basis':'no',
            'tomo_rdm':'1rdm',
            'tomo_extra':False,
            'spin_mapping':'default',
            'entangled_pairs':'d', #
            'entangler_p':'Ry_cN', #Ry with a constant N
            'entangler_q':'UCC2c', #Ry with a constant N
            'pr_e':0, # error correction
            'pr_q':0,
            'ec':True,
            'ec_method':'hyperplane',
            'Sz':0.0,
            'depth':1,
            'opt':{
                'pr_o':1,
                'shift':None,
                'N_vectors':5,
                'unity':pi/2,
                'max_iter':100,
                'optimizer':'NM',
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
                #'optimizer':'nevergrad',
                'optimizer':'gpso',
                'nevergrad_opt':'PSO',
                'pr_o':0,
                'N_vectors':6,
                'max_iter':5000,
                'conv_crit_type':'MaxDist',
                'particles':20,
                'pso_iterations':50,
                'examples':2,
                'inertia':0.7,
                'max_velocity':0.5,
                'conv_threshold':0.1,
                'unity':pi,
                'pr_o':2,
                'accel':[1.0,1.0],
                'simplex_scale':90},
            'pr_m':0,
            'region':'active_space'}
        }
    return var

def circuit():
    var = {
        'max_iter':25,
        'pr_m':0,
        'pr_g':2,
        'pr_s':1,
        'qc':{
            'qc':True,
            'backend':'qasm_simulator',
            'num_shots':2048,
            'info':'calc',
            'provider':'Aer',
            'fermion_mapping':'jordan-wigner',
            'ansatz':'nat-orb-no',
            'method':'carlson-keller',
            'transpile':'default',
            'Nq':4,
            'noise':False,
            'noise_model_location':None,
            'tomo_basis':'no',
            'tomo_rdm':'1rdm',
            'tomo_extra':False,
            'spin_mapping':'default',
            'entangled_pairs':'d', #
            'entangler_p':'Ry_cN', #Ry with a constant N
            'entangler_q':'UCC2_1s', #Ry with a constant N
            'pr_e':0, # error correction
            'pr_q':0,
            'ec':False,
            'ec_method':None,
            'Sz':0.0,
            'depth':1,
            'Ne_as':2,
            'No_as':2,
            'alpha_mos':{'active':[0,1]},
            'beta_mos':{'active':[2,3]},
            }
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
            'Nq':4,
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

