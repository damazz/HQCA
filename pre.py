# pre.py
#
# Reads the configuration file, which has been set by the main program, and then
# read in all of the parameters. Then, if any parameters have not been assigned,
# critical ones which must be assigned as set to default values. 
# 
#

def NOFT():
    var = {
        'qc_backend':'qasm_simulator',
        'qc_num_shots':'4096',
        'qc_tomography':'1RDM',
        'qc_verbose':True,
        'qc_provider':'Aer',
        'qa_fermion':'compact',
        'qa_gpc_mapping':False,
        'max_iter':25,
        'restart':False,
        'wait':True,
        'prolix':True,
        'chem_orb':'HF',
        'opt_thresh':0.0001,
        'run_type':'rdm',
        'main':{
            'algorithm':'test',
            'wf_mapping':'zeta',
            'method':'stretch',
            'optimizer':'NM',
            'opt_thresh':0.01,
            'opt_crit':'default',
            'max_iter':100,
            'gd_gradient':'numerical',
            'gd_gradient_distance':0.01,
            'nm_simplex':10,
            'verbose':True},
        'sub':{
            'method':'givens',
            'optimizer':'NM',
            'opt_thresh':0.00001,
            'opt_crit':'default',
            'max_iter':1000,
            'gd_gradient':'numerical',
            'gd_gradient_distance':0.01,
            'nm_simplex':5,
            'verbose':True,
            'tri':True,
            'method_Ntri':3,
            'load_triangle':False}
        }
    return var


