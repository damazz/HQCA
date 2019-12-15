from pyscf import gto,scf,mcscf
from os import getcwd
from math import pi
from hqca.hqca import sp,scan,circuit
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 0.678
x = d
y = d/np.sqrt(2)
mol.atom=[
        ['H',(x,0,-y)],['H',(-x,0,-y)],
        ['H',(0,x,y)],['H',(0,-x,y)]]
mol.basis='sto-3g'
mol.spin=0
mol.charge=2
mol.verbose=0
mol.build()
prog = sp(theory='noft',mol=mol,casci=True)
filename = getcwd()+'/ibmq_16_melbourne'
kw_qc = {
        'Ne_as':2,
        'No_as':3,
        'alpha_mos':{'active':[0,1,2,3]},
        'beta_mos':{'active':[4,5,6,7]},
        'Nq':8,
        'Nq_ancilla':0,
        'num_shots':8192,
        'entangler_q':'custom',
        'entangler_kw':{
            'entanglers':[
                {
                    'circ':'UCC2_1s',
                    'kw':{},
                    'np':1
                    },
                {
                    'circ':'UCC2_2s_custom',
                    'kw':{},
                    'np':1
                    },
                {
                    'circ':'UCC2_2s_custom',
                    'kw':{},
                    'np':1
                    },
                ]
            },
        'spin_mapping':'alternating',
        'depth':1,
        'info':'ibm',
        'provider':'IBMQ',
        'backend':'ibmq_16_melbourne',
        #'noise':True,
        #'noise_model_location':filename,
        'backend_initial_layout':[8,6,5,4,3,2,1,0],
        #'backend_file':filename,
        'ec_comp_ent':False,
        'ec_post':True,
        'ec_pre':False,
        'ec_pre_kw':{
            'filter_measurements':False,
            },
        'ec_syndrome':False,
        'ec_post_kw':{
            'symm_verify':True,
            'symmetries':['N','Sz'],
            'hyperplane':True,
            },
        #'tomo_basis':'pauli_symm',
        'tomo_basis':'no',
        'tomo_extra':'classical',
        'tomo_approx':'fo',
        'transpile':'default',
        'transpiler_keywords':{
            },
        'transpile':'default',
        'ansatz':'nat-orb-no',
        }
kw_opt = {
        'optimizer':'NM',
        'unity':np.pi/2,
        'nevergrad_opt':'TBPSA',
        'max_iter':5000,
        #'conv_crit_type':'MaxDist',
        'conv_crit_type':'default',
        'initial_conditions':'han',
        'conv_threshold':1e-2,
        'N_vectors':2,
        }
orb_opt = {
        'optimizer':'bfgs',
        'unity':np.pi,
        'conv_threshold':1e-8}
prog.set_print(level='diagnostic')
prog.update_var(**{'opt_thresh':0.001})
prog.update_var(target='qc',**kw_qc )
prog.update_var(target='opt',**kw_opt)
prog.update_var(target='orb_opt',**orb_opt)
prog.build()
