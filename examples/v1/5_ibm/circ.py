from pyscf import gto,scf,mcscf
from os import getcwd
from math import pi
from hqca.hqca import sp,scan,circuit
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 0.9+2.5
x = d*(1/2)
y = d*(np.sqrt(3)/2)
mol.atom=[['H',(-x,0,0)],['H',(x,0,0)],['H',(0,y,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.charge=1
mol.verbose=0
mol.build()
prog = circuit(theory='noft')#,mol=mol,casci=True)
filename = getcwd()+'/ibmq_16_melbourne'
kw_qc = {
        'Ne_as':2,
        'No_as':3,
        'alpha_mos':{'active':[0,1,2]},
        'beta_mos':{'active':[3,4,5]},
        'Nq':6,
        'Nq_ancilla':0,
        'num_shots':4096,
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
                ]
            },
        'spin_mapping':'alternating',
        'depth':1,
        'info':'tomo',
        #'provider':'IBMQ',
        #'backend':'ibmq_16_melbourne',
        #'noise':True,
        #'noise_model_location':filename,
        #'backend_initial_layout':[4,5,6,8,9,10],
        #'backend_file':filename,
        'ec_comp_ent':False,
        'ec_post':True,
        'ec_pre':False,
        'ec_pre_kw':{
            'filter_measurements':False,
            },
        'ec_syndrome':False,
        'ec_comp_ent_kw':{
            'ec_replace_quad':[ #list of quad sequences
                { #entry for gate 1
                    'replace':True,
                    'N_anc':1,
                    'circ':'pauli_UCC2_test',
                    'kw':{'seq':'xxxy'},
                    'use':'sign',
                    },
                { #entry for gate 2
                    'replace':True,
                    'N_anc':1,
                    'circ':'pauli_UCC2_2s_test',
                    'kw':{'seq':'xxxy'},
                    'use':'sign',
                    },
                ]
            },
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
        'conv_crit_type':'MaxDist',
        #'conv_crit_type':'default',
        'initial_conditions':'han',
        'conv_threshold':1e-3,
        'N_vectors':2,
        }
orb_opt = {
        'optimizer':'bfgs',
        'unity':np.pi,
        'conv_threshold':1e-8}
prog.set_print(level='diagnostic')
prog.update_var(**{'opt_thresh':0.001})
prog.update_var(target='qc',**kw_qc )
#prog.update_var(target='opt',**kw_opt)
#prog.update_var(target='orb_opt',**orb_opt)
prog.build()

#prog.execute()
