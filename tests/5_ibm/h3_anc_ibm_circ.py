from pyscf import gto,scf,mcscf
from math import pi
from hqca.hqca import sp,scan,circuit
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 0.9
x = d*(1/2)
y = d*(np.sqrt(3)/2)
mol.atom=[['H',(-x,0,0)],['H',(x,0,0)],['H',(0,y,0)]]
mol.basis='sto-3g'
mol.spin=0
mol.charge=1
mol.verbose=0
mol.build()
prog = circuit(theory='noft',mol=mol)
filename = '/home/scott/Documents/research/software/'
filename+= 'hqca/tests/2_noisy/20190514_ibmqx2'
kw_qc = {
        'Ne_as':2,
        'No_as':3,
        'alpha_mos':{'active':[0,1,2]},
        'beta_mos':{'active':[3,4,5]},
        'Nq':6,
        'Nq_ancilla':2,
        'num_shots':8192,
        'entangler_q':'custom',
        'entangler_kw':{
            'entanglers':[
                {
                    'circ':'UCC2_1s_custom',
                    'kw':{
                        'seq':'yxxx'},
                    'np':1
                    },
                {
                    'circ':'UCC2_2s_custom',
                    'kw':{'start':'yxyy'},
                    'np':1
                    },
                ]
            },
        'spin_mapping':'alternating',
        'depth':1,
        'info':'tomo',
        'provider':'IBMQ',
        'backend':'ibmq_16_melbourne',
        #'noise':True,
        #'noise_model_location':filename,
        'backend_initial_layout':[3,4,10,9,8,6,11,5],
        #'backend_file':filename,
        'ec_comp_ent':True,
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
                    'circ':'pauli_UCC2_test_1s_rev',
                    'kw':{'seq':'yxxx'},
                    'use':'sign',
                    },
                { #entry for gate 2
                    'replace':True,
                    'N_anc':1,
                    'circ':'pauli_UCC2_2s_test',
                    'kw':{'start':'yxyy'},
                    'use':'sign',
                    },
                ]
            },
        'ec_post_kw':{
            'symm_verify':True,
            'symmetries':['N','Sz'],
            },
        #'tomo_basis':'pauli_symm',
        'tomo_basis':'no',
        'tomo_extra':'sign_2e_from_ancilla',
        'tomo_approx':'h3_fo',
        'transpile':'default',
        'transpiler_keywords':{
            },
        'transpile':'default',
        'ansatz':'nat-orb-no',
        }
prog.update_var(target='qc',**kw_qc )
prog.set_print(level='diagnostic_qc')
prog.build()
