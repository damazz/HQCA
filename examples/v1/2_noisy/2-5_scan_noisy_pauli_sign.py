from pyscf import gto,scf,mcscf
from math import pi
from hqca.hqca import sp,scan
import sys
import numpy as np
from functools import reduce

mol = gto.Mole()
d = 0.9374+0.5626
mol.atom=[
        ['H',(0,0,0)],
        ['H',(d,0,0)]]
mol.basis='sto-3g'
#mol.basis='6-31g'
mol.spin=0
mol.verbose=0
mol.build()
prog = scan(theory='noft',scan_type='occ',mol=mol)
filename = '/home/scott/Documents/research/software/'
filename+= 'hqca/tests/2_noisy/20190514_ibmqx2'
kw_qc = {
        'Ne_as':2,
        'No_as':2,
        'alpha_mos':{'active':[0,1]},
        'beta_mos':{'active':[2,3]},
        'Nq':4,
        'Nq_ancilla':0,
        'num_shots':8192,
        'entangler_q':'UCC2_1s',
        'spin_mapping':'alternating',
        'depth':1,
        'info':'draw',
        'noise':True,
        'noise_model_location':filename,
        'backend_initial_layout':[0,1,2,3,4],
        'backend_file':filename,
        'ec_comp_ent':False,
        'ec_post':True,
        'ec_syndrome':False,
        'ec_post_kw':{
            'symm_verify':True,
            'symmetries':[],#['N','Sz'],
            },
        'tomo_basis':'no',
        'tomo_extra':'sign_2e_pauli',
        'tomo_approx':'full',
        'transpile':'default',
        'transpiler_keywords':{
            'optimization_level':0,
            },
        'transpile':'default',
        'ansatz':'nat-orb-no',
        }
prog.update_var(target='qc',**kw_qc )
prog.set_print(level='diagnostic_qc')
prog.build()
prog.scan(
        shift=[0],
        index=[0],
        lowers=[-pi],
        uppers=[pi],
        steps=[15],
        rdm=True
        )
