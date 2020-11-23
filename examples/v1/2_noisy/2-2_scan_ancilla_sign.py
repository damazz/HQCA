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

kw_qc = {
        'Ne_as':2,
        'No_as':2,
        'alpha_mos':{'active':[0,1]},
        'beta_mos':{'active':[2,3]},
        'Nq':4,
        'Nq_ancilla':1,
        'num_shots':4096,
        'entangler_q':'UCC2_1s',
        'spin_mapping':'alternating',
        'depth':1,
        'info':'draw',
        'ec_comp_ent':True,
        'ec_syndrome':False,
        'ec_syndrome_kw':{
            'apply_syndromes':{
                'sign':[
                    {
                    'N_anc':1,
                    'circ':'ancilla_sign',
                    'use':'sign',
                    'kw':{}
                        }
                    ],
                }
            },
        'ec_comp_ent_kw':{
            'ec_replace_quad':[ #list of quad sequences
                { #entry for gate 1
                    'replace':True,
                    'N_anc':1,
                    'circ':'pauli_UCC2_test',
                    'kw':{},
                    'use':'sign',
                    }
                ]
            },
        'tomo_extra':'sign_2e_from_ancilla',
        'tomo_approx':'fo',
        'transpile':'default',
        'transpiler_keywords':{
            'optimization_level':0,
            },
        'transpile':'default',
        'ansatz':'nat-orb-no',
        'tomo_basis':'no',
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
