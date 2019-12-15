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
        'Nq':mol.nbas*2,
        'Nq_ancilla':1,
        'num_shots':4096,
        'entangler_q':'UCC2_1s',
        'spin_mapping':'default',
        'depth':1,
        'use_radians':True,
        'transpile':True,
        'tomo_extra':'sign_2e_from_ancilla',
        'tomo_basis':'no',
        'ansatz':'nat-orb-no',
        'Ne_as':2,
        'No_as':2,
        'alpha_mos':{'active':[0,1]},
        'beta_mos':{'active':[2,3]},
        'ec_comp_ent':True,
        'ec_comp_ent_kw':{
            'ec_replace_quad':[ #list of quad sequences
                { #entry for gate 1
                    'replace':True,
                    'N_anc':1,
                    'circ':'pauli_UCC2_test',
                    'kw':{'pauli':'hhhh'},
                    'use':'sign',
                    }
                ]
            },
        }
kw_opt = {
        'optimizer':'nevergrad',
        'unity':np.pi/2,
        'nevergrad_opt':'Cobyla',
        'max_iter':500,
        'conv_crit_type':'MaxDist',
        'N_vectors':5
        }
prog.update_var(target='qc',**kw_qc )
prog.update_var(target='opt',**kw_opt)
prog.set_print(level='default')
prog.build()
prog.scan(
        shift=[0],
        index=[0],
        lowers=[-pi],
        uppers=[pi],
        steps=[20],
        rdm=True
        )

