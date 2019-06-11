from hqca.hqca import circuit
from math import pi

'''
1-1_ecc_circuit 

test for error correction circuit, with....some capacity
'''
print('Would you like to:')
print('(1) draw  - visualize the circuit')
print('(2) tomo  - visualize circuit with tomography')
print('(3) build - transpile circuit and visualize')
print('(4) qasm  - transpile circuit and save qasm')
print('(5) calc  - get simple statistics')
print('(6) stats - get more detailed statistics')

input_ok = False
while not input_ok:
    info = input('Input: ')
    if info in ['draw','tomo','build','qasm','stats','calc']:
        input_ok=True
    else:
        print('Incorrect input. Try again.')
print('Lets go. ')
print('')



prog = circuit(theory='noft')
kw_qc = {
        'Ne_as':2,
        'No_as':2,
        'alpha_mos':{'active':[0,1]},
        'beta_mos':{'active':[2,3]},
        'Nq':4,
        'Nq_ancilla':1,
        'num_shots':4096,
        'entangler_q':'UCC2_1s',
        'spin_mapping':'default',
        'depth':1,
        'info':info,
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
        'tomo_extra':'sign_2e_pauli',
        'tomo_approx':'fo',
        'transpile':'default',
        'transpiler_keywords':{
            'optimization_level':0,
            },
        'transpiler_keywords':{
            #'optimization_level':0,
            #'seed_transpiler':15,
            },
        'transpiler_keywords':{
            #'optimization_level':0,
            #'seed_transpiler':15,
            },
        'transpile':'default',
        'ansatz':'nat-orb-no',
        'tomo_basis':'no',
        }
prog.update_var(target='qc',**kw_qc )
prog.set_print(level='default')
prog.build()
