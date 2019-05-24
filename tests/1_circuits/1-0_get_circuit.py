from hqca.hqca import circuit
from math import pi

'''
get input, and then visualize circuit
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
        'Nqb':4,
        'num_shots':4096,
        'entangler_q':'UCC2_1s',
        'spin_mapping':'default',
        'depth':1,
        'info':info,
        'qc':True,
        'tomo_extra':'sign_2e_pauli',
        'tomo_approx':'fo',
        'transpile':'default',
        'ansatz':'nat-orb-no',
        'tomo_basis':'no',
        }
prog.update_var(target='qc',**kw_qc )
prog.set_print(level='default')
prog.build()
