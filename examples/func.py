'''
./examples/func.py

Contains functions for running examples.
'''

def change_config(input_file,mol_file):
    with open('./config.txt','w') as fp:
        fp.write('# Pointer for input file \n')
        fp.write('input_file= {} \n'.format(input_file))
        fp.write('# Pointer for mol file \n')
        fp.write('mol_file= {} \n'.format(mol_file))


