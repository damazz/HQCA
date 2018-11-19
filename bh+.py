#
# ./lih.py
#
# Generic molecular input for use with main.py 
#


from pyscf import gto
mol = gto.Mole()
mol.atom = '''B 0 0 0; H 2.0 0 0'''
mol.basis = 'sto-3g'
mol.spin=1
mol.charge=1
mol.verbose=2
mol.build()
