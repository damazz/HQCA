#
# ./bh+.py
#
# Generic molecular input for use with main.py 
#


from pyscf import gto
mol = gto.Mole()
mol.atom = '''B 0 0 0; H 1.3576 0 0'''
mol.basis = 'sto-3g'
mol.spin=1
mol.charge=1
mol.verbose=2
mol.build()



smol = gto.Mole()
smol.atom = '''B 0 0 0; H 1.2076 0 0'''
smol.basis = 'sto-3g'
smol.spin=1
smol.charge=1
smol.verbose=2
smol.build()
