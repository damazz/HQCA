from pyscf import gto
mol = gto.Mole()
mol.atom = '''H 0 0 0; H 0 0 -0.5374; H 0 0 0.5374'''
mol.basis = 'sto-3g'
mol.spin=1
mol.verbose=2
mol.build()
