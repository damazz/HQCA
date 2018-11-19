from pyscf import gto
mol = gto.Mole()
mol.atom = '''He 0 0 0; H -1.143 0 0; H 1.143 0 0 '''
mol.basis = 'sto-3g'
mol.spin=1
mol.charge=1
mol.verbose=2
mol.build()
