from pyscf import gto
mol = gto.Mole()
mol.atom ='''
    C  0.000  0.000 0.000;
    C -1.305 -0.660 0.000;
    C -1.305  0.660 0.000;
    H -1.867 -1.581 0.000;
    H -1.867  1.581 0.000;
    H  0.757  0.000 0.787
    '''
mol.basis = 'sto-3g'
mol.spin=1
mol.verbose=2
mol.build()
