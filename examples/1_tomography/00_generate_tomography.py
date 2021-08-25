'''

Generating simple tomographies. 

StandardTomography provides a generic tomography of the 2-RDM. ReducedTomography uses a symmetry projected operator basis. Both approaches utilize simplifcation through the clique coloring method.  


'''

from hqca.tomography import StandardTomography, ReducedTomography
from pyscf import gto
from hqca.acse import StorageACSE
from hqca.hamiltonian import MolecularHamiltonian
from hqca import QuantumStorage
from hqca.transforms import JordanWigner


mol = gto.Mole()
mol.atom=[
        ['H',(0,0,0)],
        ['H',(2.0,0,0)],
        ['H',(+2.0,1,0)],
        ['H',(-2.0,-1,0)],
        ['H',(+2.0,0,1)],
        ['H',(-2.0,0,-1)],
        ]
mol.basis='sto-3g'
mol.spin=0
mol.verbose=0
mol.build()

# we dont need it to 
ham =  MolecularHamiltonian(
        mol=mol,
        generate_operators=False,
        transform=JordanWigner)
st = StorageACSE(ham)
qs = QuantumStorage()
qs.set_algorithm(st)
qs.set_backend(
        backend='statevector_simulator',
        Nq=12,
        provider='Aer')
print('# # # # # # #')
print('Standard approach: ')
print('# # # # # # #')
tomoRe = StandardTomography(qs)
tomoRe.generate(real=True,imag=False,transform=JordanWigner,verbose=True)

print('# # # # # # #')
print('Reduced approach: ')
print('# # # # # # #')
tomoRe = ReducedTomography(qs)
tomoRe.generate(real=True,imag=False,transform=JordanWigner,verbose=True)
