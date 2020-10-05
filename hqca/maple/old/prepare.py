from hqca.tools import *
import sys
import numpy as np
import pickle
np.set_printoptions(precision=4,linewidth=300,suppress=True)

with open('pbenzyne_0915_00.log','rb') as fp:
    data = pickle.load(fp)

rdms = data['log-D']


test_rdm = RDM(order=2,
        alpha=[0,1,2,3],
        beta=[4,5,6,7],
        state='given',
        Ne=4,
        S=0,
        S2=0,
        rdm=rdms[-1].rdm)
#test_rdm.save('h4_qc_spatial',spin=False)
#test_rdm.save('h4_qc_spin',spin='ab')
spin_pure = np.loadtxt('h4_qc_spin_pure.csv',delimiter=',')
spatial_pure = np.loadtxt('h4_qc_spatial_pure.csv',delimiter=',')

spatial = test_rdm.process_for_maple(spin=False) #spatial
#sys.exit()

ab_ref = test_rdm.process_for_maple(spin='ab')
aa_ref = test_rdm.process_for_maple(spin='aa')

ab_ref =np.reshape(ab_ref,(16,16))
aa_ref =np.reshape(aa_ref,(16,16))

trdm = rdms[-1]

aa_spat = (spatial-spatial.transpose(1,0,2,3))*(1/6)
aa_spat = np.reshape(aa_spat,(16,16))
ab_spat = (2*spatial+spatial.transpose(1,0,2,3))*(1/6)
ab_spat = np.reshape(ab_spat,(16,16))

print(np.linalg.norm(ab_ref-ab_spat))

print(np.reshape(spatial,(16,16)))
print(spatial_pure)
print('--------------')

#print(aa_ref)
#print(aa_spat)
#print('--------------')
#
print(ab_ref)
#print(ab_spat)
print(spin_pure)






#test_rdm.save(name='h4_qc_spatial',dtype='rdm',spin=False)



