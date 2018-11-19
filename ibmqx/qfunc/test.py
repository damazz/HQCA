import gpc
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import rdm




##### comparing tomography and own algorithm

circuit = gpc.Run_Type_ry6p()
circuit.create_off_diagonal()
circuit.ry6p('main',[45,0,15,0,25,0],[0,2,2,1,1,2])
circuit.ry6p('err',[45,0,15,0,25,0],[0,2,2,1,2,1])

rho,unit,rdms = circuit.tomography(1)
val = rdm.rdm(unit,unitary=True)
unit_occ = np.zeros(6)
for z in range(0,3):
    unit_occ[z]   = val[z]
    unit_occ[5-z] = 1-val[z]
unit_occ.sort()

scott = np.zeros(6)
ibm_t = np.zeros(6)
ind = 0
for N in [256,512,1024,2048,4096,8192]:
#for N in [10,10,10,10,10,10]:
    results,circs = circuit.execute(N)
    main = rdm.rdm(results.get_data('main')['counts'])
    err = rdm.rdm(results.get_data('err')['counts'])
    rdms, occ, vec= rdm.construct_rdm(main,err)
    occ = np.asarray(occ)
    occ.sort()
    ###
    rho,unit,rdms = circuit.tomography(N)
    eig = []
    for i in rdms:
        eigval,eigvec = LA.eig(i)
        for j in eigval:
            eig.append(np.real(j))
    eig = np.asarray(eig)
    eig.sort()
    scott[ind] = np.sqrt(np.sum(np.square(occ[0:3]-unit_occ[0:3])))
    ibm_t[ind] = np.sqrt(np.sum(np.square(eig[0:3]-unit_occ[0:3])))
    ind+= 1


print(scott)
print(ibm_t)
x = [8,9,10,11,12,13]
plt.plot(x,scott,'rx',label='scott')
plt.plot(x,ibm_t,'bo',label='ibm_tomography')
plt.xlabel('Number of shots, $2^x$, or log base 2')
plt.ylabel('Distance from ideal point')
plt.legend()
plt.show()

