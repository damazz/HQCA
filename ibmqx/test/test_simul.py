import numpy as np
import sys
sys.path.append('/home/scott/Documents/research/3_vqa/hqca/ibmqx')
from simul import run as sim

print(sim.single_run_c3(45,45,45,[0,2,0,1,2,1]))
#sys.exit()

c = np.pi/180
theta1a = np.arange(0,1,11.25)
theta1b = np.arange(11.25,12,11.25)
theta1c = np.arange(22.5,23,11.25)
theta1d = np.arange(33.75,34,11.25)
theta1e = np.arange(45,46,11.25)
theta2 = np.arange(0,46,11.25)
theta3 = np.arange(0,46,11.25)
hold1, use1 = sim.multi_run_c3(theta1a,theta2,theta3,[[0,2,0,1,2,1]],1e-4,rad=False)
hold2, use2 = sim.multi_run_c3(theta1b,theta2,theta3,[[0,2,0,1,2,1]],1e-4,rad=False)
hold3, use3 = sim.multi_run_c3(theta1c,theta2,theta3,[[0,2,0,1,2,1]],1e-4,rad=False)
hold4, use4 = sim.multi_run_c3(theta1d,theta2,theta3,[[0,2,0,1,2,1]],1e-4,rad=False)
hold5, use5 = sim.multi_run_c3(theta1e,theta2,theta3,[[0,2,0,1,2,1]],1e-4,rad=False)
print(use1,use2,use3,use4,use5)
np.savetxt('test_a.txt',hold1,fmt='%.5f')
np.savetxt('test_b.txt',hold2,fmt='%.5f')
np.savetxt('test_c.txt',hold3,fmt='%.5f')
np.savetxt('test_d.txt',hold4,fmt='%.5f')
np.savetxt('test_e.txt',hold5,fmt='%.5f')
