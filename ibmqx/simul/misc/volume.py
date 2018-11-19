import numpy as np
import sys

# program to find the volume of the GPC polytope


def volume_gpc(cube_step):
    # start from 0.5,0.5,0.5
    vol = (cube_step)**3
    step = 0.5*cube_step
    total = 0
    for i in np.arange(0.5+step,1,cube_step):
        for j in np.arange(0.5+step,1,cube_step):
            for k in np.arange(0.5+step,1,cube_step):
                if (i>=j and j>=k and k>=(i+j-1)):
                    total += vol
                if k>j:
                    break
            if j>i:
                break
    return total

# l4 <= l5 + l6
# 1-l3 <= 2-l2-l1
# - l3 <= 1-l2-l1
# l3 >= l2+l1-1
converged = True
tolerance = 1e-5
step = 0.5
#old_vol = volume_gpc(step)
#print(old_vol)
while converged==False:
    step *= 0.5
    new_vol = volume_gpc(step)
    abs_diff = abs(old_vol - new_vol)
    if abs_diff<tolerance:
        converged=True
    else:
        old_vol = new_vol
    print(old_vol)
    print(abs_diff)
    print(1/old_vol,'\n')

   # the volume approaches 0.01041667, or 1/96
   # note the full cube is 1/8
   # half of this is 1/16
   # the triangular portion of the ensemble sate is 1/48
   # the GPC portion is 1/96

def volume_set(set_of_points,diameter):
    n = len(set_of_points)
    v_gpc = 1/96
    v_set = n*np.pi*(diameter**3)/6
    ratio = v_set/v_gpc
    return ratio,v_set

if __name__=='__main__':    
    try:
        data_set = np.loadtxt(sys.argv[1])
        diameter = float(input('Sphere diameter?'))
    except:
        data_set = np.loadtxt(input('Filename?'))
        diameter = float(input('Sphere diameter?'))
    ratio,vol = volume_set(data_set,diameter)
    print(ratio,vol)
