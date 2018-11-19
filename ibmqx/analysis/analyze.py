#
# Analysis program - particularly for generating high quality plots from data sets. 
#
## take a 
import qiskit
from qiskit import QuantumProgram,QuantumCircuit
from qiskit.tools.qi.qi import state_fidelity,purity
import matplotlib.pyplot as plt
from qiskit.tools.qcvv import tomography as tomo
from func.useful import gpc as gpc
from func.useful import mean_stdv
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import seaborn as sns
import sys
import math
import numpy as np
import pickle
sys.path.append('../gpc/')
from gpcf import gpc
# 
np.set_printoptions(suppress=True,precision=5)
# load file
# 

def_in_loc = './compiled/'

try:
    input_name = def_in_loc + sys.argv[1]
    
except Exception:
    sys.exit('Goodbye!')

with open(input_name,'rb') as fp:
    data = pickle.load(fp)

N_data = len(data)
print('You have {} data points.'.format(N_data))
print('There are the following dictionaries available.')
print(data[0].keys())

print('Here are the analysis choices:')



choices = ['histograms','gpc','gpc_error','compare','tomography']
print(choices)
choice = input('Please choose a analysis method: ')

#
# Graphing Methods:
# ____pairplot
# ____gpc
# ____histograms 
#
#
if choice=='tomography':
    if 'tomo-ON' in data[0].keys():
        print('OK! Have tomography data.')
    else:
        print('X. Do not have tomography data. Wrong analysis option.')
        sys.exit('Goodbye!')
    for item in data:
        # genrate unitary state
        test_circ = gpc.Psuedo_Quant_Algorithm(item['main-counts'],item['err-counts'],5)
        test_circ.get_unitary('ry6p',item['parameters'],item['order'])
        state = test_circ.circuit.unit_state
        rho = np.asmatrix(item['tomo-rho'])
        print('Fidelity = {}'.format(state_fidelity(rho,state)))
        print('Purity = {}'.format(purity(rho)))


elif choice=='compare':
    if 'res-ON' in data[0].keys():
        print('OK! Have shots data.')
        shots = True
    else:
        print('X. Do not have shots data.')
        shots = False
    if 'tomo-ON' in data[0].keys():
        print('OK! Have tomography data.')
        tomo = True
    else:
        print('X. Do not have tomography data.')
        tomo = False
    if 'ideal' in data[0].keys():
        print('OK! Have unitary data.')
        unit = True
    else:
        print('X. Do not have unitary data.')
        unit = False
        sys.exit()

    for point in data:
        print('Data attributes:\nNumber of shots: {}\nParameters: {}'.format(point['shots'],point['parameters']))
        unit_ON = np.asarray(point['ideal'])
        print(unit_ON)
        if shots:
            res_ON = np.asarray(point['res-ON'])
            res_err = np.sqrt(np.sum(np.square(unit_ON[3:]-res_ON[3:])))
            print('Distance from ideal point with 1-RDM tomography:\n {}'.format(res_err))
            print(res_ON)
        if tomo:
            tomo_ON = np.asarray(point['tomo-ON'])
            tomo_err = np.sqrt(np.sum(np.square(unit_ON[3:]-tomo_ON[3:])))
            print('Distance from ideal point with IBM tomography:\n {}'.format(tomo_err))
            print(tomo_ON)
            
    


elif choice == 'histograms': 
    #begin histograms
    print('What type of histograms would you like to look at?')
    print('off_diag, diag, both, or rdm?')
    type_err = input('')
    cont_hist = False
    while cont_hist==False:
        if type_err=='off_diag' or type_err=='diag' or type_err=='both' or type_err=='rdm':
            cont_hist=True
        else:
            type_err =input('Invalid selection. Please input histogram selection:  ')
    else: 
        type_err = 'off_diag'
    if type_err=='off_diag':
        pass
        print('Number of orderings: ',len(orders))
        bins = round(n_data/(len(orders)*4))
        f, axes = plt.subplots(len(orders))
            
        for ind in range(0,len(orders)):
            if len(orders)==1:
                ord_s = dataset.as_matrix()
                print(rev_orders[ind])
                for qb in range(0,3):
                    print(qb,ind)
                    axes.hist(ord_s[:,qb+6],normed=True,bins='scott',label='qb{0}'.format(qb))
                    print(mean_stdv(ord_s[:,qb+6]))
                axes.set_title('Ordering = {0}'.format(rev_orders[ind]))
            else:
                ord_s = dataset.query('ord=={0}'.format(ind))
                ord_s = ord_s.as_matrix()
                #print(ord_s)
                print(rev_orders[ind])
                for qb in range(0,3):
                    axes[ind].hist(ord_s[:,qb+6],normed=True,bins='sqrt',label='qb{0}'.format(qb))
                    print(mean_stdv(ord_s[:,qb+6]))
                axes[ind].set_title('Ordering = {0}'.format(rev_orders[ind]))
            plt.legend()
        plt.show()
    elif type_err=='rdm':
        # prepare the data
        bins = round(n_data/(len(orders)*4))
        f, axes = plt.subplots(len(orders))
        for ind in range(0,len(orders)):
            if len(orders)==1:
                ord_s = dataset.as_matrix()
                exp_dat = np.sort(ord_s[:,:6],axis=1)
                cal_dat = np.sort(ord_s[:,13:19],axis=1)
                exp_dat.sort()
                cal_dat.sort()
                euclid = np.zeros(int(n_data/len(orders)))
                for each in range(0,n_data):
                    euclid[each] = np.sqrt(np.sum(np.square(exp_dat[each,3:]-cal_dat[each,3:])))
                print(rev_orders[ind])
                for qb in range(0,1):
                    print(qb,ind)
                    axes.hist(euclid,normed=True,bins='sqrt')
                    print(mean_stdv(euclid))
                axes.set_title('Ordering = {0}'.format(rev_orders[ind]))
            else:
                ord_s = dataset.query('ord=={0}'.format(ind))
                ord_s = ord_s.as_matrix()
                #print(ord_s)
                print(rev_orders[ind])
                for qb in range(0,3):
                    axes[ind].hist(ord_s[:,qb],normed=True,bins='sqrt',label='qb{0}'.format(qb))
                    print(mean_stdv(ord_s[:,qb+6]))
                axes[ind].set_title('Ordering = {0}'.format(rev_orders[ind]))
            plt.legend()
        plt.show()
    elif type_err=='diag':
        diag_type = input('\'euc\' or \'qb\'')
        f, axes = plt.subplots(len(orders))
        for ind in range(0, len(orders)):
            if len(orders)==1:
                ordered_set = dataset.as_matrix()
                exp_dat = ordered_set[:,:6]
                err_dat = ordered_set[:,6:9]
                cal_dat = np.sort(ordered_set[:,13:19],axis=1)
                diag_err = np.zeros((n_data,6))
                for each in range(0,n_data):
                    for block in range(0,3):
                        l1 = exp_dat[each,block*2]
                        l2 = exp_dat[each,1+block*2]
                        err = err_dat[each,block] 
                        print(l1,l2,err)
                        diag_err[each,block*2] = 0.5-0.5*np.sqrt(1-4*(l1*l2+err**2))
                        diag_err[each,1+block*2] = 1-diag_err[each,block*2]
                        if math.isnan(float(diag_err[each,block*2])):
                            diag_err[each,block*2]=0
                            diag_err[each,block*2+1]=0
                d_err = np.sort(diag_err,axis=1)
                err_diff = abs(cal_dat - d_err)
                euclid = np.zeros(int(n_data/len(orders)))
                euclid2 = np.zeros(int(n_data/len(orders)))
                if diag_type=='euc':
                    for each in range(0,n_data):
                        euclid[each] = np.sqrt(np.sum(np.square(d_err[each,3:]-cal_dat[each,3:])))
                    for qb in range(0,1):
                        print(qb,ind)
                        axes.hist(euclid,normed=True,bins='sqrt')
                        print(mean_stdv(euclid))
                    axes.set_title('Euclidean Distance for Diagonal vs. Calc, Ordering = {0}'.format(rev_orders[ind]))
                        
                    plt.show()
                    f, axes = plt.subplots(len(orders))
                    for each in range(0,n_data):
                        euclid2[each] = np.sqrt(np.sum(np.square(err_dat[each,:])))
                    for qb in range(0,1):
                        print(qb,ind)
                        axes.hist(euclid2,normed=True,bins='sqrt')
                        print(mean_stdv(euclid2))
                    axes.set_title('Euclidean Distance for Off-Diag vs. Calc, Ordering = {0}'.format(rev_orders[ind]))
                elif diag_type=='qb':
                    for qb in range(0,3):
                        print(qb,ind)
                        axes.hist(err_diff[:,qb*2],normed=True,bins='sqrt',label='qb{0}'.format(qb))
                        print(mean_stdv(err_diff[:,qb*2]))
                    axes.set_title('Ordering = {0}'.format(rev_orders[ind]))
                    
            else:
                sys.exit('Not supported currently. Goodbye!')
                
        plt.show()


elif choice == 'gpc_error':

    # first, set data
    datagpc = dataset.as_matrix()
    cal_points = datagpc[:,:6]
    exp_points = datagpc[:,-6:]
    cal_points = np.sort(cal_points,axis=1)
    exp_points = np.sort(exp_points,axis=1)
    print(cal_points,exp_points)
    x1 = cal_points[:,5]
    y1 = cal_points[:,4]
    z1 = cal_points[:,3]
    x2 = exp_points[:,5]
    y2 = exp_points[:,4]
    z2 = exp_points[:,3]

    # now, begin plot
    fig = plt.figure() #set up matplotlib figure object 
    ax = fig.add_subplot(111,projection='3d')

    ax.zaxis.set_rotate_label(False) 
    ax.xaxis.set_rotate_label(False) 
    ax.yaxis.set_rotate_label(False) 
    ax.set_xlabel('$\lambda$ 1',rotation='horizontal') 
    ax.set_ylabel('$\lambda$ 2',rotation='horizontal')
    ax.set_zlabel('$\lambda$ 3',rotation='horizontal')
    ax.set_xlim(0.5,1)
    ax.set_ylim(0.5,1)
    ax.set_zlim(0.5,1)
    ax.scatter(x1,y1,z1,c='r')
    ax.scatter(x2,y2,z2,c='b',edgecolors='k')

    rt2 = 0.5 * np.sqrt(2)
    
    verts1 = [[0.5,0.5,0.5],[1,1,1],[0.75,0.75,0.5]]
    verts2 = [[1, 1, 1], [0.5,0.5, 0.5],[0.75,0.75,0.5]]
    verts3 = [[1, 1, 1], [0.5, 0.5, 0.5], [1, 0.5, 0.5]]
    verts4 = [[1,1,1],[0.75,0.75,0.5],[1,0.5,0.5]]
    
    face1= Poly3DCollection([verts1],linewidth=2,alpha=0.33)
    face2= Poly3DCollection([verts2],linewidth=2,alpha=0.33)
    face3= Poly3DCollection([verts3],linewidth=2,alpha=0.33)
    face4= Poly3DCollection([verts4],linewidth=2,alpha=0.33)
        
    aleph = 0.5
    face1.set_facecolor((0,0.5,0.5,aleph))
    face2.set_facecolor((0,0.5,0.5,aleph))
    face3.set_facecolor((0,0.5,0.5,aleph))
    face4.set_facecolor((0,0.5,0.5,aleph))
    
    ax.add_collection3d(face1)
    ax.add_collection3d(face2)
    ax.add_collection3d(face3)
    ax.add_collection3d(face4)
    
    ax.zaxis.set_rotate_label(False) 
    ax.xaxis.set_rotate_label(False) 
    ax.yaxis.set_rotate_label(False) 
    ax.set_xlabel('$\lambda$ 1',rotation='horizontal',labelpad=20) 
    ax.set_ylabel('$\lambda$ 2',rotation='horizontal',labelpad=20)
    ax.set_zlabel('$\lambda$ 3',rotation='horizontal',labelpad=8)
    ax.set_xlim(0.5,1)
    ax.set_ylim(0.5,1)
    ax.set_zlim(0.5,1)
    
    
    
    
    plt.show()
elif choice == 'gpc':
    # set data

    datagpc = np.zeros((N_data,6))
    data_type = input('Would you like exp, ideal, or filt? ')
    for i in range(0,N_data):
        if data_type =='exp':
            datagpc[i,:] = data[i]['exp-ON']
        elif data_type == 'filt':
            datagpc[i,:] = data[i]['diag-ON']
        else:
            datagpc[i,:] = data[i]['ideal']
    xs = datagpc[:,5]
    ys = datagpc[:,4]
    zs = datagpc[:,3]

    # begin plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.zaxis.set_rotate_label(False) 
    ax.xaxis.set_rotate_label(False) 
    ax.yaxis.set_rotate_label(False) 
    ax.set_xlabel('$\lambda$ 1',rotation='horizontal') 
    ax.set_ylabel('$\lambda$ 2',rotation='horizontal')
    ax.set_zlabel('$\lambda$ 3',rotation='horizontal')
    ax.set_xlim(0.5,1)
    ax.set_ylim(0.5,1)
    ax.set_zlim(0.5,1)
    for xm, ym, zm in zip(xs,ys,zs):
        if (zm + 1) < (ym + xm):
            col = 'r'
        else:
            col = 'b'
        ax.scatter(xm,ym,zm,c=col,edgecolors='k',s=60)
    
    #ax.scatter(xs,ys,zs,c=colors)
    
    #ax = Axes3D(fig)
    rt2 = 0.5 * np.sqrt(2)
    
    verts1 = [[0.5,0.5,0.5],[1,1,1],[0.75,0.75,0.5]]
    verts2 = [[1, 1, 1], [0.5,0.5, 0.5],[0.75,0.75,0.5]]
    verts3 = [[1, 1, 1], [0.5, 0.5, 0.5], [1, 0.5, 0.5]]
    verts4 = [[1,1,1],[0.75,0.75,0.5],[1,0.5,0.5]]
    
    face1= Poly3DCollection([verts1],linewidth=2,alpha=0.33)
    face2= Poly3DCollection([verts2],linewidth=2,alpha=0.33)
    face3= Poly3DCollection([verts3],linewidth=2,alpha=0.33)
    face4= Poly3DCollection([verts4],linewidth=2,alpha=0.33)
        
    aleph = 0.5
    face1.set_facecolor((0,0.5,0.5,aleph))
    face2.set_facecolor((0,0.5,0.5,aleph))
    face3.set_facecolor((0,0.5,0.5,aleph))
    face4.set_facecolor((0,0.5,0.5,aleph))
    
    ax.add_collection3d(face1)
    ax.add_collection3d(face2)
    ax.add_collection3d(face3)
    ax.add_collection3d(face4)
    
    ax.zaxis.set_rotate_label(False) 
    ax.xaxis.set_rotate_label(False) 
    ax.yaxis.set_rotate_label(False) 
    ax.set_xlabel('$\lambda$ 1',rotation='horizontal',labelpad=20) 
    ax.set_ylabel('$\lambda$ 2',rotation='horizontal',labelpad=20)
    ax.set_zlabel('$\lambda$ 3',rotation='horizontal',labelpad=8)
    ax.set_xlim(0.5,1)
    ax.set_ylim(0.5,1)
    ax.set_zlim(0.5,1)
    
    
    
    
    plt.show()

elif choice == 'gpc_mod':
    # gpc, with two sections highlighted....shows GPC and possible set of orderings
   
    # set data
    datagpc = dataset.as_matrix()
    datagpc = datagpc[:,:6]
    datagpc = np.sort(datagpc,axis=1)
    print(datagpc)
    xs = datagpc[:,5]
    ys = datagpc[:,4]
    zs = datagpc[:,3]

    # begin plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.zaxis.set_rotate_label(False) 
    ax.xaxis.set_rotate_label(False) 
    ax.yaxis.set_rotate_label(False) 
    ax.set_xlabel('$\lambda$ 1',rotation='horizontal') 
    ax.set_ylabel('$\lambda$ 2',rotation='horizontal')
    ax.set_zlabel('$\lambda$ 3',rotation='horizontal')
    ax.set_xlim(0.5,1)
    ax.set_ylim(0.5,1)
    ax.set_zlim(0.5,1)
    for xm, ym, zm in zip(xs,ys,zs):
        if (zm + 1) < (ym + xm):
            col = 'r'
        else:
            col = 'b'
        ax.scatter(xm,ym,zm,c=col,edgecolors='k',s=60)
    
    #ax.scatter(xs,ys,zs,c=colors)
    
    #ax = Axes3D(fig)
    rt2 = 0.5 * np.sqrt(2)
    
    verts11 = [[0.5,0.5,0.5],[1,1,1],[0.75,0.75,0.5]]
    verts12 = [[1, 1, 1], [0.5,0.5, 0.5],[0.75,0.75,0.5]]
    verts13 = [[1, 1, 1], [0.5, 0.5, 0.5], [1, 0.5, 0.5]]
    verts14 = [[1,1,1],[0.75,0.75,0.5],[1,0.5,0.5]]
    
    face11= Poly3DCollection([verts11],linewidth=2,alpha=0.33)
    face14= Poly3DCollection([verts14],linewidth=2,alpha=0.33)
        
    aleph = 0.5
    face11.set_facecolor((0,0.5,0.5,aleph))
    face12.set_facecolor((0,0.5,0.5,aleph))
    face13.set_facecolor((0,0.5,0.5,aleph))
    face14.set_facecolor((0,0.5,0.5,aleph))
    #face1.set_edgecolor((1,1,1,1))
    
    ax.add_collection3d(face11)
    ax.add_collection3d(face12)
    ax.add_collection3d(face13)
    ax.add_collection3d(face14)
    
    verts21 = [[1,1,0.5],[1,1,1],[0.75,0.75,0.5]]
    verts22 = [[1, 1, 1], [1,1, 0.5],[0.75,0.75,0.5]]
    verts23 = [[1, 1, 1], [1,1, 0.5], [1, 0.5, 0.5]]
    verts24 = [[1,1,1],[0.75,0.75,0.5],[1,0.5,0.5]]
    
    face21= Poly3DCollection([verts21],linewidth=2,alpha=0.1)
    face22= Poly3DCollection([verts22],linewidth=2,alpha=0.1)
    face23= Poly3DCollection([verts23],linewidth=2,alpha=0.1)
    face24= Poly3DCollection([verts24],linewidth=2,alpha=0.1)
        
    aleph = 0.01
    face21.set_facecolor((0.5,0,0,aleph))
    face22.set_facecolor((0.5,0,0,aleph))
    face23.set_facecolor((0.5,0,0,aleph))
    face24.set_facecolor((0.5,0,0,aleph))
    ax.set_facecolor('w')
    ax.add_collection3d(face21)
    ax.add_collection3d(face22)
    ax.add_collection3d(face23)
    ax.add_collection3d(face24)
    ax.zaxis.set_rotate_label(False) 
    ax.xaxis.set_rotate_label(False) 
    ax.yaxis.set_rotate_label(False) 
    ax.set_xlabel('$\lambda$ 1',rotation='horizontal',labelpad=20) 
    ax.set_ylabel('$\lambda$ 2',rotation='horizontal',labelpad=20)
    ax.set_zlabel('$\lambda$ 3',rotation='horizontal',labelpad=8)
    ax.set_xlim(0.5,1)
    ax.set_ylim(0.5,1)
    ax.set_zlim(0.5,1)
    
    plt.show()

else:
    sys.exit('Goodbye!')
