# take a 
import matplotlib.pyplot as plt
from func.useful import gpc as gpc
from func.useful import mean_stdv
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import seaborn as sns
import sys
import math
import numpy as np
# 
np.set_printoptions(suppress=True,precision=5)
# load file
# 

def_in_loc = './compiled/'

try:
    input_name = def_in_loc + sys.argv[1]
    
except Exception:
    sys.exit('Goodbye!')

raw_data = np.loadtxt(input_name)

len_data = raw_data.shape[1]
n_data   = raw_data.shape[0]
if len_data == 18:
    titles = ['l0','l1','l2','l3','l4','l5','e0','e1','e2','t1','t2','t3','ord']
    data = np.zeros((raw_data.shape[0],13))
    orders = {'020121':0}
    for ind in range(0,raw_data.shape[0]):
        data[ind,0:12] = raw_data[ind,0:12]
        temp_str = ''.join(str(int(x)) for x in raw_data[ind,12:])
        if temp_str not in orders:
            orders[temp_str]=len(orders)   
        data[ind,12] = orders[temp_str]
elif len_data == 12:
    titles = ['l0','l1','l2','l3','l4','l5','e0','e1','e2','t1','t2','t3','ord']
    data = np.zeros((raw_data.shape[0],13))
    orders = {'020121':0}
    for ind in range(0,raw_data.shape[0]):
        data[ind,0:12] = raw_data[ind,0:12]
        data[ind,12] = 0

elif len_data == 16:
    titles = ['l0','l1','l2','l3','l4','l5','e0','e1','e2','t1','t2','t3','ord']
    orders = {'0212':0}
    data = np.zeros((raw_data.shape[0],13))
    for ind in range(0,raw_data.shape[0]):
        data[ind,0:12] = raw_data[ind,0:12]
        temp_str = ''.join(str(int(x)) for x in raw_data[ind,12:])
        if temp_str not in orders:
            orders[temp_str]=len(orders)   
        data[ind,12] = orders[temp_str]
elif len_data == 17:
    titles = ['l0','l1','l2','l3','l4','l5','e0','e1','e2','t1','t2','o0','o1','o2','o3','04','05']
    orders = {'020121':0}
elif len_data == 24:
    titles = ['l0','l1','l2','l3','l4','l5','e0','e1','e2','t1','t2','t3','ord','s1','s2','s3','s4','s5','s6','euclid']
    data = np.zeros((raw_data.shape[0],20))
    exp_dat = np.sort(raw_data[:,:6],axis=1)
    cal_dat = np.sort(raw_data[:,-6:],axis=1)

    exp_dat.sort()
    cal_dat.sort()
    for each in range(0,n_data):
        data[each,19] = np.sqrt(np.sum(np.square(exp_dat[each,3:]-cal_dat[each,3:])))
    orders = {'020121':0}
    for ind in range(0,raw_data.shape[0]):
        data[ind,0:12] = raw_data[ind,0:12]
        temp_str = ''.join(str(int(x)) for x in raw_data[ind,12:18])
        if temp_str not in orders:
            orders[temp_str]=len(orders)   
        data[ind,12] = orders[temp_str]
        data[ind,13:19] = raw_data[ind,18:]
else:
    sys.exit('Not a good file. Goodbye!')
rev_orders = {n:ord for ord, n in orders.items()}



dataset  = pd.DataFrame(data,index=None,columns=titles)
print('Here are the titles:')
print(titles)
filter = input('Would you like to filter any of the results?')
if filter=='yes' or filter=='Yes' or filter=='Y' or filter=='y':
    stop=False
    while stop==False:
        filter = input('Filter: ')
        if filter=='stop' or filter=='end':
            stop=True
        try:
            dataset = dataset.query(filter)
        except Exception as e:
            print('Some sort of error:')
            print(e)        
            print('Syntax is: a<b, a==b, a>b, etc.')
else:
    pass

print('Here are the analysis choices:')

graph_list = ['pairplot','histograms','gpc','gpc_error']
print(graph_list)
graph = input('Please choose a graphing method: ')

#
# Graphing Methods:
# ____pairplot
# ____gpc
# ____histograms 
#
#




if graph =='pairplot': 
    print('Set x variables:')
    print(titles)
    print('\'end\' or \'stop\' to finish.')
    stop=False
    X_VAR = []
    while stop==False:
        var = input('X variable: ')
        if var=='stop' or var=='end':
            stop=True
        else:
            X_VAR.append(var) 
    
    print('Set y variables:')
    stop=False
    Y_VAR = []
    while stop==False:
        var = input('Y variable: ')
        if var=='stop' or var=='end':
            stop=True
        else:
            Y_VAR.append(var) 
    
    input_hue = input('Specify hue, or \'n/no\'):   ')
    if input_hue=='n' or input_hue=='no' or input_hue=='No' or input_hue=='N':
        sns.pairplot(dataset,x_vars=X_VAR,y_vars=Y_VAR)
    else:
        found = False
        while not found: 
            try:
                sns.pairplot(dataset,x_vars=X_VAR,y_vars=Y_VAR,hue=input_hue,palette='hls')   
                found = True
            except Exception as e:
                print(e)
                input_hue = input('Try another item:  ')
    plt.show()

elif graph == 'histograms': 
    #begin histograms
    if len(titles)>13:
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


elif graph == 'gpc_error':
    fig = plt.figure() #set up matplotlib figure object 
    ax = fig.add_subplot(111,projection='3d')
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
    #face1.set_edgecolor((1,1,1,1))
    
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
elif graph == 'gpc':
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    datagpc = dataset.as_matrix()
    datagpc = datagpc[:,:6]
    datagpc = np.sort(datagpc,axis=1)
    print(datagpc)
    xs = datagpc[:,5]
    ys = datagpc[:,4]
    zs = datagpc[:,3]
    #colors = ['b' if gpc(hold[x,:])==1 else 'r' for x in Nlist]
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
        if (zm + 1) <= (ym + xm):
            col = 'r'
        else:
            col = 'b'
        ax.scatter(xm,ym,zm,c=col,edgecolors='k')
    
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
    #face1.set_edgecolor((1,1,1,1))
    
    ax.add_collection3d(face1)
    ax.add_collection3d(face2)
    ax.add_collection3d(face3)
    ax.add_collection3d(face4)
    
    #ax.add_collection3d(Poly3DCollection([verts1],facecolors=['g'],alpha=0.33))
    #ax.add_collection3d(Poly3DCollection([verts2],facecolors=['g'],alpha=0.33))
    #ax.add_collection3d(Poly3DCollection([verts3],facecolors=['g'],alpha=0.33))
    #ax.add_collection3d(Poly3DCollection([verts4],facecolors=['g'],alpha=0.33))
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

elif graph == 'gpc_mod':
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    datagpc = dataset.as_matrix()
    datagpc = datagpc[:,:6]
    datagpc = np.sort(datagpc,axis=1)
    print(datagpc)
    xs = datagpc[:,5]
    ys = datagpc[:,4]
    zs = datagpc[:,3]
    #colors = ['b' if gpc(hold[x,:])==1 else 'r' for x in Nlist]
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
        if (zm + 1) <= (ym + xm):
            col = 'r'
        else:
            col = 'b'
        ax.scatter(xm,ym,zm,c=col,edgecolors='k')
    
    #ax.scatter(xs,ys,zs,c=colors)
    
    #ax = Axes3D(fig)
    rt2 = 0.5 * np.sqrt(2)
    
    verts11 = [[0.5,0.5,0.5],[1,1,1],[0.75,0.75,0.5]]
    verts12 = [[1, 1, 1], [0.5,0.5, 0.5],[0.75,0.75,0.5]]
    verts13 = [[1, 1, 1], [0.5, 0.5, 0.5], [1, 0.5, 0.5]]
    verts14 = [[1,1,1],[0.75,0.75,0.5],[1,0.5,0.5]]
    
    face11= Poly3DCollection([verts11],linewidth=2,alpha=0.33)
    face12= Poly3DCollection([verts12],linewidth=2,alpha=0.33)
    face13= Poly3DCollection([verts13],linewidth=2,alpha=0.33)
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
    #face1.set_edgecolor((1,1,1,1))
    ax.set_facecolor('w')
    ax.add_collection3d(face21)
    ax.add_collection3d(face22)
    ax.add_collection3d(face23)
    ax.add_collection3d(face24)
    #ax.add_collection3d(Poly3DCollection([verts1],facecolors=['g'],alpha=0.33))
    #ax.add_collection3d(Poly3DCollection([verts2],facecolors=['g'],alpha=0.33))
    #ax.add_collection3d(Poly3DCollection([verts3],facecolors=['g'],alpha=0.33))
    #ax.add_collection3d(Poly3DCollection([verts4],facecolors=['g'],alpha=0.33))
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
# 
# start to title and arrange the user file
#
 
