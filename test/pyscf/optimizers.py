import numpy as np


    


def gradient_descent(set_data,gamma,dx,initial=False):
    # simple gradient descent function, produces gradient
    # a.k.a., method of steepest descent
    new_runs = []
    # if initial, we are just generating 
    if initial==True:
        ref_point = set_data[0]
        num_para = len(ref_point)
        new_runs.append(ref_point)
        for i in range(0,num_para):
            new_para = ref_point.copy()
            new_para[i]=dx
            new_runs.append(new_para)
        gradient=[]
    else:
        #print(set_data)
        ref_par = np.asarray(set_data[0][0:-1])
        f_ref   = np.asarray(set_data[0][-1])
        npar = len(ref_par)
        set_par = np.zeros((npar,npar))
        set_f = np.zeros(npar)
        for i in range(0,npar):
            set_par[i][:] = set_data[i+1][0:-1]
            set_f[i] = set_data[i+1][-1]
        gradient = np.zeros(npar)
        for i in range(0,npar):
            ref_point = np.asarray(ref_par)
            df = set_f[i]-f_ref
            dx_i = set_par[i][:]-ref_par
            for j in range(0,npar):
                if dx_i[j]==0:
                    pass
                else:
                    gradient[j]+= df/dx_i[j]
        # calculating new point
        delta = np.sqrt(np.sum(np.square(gradient)))
        new_ref = ref_par - gamma*gradient
        new_runs.append(new_ref.tolist())
        for i in range(0,npar):
            new_para = new_ref.copy()
            new_para[i]+=dx
            new_runs.append(new_para.tolist())
        #print('Algorithm, ',new_runs)
    return new_runs, gradient

def f(x):
    return x**4 - 3*x**3 +2

'''
thresh = 1e-8
cur = 1
x = 6
gamma = 0.01
a = [[x,f(x)],[x+0.01,f(x+0.01)]]
while abs(cur)>thresh:
    b, grad = gradient_descent(a,gamma,step)
    cur = grad
    print(grad,b[0])
    a = [[b[0][0],f(b[0][0])],[b[1][0],f(b[1][0])]]

'''
