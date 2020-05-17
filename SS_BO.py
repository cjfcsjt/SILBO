import numpy as np
import math
import test_SIR
import functions
import GPy
import kernel_inputs
from scipy.linalg import eigh
import timeit
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.linalg import block_diag
import cma
import acquisition
import ristretto.svd as svd

from scipy.optimize import minimize, rosen, rosen_der

def ucb(mu, var, kappa=0.1):
    return mu + kappa * var


def Run_Main(low_dim=2, high_dim=20, initial_n=20, total_itr=100, func_type='Branin',
             matrix_type='simple', kern_inp_type='Y', A_input=None, s=None,xl=None,xu=None, active_var=None,
             hyper_opt_interval=20, ARD=False, variance=1., length_scale=None, box_size=None,
             noise_var=0,slice_number=None):

    if slice_number is None:
        slice_number = low_dim+1
    if active_var is None:
        active_var= np.arange(high_dim)
    if box_size is None:
        box_size = math.sqrt(low_dim)
    if hyper_opt_interval is None:
        hyper_opt_interval = 10
    # Specifying the type of objective function
    if func_type == 'Branin':
        test_func = functions.Branin(active_var, noise_var=noise_var)
    elif func_type == 'Rosenbrock':
        test_func = functions.Rosenbrock(active_var, noise_var=noise_var)
    elif func_type == 'Hartmann6':
        test_func = functions.Hartmann6(active_var, noise_var=noise_var)
    elif func_type == 'Col':
        test_func = functions.colville(active_var, noise_var=noise_var)
    elif func_type == 'CAMEL':
        test_func = functions.camel3(active_var, noise_var=noise_var)
    elif func_type == 'MNIST':
        test_func = functions.MNIST(active_var)
    else:
        TypeError('The input for func_type variable is invalid, which is', func_type)
        return

    best_results = np.zeros([1, total_itr + initial_n])
    elapsed = np.zeros([1, total_itr + initial_n])


    # generate embedding matrix via samples
    #f_s = test_func.evaluate(np.array(xl))
    f_s_true = test_func.evaluate_true(xl)
    # get project matrix B using Semi-LSIR
    B = SSIR(low_dim, xl,f_s_true , xu, slice_number, k=3)

    embedding_sample = np.matmul(xl,B)
    for i in range(initial_n):
        best_results[0, i] = np.max(f_s_true[0:i + 1])
    for i in range(initial_n):
        best_results[0,i]=np.max(f_s_true[0:i+1])

    # Specifying the input type of kernel
    if kern_inp_type == 'Y':
        kern_inp = kernel_inputs.InputY(B)
        input_dim = low_dim
    elif kern_inp_type == 'X':
        kern_inp = kernel_inputs.InputX(B)
        input_dim = high_dim
    elif kern_inp_type == 'psi':
        kern_inp = kernel_inputs.InputPsi(B)
        input_dim = high_dim
    else:
        TypeError('The input for kern_inp_type variable is invalid, which is', kern_inp_type)
        return


    # Generating GP model
    k = GPy.kern.Matern52(input_dim=input_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(kern_inp.evaluate(embedding_sample), f_s_true, kernel=k)
    m.likelihood.variance = 1e-6
    bounds = np.zeros((high_dim,2))
    bounds[:,0]=-1
    bounds[:,1]=1
    ac = acquisition.ACfunction(B,m,initial_size=initial_n,low_dimension=low_dim)

    for i in range(total_itr):
        start = timeit.default_timer()
        #Updating GP model
        m.set_XY(kern_inp.evaluate(embedding_sample),f_s_true)
        m.optimize()


        #find X to max UCB(BX)
        es = cma.CMAEvolutionStrategy(high_dim * [0], 0.5, {'bounds': [-1, 1]})
        iter = 0
        u=[]
        ac.set_fs_true(max(f_s_true))
        #_, maxD = ac.acfunctionEI(max(f_s), low_dim)
        if i != 0 and (i) % 20 == 0:
            print("update")

            while not es.stop() and iter !=2:
                iter+=1
                X = es.ask()
                es.tell(X, [ac.newfunction(x) for x in X]) #set UCB or EI in newfunction() manually
                # if i != 0 and (i) % 10 == 0:
                u.append(es.result[0])
                 #es.disp()  # doctest: +ELLIPSIS
             #return candidate X
            maxx = es.result[0].reshape((1,high_dim))
        else:
            while not es.stop() and iter !=2:
                iter+=1
                X = es.ask()
                es.tell(X, [ac.newfunction(x) for x in X]) #set UCB or EI in newfunction() manually
               #es.disp()  # doctest: +ELLIPSIS
             #return candidate X
            maxx = es.result[0].reshape((1,high_dim))

        #_,maxD = ac.acfunctionEI(max(f_s),low_dim)

        #initial qp
        #ac.updateQP(maxD)
        #solve qp
        #res = minimize(ac.qp,np.zeros((high_dim,1)),method='SLSQP',bounds=bounds,options={'maxiter': 5, 'disp': True})

        #maxx = res.x
        #print("qp fun = ",res.fun)


        es = np.matmul(maxx,B) #maxx:1000*1  B:1000*6
        embedding_sample = np.append(embedding_sample, es, axis=0)
        xl = np.append(xl,maxx,axis=0)
        #f_s = np.append(f_s, test_func.evaluate(maxx), axis=0)
        f_s_true = np.append(f_s_true, test_func.evaluate_true(maxx), axis=0)

        #update project matrix B
        if i != 0 and (i) % 20 == 0:
            print("update")
            #get top "inital_n" from xl
            xlidex = np.argsort(-f_s_true, axis=0).reshape(-1)[:initial_n]
            f_s_special = f_s_true[xlidex]
            xl_special = xl[xlidex]
            #get top unlabeled data from xu
            xu = np.array(u)
            B = SSIR(low_dim, xl_special, f_s_special,xu, slice_number, k=3)
            embedding_sample = np.matmul(xl, B)
            ac.resetflag(B)


        # Collecting data
        stop = timeit.default_timer()
        print("iter = ", i, "maxobj = ", np.max(f_s_true))
        best_results[0, i + initial_n] = np.max(f_s_true)
        elapsed[0, i + initial_n] = stop - start



    return best_results, elapsed, embedding_sample, f_s_true

def SSIR(r,xl,y,xu,slice_number,k,alpha = 0.1):
    #xt = np.concatenate( (xl,xu) ,axis=0) #xtotal
    nl = xl.shape[0] #label_sample_size
    nu = xu.shape[0] #unlabel_sample_size
    dim = xl.shape[1] #dimension
    xly = np.concatenate( (xl,y) , axis=1)
    sort_xl = xly[np.argsort(xly[:,-1])][:,:-1] #根据Y值进行排序
    xt = np.concatenate( (sort_xl,xu) ,axis=0)
    xt_mean = np.mean(xt,axis=0)

    sizeOfSlice = int(nl/slice_number)

    slice_mean = np.zeros((slice_number,dim))
    smean_c = np.zeros((slice_number,dim))
    W = []
    #计算每一个slice mean
    for i in range(slice_number):
        if i ==slice_number -1:
            temp = sort_xl[i*sizeOfSlice:,:]
        else :
            temp = sort_xl[i*sizeOfSlice:(i+1)*sizeOfSlice,:]
        ni = temp.shape[0] #?
        numNNi = min(ni,k)
        W.append(getWbyKNN(temp,numNNi,numNNi))
    W = block_diag(*W)
    zero = np.zeros((nu,nu))
    W = block_diag(W,zero)
    #fast svd
    cX = xt - np.tile(xt_mean, ((nl + nu), 1))
    #R = np.matmul(W,cX)
    Ra = np.matmul(cX.T,W)   #
    #U,S,Vh = svd.compute_rsvd(R,6)
    U1,S1,V1h = svd.compute_rsvd(Ra,r) #


    W = getWbyKNN(xt,k,1)
    L = laplacian_matrix(W)
    Il = np.identity(nl)
    I = block_diag(Il,zero)
    M = I + alpha* L+0.01*np.identity((nl+nu))

    L = np.linalg.cholesky(M) #
    u,s,vh = np.linalg.svd(M)
    ss = np.diag(np.sqrt(s))
    J = np.matmul(ss,vh)
    # J*cX*U*S^-1

    aa = np.matmul(np.diag(1.0/S1),U1.T)#
    bb = np.matmul(cX.T, L)#
    cc = np.matmul(aa,bb)#

    u2,s2,vh2 = np.linalg.svd(cc) #
    eii  = np.matmul( np.matmul(U1,np.diag(1.0/S1)),u2) #

    return eii
def distance(x1,x2,sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def getWbyKNN(X,k,a=0):
    num = X.shape[0]
    S = np.zeros((num,num))
    W = np.zeros((num,num))
    for i in range(num):
        for j in range(i+1, num):
            S[i][j] = 1.0 * distance(X[i],X[j])
            S[j][i] = S[i][j]
    for i in range(num):
        index_array = np.argsort(S[i])
        W[i][index_array[1:k+1]] = 1
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2
    if a!=0:
        for i in range(num):
            for j in range(num):
                if W[i][j]!=0:
                    W[i][j] = 1/a
    return W

def laplacian_matrix(M,normalize= False):
    #compute the D = sum(A)
    D = np.sum(M,axis=1)
    #compute the laplacian matrix: L=D-A
    L = np.diag(D)-M
    #normalize
    if normalize:
        sqrtD = np.diag(1.0/(D**(0.5)))
        return np.dot(np.dot(sqrtD,L),sqrtD)
    return L




