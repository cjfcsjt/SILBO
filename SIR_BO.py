import numpy as np
import math
import functions
import GPy
import kernel_inputs
from scipy.linalg import eigh
import timeit
import test_SIR
import cma
import acquisition
from pyDOE import lhs


def ucb(mu, var, kappa=0.1):
    return mu + kappa * var


def Run_Main(low_dim=2, high_dim=20, initial_n=20, total_itr=100, func_type='Branin',
             matrix_type='simple', kern_inp_type='Y', A_input=None, s=None, active_var=None,
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
    elif func_type == 'MNIST':
        test_func = functions.MNIST(active_var)
    elif func_type == 'CAMEL':
        test_func = functions.camel3(active_var, noise_var=noise_var)
    else:
        TypeError('The input for func_type variable is invalid, which is', func_type)
        return

    best_results = np.zeros([1, total_itr + initial_n])
    elapsed = np.zeros([1, total_itr + initial_n])


    # generate embedding matrix via samples
    #f_s = test_func.evaluate(np.array(s))
    f_s_true = test_func.evaluate_true(s)
    B = SIR(low_dim,s,f_s_true,slice_number)



    embedding_sample = np.matmul(s,B)
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
        #CMA_ES
        # D = lhs(high_dim, 2000) * 2 * box_size - box_size
        # ac_value = AC_function(m,B,D)
        # solution = np.concatenate((D,ac_value),axis=1)
        #
        # for item in solution:
        #     solutions.append((item[:-1],item[-1]))
        # cma_iteration = 5
        # res = np.zeros((cma_iteration,high_dim))
        # keep = 0
        # for generation in range(cma_iteration):
        #     solutions = []
        #     for _ in range(cma_es.population_size):
        #         x = cma_es.ask()
        #         value = -AC_function(m, B, x).reshape(1)
        #         solutions.append((x, float(value)))
        #         #print("generation = ", generation,"value = ",value,"\n")
        #     cma_es.tell(solutions)
        #     a = 0
        #     for sol in solutions:
        #         if sol[1]<keep:
        #             keep = sol[1]
        #             a =np.array(sol[0]).reshape((1,high_dim))
        #     res[generation] = a[:]
        # maxx = res[cma_iteration-1]
        #*****
        #D = lhs(high_dim, 2000) * 2 * box_size - box_size
        #test = ac.acfunctionUCB(D)


        #*****
        es = cma.CMAEvolutionStrategy(high_dim * [0], 0.5, {'bounds': [-1, 1]})
        iter = 0
        while not es.stop() and iter !=2:
            iter+=1
            X = es.ask()
            es.tell(X, [ac.acfunctionUCB(x) for x in X])
            #es.disp()  # doctest: +ELLIPSIS
        #es.optimize(cma.ff.rosen)
        #es.optimize(acfunction.acfunction)
        maxx = es.result[0]

        s = np.matmul(maxx.T,B).reshape((1,low_dim)) #maxx:1000*1  B:1000*6
        embedding_sample = np.append(embedding_sample, s, axis=0)
        #f_s = np.append(f_s, test_func.evaluate(maxx), axis=0 )
        f_s_true = np.append(f_s_true, test_func.evaluate_true(maxx),axis=0)

        # Collecting data
        stop = timeit.default_timer()
        print("iter = ", i, "maxobj = ", np.max(f_s_true))
        best_results[0, i + initial_n] = np.max(f_s_true)
        elapsed[0, i + initial_n] = stop - start



    return best_results, elapsed, embedding_sample, f_s_true

def SIR(r,sample_x,sample_y,slice_number):
    nl = sample_x.shape[0]
    dim = sample_x.shape[1]
    xly = np.concatenate( (sample_x,sample_y) , axis=1)
    sort_xl = xly[np.argsort(xly[:,-1])][:,:-1]
    sample_mean = np.mean(sample_x,axis=0)

    sizeOfSlice = int(nl/slice_number)

    slice_mean = np.zeros((slice_number,dim))
    smean_c = np.zeros((slice_number,dim))
    #计算每一个slice mean
    W = np.zeros((nl, slice_number))
    for i in range(slice_number):
        if i ==slice_number -1:
            for j in range(i * sizeOfSlice, nl):
                W[j][i] = 1
        else :
            for j in range(i*sizeOfSlice,(i+1)*sizeOfSlice):
                W[j][i] = 1


    #解决下面的去generalized eigenvalue problem：Cov(HX)*V = lamda*Cov(X)*V
    cX = sample_x - np.tile(sample_mean, ((nl, 1)))
    Cov_X = np.matmul(cX.T,cX)
    WX = np.matmul(cX.T,W)
    Sigma_X = np.matmul(WX,WX.T)
    eigvals, eigvecs = eigh(Sigma_X, Cov_X + 0.01 * np.identity(dim), eigvals_only=False)
    B = eigvecs[:, dim - r:]
    return B

def distance(x1,x2,sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def getWbyKNN(X,k,numNNi=0):
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
    if numNNi!=0:
        for i in range(num):
            for j in range(num):
                if W[i][j]!=0:
                    W[i][j] = 1/numNNi
    return W

def AC_function(GP,B,X):
    X = X.reshape((-1,1))
    Y = np.matmul(X.T,B)
    mu,var = GP.predict(Y)
    ucb_d = ucb(mu,var).reshape(1)
    return ucb_d