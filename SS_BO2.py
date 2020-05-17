import numpy as np
import math
import functions
import GPy
import kernel_inputs
import projections
from pyDOE import lhs
from scipy.stats import norm
from scipy.linalg import block_diag
from scipy.linalg import eigh
from scipy.linalg import pinv
import timeit
import ristretto.svd as svd
import acquisition

def EI(D_size,f_max,mu,var):
    """
    :param D_size: number of points for which EI function will be calculated
    :param f_max: the best value found for the test function so far
    :param mu: a vector of predicted values for mean of the test function
        corresponding to the points
    :param var: a vector of predicted values for variance of the test function
        corresponding to the points
    :return: a vector of EI values of the points
    """
    ei=np.zeros((D_size,1))
    std_dev=np.sqrt(var)
    for i in range(D_size):
        if var[i]!=0:
            z= (mu[i] - f_max) / std_dev[i]
            ei[i]= (mu[i]-f_max) * norm.cdf(z) + std_dev[i] * norm.pdf(z)
    return ei



def Run_Main(low_dim=2, high_dim=20, initial_n=20, total_itr=100, func_type='Branin',
             matrix_type='simple', kern_inp_type='Y', A_input=None, s=None,xl=None,xu=None, active_var=None,
             hyper_opt_interval=10, ARD=False, variance=1., length_scale=None, box_size=None,
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
    elif func_type == 'StybTang':
        test_func = functions.StybTang(active_var, noise_var=noise_var)
    elif func_type == 'Col':
        test_func = functions.colville(active_var, noise_var=noise_var)
    elif func_type == 'MNIST':
        test_func = functions.MNIST(active_var)
    else:
        TypeError('The input for func_type variable is invalid, which is', func_type)
        return

    best_results = np.zeros([1, total_itr + initial_n])
    elapsed = np.zeros([1, total_itr + initial_n])
    total_best_results =  np.zeros([1, total_itr + initial_n])

    # generate embedding matrix via samples
    #f_s = test_func.evaluate(xl)
    f_s_true = test_func.evaluate_true(xl)
    B = SSIR(low_dim, xl,f_s_true , xu, slice_number, k=3)
    Bplus = pinv(B).T #6*100 with T ; otherwise, 100*6
    cnv_prj = projections.ConvexProjection(Bplus)
    #embedding_sample = np.matmul(xl,B.T)
    box = np.sum(B, axis=1)
    print(box)
    #box_bound = np.empty((2, low_dim))
    # for i in range(low_dim):
    #     for j in range(2):
    #         if j == 0:
    #             box_bound[j][i] = -np.abs(box[i])
    #         else:
    #             box_bound[j][i] = np.abs(box[i])

    # Initiating first sample
    if s is None:
        #s = lhs(low_dim, initial_n) * 2 * box_size - box_size
        # D = []
        # for i in range(low_dim):
        #     D.append(lhs(1, initial_n) * 2 * np.abs(box[i]) - np.abs(box[i]))
        s = lhs(low_dim, 2000) * 2 * np.max(np.abs(box)) - np.max(np.abs(box))
        #s = np.array(D).reshape((initial_n,low_dim))

    # get low-dimensional representations
    s = np.matmul(xl,B.T)

    for i in range(initial_n):
        best_results[0, i] = np.max(f_s_true[0:i + 1])
    for i in range(initial_n):
        best_results[0,i]=np.max(f_s_true[0:i+1])

    # Specifying the input type of kernel
    kern_inp, input_dim = specifyKernel("Y",Bplus=Bplus,low_dim=low_dim,high_dim=high_dim)


    # Generating GP model
    k = GPy.kern.Matern52(input_dim=input_dim, ARD=ARD, variance=variance, lengthscale=length_scale)
    m = GPy.models.GPRegression(kern_inp.evaluate(s), f_s_true, kernel=k)
    m.likelihood.variance = 1e-6
    ac = acquisition.ACfunction(B, m, initial_size=initial_n, low_dimension=low_dim)
    # Main loop of the algorithm
    ei_d = 0
    D=0
    for i in range(total_itr):
        print("i = ", i)
        start = timeit.default_timer()
        #update project matrix every 20 iterations
        if i!=0 and (i)%20 == 0:
            print("update")
            idx = np.argsort(np.array(-ei_d),axis=0).reshape(-1)[:100]#get 100 unlabeled data index
            xu = cnv_prj.evaluate(D[idx]) #project the unlabeled data to high-dimensional space
            xlidex = np.argsort(-f_s_true,axis=0).reshape(-1)[:initial_n] # get 'inital_n' labeled data index
            xl_special = cnv_prj.evaluate(s[xlidex]) #project the labeled data to high-dimensional space
            f_s_special = f_s_true[xlidex] # evaluate the labeled data to get response value
            B = SSIR(low_dim, xl_special, f_s_special,xu, slice_number, k=3) # perform SEMI-LSIR to update B
            Bplus = pinv(B).T
            specifyKernel("Y",B,low_dim,high_dim)
            cnv_prj = projections.ConvexProjection(Bplus)
            box = np.sum(B, axis=1) # update low-dimensional search space
            #f_s = test_func.evaluate(cnv_prj.evaluate(s))
            f_s_true = test_func.evaluate_true(cnv_prj.evaluate(s))
            print(box)


        # Updating GP model
        m.set_XY(kern_inp.evaluate(s), f_s_true)
        #if (i + initial_n <= 25 and i % 5 == 0) or (i + initial_n > 25 and i % hyper_opt_interval == 0):
        m.optimize()

        # finding the next point for sampling
        # D = []
        # for a in range(low_dim):
        #     D.append(lhs(1, 2000) * 2 * np.abs(box[a])  - np.abs(box[a]))
        # D = np.array(D).reshape((2000, low_dim))
        D = lhs(low_dim, 2000) * 2 * np.max(np.abs(box)) - np.max(np.abs(box))
        #D = lhs(low_dim, 2000) * 2 * box_size - box_size
        #test = kern_inp.evaluate(D)

        mu, var = m.predict(kern_inp.evaluate(D))
        #UCB
        ei_d = ac.originalUCB(mu,var)
        #EI
        #ei_d = EI(len(D), max(f_s_true), mu, var)
        index = np.argmax(ei_d)


        #xl = np.append(xl,cnv_prj.evaluate([D[index]]),axis=0)
        s = np.append(s, [D[index]], axis=0)
        #f_s = np.append(f_s, test_func.evaluate(cnv_prj.evaluate([D[index]])), axis=0)
        f_s_true = np.append(f_s_true, test_func.evaluate_true(cnv_prj.evaluate([D[index]])), axis=0)

        # Collecting data
        stop = timeit.default_timer()
        print("iter = ", i, "maxobj = ", np.max(f_s_true))
        best_results[0, i + initial_n] = np.max(f_s_true)
        elapsed[0, i + initial_n] = stop - start
    for i in range(initial_n + total_itr):
        total_best_results[0, i] = np.max(best_results[0,:i+1])


    # if func_type == 'WalkerSpeed':
    #     eng.quit()

    return total_best_results, elapsed, s, f_s_true #cnv_prj.evaluate(s)


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
    #compute slice mean
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
    #a = np.matmul(J,cX)
    #b = np.matmul(Vh.T,np.diag(1.0/S))
    #c = np.matmul(a,b)
    #u1,s1,vh1 = np.linalg.svd(c)
    u2,s2,vh2 = np.linalg.svd(cc) #
    eii  = np.matmul( np.matmul(U1,np.diag(1.0/S1)),u2) #
    #Cov = np.matmul(np.matmul(cX,M),cX.T)
    #eigvecs = np.matmul(np.matmul(Vh.T,np.diag(1.0/S)),vh1.T)

    #eigvals, eigvecs = eigh(Sigma, b = Cov)
    #embedding_matrix = eigvecs[:,dim-4:] # dim*4 matrix
    return eii.T
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

def specifyKernel(kern_inp_type="X",Bplus=None,low_dim=None,high_dim=None):
    if kern_inp_type == 'Y':
        kern_inp = kernel_inputs.InputY(Bplus)
        input_dim = low_dim
        return kern_inp, input_dim
    elif kern_inp_type == 'X':
        kern_inp = kernel_inputs.InputX(Bplus)
        input_dim = high_dim
        return kern_inp, input_dim
    elif kern_inp_type == 'psi':
        kern_inp = kernel_inputs.InputPsi(Bplus)
        input_dim = high_dim
        return kern_inp,input_dim
    else:
        TypeError('The input for kern_inp_type variable is invalid, which is', kern_inp_type)
        return