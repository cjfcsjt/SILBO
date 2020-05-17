import os
import REMBO
import SIR_BO
import SS_BO
import SS_BO2
import count_sketch
import numpy as np
import pickle
import timeit
import sys
import json
from random import sample
from pyDOE import lhs

def REMBO_experiments(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100,
                      low_dim=2, high_dim=25, initial_n=20, opt_interval=20, ARD=False,
                      box_size=None, noise_var=0):
    if box_size is None:
        box_size=np.sqrt(low_dim)

    all_A = np.random.normal(0, 1, [stop_rep, low_dim, high_dim])
    all_s = np.empty((stop_rep, initial_n, low_dim))

    for i in range(stop_rep):
        all_s[i] = lhs(low_dim, initial_n) * 2 * box_size - box_size

    result_x_obj = np.empty((0, total_itr+initial_n))
    result_y_obj = np.empty((0, total_itr+initial_n))
    result_psi_obj = np.empty((0, total_itr+initial_n))

    elapsed_x = np.empty((0, total_itr + initial_n))
    elapsed_y = np.empty((0, total_itr + initial_n))
    elapsed_psi = np.empty((0, total_itr + initial_n))

    result_x_s = np.empty((0, initial_n + total_itr, low_dim))
    result_x_f_s = np.empty((0, initial_n + total_itr, 1))
    result_y_s = np.empty((0, initial_n + total_itr, low_dim))
    result_y_f_s = np.empty((0, initial_n + total_itr, 1))
    result_psi_s = np.empty((0, initial_n + total_itr, low_dim))
    result_psi_f_s = np.empty((0, initial_n + total_itr, 1))

    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        active_var = sample(range(high_dim), low_dim)

        # Running different algorithms to solve Hartmann6 function
        temp_result, temp_elapsed, temp_s, temp_f_s, _, _ = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                           total_itr=total_itr, func_type=test_func, A_input=all_A[i],
                                                                           s=all_s[i], kern_inp_type='Y', matrix_type='simple',
                                                                           hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                           noise_var=noise_var)
        result_y_obj = np.append(result_y_obj, temp_result, axis=0)
        elapsed_y = np.append(elapsed_y, temp_elapsed, axis=0)
        result_y_s = np.append(result_y_s, [temp_s], axis=0)
        result_y_f_s = np.append(result_y_f_s, [temp_f_s], axis=0)

        temp_result, temp_elapsed, temp_s, temp_f_s, _, _ = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                           total_itr=total_itr, func_type=test_func, A_input=all_A[i],
                                                                           s=all_s[i], kern_inp_type='X', matrix_type='simple',
                                                                           hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                           noise_var=noise_var)
        result_x_obj = np.append(result_x_obj, temp_result, axis=0)
        elapsed_x = np.append(elapsed_x, temp_elapsed, axis=0)
        result_x_s = np.append(result_x_s, [temp_s], axis=0)
        result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)

        temp_result, temp_elapsed, temp_s, temp_f_s, _, _ = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                           total_itr=total_itr, func_type=test_func, A_input=all_A[i],
                                                                           s=all_s[i], kern_inp_type='psi', matrix_type='simple',
                                                                           hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                           noise_var=noise_var)
        result_psi_obj = np.append(result_psi_obj, temp_result, axis=0)
        elapsed_psi = np.append(elapsed_psi, temp_elapsed, axis=0)
        result_psi_s = np.append(result_psi_s, [temp_s], axis=0)
        result_psi_f_s = np.append(result_psi_f_s, [temp_f_s], axis=0)

        stop = timeit.default_timer()

        print(i)
        print(stop - start)

    # Saving the results
    if test_func=='Rosenbrock':
        file_name = 'result/rosenbrock_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func=='Branin':
        file_name = 'result/branin_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'Hartmann6':
        file_name = 'result/hartmann6_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'StybTang':
        file_name = 'result/stybtang_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'WalkerSpeed':
        file_name = 'result/walkerspeed_results_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)
    elif test_func == 'MNIST':
        file_name = 'result/mnist_results_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)

    fileObject = open(file_name, 'wb')
    pickle.dump(result_y_obj, fileObject)
    pickle.dump(result_x_obj, fileObject)
    pickle.dump(result_psi_obj, fileObject)

    pickle.dump(elapsed_y, fileObject)
    pickle.dump(elapsed_x, fileObject)
    pickle.dump(elapsed_psi, fileObject)

    pickle.dump(result_y_s, fileObject)
    pickle.dump(result_x_s, fileObject)
    pickle.dump(result_psi_s, fileObject)

    pickle.dump(result_y_f_s, fileObject)
    pickle.dump(result_x_f_s, fileObject)
    pickle.dump(result_psi_f_s, fileObject)
    fileObject.close()

def SIR_BO_experiment(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100,
                      low_dim=2, high_dim=25, initial_n=20, kern_inp_type="Y" ,opt_interval=20, ARD=False,
                      box_size=1, noise_var=0):
    all_s = np.empty((stop_rep, initial_n, high_dim))

    #here all_s is sample from high dimension


    # result_x_obj = np.empty((0, total_itr + initial_n))
    # elapsed_x = np.empty((0, total_itr + initial_n))
    # result_x_s = np.empty((0, initial_n + total_itr, low_dim))
    # result_x_f_s = np.empty((0, initial_n + total_itr, 1))
    # result_high_s = np.empty((0, initial_n + total_itr, high_dim))
    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        active_var = sample(range(high_dim), low_dim)
        while (True):
            print("try again = ",i)
            for ss in range(stop_rep):
                all_s[ss] = lhs(high_dim, initial_n) * 2 * box_size - box_size
            try:
                temp_result, temp_elapsed, temp_s, temp_f_s = SIR_BO.Run_Main(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                                             total_itr=total_itr, func_type=test_func,
                                                                                             s=all_s[i], kern_inp_type=kern_inp_type, matrix_type='simple',
                                                                                             hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                                             noise_var=noise_var)
                # result_x_obj = np.append(result_x_obj, temp_result, axis=0)
                # elapsed_x = np.append(elapsed_x, temp_elapsed, axis=0)
                # #result_high_s = np.append(result_high_s, [temp_high_s], axis=0)
                # result_x_s = np.append(result_x_s, [temp_s], axis=0)
                # result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)


                stop = timeit.default_timer()

                print(i)
                print(stop - start)
                file_name = 0
                output_path = 0
                if test_func == 'Rosenbrock':
                    file_name = 'rosenbrock_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SIR_result/rosenbrock' + str(high_dim) + '/'
                elif test_func == 'Branin':
                    file_name = 'branin_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SIR_result/branin' + str(high_dim) + '/'
                elif test_func == 'Hartmann6':
                    file_name = 'hartmann6_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SIR_result/hartmann6' + str(high_dim) + '/'
                elif test_func == 'StybTang':
                    file_name = 'stybtang_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SIR_result/tang' + str(high_dim) + '/'
                elif test_func == 'Col':
                    file_name = 'col_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(
                        high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SIR_result/col' + str(high_dim) + '/'
                elif test_func == 'MNIST':
                    file_name = 'mnist_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SIR_result/mnist' + str(high_dim) + '/'

                if output_path is not None:
                    data = dict()
                    data["time"] = temp_elapsed.tolist()
                    data["objmaxvalue"] = temp_result.tolist()
                    # data["lows"] = result_x_s.tolist()
                    # data["highs"] = result_high_s.tolist()
                    # data["objvalue"] = result_x_f_s.tolist()

                    json.dump(data, open(os.path.join(output_path, "%s.json" % file_name), "w"))

                break
            except np.linalg.LinAlgError:
                print("except i = ",i)


def REMBO_separate(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100, low_dim=2,
                   high_dim=25, initial_n=20, opt_interval=20, ARD=False, box_size=None,
                   kern_inp_type='Y', noise_var=0):
    if box_size is None:
        box_size=np.sqrt(low_dim)

    all_A = np.random.normal(0, 1, [stop_rep, low_dim, high_dim])
    all_s = np.empty((stop_rep, initial_n, low_dim))



    result_x_obj = np.empty((0, total_itr+initial_n))
    elapsed_x = np.empty((0, total_itr + initial_n))
    result_x_s = np.empty((0, initial_n + total_itr, low_dim))
    result_x_f_s = np.empty((0, initial_n + total_itr, 1))
    result_high_s = np.empty((0, initial_n + total_itr, high_dim))
    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        active_var = sample(range(high_dim), low_dim)
        while (True):
            print("try again now =",i)
            for ss in range(stop_rep):
                all_s[ss] = lhs(low_dim, initial_n) * 2 * box_size - box_size
            try:
                # Running different algorithms to solve Hartmann6 function
                temp_result, temp_elapsed, temp_s, temp_f_s  = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                                             total_itr=total_itr, func_type=test_func, A_input=all_A[i],
                                                                                             s=all_s[i], kern_inp_type=kern_inp_type, matrix_type='simple',
                                                                                             hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                                             noise_var=noise_var)
                result_x_obj = np.append(result_x_obj, temp_result, axis=0)
                elapsed_x = np.append(elapsed_x, temp_elapsed, axis=0)
                # result_high_s = np.append(result_high_s, [temp_high_s], axis=0)
                # result_x_s = np.append(result_x_s, [temp_s], axis=0)
                # result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)


                stop = timeit.default_timer()

                print(i)
                print(stop - start)

                # Saving the results
                file_name = 0
                output_path = 0
                if test_func=='Rosenbrock':
                    file_name = 'rosenbrock_results_' + kern_inp_type + '_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './REMBO_result/rosenbrock' + str(high_dim) + '/'
                elif test_func=='Branin':
                    file_name = 'branin_results_' + kern_inp_type + '_d'+str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './REMBO_result/branin' + str(high_dim) + '/'
                elif test_func == 'Hartmann6':
                    file_name = 'hartmann6_results_' + kern_inp_type + '_d' + str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './REMBO_result/hartmann6' + str(high_dim) + '/'
                elif test_func == 'StybTang':
                    file_name = 'stybtang_results_' + kern_inp_type + '_d' + str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './REMBO_result/tang' + str(high_dim) + '/'
                elif test_func == 'Col':
                    file_name = 'col_results_' + kern_inp_type + '_d' + str(low_dim)+'_D'+str(high_dim)+'_n'+str(initial_n)+'_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './REMBO_result/col' + str(high_dim) + '/'
                elif test_func == 'MNIST':
                    file_name = 'mnist_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './REMBO_result/mnist' + str(high_dim) + '/'
                if output_path is not None:
                    data = dict()
                    data["time"] = elapsed_x.tolist()
                    data["objmaxvalue"] = result_x_obj.tolist()
                    # data["lows"] = result_x_s.tolist()
                    # data["highs"] = result_high_s.tolist()
                    # data["objvalue"] = result_x_f_s.tolist()

                    json.dump(data, open(os.path.join(output_path, "%s.json" % file_name), "w"))
                break
            except np.linalg.LinAlgError:
                print("except i =",i)


def xREMBO_separate(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100, low_dim=2,
                   high_dim=25, initial_n=20, opt_interval=20, ARD=False, box_size=None,
                   kern_inp_type='Y', noise_var=0):
    if box_size is None:
        box_size=np.sqrt(low_dim)

    all_A = np.random.normal(0, 1, [stop_rep, low_dim, high_dim])
    all_s = np.empty((stop_rep, initial_n, low_dim))



    result_x_obj = np.empty((0, total_itr+initial_n))
    elapsed_x = np.empty((0, total_itr + initial_n))
    result_x_s = np.empty((0, initial_n + total_itr, low_dim))
    result_x_f_s = np.empty((0, initial_n + total_itr, 1))
    result_high_s = np.empty((0, initial_n + total_itr, high_dim))
    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        active_var = sample(range(high_dim), low_dim)

        # Running different algorithms to solve Hartmann6 function
        while (True):
            print("try again now =",i)
            for ss in range(stop_rep):
                all_s[ss] = lhs(low_dim, initial_n) * 2 * box_size - box_size
            try:
                temp_result, temp_elapsed, temp_s, temp_f_s  = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                                         total_itr=total_itr, func_type=test_func, A_input=all_A[i],
                                                                                         s=all_s[i], kern_inp_type=kern_inp_type, matrix_type='simple',
                                                                                         hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                                         noise_var=noise_var)
                result_x_obj = np.append(result_x_obj, temp_result, axis=0)
                elapsed_x = np.append(elapsed_x, temp_elapsed, axis=0)
                # result_high_s = np.append(result_high_s, [temp_high_s], axis=0)
                # result_x_s = np.append(result_x_s, [temp_s], axis=0)
                # result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)

                stop = timeit.default_timer()

                print(i)
                print(stop - start)

                file_name = 0
                output_path = 0
                if test_func == 'Rosenbrock':
                    file_name = 'rosenbrock_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(
                        high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './xREMBO_result/rosenbrock' + str(high_dim) + '/'
                elif test_func == 'Branin':
                    file_name = 'branin_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './xREMBO_result/branin' + str(high_dim) + '/'
                elif test_func == 'Hartmann6':
                    file_name = 'hartmann6_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(
                        high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './xREMBO_result/hartmann6' + str(high_dim) + '/'
                elif test_func == 'StybTang':
                    file_name = 'stybtang_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(
                        high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './xREMBO_result/tang' + str(high_dim) + '/'
                elif test_func == 'Col':
                    file_name = 'col_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(
                        high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './xREMBO_result/col' + str(high_dim) + '/'
                elif test_func == 'MNIST':
                    file_name = 'mnist_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './xREMBO_result/mnist' + str(high_dim) + '/'

                if output_path is not None:
                    data = dict()
                    data["time"] = elapsed_x.tolist()
                    data["objmaxvalue"] = result_x_obj.tolist()
                    # data["lows"] = result_x_s.tolist()
                    # data["highs"] = result_high_s.tolist()
                    # data["objvalue"] = result_x_f_s.tolist()

                    json.dump(data, open(os.path.join(output_path, "%s.json" % file_name), "w"))

                break

            except np.linalg.LinAlgError:
                print("except i =", i)





def psiREMBO_separate(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100, low_dim=2,
                    high_dim=25, initial_n=20, opt_interval=20, ARD=False, box_size=None,
                    kern_inp_type='Y', noise_var=0):
    if box_size is None:
        box_size = np.sqrt(low_dim)

    all_A = np.random.normal(0, 1, [stop_rep, low_dim, high_dim])
    all_s = np.empty((stop_rep, initial_n, low_dim))



    result_x_obj = np.empty((0, total_itr + initial_n))
    elapsed_x = np.empty((0, total_itr + initial_n))
    # result_x_s = np.empty((0, initial_n + total_itr, low_dim))
    # result_x_f_s = np.empty((0, initial_n + total_itr, 1))
    # result_high_s = np.empty((0, initial_n + total_itr, high_dim))
    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        active_var = sample(range(high_dim), low_dim)

        # Running different algorithms to solve Hartmann6 function
        while(True):
            print("try again now =",i)
            for ss in range(stop_rep):
                all_s[ss] = lhs(low_dim, initial_n) * 2 * box_size - box_size
            try:
                temp_result, temp_elapsed, temp_s, temp_f_s = REMBO.RunRembo(low_dim=low_dim, high_dim=high_dim,
                                                                                             initial_n=initial_n,
                                                                                             total_itr=total_itr,
                                                                                             func_type=test_func,
                                                                                             A_input=all_A[i],
                                                                                             s=all_s[i],
                                                                                             kern_inp_type=kern_inp_type,
                                                                                             matrix_type='simple',
                                                                                             hyper_opt_interval=opt_interval,
                                                                                             ARD=ARD, box_size=box_size,
                                                                                             noise_var=noise_var)

                result_x_obj = np.append(result_x_obj, temp_result, axis=0)
                elapsed_x = np.append(elapsed_x, temp_elapsed, axis=0)
                # result_high_s = np.append(result_high_s, [temp_high_s], axis=0)
                # result_x_s = np.append(result_x_s, [temp_s], axis=0)
                # result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)

                stop = timeit.default_timer()

                print(i)
                print(stop - start)

                file_name = 0
                output_path = 0
                if test_func == 'Rosenbrock':
                    file_name = 'rosenbrock_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './psiREMBO_result/rosenbrock' + str(high_dim) + '/'
                elif test_func == 'Branin':
                    file_name = 'branin_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './psiREMBO_result/branin' + str(high_dim) + '/'
                elif test_func == 'Hartmann6':
                    file_name = 'hartmann6_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './psiREMBO_result/hartmann6' + str(high_dim) + '/'
                elif test_func == 'StybTang':
                    file_name = 'stybtang_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './psiREMBO_result/tang' + str(high_dim) + '/'
                elif test_func == 'Col':
                    file_name = 'col_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(
                        high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './psiREMBO_result/col' + str(high_dim) + '/'
                elif test_func == 'MNIST':
                    file_name = 'mnist_results_' + kern_inp_type + '_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './psiREMBO_result/mnist' + str(high_dim) + '/'


                if output_path is not None:
                    data = dict()
                    data["time"] = elapsed_x.tolist()
                    data["objmaxvalue"] = result_x_obj.tolist()
                    # data["lows"] = result_x_s.tolist()
                    # data["highs"] = result_high_s.tolist()
                    # data["objvalue"] = result_x_f_s.tolist()

                    json.dump(data, open(os.path.join(output_path, "%s.json" % file_name), "w"))
                break
            except np.linalg.LinAlgError:
                print("except i =",i)

def count_sketch_BO_experiments(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100,
                                low_dim=2, high_dim=25, initial_n=20, ARD=False, box_size=None,
                                noise_var=0):

    result_obj = np.empty((0, total_itr+initial_n))
    elapsed = np.empty((0, total_itr + initial_n))
    result_s = np.empty((0, initial_n + total_itr, low_dim))
    result_f_s = np.empty((0, initial_n + total_itr, 1))
    result_high_s = np.empty((0, initial_n + total_itr, high_dim))

    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        while (True):
            print("try again now =",i)
            try:
                temp_result, temp_elapsed, temp_s, temp_f_s,  temp_high_s = count_sketch.RunMain(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                                      total_itr=total_itr, func_type=test_func, s=None, ARD=ARD,
                                                                                      box_size=box_size, noise_var=noise_var)

                result_obj = np.append(result_obj, temp_result, axis=0)
                elapsed = np.append(elapsed, temp_elapsed, axis=0)
                # result_s = np.append(result_s, [temp_s], axis=0)
                # result_f_s = np.append(result_f_s, [temp_f_s], axis=0)
                # result_high_s = np.append(result_high_s, [temp_high_s], axis=0)

                stop = timeit.default_timer()

                print(i)
                print(stop - start)

                    # Saving the results for Hartmann6 in a pickle
                file_name = 0
                output_path = 0
                if test_func == 'Rosenbrock':
                    file_name = 'rosenbrock_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) +str(i)
                    output_path = './HeSBO_result/rosenbrock'+str(high_dim)+'/'
                elif test_func == 'Branin':
                    file_name = 'branin_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './HeSBO_result/branin' + str(high_dim) + '/'
                elif test_func == 'Hartmann6':
                    file_name = 'hartmann6_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './HeSBO_result/hartmann6' + str(high_dim) + '/'
                elif test_func == 'StybTang':
                    file_name = 'stybtang_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './HeSBO_result/tang' + str(high_dim) + '/'
                elif test_func == 'WalkerSpeed':
                    file_name = 'walkerspeed_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './HeSBO_result/branin' + str(high_dim) + '/'
                elif test_func == 'Col':
                    file_name = 'col_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './HeSBO_result/col' + str(high_dim) + '/'
                elif test_func == 'MNIST':
                    file_name = 'mnist_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './HeSBO_result/mnist' + str(high_dim) + '/'


                if output_path is not None:
                    data = dict()
                    data["time"] = elapsed.tolist()
                    data["objmaxvalue"] = result_obj.tolist()
                    # data["lows"] = result_s.tolist()
                    # data["highs"] = result_high_s.tolist()
                    # data["objvalue"] = result_f_s.tolist()

                    json.dump(data, open(os.path.join(output_path, "%s.json" % file_name), "w"))
                break

            except np.linalg.LinAlgError:
                print("except i =", i)

def SSIR_BO_experiment(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100,
                      low_dim=2, high_dim=25, initial_n=20, kern_inp_type="Y" ,opt_interval=20, ARD=False,
                      box_size=1, noise_var=0):
    xl = np.empty((stop_rep, initial_n, high_dim))
    xu = np.empty((stop_rep, initial_n, high_dim))

    for i in range(start_rep - 1, stop_rep):

        start = timeit.default_timer()
        active_var = sample(range(high_dim), low_dim)
        while(True):
            print("try again now = ",i)
            for ss in range(stop_rep):
                xl[ss] = lhs(high_dim, initial_n) * 2 * box_size - box_size
                xu[ss] = lhs(high_dim, initial_n) * 2 * box_size - box_size
            try:
                # Running different algorithms to solve Hartmann6 function
                temp_result, temp_elapsed, temp_s, temp_f_s = SS_BO.Run_Main(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                                             total_itr=total_itr, func_type=test_func,
                                                                                             xl=xl[i],xu=xu[i], kern_inp_type=kern_inp_type, matrix_type='simple',
                                                                                             hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                                             noise_var=noise_var)
                # result_x_obj = np.append(result_x_obj, temp_result, axis=0)
                # elapsed_x = np.append(elapsed_x, temp_elapsed, axis=0)
                # #result_high_s = np.append(result_high_s, [temp_high_s], axis=0)
                # result_x_s = np.append(result_x_s, [temp_s], axis=0)
                # result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)


                stop = timeit.default_timer()

                print(i)
                print(stop - start)

            # Saving the results for Hartmann6 in a pickle
                file_name = 0
                output_path = 0
                if test_func == 'Rosenbrock':
                    file_name = 'rosenbrock_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) +str(i)
                    output_path = './SSSIR_result/rosenbrock2'+str(high_dim)+'/'
                elif test_func == 'Branin':
                    file_name = 'branin_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './SSSIR_result/branin2' + str(high_dim) + '/'
                elif test_func == 'Hartmann6':
                    file_name = 'hartmann6_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './SSSIR_result/hartmann62' + str(high_dim) + '/'
                elif test_func == 'StybTang':
                    file_name = 'stybtang_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './SSIR_result/tang' + str(high_dim) + '/'
                elif test_func == 'Col':
                    file_name = 'col_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './SSSIR_result/col' + str(high_dim) + '/'
                elif test_func == 'MNIST':
                    file_name = 'mnist_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep)+str(i)
                    output_path = './SSSIR_result/mnist2' + str(high_dim) + '/'

                if output_path is not None:
                    data = dict()
                    data["time"] = temp_elapsed.tolist()
                    data["objmaxvalue"] = temp_result.tolist()
                    #data["lows"] = result_x_s.tolist()
                    #data["highs"] = result_high_s.tolist()
                    #data["objvalue"] = result_x_f_s.tolist()

                    json.dump(data, open(os.path.join(output_path, "%s.json" % file_name), "w"))
                break
            except np.linalg.LinAlgError:
                print("except i = ",i)

def SSIR_BO2_experiment(start_rep=1, stop_rep=50, test_func='Rosenbrock', total_itr=100,
                      low_dim=2, high_dim=25, initial_n=20, kern_inp_type="Y" ,opt_interval=20, ARD=False,
                      box_size=1, noise_var=0):
    xl = np.empty((stop_rep, initial_n, high_dim))
    xu = np.empty((stop_rep, initial_n, high_dim))

    #here all_s is sample from high dimension


    # result_x_obj = np.empty((0, total_itr + initial_n))
    # elapsed_x = np.empty((0, total_itr + initial_n))
    # result_x_s = np.empty((0, initial_n + total_itr, low_dim))
    # result_x_f_s = np.empty((0, initial_n + total_itr, 1))
    # result_high_s = np.empty((0, initial_n + total_itr, high_dim))
    for i in range(start_rep - 1, stop_rep):
        start = timeit.default_timer()
        active_var = sample(range(high_dim), low_dim)
        while(True):
            print("try again now = ",i)
            for ss in range(stop_rep):
                xl[ss] = lhs(high_dim, initial_n) * 2 * box_size - box_size
                xu[ss] = lhs(high_dim, initial_n) * 2 * box_size - box_size
            try:
                # Running SILBO-BU
                temp_result, temp_elapsed, temp_s, temp_f_s = SS_BO2.Run_Main(low_dim=low_dim, high_dim=high_dim, initial_n=initial_n,
                                                                                             total_itr=total_itr, func_type=test_func,
                                                                                             xl=xl[i],xu=xu[i], kern_inp_type=kern_inp_type, matrix_type='simple',
                                                                                             hyper_opt_interval=opt_interval, ARD=ARD, box_size=box_size,
                                                                                             noise_var=noise_var)
                # result_x_obj = np.append(result_x_obj, temp_result, axis=0)
                # elapsed_x = np.append(elapsed_x, temp_elapsed, axis=0)
                # result_high_s = np.append(result_high_s, [temp_high_s], axis=0)
                # result_x_s = np.append(result_x_s, [temp_s], axis=0)
                # result_x_f_s = np.append(result_x_f_s, [temp_f_s], axis=0)


                stop = timeit.default_timer()

                print(i)
                print(stop - start)

                # Saving the results for Hartmann6 in a pickle
                file_name = 0
                output_path = 0
                if test_func == 'Rosenbrock':
                    file_name = 'rosenbrock_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SSIR2_result/rosenbrockUCB' + str(high_dim) + '/'
                elif test_func == 'Branin':
                    file_name = 'branin_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SSIR2_result/braninUCB' + str(high_dim) + '/'
                elif test_func == 'Hartmann6':
                    file_name = 'hartmann6_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SSIR2_result/hartmann6UCB' + str(high_dim) + '/'
                elif test_func == 'StybTang':
                    file_name = 'stybtang_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SSIR2_result/tang' + str(high_dim) + '/'
                elif test_func == 'WalkerSpeed':
                    file_name = 'walkerspeed_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SSIR2_result/branin' + str(high_dim) + '/'
                elif test_func == 'Col':
                    file_name = 'col_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SSIR2_result/colUCB' + str(high_dim) + '/'
                elif test_func == 'MNIST':
                    file_name = 'mnist_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + str(i)
                    output_path = './SSIR2_result/mnistUCB' + str(high_dim) + '/'

                if output_path is not None:
                    data = dict()
                    data["time"] = temp_elapsed.tolist()
                    data["objmaxvalue"] = temp_result.tolist()
                    # data["lows"] = result_x_s.tolist()
                    # data["highs"] = result_high_s.tolist()
                    # data["objvalue"] = result_x_f_s.tolist()

                    json.dump(data, open(os.path.join(output_path, "%s.json" % file_name), "w"))
                break
            except np.linalg.LinAlgError:
                print("except i=",i)






if __name__=='__main__':
    RunName = ['SSIRBO']#,'SSIRBO2','xREMBO','psiREMBO'

    for st in RunName:
        start_rep = 1
        stop_rep = 100
        test_func = "Col" #MNIST Branin Hartmann6
        total_iter = 200
        low_dim = 4
        high_dim = 1000
        initial_n = 50
        variance = 0
        argv = st #ssir ssir2 sir
        #sys.argv[1] = "REMBO"

        if argv=="REMBO":
            REMBO_separate(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, kern_inp_type="Y", noise_var=variance)
        elif argv=="HeSBO":
            count_sketch_BO_experiments(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, box_size=1, noise_var=variance)
        elif argv=="SIRBO":
            SIR_BO_experiment(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter,
                              low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, box_size=1, noise_var=variance)
        elif argv=="SSIRBO2": #SILBO-BU
            SSIR_BO2_experiment(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, box_size=1, noise_var=variance)
        elif argv=="SSIRBO": #SILBO-TD
            SSIR_BO_experiment(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter, low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, box_size=1, noise_var=variance)

        elif argv=="xREMBO":#REMBO-KX
            xREMBO_separate(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter,
                           low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, kern_inp_type="X",
                           noise_var=variance)


        elif argv=="psiREMBO":#REMBO-Kpsi
            psiREMBO_separate(start_rep=start_rep, stop_rep=stop_rep, test_func=test_func, total_itr=total_iter,
                           low_dim=low_dim, high_dim=high_dim, initial_n=initial_n, ARD=True, kern_inp_type="psi",
                           noise_var=variance)

