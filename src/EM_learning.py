#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:53:03 2022

@author: dama-f
"""

import numpy as np
import concurrent.futures
from real_covariates import Real_Covariates
from event_sequence_covariates import Event_Seq_Covariates

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                             "src")))
from gaussian_innovation import Gaussian_X
from scaled_modified_forward_backward_recursion import modified_FB


################################ BEGIN UTILS FUNCTIONS

## @fn compute_ll
#  @brief Another way to compute log-likelihood ll.
#   Likelihood of X_t can be computed as the weighted sum of the likehood within 
#   each regime:
#   P(X_t|past_values) = \sum_k P(X_t|past_values, Z_t=k) * P(Z_t=k).
#   Note that weights are regimes' probability at timestep t provided in
#   parameter list_Gamma.
#   In the end ll is equal to the sum of log( P(X_t|past_values) ) over all
#   timestep for all time series.  
#    
#  NB: Note that initial values are not included.
#    
def compute_ll(X_process, list_Gamma):
    
    log_ll = np.float64(0.0)
    
    for s in range(len(list_Gamma)):
        
        LL_s = X_process.total_likelihood_s(s)
        
        for t in range(list_Gamma[s].shape[0]):
            
            log_ll = log_ll + np.log(np.sum( LL_s[t, :] * \
                                            list_Gamma[s][t, :]))
            
    return log_ll


## @fn run_modified_FB_s
#  @brief Run modified_FB algorithm on the s^{th} observation sequence.
#
#  @return modified_FB outputs and sequence index which is used in main 
#   process to properly collect the results of different child processes.
#
def run_modified_FB_s(M, LL, B, Pi, s):
    
    #run modified backward-forward-backward on the s^th sequence
    return (s,  modified_FB(M, LL, B, Pi)  )
    

## @fn M_X_step M_Z_step
#  @brief Run the step M_.. of EM in which NH_HMC_process parameters are updated.
#
def M_X_step(X_process, list_Gamma):
   return X_process.update_parameters(list_Gamma)

def M_Z_step(NH_HMC_process, list_Xi, list_Gamma):
   return NH_HMC_process.update_parameters(list_Xi, list_Gamma)



## @fn compute_norm
#  @brief Compute the L1-norm of the difference between previous estimation and
#   current estumation of parameters.
#
def compute_norm(prev_estimated_param, NH_HMC_process, X_process):
    
    A = prev_estimated_param[1]
    Pi = prev_estimated_param[2]
    Y_params = prev_estimated_param[9]
    ar_coefficients = prev_estimated_param[5]
    ar_intercepts = prev_estimated_param[7]  
    Sigma = prev_estimated_param[6]
    
    (norm_diff, norm_prev) = NH_HMC_process.l1_norm_of_diff(Pi, A, Y_params)
    
    for k in range(X_process.nb_regime):
    
        norm_diff += np.sum(np.abs(X_process.intercept[k] - ar_intercepts[k]))
        norm_diff += np.sum(np.abs(X_process.sigma[k] - Sigma[k]))
        
        norm_prev += np.sum(np.abs(ar_intercepts[k]))
        norm_prev += np.sum(np.abs(Sigma[k]))
    
        for i in range(X_process.order):
            norm_diff += np.sum(np.abs(X_process.coefficients[k][i] - \
                                  ar_coefficients[k][i]))
            
            norm_prev += np.sum(np.abs(ar_coefficients[k][i]))
            
    return (norm_diff,  norm_diff/norm_prev)


################################ END UTILS FUNCTIONS
   
    

################################ BEGIN RANDOM INITIALIZATION OF EM

## @fn
#  @brief Initialize model parameters randomly then run nb_iters_per_init 
#   steps of EM and return the resulting estimation of parameters.
#
def run_EM_on_init_param(X_dim, X_order, nb_regimes, data, initial_values, \
                         innovation, var_init_method, nb_covariates, \
                         covariate_type, covariate_data, kappa_data, \
                         nb_iters_per_init):
    
    # re-seed the pseudo-random number generator.
    # This guarantees that each process/thread runs this function with
    # a different seed.
    np.random.seed()
    
    #----VAR process initialization  
    X_process = Gaussian_X(X_dim, X_order, nb_regimes, data, initial_values, \
                           var_init_method)              
            
    #----NH-HMC process initialization
    if(covariate_type == "real"):
        NH_HMC_process = Real_Covariates(nb_regimes, nb_covariates, \
                                         covariate_data)
        
    if(covariate_type == "event.sequences"):
        NH_HMC_process = Event_Seq_Covariates(nb_regimes, nb_covariates, \
                                              covariate_data, kappa_data) 
    
    #----run EM nb_iters_per_init times
    return EM(X_process, NH_HMC_process, nb_iters_per_init, epsilon=1e-6, \
              log_info=False)
 

## @fn
#  @brief EM initialization procedure.
#   Parameters are initialized nb_init times.
#   For each initialization, nb_iters_per_init of EM is executed.
#   Finally, the set of parameters that yield maximum likelihood is returned.
#
#  @param nb_init Number of initializations.
#  @param nb_iters_per_init Number of EM iterations.
#
#  @return The set of parameters having the highest likelihood value.
#   (ar_intercept, ar_coefficient, ar_sigma, Pi_, A_, Y_params).
#
def random_init_EM(X_dim, X_order, nb_regimes, data, initial_values, \
                   innovation, var_init_method, nb_covariates, covariate_type, \
                   covariate_data, kappa_data, nb_init, nb_iters_per_init):
        
    print("********************EM initialization begins**********************")        
    
    #------Runs EM nb_iters_per_init times on each initial values 
    output_params = []
    
    nb_workers = min(30, nb_init)
    
    #### Begin multi-process  parallel execution     
    with concurrent.futures.ProcessPoolExecutor(max_workers=nb_workers) as executor:
        futures = []
        for _ in range(nb_init):
            futures.append( executor.submit(run_EM_on_init_param, X_dim, \
                                            X_order, nb_regimes, data, \
                                            initial_values, innovation, \
                                            var_init_method, nb_covariates, \
                                            covariate_type, covariate_data, \
                                            kappa_data, nb_iters_per_init) )
             
        for f in concurrent.futures.as_completed(futures):
            print("Child process_X, I am done !") 
            try:
                output_params.append(f.result())
            except Exception as exc:
                print("Warning: EM initialization generates an exception: ", exc)                       
    #### End multi-proces parallel execution    
        
    #------Find parameters that yields maximum likehood
    #maximum likeliked within [-inf, 0]
    print("==================LIKELIHOODS====================")
    print("Number of EM initializations = ", nb_init)
    print("Number of SUCCESSFUL EM initializations = ", len(output_params))
    
    max_ll = -1e300  
    max_ind = -1
    nb_succ_init = len(output_params)
    
    for i in range(nb_succ_init):     
        print(output_params[i][0])
        
        if (output_params[i][0] > max_ll):
            max_ll = output_params[i][0]
            max_ind = i
        
    print("==================INITIAL VALUES OF PARAMETERS====================")
    print("------------------VAR process----------------------")
    print("total_log_ll= ", output_params[max_ind][0])
    print("ar_coefficients=", output_params[max_ind][5])
    print("intercept=", output_params[max_ind][7])
    print("sigma=", output_params[max_ind][6])
    print()
    print("------------------Markov chain----------------------")
    print("Pi=", output_params[max_ind][2])
    print("A=", output_params[max_ind][1])
    print("Y_params=", output_params[max_ind][9])
    
    print("*********************EM initialization ends***********************")
    
    return (output_params[max_ind][7], output_params[max_ind][5], \
            output_params[max_ind][6], output_params[max_ind][2], \
            output_params[max_ind][1], output_params[max_ind][9])
"""
((total_log_ll + X_process.init_val_ll()), NH_HMC_process.A, NH_HMC_process.Pi, \
list_Gamma, list_Alpha, X_process.coefficients, X_process.sigma, \
X_process.intercept, X_process.psi, NH_HMC_process.Y_params)
"""

################################ END RANDOM INITIALIZATION OF EM



################################ BEGIN  LEARNING 

## @fn
#  @brief Learn PHMC-LAR model parameters by EM algortihm.
#
#  @param X_order Autoregressive order, must be positive.
#  @param nb_regimes Number of switching regimes.
#  @param data List of length S, where S is the number of observed
#  time series. data[s], the s^th time series, is a T_s x dimension matrix
#  where T_s denotes its size starting at timestep t = order + 1 included.
#  @param initial_values List of length S. initial_values[s] is a column vector 
#   orderx1 of initial values associated with the s^th time series.
#  @param innovation Law of model error terms, only 'gaussian' noises are supported  
#
#  @param nb_covariates 
#  @param covariate_type Two possible values "real" of "event.sequences"
#  @param q Transition probabilities only depends of Y_{t-q} avec q <= X_order
#   if covariate_type == "real" and q=0 if covariate_type"== "event.sequences".
#  @param covariate_data Covariate Y_t data. List of length S.
#  @param kappa_data Observed time series sampling time. 
#   List of S T_s length arrays. 
#   Only used when covariate_type equals "event.sequences"
#
#  @param homo_msvar_params Initialize EM with homogeneous MSVAR parameters.
#   A dictionary having the following keys: "ar_inter", "ar_coef", "ar_sigma", 
#   "Pi" and "A". Parameters associated with covariates are all set at zeros.
#
#  @param nb_iters Maximum number of EM iterations.
#  @param epsilon Convergence precision. EM will stops when the shift 
#   in parameters' estimate between two consecutive iterations is less than
#   epsilon. L1-norm was used.
#  @param nb_init, nb_iters_per_init Number of random initialization of EM and
#   number of EM iterations per initialization. Only used when no initial
#   parameters are given, that is init_parameters==None
#  @param nb_iters_per_init
#
#  @return Parameters' estimation computed by EM algorithm.
#
def hmc_var_parameter_learning(X_dim, X_order, nb_regimes, data, \
                               initial_values, innovation, \
                               nb_covariates, covariate_type, \
                               covariate_data, kappa_data={}, \
                               homo_msvar_params=None, \
                               nb_iters=500, epsilon=1e-6, \
                               nb_init=10, nb_iters_per_init=5):
        
    if(innovation != "gaussian"):
        print()
        print("ERROR: file EM_learning.py: the given distribution is not supported!")
        exit (1)
        
    if(covariate_type != "real" and covariate_type != "event.sequences"):
        print()
        print("ERROR: file EM_learning.py: covariate_type must be either "
              "'real' or 'event.sequences'!")
        exit (1)

    if(covariate_type == "event.sequences" and nb_covariates != 0 and kappa_data == {}):
        print()
        print("ERROR: file EM_learning.py: when covariate_type equals "
              "'event.sequences', kappa_data must be provided!")
        exit (1)
        
    # check time series data and covariates dimension conformity
    S = len(data)
    OK = S == len(covariate_data)
    for s in range(S):
        OK = OK and (data[s].shape[0] == covariate_data[s].shape[0])
        
    if(not OK):
        print()
        print("ERROR: file EM_learning.py: time series data and covariate "
              "must have the same time-step length!\n")
        exit (1)
        
    # method used to initialize VAR process
    var_init_method = "rand2"
    
    # NH-HMC process initialization
    if(covariate_type == "real"):
        NH_HMC_process = Real_Covariates(nb_regimes, nb_covariates, \
                                         covariate_data)
    if(covariate_type == "event.sequences"):
        NH_HMC_process = Event_Seq_Covariates(nb_regimes, nb_covariates, \
                                              covariate_data, kappa_data) 
    # VAR process initialization 
    X_process = Gaussian_X(X_dim, X_order, nb_regimes, data, initial_values, \
                           var_init_method)
    
    #---------------EM initialization
    #----random initialization of EM: run EM nb_init times then take the set 
    # of parameters that yields maximum likelihood
    if(homo_msvar_params==None):
        (ar_inter, ar_coef, ar_sigma, Pi_, A_, Y_params) = \
                                random_init_EM(X_dim, X_order, nb_regimes, \
                                               data, initial_values, innovation, \
                                               var_init_method, \
                                               nb_covariates, covariate_type, \
                                               covariate_data, kappa_data, \
                                               nb_init, nb_iters_per_init)
        X_process.set_parameters(ar_inter, ar_coef, ar_sigma)
        NH_HMC_process.set_parameters(Pi_, A_, Y_params)
        
    else:
        #----initialize EM with init_parameters
        X_process.set_parameters(homo_msvar_params["ar_inter"], \
                    homo_msvar_params["ar_coef"], homo_msvar_params["ar_sigma"])
        # Y_params are set at zeros
        NH_HMC_process.set_at_homogeneous_HMC(homo_msvar_params["Pi"], \
                                              homo_msvar_params["A"])
    
    #---------------psi parameters estimation
    X_process.estimate_psi_MLE()
    
           
    #---------------run EM
    return EM(X_process, NH_HMC_process, nb_iters, epsilon)
    

  
## @fn
#  @brief
#
#  @param X_process Object of class RSVARM.
#  @param NH_HMC_process Object of class PHMC.
#  @param nb_iters
#  @param epsilon
#  @param log_info
#
#  @return Parameters' estimation computed by EM algorithm. 
#
def EM(X_process, NH_HMC_process, nb_iters, epsilon, log_info=True):
        
    #nb observed sequences
    S = len(X_process.data)
    #nb regimes
    M = X_process.nb_regime
    # training sequences' length
    train_seq_len = [X_process.data[s].shape[0] for s in range(S)]
      
    #total_log_ll is in [-inf, 0], can be also greater than 0
    prec_total_log_ll = -1e300  
    
    #current/previous estimated parameters
    prev_estimated_param = (-np.inf,  NH_HMC_process.A.copy(), \
                            NH_HMC_process.Pi.copy(), [], [], \
                            X_process.coefficients.copy(), \
                            X_process.sigma.copy(), X_process.intercept.copy(),  \
                            X_process.psi, NH_HMC_process.Y_params.copy())
    curr_estimated_param = {}
    
    #fix max_workers
    if(log_info):
        nb_workers=60 
    else:
        # initialization: 30*2 = 60 where 30 = max_workers_init  
        nb_workers=2
        
    #--------------------------------------------nb_iters of EM algorithm
    for ind in range(nb_iters):
    
        # initialization
        list_Gamma = [{} for _ in range(S)]
        list_Alpha = [{} for _ in range(S)]
        list_Xi = [{} for _ in range(S)]
        total_log_ll = np.float64(0.0)
        
        # compute non-stationary transition probabilities using parameters's
        # estimation at previous iterations. 
        # List of S arrays where list_mat_B[s] is a T_sxMxM matrix with   
        # list_mat_B[t, i, j] = P(Z_t=j | Z_{t-1}=i, t) 
        list_mat_B = NH_HMC_process.compute_transition_mat(train_seq_len)
        
        #----------------------------begin E-step  
        #### begin multi-process  parallel execution
        #list of tasks, one per sequence
        list_futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=nb_workers) as executor:
            #run the modified forward backward algorithm on each equence
            for s in range(S):             
                list_futures.append( executor.submit(run_modified_FB_s, M, \
                                    X_process.total_likelihood_s(s), \
                                    list_mat_B[s], NH_HMC_process.Pi, s) )

            #collect the results as tasks complete
            for f in concurrent.futures.as_completed(list_futures):
                #results of the s^{th} sequence
                (s, (log_ll_s, Xi_s, Gamma_s, Alpha_s)) = f.result()           
                #Xi, Gamma and Alpha probabilities computed on the s^th sequence
                list_Xi[s] = Xi_s
                list_Gamma[s] = Gamma_s
                list_Alpha[s] = Alpha_s          
                #total log-likelihood over all observed sequence
                total_log_ll = total_log_ll + log_ll_s  
        #### end multi-process  parallel execution
        #----------------------------end E-step
        
        #----------------------------begin M-steps
        #### begin multi-process parallel execution  
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor: 
            #-----------M-X substep 
            # Note that the child process runs on a copy of object X_process
            task_m_x = executor.submit(M_X_step, X_process, list_Gamma)
            
            #-----------M-Z substep 
            # Because the M_Z step is executed by the master process, Numba 
            # Parallel Accelerator can be used in parameter updating functions 
            # provided that these function are not executed by child processes
            # (instantiated by concurrent.futures for instance) 
            NH_HMC_process.update_parameters(list_Xi, list_Gamma)
        
            #collect result of task and update Theta_X
            (intercept, coefficients, sigma) = task_m_x.result()
            X_process.set_parameters(intercept, coefficients, sigma)           
        #### end multi-process parallel execution
        #----------------------------end M-steps 
        
        #-----------------------begin EM stopping condition        
        delta_log_ll = total_log_ll - prec_total_log_ll       
        log_ll_incr = delta_log_ll / np.abs(total_log_ll + prec_total_log_ll)
        
        (abs_of_diff, normaliz_abs_of_diff) = \
                compute_norm(prev_estimated_param, NH_HMC_process, X_process) 
        curr_estimated_param = (total_log_ll, NH_HMC_process.A.copy(), \
                                NH_HMC_process.Pi.copy(), list_Gamma, \
                                list_Alpha, X_process.coefficients.copy(), \
                                X_process.sigma.copy(), X_process.intercept.copy(),  \
                                X_process.psi, NH_HMC_process.Y_params.copy())
            
        if(np.isnan(abs_of_diff)):
            print("--------------EM stops with a warning------------")
            print("At iteration {}, NAN values encountered in the estimated ", \
                  "parameters".format(ind+1))
            print("PARAMETERS AT ITERATION {} ARE RETURNED".format(ind))            
    
            return prev_estimated_param
        
        #EM stops when log_ll decreases
        if(delta_log_ll < 0.0):
            print("--------------EM CONVERGENCE------------")
            print("delta_log_ll = {}, is negative".format(delta_log_ll))
            print("PARAMETERS AT ITERATION {} ARE RETURNED".format(ind)) 
            
            return prev_estimated_param
        
        #convergence criterion
        if(abs_of_diff < epsilon) or (delta_log_ll < epsilon):            
            #LOG-info
            if(log_info):
                print("--------------EM CONVERGENCE------------")
                print("#EM_converges after {} iterations".format(ind+1))
                print("log_ll = {}".format(total_log_ll))
                print("delta_log_ll = {}".format(delta_log_ll))
                print("log_ll_increase = {}".format(log_ll_incr))
                print("shift in parameters = {}".format(abs_of_diff))
                print("normalized shift in parameters = {}".format(normaliz_abs_of_diff))
            break       
        else:
            #LOG-info
            if(log_info):
                print("iterations = {}".format(ind+1))
                print("log_ll_alpha = {}".format(total_log_ll))
                print("delta_log_ll = {}".format(delta_log_ll))
                print("log_ll_increase = {}".format(log_ll_incr))
                print("shift in parameters = {}".format(abs_of_diff))
                print("normalized shift in parameters = {}".format(normaliz_abs_of_diff))
                                           
            #update prec_total_log_ll
            prec_total_log_ll = total_log_ll
        
            #update prev_estimated_param
            prev_estimated_param = curr_estimated_param
        #-----------------------end EM stopping condition  
             
    #another way to compute log_likelihood
    print("another way to compute log_likelihood = ", compute_ll(X_process, \
                                                                 list_Gamma) )
           
    return curr_estimated_param
      
################################ END LEARNING  



