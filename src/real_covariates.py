#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:24:39 2022

@author: dama-f
"""

import numpy as np
import concurrent.futures
from scipy import optimize
from scipy.stats import dirichlet
from numba import jit

from non_homogeneous_HMC import NHHMC


############################################################################
## @class Real_Covariate
#
class Real_Covariates(NHHMC):

    ## @fn
    #  @brief In this sub-class parameters associated with covariates denoted
    #  by Y_params is a nb_states(i) x nb_states(j) x nb_covariates(l) matrix.
    #
    #  @param nb_states
    #  @param nb_covariates
    #  @param covariate_data Real valued covariates.
    #   List of S T_s x nb_covariates matrices.
    #
    def __init__(self, nb_states, nb_covariates, covariate_data):
        
        #---assertions
        assert(nb_states > 0)
        assert(nb_covariates >= 0)
        
        if(nb_covariates != 0 and covariate_data[0].shape[1] != nb_covariates):
            print()
            print("ERROR: class Real_Covariates: nb_covariates must be equal ", \
                  "to the number of column within covariate_data!\n")
            exit(1)
        
        # hyper-parameters setting
        self.covariate_type = "real"
        self.nb_covariates = nb_covariates
        self.nb_states = nb_states
        self.covariate_data = covariate_data
        
        #----random initialization of Pi and A
        concentration_param = [1 for i in range(nb_states)]
        self.Pi = np.zeros(dtype=np.float64, shape=(1, nb_states))
        self.Pi[0, :] = dirichlet.rvs(concentration_param, 1)[0]
            
        self.A = np.zeros(dtype=np.float64, shape=(nb_states, nb_states))
        for i in range(nb_states):
            self.A[i, :] = dirichlet.rvs(concentration_param, 1)[0]
            
        #----random initialization of Y_params
        self.Y_params = np.zeros(dtype=np.float64, \
                                 shape=(nb_states, nb_states, nb_covariates)) 
        # Y_params[i, j, l] are uniformly drawn in [-1, 1] under contraints
        # sum_l Y_params[i, j, l] = 0  for all (i, j) fixed
        for i in range(nb_states):
            for j in range(nb_states):
                self.Y_params[i, j, :] = np.random.uniform(-1, 1, nb_covariates)
                
                self.Y_params[i, j, 0] = -np.sum(self.Y_params[i, j, 1:])
                
        return
    
    
    ## @fn
    #  @brief
    #
    def set_at_homogeneous_HMC(self, Pi, A):
        
        self.Pi = Pi
        self.A = A
        
        # set Y_params at zeros
        self.Y_params = np.zeros(dtype=np.float64, shape=(self.nb_states, \
                                                          self.nb_states, \
                                                          self.nb_covariates))
        
        return
    
       
    ## @fn update_parameters
    #  @brief Computes the step M-S of EM.
    # 
    #  @param list_Xi
    #  @param list_Gamma
    # 
    #  @return The new estimated parameters
    #
    def update_parameters(self, list_Xi, list_Gamma):
        
        #---update initial law Pi
        self.update_Pi(list_Gamma)
        
        #---Homogeneous HMM case
        if(self.nb_covariates == 0):
            self.update_homogeneous_HMM(list_Xi, list_Gamma)
        
        else:
            #---update transition probability parameters: A and Y_params
            # for each state i, estimate parameters related to transitions 
            # i-->j in parallel over i
        
            #### BEGIN multi-process parallel execution 
            futures = []        
    
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.nb_states) as executor:               
                #lauch tasks
                for i in range(self.nb_states):
                    futures.append( executor.submit(call_update_params_trans_from_i, \
                                                    self, list_Xi, i) )      
                
                #collect the results as tasks complete
                for f in concurrent.futures.as_completed(futures):
                    #update regime k parameters
                    (A_i, Y_params_i, i) = f.result() 
                    self.A[i, :] = A_i
                    self.Y_params[i, :, :] = Y_params_i                 
            #### End multi-process parallel execution    
                                
        return (self.Pi, self.A, self.Y_params)
    
    
    ## @fn update_params_trans_from_i
    #  @brief
    #
    #  @param
    #
    #  @return 
    #   * A_i_*: nb_regimes-length array
    #   * Y_params[i,*,*]: nb_regimes x nb_covariate matrix
    #
    def update_params_trans_from_i(self, list_Xi, i):
    
        # local variables
        K = self.nb_states
        nb_covariates = self.nb_covariates
        nb_params = K + K*nb_covariates
        
        #---Initial values of parameters: equal to the current estimate
        init_parameters = np.zeros(dtype=np.float64, shape=nb_params)
        init_parameters[0:K] = self.A[i, :]
        # flatten Y_params[i, *(j), *(l)] by row 
        init_parameters[K:] = self.Y_params[i, :, :].flatten(order='C')
    
        #---Parameter bounds: they are searched within [lower_b, upper_b]
        # Y_params[i, :, :]'s are in R
        lower_b = np.repeat(-np.inf, nb_params)
        upper_b = np.repeat(+np.inf, nb_params) 
        # A_i's are in  [0, 1]
        lower_b[0:K] = np.repeat(1e-100, K)
        upper_b[0:K] = np.repeat(1.0001, K)
        
        # build Bound object
        bounds = optimize.Bounds(lb=lower_b, ub=upper_b, keep_feasible=True) 
        
        #---Constraints
        # Ours constraints are formulated as follows: 
        # sum_l x_l = C  <==> C <= (1, ..., 1) x (x_1, ..., x_L)' <= C
        # We have (nb_covariates + 1) constraints.
        # For each constraints we define the parameters involved and C value
        
        # list of nb_constr element: each entry correspond to one constraint
        c_values = []
        # list of nb_params-length array: each array correspond to one constraint
        constr_coefs = []
        
        # first constraint: sum_j A_i_j = 1
        tmp = np.zeros(dtype=np.float64, shape=nb_params)
        tmp[0:K] = np.repeat(1, K)
        constr_coefs.append(tmp)
        c_values.append(1)

        # K constraints left
        for j in range(K):
            # sum_l Y_params[i,j,l] = 0
            y_param_i_ = np.zeros(dtype=np.float64, shape=(K, nb_covariates))
            y_param_i_[j, :] = np.repeat(1, nb_covariates)
            tmp = np.zeros(dtype=np.float64, shape=nb_params)
            tmp[K:] = y_param_i_.flatten(order='C')
            constr_coefs.append(tmp)            
            c_values.append(0)

        # build LinearConstraint object
        c_values = np.array(c_values)
        constr_coefs = np.array(constr_coefs)
        linear_constraints = \
                optimize.LinearConstraint(constr_coefs, lb=c_values, \
                                          ub=c_values, keep_feasible=True)
        
        #---Numerical optimization
        # * jac: either callable or one of the supported finite-difference
        #   approximation schemes. If callable (grad_minus_Q_S_i) it is called 
        #   with the same arguments as the objective function.
        #   NB: When a finite-difference scheme is used, bound constraints can
        #   be violated during gradient approximation (for instance when 
        #   it is computed at the limiting values). This raises
        #   """ValueError: `x0` violates bound constraints""".
        #
        # * hess: Etheir quasi-Newton hessian approximation strategies 
        #   (optimize.BFGS(), optimize.SR1()) or finite-differrence approximation
        #   NB1: quasi-Newton approximation strategies sometimes raise  
        #    """UserWarning: delta_grad == 0.0. Check if the approximated 
        #       function is linear. If the function is linear better results 
        #       can be obtained by defining the Hessian as zero instead of 
        #       using quasi-Newton approximations"""
        #   Obviously, my objective function is non-linear (w.r.t. each of
        #   its parameters). It looks like that the algorithm might be
        #   mislead by the fact that my objective can be very flat and therfore
        #   it behaves as if my objective was linear. 
        #   Because quasi-Newton hessian approximation is too bad when dealing
        #   with linear objective functions, It would be better to use 
        #   finite-differrence approximation. I used the 2-point scheme.
        #  SOURCE: https://github.com/scipy/scipy/issues/8644
        #  
        res = optimize.minimize(fun=minus_Q_S_i, x0=init_parameters, \
                                args=(i, list_Xi, self.covariate_data), \
                                method="trust-constr", \
                                jac=grad_minus_Q_S_i, hess='2-point', \
                                constraints=linear_constraints, bounds=bounds,  \
                                options={'verbose': 0, 'maxiter': 1000})
        
        # Warning
        if(not res.success):
            print("Warning: class real_covariates: numerical optimization", \
                  "did not converge while updating probabilities of transition", \
                  " from state {}. Failure message: {}.".format(i, res.message))
            
        #---Output building 
        estimated_A_i_ = res.x[0:K]
        estimated_Y_param_i = np.reshape(res.x[K:], newshape=(K, nb_covariates), \
                                         order='C')
        
        return (estimated_A_i_, estimated_Y_param_i, i)
    
    
    ## @fn compute_norm
    #  @brief Compute the L1-norm of the difference between self and the given
    #  NH_HMC 
    #
    def l1_norm_of_diff(self, Pi, A, Y_params):
           
        norm_diff = np.sum(np.abs(self.A - A)) + np.sum(np.abs(self.Pi - Pi)) + \
                    np.sum(np.abs(self.Y_params - Y_params))
                
        norm_given = np.sum(np.abs(A)) + np.sum(np.abs(Pi)) + \
                     np.sum(np.abs(Y_params))
    
        return (norm_diff, norm_given)
    
    
    ## @fn compute_transition_mat
    #  @brief Compute the probabilities of observing a transition from 
    #   state i to state j given covariates at time t: P(Z_t=j | Z_{t-1}=i, Y_t)
    #
    #  @param train_seq_len Length of training seqences
    #
    #  @return A list of S T_s(t)xK(i)xK(j) matrices where list_mat_B[s] is 
    #   the set of probabilities associated with the s^th sequence. 
    #
    def compute_transition_mat(self, train_seq_len):
        
        #---Homogeneous HMM case
        if(self.nb_covariates == 0):  
            
            S = len(train_seq_len)
            list_mat_B = [ -1 * np.ones(dtype=np.float64, \
                                shape=(T, self.nb_states, self.nb_states)) \
                            for T in train_seq_len ]  
        
            for s in range(S):
                for t in range(train_seq_len[s]):
                    list_mat_B[s][t, :, :] = self.A
                    
            return list_mat_B
        
        else:
            #---Non-Homogeneous case
            return compute_transition_mat_realCov(self.A, self.Y_params, \
                                       self.covariate_data, self.nb_states, \
                                       self.nb_covariates, train_seq_len)

 
############################### BEGIN UTILS FUNCTIONS 
                      
## @fn
#
def compute_transition_mat_realCov(A, Y_params, covariate_data, K, \
                                   nb_covariates, train_seq_len):
    
    # number of training sequences
    S = len(train_seq_len)
    
    # output initialisation
    list_mat_B = [ -1 * np.ones(dtype=np.float64, shape=(T, K, K)) \
                  for T in train_seq_len ]
     
    for s in range(S):        
        for t in range(train_seq_len[s]):
            # compute probabilities of transition i --> *
            for i in range(K):
                # compute G(y_t_s; i, j) for all j: K length array
                # in case of overflow in exp, +inf are replaced by 1e250
                #
                # NB: Y_params[i, :, :] results in '''NumbaPerformanceWarning: 
                # np.dot() is faster on contiguous arrays'''
                log_g_yt_s_i = np.dot(Y_params[i], covariate_data[s][t, :])           
                g_yt_s_i = np.exp(log_g_yt_s_i)
                
                tmp_prob = A[i, :] * g_yt_s_i   
                # normalization
                list_mat_B[s][t, i, :] = tmp_prob / np.sum(tmp_prob)
                
    return list_mat_B

    
## @fn call_update_params_trans_from_i
#  @brief Call function update_params_trans_from_i on the given nhhmc object
#
#  @param nhhmc_object
#  @param list_Xi
#  @param i
#
def call_update_params_trans_from_i(nhhmc_object, list_Xi, i):
    return nhhmc_object.update_params_trans_from_i(list_Xi, i) 

    
############################### BEGIN FUNCTION IMPLEMENTATION
## @fn minus_Q_S_i
#  @brief Minimizing minus_Q_S_i is equivalent to maximizing Q_S_i
#
#  @param parameters_i (K+K*nb_covariates) length array of parameters organized
#   as follows
#   * parameters_i[0:K] = A_i_*
#   * parameters_i[K:] = Y_param[i, *(j), *(l)] flatten by row 
#
#  @param covariate_data List of S T_s x nb_covariate matrices
#
#  @param return A real value
#
def minus_Q_S_i(parameters_i, i, list_Xi, covariate_data):

    # local variables
    K = list_Xi[0][0].shape[0]
    nb_cov = covariate_data[0].shape[1]
    
    # parameters's structure building
    A_i = parameters_i[0:K]
    Y_param_i = np.reshape(parameters_i[K:], newshape=(K, nb_cov), order="C")

    # compute minus_Q_S_i
    func_eval = - compute_Q_S_i(A_i, Y_param_i, i, list_Xi, covariate_data)
    
    
    return func_eval


## @fn compute_Q_S_i
#
#  * nopython: function is fully-compiled, i.e. no usage of Python interpretor
#  * nogil: release GIL (Generalized Interpretor Lock), this enables 
#    numba-compiled function to be run concurrently by different threads
#  * parallel: automatic parallelization of map-reduce operations over array.
#    Explicite loop parallelization through numba.prange can be also used.
#    The user is required to make sure that the loop does not have cross 
#    iteration dependencies except for supported reductions. 
#    NB: Care should be taken, however, when reducing into slices or elements of 
#    an array if the elements specified by the slice or index are written to 
#    simultaneously by multiple parallel threads. The compiler may not detect 
#    such cases and then a RACE CONDITION would occur: this can result in an 
#    incorrect return value.
#    See for details  https://numba.pydata.org/numba-doc/latest/user/parallel.html
#
#  WARNING: Numba Parallel Accelerator cannot be combined with multi-threading/
#  process computing (for instance with Concurrent.Futures module). 
#  Therefore, Numba Parallel Accelerator should not be used in functions 
#  executed by child processes/threads. 
#
#  NB1: Comment line @jit(...) in order to get the non-boosted version.
#  NB2: Because Numba does not support numpy.matmul function, it has been 
#  replaced by numpy.dot 
#
@jit(nopython=True, nogil=True)
def compute_Q_S_i(A_i, Y_param_i, i, list_Xi, covariate_data):
    
    # number of training sequences
    S = len(covariate_data)
    # logarithm of A_i
    log_A_i = np.log(A_i + np.finfo(0.).tiny)
    
    # computing starts
    val = np.float64(0.0)
    
    for s in range(S):
        T_s = covariate_data[s].shape[0]
        for t in range(1, T_s):
            # compute G(y_t_s; i, j) for all j which represent the effect of 
            # covariates y_t_s on transitions i --> j. 
            # K length array
            log_g_yt_s_i = np.dot(Y_param_i, covariate_data[s][t, :])
            
            # in case of overflow in exp, +inf are replaced by 1e250 
            g_yt_s_i = np.array([min(1e250, elt) for elt in np.exp(log_g_yt_s_i)])
            
            # log of the normalization constant of transition probs i-->*
            log_norm_cst = np.log(np.sum(A_i * g_yt_s_i))
            
            tmp = (log_A_i + log_g_yt_s_i - log_norm_cst) * list_Xi[s][t, i, :]
            val += np.sum(tmp)
            
    return val



## @fn grad_minus_Q_S_i
#  @brief Compute the gradient of function minus_Q_S_i.
#   See function minus_Q_S_i for parameters details.
#   Note that grad(-Q_S_i) = - grad(Q_S_i).
#
#  @param parameters_i (K+K*nb_covariates) length array of function parameters
#   organized as follows
#   * parameters_i[0:K] = A_i_*
#   * parameters_i[K:] = Y_param[i, *(j), *(l)] flatten by row  
#
#  @return (K+K*nb_covariates) length array of gradients computed w.r.t. each 
#   variables
#
def grad_minus_Q_S_i(parameters_i, i, list_Xi, covariate_data):
    
    # local variables
    K = list_Xi[0][0].shape[0]
    nb_cov = covariate_data[0].shape[1]
    
    # parameters's structure building
    A_i = parameters_i[0:K]
    Y_param_i = np.reshape(parameters_i[K:], newshape=(K, nb_cov), order="C")

    # compute grad_minus_Q_S_i
    grad_verctor = - compute_grad_Q_S_i(A_i, Y_param_i, i, list_Xi, \
                                        covariate_data, parameters_i.shape[0])
    return grad_verctor
    
    


## @fn compute_grad_Q_S_i
#
@jit(nopython=True, nogil=True)
def compute_grad_Q_S_i(A_i, Y_param_i, i, list_Xi, covariate_data, nb_params):
    
    # output initialization
    grad_vec = np.zeros(dtype=np.float64, shape=nb_params)
    
    # number of training sequences
    S = len(covariate_data)
    # number of states
    K = list_Xi[0][0].shape[0]
    # number of covariates
    nb_cov = covariate_data[0].shape[1]
    # to add before computing the inverse of A
    tiny = np.finfo(0.).tiny
    
    for s in range(S):
        T_s = covariate_data[s].shape[0]
        for t in range(1, T_s):
            
            # compute G(y_t_s; i, j) for all j which represent the effect of 
            # covariates y_t_s on transitions i --> j. 
            # K length array
            log_g_yt_s_i = np.dot(Y_param_i, covariate_data[s][t, :])
            # in case of overflow in exp, +inf are replaced by 1e250 
            g_yt_s_i = np.array([min(1e250, elt) for elt in np.exp(log_g_yt_s_i)])
            
            # the normalization constant of transition probs i-->*
            norm_cst = np.sum(A_i * g_yt_s_i) 
            
            # sum xi_t_i_*
            sum_xi_t_i_ = np.sum(list_Xi[s][t, i, :]) 
            
            #---gradient w.r.t. A_i_* (K variables)
            tmp_A = list_Xi[s][t, i, :] / (A_i + tiny) - \
                                            g_yt_s_i * sum_xi_t_i_ / norm_cst
            grad_vec[0:K] += tmp_A
            
            #---gradient w.r.t. Y_param_i[J] (nb_cov variables)
            for J in range(K):
                tmp_Y_param_J = list_Xi[s][t, i, J] * covariate_data[s][t, :]
                tmp_Y_param_J -= A_i[J] * g_yt_s_i[J] * covariate_data[s][t, :] * \
                                    sum_xi_t_i_ / norm_cst
                
                # indice of variables Y_param_i[J]'s within grad_vec
                b_ind = K + J*nb_cov
                e_ind = b_ind + nb_cov
                grad_vec[b_ind:e_ind] += tmp_Y_param_J
    
    return grad_vec
