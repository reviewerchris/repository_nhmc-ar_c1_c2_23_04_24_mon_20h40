#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:24:59 2022

@author: dama-f
"""

import numpy as np
from scipy import optimize
from scipy.stats import dirichlet
from numba import jit

from non_homogeneous_HMC import NHHMC


############################################################################
## @class Event_Seq_Covariates
#
class Event_Seq_Covariates(NHHMC):

    ## @fn
    #  @brief In this sub-class parameters associated with covariates denoted
    #  by Y_params. Let nb_inter_terms = (nb_covariates*(nb_covariates-1)/2)
    #  the number of pairs (l,l') such that l < l' and l,l'=1, ..., nb_covariates.
    #  Y_params is a dictionary with the following keys:
    #    * "phi": nb_states(j) x nb_covariates(l)
    #    * "delta1": array of length nb_covariates
    #    * "psi": nb_states(j) x nb_inter_terms
    #    * "delta2": array of length nb_inter_terms
    #
    #  @param nb_states
    #  @param nb_covariates
    #  @param covariate_data List of S T_s x nb_covariates matrices.
    #   covariate_data[s] represents the time of the last occurrence of each
    #   event category. Note that nb_covariates is the number of event categories.
    #  @param kappa_data List of S T_s length arrays.
    #   kappa_data[s] contains sampling time of the sth observed time series
    #   from time-step 1 to time-step T_s. Time unit is the one used for
    #   covariate_data.
    #
    #
    def __init__(self, nb_states, nb_covariates, covariate_data, kappa_data):
        
        #---assertions
        assert(nb_states > 0)
        
        if(nb_covariates < 0 or nb_covariates == 1):
            print()
            print("ERROR: class event_sequence_Covariates: uncorrect "
                  "nb_covariates. Give 0 or more than one covariates!\n")
            exit(1)
        
        if(nb_covariates != 0 and covariate_data[0].shape[1] != nb_covariates):
            print()
            print("ERROR: class event_sequence_Covariates: nb_covariates must"
                  " be equal to the number of column within covariate_data!\n")
            exit(1)
        
        # hyper-parameters setting
        self.covariate_type = "event.sequences"
        self.nb_covariates = nb_covariates
        self.nb_states = nb_states
        self.covariate_data = covariate_data
        self.kappa_data = kappa_data
        
        #---random initialization of Pi and A
        concentration_param = [1 for i in range(nb_states)]
        self.Pi = np.zeros(dtype=np.float64, shape=(1, nb_states))
        self.Pi[0, :] = dirichlet.rvs(concentration_param, 1)[0]
            
        self.A = np.zeros(dtype=np.float64, shape=(nb_states, nb_states))
        for i in range(nb_states):
            self.A[i, :] = dirichlet.rvs(concentration_param, 1)[0] 
        
        #---random initialization of Y_params
        nb_inter_terms = int(nb_covariates*(nb_covariates-1)/2) 
        self.Y_params = {}
        
        # "phi" parameters are chosen within [-1, 1] under some identification 
        #  constraints
        self.Y_params["phi"] = np.zeros(dtype=np.float64, \
                                         shape=(nb_states, nb_covariates))
        
        # sum_l Y_params["phi"][j, l] = 0 for all j
        for j in range(nb_states):
            self.Y_params["phi"][j, :] = np.random.uniform(-1, 1, nb_covariates)
            self.Y_params["phi"][j, 0] = -np.sum(self.Y_params["phi"][j, 1:])
        
            
        # "psi" parameters are also chosen within [-1, 1] under some 
        # identification constraints
        self.Y_params["psi"] = np.zeros(dtype=np.float64, \
                                        shape=(nb_states, nb_inter_terms))
        
        # sum_int_term Y_params["psi"][j, int_term] = 0 for all j
        for j in range(nb_states):
            self.Y_params["psi"][j, :] = np.random.uniform(-1, 1, nb_inter_terms)
            self.Y_params["psi"][j, 0] = -np.sum(self.Y_params["psi"][j, 1:])
        
        # "delta1" and "delta2" are chosen in [0, 1]  
        self.Y_params["delta1"] = np.random.uniform(0, 1, nb_covariates)
        self.Y_params["delta2"] = np.random.uniform(0, 1, nb_inter_terms)
                       
        
        return
    
    
    ## @fn
    #  @brief
    #
    def set_at_homogeneous_HMC(self, Pi, A):
        
        self.Pi = Pi
        self.A = A
        
        # set Y_params at zeros
        nb_cov = self.nb_covariates
        nb_states = self.nb_states
        self.Y_params["phi"] = np.zeros(dtype=np.float64, \
                                         shape=(nb_states, nb_cov))
        self.Y_params["delta1"] = np.zeros(dtype=np.float64, shape=nb_cov)

        nb_inter_terms = int(nb_cov*(nb_cov-1)/2) 
        self.Y_params["psi"] = np.zeros(dtype=np.float64, \
                                        shape=(nb_states, nb_inter_terms))
        self.Y_params["delta2"] = np.zeros(dtype=np.float64, shape=nb_inter_terms)
        
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
            (A, Phi, Delta1, Psi, Delta2) = self.update_A_Y_params(list_Xi)
            
            self.A = A
            self.Y_params["phi"] = Phi
            self.Y_params["delta1"] = Delta1
            self.Y_params["psi"] = Psi
            self.Y_params["delta2"] = Delta2
        
        return (self.Pi, self.A, self.Y_params)   


    ## @fn update_A_Y_params
    #  @brief
    #
    #  @param list_Xi
    #
    def update_A_Y_params(self, list_Xi):

        # local variables
        K = self.nb_states
        nb_cov = self.nb_covariates
        #the number of interaction terms 
        nb_inter_terms = int(nb_cov*(nb_cov-1)/2)   
        nb_params = K*K + K*nb_cov + nb_cov + K*nb_inter_terms + nb_inter_terms                   

        #---Initial values of parameters: equal to the current estimate
        init_parameters = flatten_params(self.A, self.Y_params, K, nb_cov,  \
                                         nb_inter_terms, nb_params)
        # end index (excluded) of different parameters 
        e_ind_A = K*K
        e_ind_phi = e_ind_A + K*nb_cov
        e_ind_delta1 = e_ind_phi + nb_cov
        e_ind_psi = e_ind_delta1 + K*nb_inter_terms

        #---Parameter bounds: they are searched within [lower_b, upper_b]
        # Y_params["phi"] and Y_params["psi"] are in R
        lower_b = np.repeat(-np.inf, nb_params)
        upper_b = np.repeat(np.inf, nb_params)
        # A_i_j's are in  [0, 1]
        lower_b[0:e_ind_A] = np.repeat(1e-100, K*K)
        upper_b[0:e_ind_A] = np.repeat(1.0001, K*K)
        # Y_params["delta1"] and Y_params["delta2"] are in R+
        lower_b[e_ind_phi:e_ind_delta1] = np.repeat(0., nb_cov)
        lower_b[e_ind_psi:] = np.repeat(0., nb_inter_terms)
        
        # build Bound object
        # Note that bounds are (internally) transformed into inequality
        # constraints.
        bounds = optimize.Bounds(lb=lower_b, ub=upper_b, keep_feasible=True) 

        #---Constraints
        # Ours constraints are formulated as follows: 
        # sum_l x_l = C  <==> C <= (1, ..., 1) x (x_1, ..., x_L)' <= C
        # We have (K + nb_cov + nb_inter_terms) constraints.
        # For each constraints we define the parameters involved and C value
        
        # list of nb_constr element: each entry correspond to one constraint
        c_values = []
        # list of nb_params-length array: each array correspond to one constraint
        constr_coefs = []

        # first K constraint over A_i_j's
        for i in range(K):
            # sum_j A_i_j = 1
            c_values.append(1)
            tmp = np.zeros(dtype=np.float64, shape=nb_params)
            b_ind = K*i
            e_ind = b_ind + K
            tmp[b_ind:e_ind] = np.repeat(1, K)
            constr_coefs.append(tmp)

        # K constraints over "phi"'s
        for j in range(K):
            # sum_l Y_params["phi"][j, l] = 0
            phi_ = np.zeros(dtype=np.float64, shape=(K, nb_cov))
            phi_[j, :] = np.repeat(1, nb_cov)
            tmp = np.zeros(dtype=np.float64, shape=nb_params)
            tmp[e_ind_A:e_ind_phi] = phi_.flatten(order='C')
            constr_coefs.append(tmp)
            c_values.append(0)
        
        # K constraints over "psi"'s
        for j in range(K):
            # sum_inter_t Y_params["psi"][j, inter_t] = 0
            psi_ = np.zeros(dtype=np.float64, shape=(K, nb_inter_terms))
            psi_[j, :] = np.repeat(1, nb_inter_terms)
            tmp = np.zeros(dtype=np.float64, shape=nb_params)
            tmp[e_ind_delta1:e_ind_psi] = psi_.flatten(order='C')
            constr_coefs.append(tmp)
            c_values.append(0)

        # build LinearConstraint object
        c_values = np.array(c_values)
        constr_coefs = np.array(constr_coefs)
        linear_constraints = \
                optimize.LinearConstraint(constr_coefs, lb=c_values, \
                                          ub=c_values, keep_feasible=True)
        
        #---Numerical optimization
        # * method="COBYLA": Constrained Optimization BY Linear Approximation
        #   Only support constraints, not bounds
        #
        # * method="SLSQP: Sequential Least Square Programming
        #   Constraint option 'keep_feasible' is ignored by this method.
        #   Bound option 'keep_feasible' seems to be taken into account
        #
        # * method="trust-constr" trust-region algorithm 
        #   Both constraint and bound options 'keep_feasible' are taken into 
        #   account
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
        #   finite-differrence approximation. I used the 2-point scheme 
        #  SOURCE: https://github.com/scipy/scipy/issues/8644
        #  
        res = optimize.minimize(fun=minus_Q_S, x0=init_parameters, \
                        args=(list_Xi, self.covariate_data, self.kappa_data), \
                        method="trust-constr", \
                        jac=grad_minus_Q_S, hess='2-point', \
                        constraints=linear_constraints, bounds=bounds, \
                        options={'verbose': 0, 'maxiter': 1000})
        # Warning
        if(not res.success):
            print("Warning: class real_covariates: numerical optimization", \
                  "did not converge while updating transition probabilities. ", \
                  "Failure message: {}.".format(res.message))

        #---update parameters
        (A, Phi, Delta1, Psi, Delta2) = build_params_struc(res.x, K, nb_cov,  \
                                                           nb_inter_terms)
        
        return (A, Phi, Delta1, Psi, Delta2)

        
    ## @fn compute_norm
    #  @brief Compute the L1-norm of the difference between self and the given
    #   NH_HMC 
    #
    def l1_norm_of_diff(self, A, Pi, Y_params):
           
        norm_diff = np.sum(np.abs(self.A - A)) + np.sum(np.abs(self.Pi - Pi)) + \
                    np.sum(np.abs(self.Y_params["phi"] - Y_params["phi"])) + \
                    np.sum(np.abs(self.Y_params["psi"] - Y_params["psi"])) + \
                    np.sum(np.abs(self.Y_params["delta1"] - Y_params["delta1"])) + \
                    np.sum(np.abs(self.Y_params["delta2"] - Y_params["delta2"]))
                    
        norm_given = np.sum(np.abs(A)) + np.sum(np.abs(Pi)) + \
                     np.sum(np.abs(Y_params["phi"])) + \
                     np.sum(np.abs(Y_params["psi"])) + \
                     np.sum(np.abs(Y_params["delta1"])) + \
                     np.sum(np.abs(Y_params["delta2"]))
        
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
        
        #---Homogeneous HMM case: transition matrix is constant over time
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
            return compute_transition_mat_eventSeqCov(self.A, self.Y_params["phi"], \
                                self.Y_params["delta1"], self.Y_params["psi"], \
                                self.Y_params["delta2"], self.covariate_data, \
                                self.kappa_data, self.nb_states, \
                                self.nb_covariates, train_seq_len)


############################### BEGIN UTILS FUNCTIONS
        
## @fn
#
def compute_transition_mat_eventSeqCov(A, Phi, Delta1, Psi, Delta2, \
                                       covariate_data, kappa_data, K, \
                                       nb_covariates, train_seq_len):    
                         
    # number of training sequences
    S = len(train_seq_len)
    
    # output initialisation
    list_mat_B = [ -1 * np.ones(dtype=np.float64, shape=(T, K, K)) \
                  for T in train_seq_len ]
        
    for s in range(S):        
        for t in range(train_seq_len[s]):
            # compute G(y_t_s; j) for all j, K length array
            # in case of overflow in exp, +inf are replaced by 1e250 
            log_g_yt_s = log_G_y(Phi, Delta1, Psi, Delta2, \
                                 covariate_data[s][t, :], kappa_data[s][t])
            g_yt_s = np.exp(log_g_yt_s)
            
            # compute probabilities of transition i --> *
            for i in range(K):
                tmp_prob = A[i, :] * g_yt_s
                # normalization
                list_mat_B[s][t, i, :] = tmp_prob / np.sum(tmp_prob)
                
    return list_mat_B


## @fn flatten_params
#  @brief This function flattens parameters (A, Y_params) into a single 1D 
#   vector in the following order: A, Phi, Delta1, Psi and Delta2 
#
#  @param A KxK array
#  @param Y_params Dictionary (for more details see function __init__)
#
#  @return nb_params length array
#
def flatten_params(A, Y_params, K, nb_cov, nb_inter_terms, nb_params):

    # output
    flatten_parameters = np.zeros(dtype=np.float64, shape=nb_params)

    # pameters are stored in this order: A, Phi, Delta1, Psi and Delta2 
    e_ind_A = K*K
    e_ind_phi = e_ind_A + K*nb_cov
    e_ind_delta1 = e_ind_phi + nb_cov
    e_ind_psi = e_ind_delta1 + K*nb_inter_terms
    
    # store A, from index 0 to e_ind_A
    flatten_parameters[0:e_ind_A] = A.flatten(order='C')

    # store Y_params["phi"] from index e_ind_A to e_ind_phi
    flatten_parameters[e_ind_A:e_ind_phi] = Y_params["phi"].flatten(order='C')

    # store Y_params["delta1"] from index e_ind_phi to e_ind_delta1
    flatten_parameters[e_ind_phi:e_ind_delta1] = Y_params["delta1"]
    
    # store Y_params["psi"] from index e_ind_delta1 to e_ind_psi
    flatten_parameters[e_ind_delta1:e_ind_psi] = Y_params["psi"].flatten(order='C')

    # store Y_params["delta2"] from index e_ind_psi to the end
    flatten_parameters[e_ind_psi:] = Y_params["delta2"]

    return flatten_parameters


## @fn build_params_struc
#  @brief This function build parameters structure from a flatten vector of parameters.
#
#  @param flatten_parameters
#
#  @return 
#   * A       KxK array 
#   * Phi     nb_states(j) x nb_covariates(l) array
#   * Delat1 (nb_covariates, )  array
#   * Psi    nb_states(j) x nb_inter_terms array
#   * Delat2 (nb_inter_terms, ) array
#
def build_params_struc(flatten_parameters, K, nb_cov, nb_inter_terms):

    # pameters are stored in this order: A, Phi, Delta1, Psi and Delta2 
    e_ind_A = K*K
    e_ind_phi = e_ind_A + K*nb_cov
    e_ind_delta1 = e_ind_phi + nb_cov
    e_ind_psi = e_ind_delta1 + K*nb_inter_terms
        
    A = np.reshape(flatten_parameters[0:e_ind_A], newshape=(K,K), order='C')
    Phi = np.reshape(flatten_parameters[e_ind_A:e_ind_phi], \
                     newshape=(K, nb_cov), order='C')
    Delta1 = flatten_parameters[e_ind_phi:e_ind_delta1]
    Psi = np.reshape(flatten_parameters[e_ind_delta1:e_ind_psi], \
                    newshape=(K, nb_inter_terms), order='C')
    Delta2 = flatten_parameters[e_ind_psi:]
        
    return (A, Phi, Delta1, Psi, Delta2)
    

############################### BEGIN FUNCTION IMPLEMENTATION
## @fn log_G_y
#  @brief Compute G(y_t; j) for j = 1, ..., nb_regimes where G 
#   models the effect of covariates y_t on transitions * --> j.
#
#  @param Y_params = (Phi, Delta1, Psi, Delta2) 
#  @param y_t array of length nb_covariates. 
#  @param kappa_t Real value.
#
#  @return An array of length K where the jth entry corresponds to log(G(y_t; j))
#
@jit(nopython=True, nogil=True)
def log_G_y(Phi, Delta1, Psi, Delta2, y_t, kappa_t):
    
    # compute temporal horizon of dependencies at time-step t
    # h1: (nb_cov, ) array, h2: (nb_inter_terms, ) array
    (h1, h2, _, _) = temporal_dependence_scope(Delta1, Delta2, y_t, kappa_t)
     
    # compute G
    outputs = np.dot(Phi, h1) + np.dot(Psi, h2)

    return outputs
    

## @fn temporal_dependence_scope
#  @brief Compute the scope of temporal dependencies h1(kappa_t_s, y_t_s; Delta1), 
#  h2(kappa_t_s, y_t_s; Delta2) and their gradient.
#  
#  @param Delta1, Delta2 1 D arrays
#  @param y_t array of length nb_covariates. 
#  @param kappa_t Real value.
#
#  @return
#   * h1: (nb_cov, ) arrays
#   * h2: (nb_inter_terms, ) arrays, with 
#         nb_inter_terms = nb_covariates * (nb_covariates - 1) /2
#   * grad_h1 w.r.t. Delta1: (nb_cov, ) arrays
#   * grad_h2 w.r.t Delta2: (nb_inter_terms, ) arrays
#
@jit(nopython=True, nogil=True)
def temporal_dependence_scope(Delta1, Delta2, y_t, kappa_t):
    
    # local variables
    nb_cov = y_t.shape[0]
    
    #----compute function h1 and grad_h1
    h1 = np.exp(-Delta1 * np.abs(kappa_t - y_t))
    grad_h1 = - np.abs(kappa_t - y_t) * h1

    #----compute function h2 and grad_h2
    # second order interaction terms: set of  pairs (y_t_l, y_t_l') such 
    # that l < l'. nb_inter_terms x 2 matrix where 
    pairs_of_y_t = np.array([ [y_t[l], y_t[l_prim]] for l in range(nb_cov) \
                              for l_prim in range(nb_cov) if l < l_prim ])
    
    dist1 = np.abs(kappa_t - pairs_of_y_t)
    dist2 = np.abs(pairs_of_y_t[:,0] - pairs_of_y_t[:,1])
    
    h2 = np.exp(-Delta2 * (dist1[:, 0] + dist1[:, 1] + dist2))
    grad_h2 = -(dist1[:, 0] + dist1[:, 1] + dist2) * h2
    
    return (h1, h2, grad_h1, grad_h2)

    
## @fn minus_Q_S
#  @brief Minimizing minus_Q_S is equivalent to maximizing Q_S
#
#  @param parameters
#
#  @param covariate_data List of S T_s x nb_covariate matrices
#  @param kappa_data List of S T_s length arrays
#
def minus_Q_S(parameters, list_Xi, covariate_data, kappa_data):
    
    # local variables
    K = list_Xi[0][0].shape[0]
    nb_cov = covariate_data[0].shape[1]
    nb_inter_terms = int(nb_cov*(nb_cov-1)/2)

    # parameters's structure building
    (A, Phi, Delta1, Psi, Delta2) = build_params_struc(parameters, K, nb_cov, \
                                                       nb_inter_terms)
    # compute minus_Q_S_i
    func_eval = - compute_Q_S(A, Phi, Delta1, Psi, Delta2, list_Xi, \
                                     covariate_data, kappa_data)
    
    return func_eval
    
    
## @fn compute_Q_S
#
#  @parameter A, Phi, Delta1, Psi, Delta2
#  for more details about parameters' dimension see function "build_params_struc"
#
#  NB1: Comment line @jit(...) in order to get the non-boosted version.
#  NB2: The version with Numba Parallel Accelerator (parallel=True, where range 
#  is replaced by numba.prange) is slower. 
#
@jit(nopython=True, nogil=True)
def compute_Q_S(A, Phi, Delta1, Psi, Delta2, list_Xi, covariate_data, \
                kappa_data):
    # local variables
    K = A.shape[0]
    S = len(covariate_data)
    # logarithm of A_i
    log_A = np.log(A + np.finfo(0.).tiny)

    # computing starts
    val = np.float64(0.0)

    for s in range(S):
        T_s = covariate_data[s].shape[0]
        for t in range(1, T_s):
            # compute G(y_t_s; j) for all j which represent the effect of 
            # covariates y_t_s on transitions * --> j. 
            # K length array
            log_g_yt_s = log_G_y(Phi, Delta1, Psi, Delta2, \
                                 covariate_data[s][t, :], kappa_data[s][t])
            
            # in case of overflow in exp, +inf are replaced by 1e250
            g_yt_s = np.array([min(1e250, elt) for elt in np.exp(log_g_yt_s)])
            
            # for each fixed i, log of the normalization constant of 
            # transition probs i-->*. K length array
            log_norm_cst = np.log(np.dot(A, g_yt_s))

            for i in range(K):
                tmp = (log_A[i,:] + log_g_yt_s - log_norm_cst[i]) * list_Xi[s][t, i, :]          
                val += np.sum(tmp)        

    return val



## @fn grad_minus_Q_S
#  @brief Compute the gradient of function minus_Q_S.
#   See function minus_Q_S for parameters details.
#   Note that grad(-Q_S) = - grad(Q_S).
#
#  @param parameters 
#
#  @return 
#
def grad_minus_Q_S(parameters, list_Xi, covariate_data, kappa_data):
            
    # local variables
    K = list_Xi[0][0].shape[0]
    nb_cov = covariate_data[0].shape[1]
    nb_inter_terms = int(nb_cov*(nb_cov-1)/2)
    
    # parameters's structure building
    (A, Phi, Delta1, Psi, Delta2) = build_params_struc(parameters, K, nb_cov, \
                                                       nb_inter_terms)

    # compute grad_minus_Q_S
    grad_verctor = - compute_grad_Q_S(A, Phi, Delta1, Psi, Delta2, list_Xi, \
                                      covariate_data, kappa_data, \
                                      parameters.shape[0])
    return grad_verctor
    


## @fn compute_grad_Q_S
#
#  @parameter A, Phi, Delta1, Psi, Delta2
#  for more details about parameters' dimension see function "build_params_struc"
#
#  NB1: Comment line @jit(...) in order to get the non-boosted version.
#  NB2: The version with Numba Parallel Accelerator (parallel=True, without 
#  numba.prange) is slower
#
@jit(nopython=True, nogil=True)
def compute_grad_Q_S(A, Phi, Delta1, Psi, Delta2, list_Xi, covariate_data, \
                     kappa_data, nb_params):
    
    # number of training sequences
    S = len(covariate_data)
    # number of states
    K = list_Xi[0][0].shape[0]
    # number of covariates
    nb_cov = covariate_data[0].shape[1]
    # number of interaction terms
    nb_inter_terms = int(nb_cov*(nb_cov-1)/2)
    # to add before computing the inverse of A
    tiny = np.finfo(0.).tiny
    
    # pameters are stored in this order: A, Phi, Delta1, Psi and Delta2 
    # Limiting indice
    e_ind_A = K*K
    e_ind_phi = e_ind_A + K*nb_cov
    e_ind_delta1 = e_ind_phi + nb_cov
    e_ind_psi = e_ind_delta1 + K*nb_inter_terms
    
    # output initialization
    grad_vec = np.zeros(dtype=np.float64, shape=nb_params)
    
    for s in range(S):
        T_s = covariate_data[s].shape[0]
        for t in range(1, T_s):
            
            # compute G(y_t_s; j) for all j, K length array
            # in case of overflow in exp, +inf are replaced by 1e250 
            log_g_yt_s = log_G_y(Phi, Delta1, Psi, Delta2, \
                                 covariate_data[s][t, :], kappa_data[s][t])
            g_yt_s = np.array([min(1e250, elt) for elt in np.exp(log_g_yt_s)])
            
            # normalization constant of transition probs i-->*, K length array
            norm_cst = np.dot(A, g_yt_s)
            
            # compute temporal horizon h1, h2 and their gradients
            (h1_yt_s, h2_yt_s, grad_h1_yt_s, grad_h2_yt_s) = \
                          temporal_dependence_scope(Delta1, Delta2, \
                                                    covariate_data[s][t, :], \
                                                    kappa_data[s][t])
            
            #--------gradient w.r.t. A (KxK variables)
            tmp_cst = np.sum(list_Xi[s][t, :, :] / norm_cst.reshape((-1,1)))
            tmp_A = list_Xi[s][t, :, :] / (A + tiny) - tmp_cst * g_yt_s
            # order='C' by default
            # note that order argument is not supported by numba compiler
            grad_vec[0:e_ind_A] += tmp_A.flatten()    
            
            #--------gradient w.r.t. Phi and Psi
            # gradient w.r.t. Phi_J (nb_cov variables) and 
            # Psi_J (nb_inter_terms variables)
            for J in range(K):     
                tmp_cst_J = np.sum( list_Xi[s][t, :, :] * \
                                       (A[:, J] / norm_cst).reshape((-1,1)) )
                tmp_cst_J = np.sum(list_Xi[s][t, :, J]) - g_yt_s[J] * tmp_cst_J
                
                #---grandient w.r.t. Phi_J
                tmp_Phi_J = h1_yt_s * tmp_cst_J
                
                tmp_b_ind = e_ind_A + J*nb_cov
                tmp_e_ind = tmp_b_ind + nb_cov
                grad_vec[tmp_b_ind:tmp_e_ind] += tmp_Phi_J 
                
                #---grandient w.r.t. Psi_J
                tmp_Psi_J = h2_yt_s * tmp_cst_J
                            
                tmp_b_ind = e_ind_delta1 + J*nb_inter_terms
                tmp_e_ind = tmp_b_ind + nb_inter_terms
                grad_vec[tmp_b_ind:tmp_e_ind] += tmp_Psi_J
        
            #--------gradient w.r.t. Delta1 (nb_cov variables) and
            # Delta2 (nb_inter_terms variables)
            tmp_sum_Delta1 = np.zeros(dtype=np.float64, shape=nb_cov)
            tmp_sum_Delta2 = np.zeros(dtype=np.float64, shape=nb_inter_terms)
            
            for i in range(K):     
                # column vector
                tmp_i = (A[i,:] * g_yt_s).reshape((-1,1))
                # nb_cov length array
                tmp_vec_i_delta1 = np.sum(Phi * tmp_i, axis=0) / norm_cst[i]
                # nb_inter_terms length array
                tmp_vec_i_delta2 = np.sum(Psi * tmp_i, axis=0) / norm_cst[i]
                
                for j in range(K):
                    tmp_sum_Delta1 += (Phi[j, :] - tmp_vec_i_delta1) * \
                                        list_Xi[s][t, i, j]
                                        
                    tmp_sum_Delta2 += (Psi[j, :] - tmp_vec_i_delta2) * \
                                        list_Xi[s][t, i, j] 
                                        
            #---grandient w.r.t. Delta1
            tmp_Delta1 = grad_h1_yt_s * tmp_sum_Delta1
            grad_vec[e_ind_phi:e_ind_delta1] += tmp_Delta1 
                
            #---grandient w.r.t. Delta2
            tmp_Delta2 = grad_h2_yt_s * tmp_sum_Delta2
            grad_vec[e_ind_psi:] += tmp_Delta2
            
             
    return grad_vec




