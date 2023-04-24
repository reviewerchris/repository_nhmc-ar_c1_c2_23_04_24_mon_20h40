#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:22:38 2022

@author: dama-f
"""

###############################################################################
## @module modified_FB 
#  @brief This module impements the modidied Baum-Welsh algorithm dedicated
#   to Non-Homogeneous MSVAR models in which transition probabilities are 
#   no more stationary in constrast to the standard MSVAR models.
#   In fact transitions from one state to another depends of a set of 
#   covariates Y 
#   
#  This algorithm computes the probabilities 
#  Xi_t(i,j) = P(Z_t=i, Z_{t+1}=j | X, Y, Theta), 
#  gamma_t = P(Z_t=i | X, Y, Theta) and the likelihood of observation 
#  data P(X).
#
###############################################################################

import numpy as np
import math


"""****************BEGIN*********************************************"""
## @fn BFB
#  @brief This function runs BFB algorithm.
#
#  @param M The number of states.
#  @param LL The likelihood matrix of dimension TxM where: \n
#  * LL[t,z] = g(x_t | x_{t-order}^{t-1}, Z_t=z ). \n
#
#  @param B  Array of dimension TxMxM where B[t, i, j] = P(Z_t=j | Z_{t-1}=i, t) 
#  @param Pi Row vector 1xM, initial state probabilities, Pi[0,k] = P(Z_1 = k).
#
#  @return the log-likelihood of observation data 
#   P(X | X_{1-p}^0, Y; Theta)
#  @return joint probabilities Xi_t
#  @return marginal probabilities gamma_t
#  
def modified_FB(M, LL, B, Pi):
                              
    #----------modified FB algorithm begins
    #Forward step
    (Alpha_tilde, C_t) = forward_step(M, LL, B, Pi)
    
    #Backward step
    Beta_tilde = backward_step( M, LL, B, Pi, C_t)
    
    #Xi computing
    Xi = compute_Xi(LL, B, Pi, Alpha_tilde, Beta_tilde)
    
    #Gamma computing
    Gamma = compute_gamma(Xi)
    """
    #another way to compute gamma
    Gamma = compute_gamma_bis(Alpha_tilde, Beta_tilde, C_t)
    """
    #ll computing
    log_ll = likelihood(C_t)
    
    
    return (log_ll, Xi, Gamma, Alpha_tilde)


"""********************Forward step****************************************"""
## @fn forward_step
#  @brief This function runs the forward step of BFB.
#
#  @param M 
#  @param LL 
#  @param B
#  @param Pi 
#
#  @return 
#   * TxM matrix denoted Alpha with 
#       Alpha[t,i] = P(X_1^T, Z_t=i | Y, X_0; Theta) \n
#   * T length array C_t, with C_t[t] = P(X_t=x_t | X_{1-p}^{t-1}, Y; Theta)
#  
def forward_step(M, LL, B, Pi):
    
    #initialization
    T = LL.shape[0]
    Alpha_tilde = np.zeros(dtype=np.float64, shape=(T, M))
    C_t = np.zeros(dtype=np.float64, shape=T)
    
    #----------base case of induction, t = 0
    #---compute C_1
    C_t[0] = np.sum(LL[0, :] * Pi[0, :]) + np.finfo(0.).tiny
    
    #---compute Alpha_tilde_1
    Alpha_tilde[0, :] =  (LL[0, :] * Pi[0, :]) / C_t[0]
    
    #assertion: valid value domain
    for s in range(M):
        assert(not math.isnan(Alpha_tilde[0, s]))
        assert(Alpha_tilde[0, s] >= 0.) 
    
    #----------recursive case
    for t in range(1, T):
        
        #---compute C_t
        tmp_sum = 0.0
        for j in range(M):
            loc_tmp = np.sum( Alpha_tilde[t-1, :] * B[t, :, j] )
            tmp_sum = tmp_sum + ( LL[t, j] * loc_tmp )
                
        C_t[t] = tmp_sum + np.finfo(0.).tiny
        
        #---compute Alpha_tilde_t
        for s in range(M):
            loc_tmp = np.sum(Alpha_tilde[t-1, :] * B[t, :, s])
            Alpha_tilde[t, s] = (LL[t, s] * loc_tmp) / C_t[t]
         
            #assertion: valid value domain
            assert(not math.isnan(Alpha_tilde[t, s]))
            assert(Alpha_tilde[t, s] >= 0.)
            
            
    return (Alpha_tilde, C_t)


"""***************************Backward step*********************************"""
## @fn backward_step
#  @brief This function runs the second backward step of BFB.
#
#  @param M The number of states.
#  @param LL The likelihood matrix of dimension TxM
#  @param A MxM matrix
#  @param Pi Row vector 1xM
#  @param C_t
#
#  @return TxM matrix denoted Beta with 
#   Beta[t,i] = P(X_{t+1}^T | Z_t=i, X_{t+1-p}^t, Y; Theta).
#
def backward_step(M, LL, B, Pi, C_t):
    
    #initialization
    T = LL.shape[0]
    Beta_tilde = np.zeros(dtype=np.float64, shape=(T, M))
                
    #----------base case of induction, t = T-1
    Beta_tilde[T-1, :] = np.ones(dtype=np.float64, shape=M) / C_t[T-1]
    
    #----------recursion case
    for t in range(T-2, -1, -1):
        for s in range(M):
            
            loc_tmp = np.sum( Beta_tilde[t+1, :] * LL[t+1, :] * B[t+1, s, :] )
            loc_tmp = loc_tmp / C_t[t]
            beta_t_s = min(1e300, loc_tmp)
            
            #assertion: valid value domain
            assert(not math.isnan(beta_t_s))
            assert(beta_t_s >= 0.)
            
            Beta_tilde[t, s] = beta_t_s
            
    return Beta_tilde


"""**************Log-likelihood and gamma computing************************"""
## @fn log_likelihood
#  @brief This function computes the likelihood of observations
#
#  @param C_t T length array with 
#   C_t[t] = P(X_t | X_{1-p}^{t-1}, Y; Theta)
#
#  @return The log likelihood P(X | X_{1-p}^0, Y; Theta) (real value).
#  
def likelihood(C_t):
    
    T = C_t.shape[0]
    
    log_ll = 0.0    
    for t in range(T):
        log_ll = log_ll + np.log(C_t[t] + np.finfo(0.).tiny)
         
    #assertion: value domain
    assert(not math.isinf(log_ll))
    assert(not math.isnan(log_ll))
         
    return log_ll


## @fn compute_gamma_bis
#  @brief This function computes gamma probabilities from Alpha_tilde, 
#   Beta_tilde and C_t.
#  
#  @param Alpha_tilde
#  @param Beta_tilde
#  @param C_t
#
#  @return A matrix TxM Gamma with Gamma[t,i] = P(Z_t=z | X, Y, Theta).
#
def  compute_gamma_bis(Alpha_tilde, Beta_tilde, C_t):
      
    (T, M) = Alpha_tilde.shape
    Gamma = np.zeros(dtype=np.float64, shape=(T, M))
    
    for t in range(T):
        for z in range(M):
            Gamma[t, z] = Alpha_tilde[t, z] * Beta_tilde[t, z] * C_t[t]
                              
    return Gamma


## @fn compute_gamma
#  @brief This function computes gamma probabilities from Xi probabilities.
#  
#  @param Xi 3D matrix of dimension TxMxM
#
#  @return A matrix TxM Gamma with Gamma[t,i] = P(Z_t=i | X, Y, Theta).
#
def  compute_gamma(Xi):
      
    (T, M, M) = Xi.shape
    Gamma = np.zeros(dtype=np.float64, shape=(T, M))
    
    #---first time-step
    for i in range(M):
        Gamma[0, i] = np.sum(Xi[1, i, :])
    
    #---other time-steps
    for t in range(1, T):
        for j in range(M):
            Gamma[t, j] = np.sum(Xi[t, :, j])
                    
    return Gamma


"""**************BEGIN (f)************************************************"""
## @fn compute_Xi
#  @brief This function runs the second backward step of BFB.
#  
#  @param LL TxM matrix 
#  @param MxM matrix 
#  @param Pi Row vector 1xM
#  @param Alpha_tilde Matrix TxM, foreward propagation terms
#  @param Beta_tilde Matrix TxM, second backward propagation terms
#
#  @return A 3D matrix Xi with Xi[t, , ] a MxM matrix and  
#   Xi[t, i, j] = P(Z_{t-1}=i, Z_t=j | X, Y, \Theta). \n
#  Xi[1, , ] is a matrix MxM of zeros because Xi is not defined at the first
#  time-step.
#
def compute_Xi(LL, B, Pi, Alpha_tilde, Beta_tilde):
    
    #initialization
    (T, M) = np.shape(Beta_tilde)
    Xi = np.zeros(dtype=np.float64, shape=(T, M, M))
        
    for t in range(1,T):
        for i in range(M):
            for j in range(M):
                
                Xi[t, i, j] = Beta_tilde[t, j] * B[t, i, j] * LL[t, j] * \
                                Alpha_tilde[t-1, i]
                                    
                #assertion: valid value domain
                assert(not math.isnan(Xi[t, i, j]))
                assert(Xi[t, i, j]  >= 0.)
                                 
    return Xi
    

