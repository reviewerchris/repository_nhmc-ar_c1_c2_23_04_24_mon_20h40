#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:23:43 2022

@author: dama-f
"""

import numpy as np
from scipy.stats import dirichlet

#####################################################################################
##  @package NH-HMC
#   Non-Homogeneous Hidden Markov Chain is a generalization of the standard
#   HMC in which the transition matrix depends of the time throughout a set 
#   of covariates Y_t.
#
#
     

class NHHMC():
    
    # Class attributes
               
    def __init__(self):   
        
        ## @brief The type of covariates used. It can be either "real" or 
        # "event.sequences"
        #
        self.covariate_type = ""
        
        ## @brief The number of covariates
        #
        self.nb_covariates = 0
        
        ## @brief The number of states.
        #
        self.nb_states = 0
        
        ## @brief The initial law: 1 x nb_states array
        #
        self.Pi = {}
        
        ## @brief Baseline transition matrix: nb_states x nb_states matrix
        #
        self.A = {}
        
        ## @brief The parameters related to covariates Y_t 
        #
        self.Y_params = {}
        
        ## @brief List of length S, where S is the number of training sequences.
        #  self.covariate_data[s] is a T_s x nb_covariates matrix
        #
        self.covariate_data = {}
                
        return 
    
    
    ## @fn set_parameters
    #  @brief 
    # 
    #  @param Pi
    #  @param A
    #  @param Y_params
    #
    def set_parameters(self, Pi, A, Y_params):
        
        self.Pi = Pi
        self.A = A
        self.Y_params = Y_params
        
        return 
            
        
    ## @fn
    #  @brief
    #
    def update_Pi(self, list_Gamma):
        
        #nb regimes
        M = self.nb_states
        #nb sequences
        S = len(list_Gamma)
    
        Pi = np.zeros(dtype=np.float64, shape=(1, M))
    
        # states 0 to M-2
        for i in range(M-1):            
            Pi[0, i] = np.mean([list_Gamma[s][0, i] for s in range(S)])
        
        # state M-1
        Pi[0, M-1] = max(0, 1 - np.sum(Pi[0, 0:(M-1)]))
        #normalization: probs sum at one
        Pi[0, :] = Pi[0, :] / np.sum(Pi[0, :])
        
        #assertion: valid value domain
        assert(np.sum(np.isnan(Pi[0, :])) == 0)
        assert(np.sum(Pi[0, :] < 0.) == 0)
        assert(np.round(np.sum(Pi), 5) == 1.)

        self.Pi = Pi
        
        return          
    
    
    ## @fn
    #  @brief
    #
    def update_homogeneous_HMM(self, list_Xi, list_Gamma):
        
        #nb regimes
        M = self.nb_states
        #nb sequences
        S = len(list_Gamma)
                
        # compute transitions' frequency
        F = np.zeros(dtype=np.float64, shape=(M, M))
        for s in range(S):
            T_s = list_Xi[s].shape[0]
            for t in range(T_s):
                F += list_Xi[s][t, :, :]
                
        # transition probs
        for i in range(M):
            state_i_freq = 0.0
            for s in range(S):
                state_i_freq = state_i_freq + np.sum(list_Gamma[s][:, i])                   
            
            #NB: if state_i_freq = 0 then state i never reached
            # states 0 to M-2
            F[i, :] = F[i, :] / (state_i_freq + np.finfo(0.).tiny)
            # state M - 1
            F[i, M-1] = max(0, 1 - np.sum(F[i, 0:(M-1)]))
            # normalization: probs sum at one                     
            F[i, :] = F[i, :] / np.sum(F[i, :])
                        
            #assertion: valid value domain
            assert(np.sum(np.isnan(F[i, :])) == 0)
            assert(np.sum(F[i, :] < 0.) == 0)
            assert(np.round(np.sum(F[i, :]), 5) == 1.)
        
        self.A = F
        
        return
        
    
    

    
    
    
