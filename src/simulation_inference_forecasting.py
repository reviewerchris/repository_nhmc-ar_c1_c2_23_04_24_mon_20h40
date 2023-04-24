#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:11:54 2022

@author: dama-f
"""


import numpy as np
import pickle
from scipy.stats import multivariate_normal

from sys import exit
import os
import sys
from scaled_modified_forward_backward_recursion import modified_FB

# function specific to eventSeqCov NH-MSAR model
from event_sequence_covariates import compute_transition_mat_eventSeqCov
from event_seq_simulation_forecasting import eventSeqCov_sliding_forecasting, \
    eventSeqCov_sliding_forecasting_error, eventSeqCov_compute_model_residuals

# function specific to realCov NH-MSAR model
from real_covariates import compute_transition_mat_realCov
from real_cov_simulation_forecasting import realCov_sliding_forecasting, \
    realCov_sliding_forecasting_error, realCov_compute_model_residuals

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                            "../../PHMC-VAR/src/")))
from utils import compute_LL, compute_prediction_probs, cond_density_sampling, \
     compute_means


#/////////////////////////////////////////////////////////////////////////////

## @fn compute_transition_mat
## @brief Compute the time dependent transition matrices B where B dependents 
#   of covariates Y_t which can be either known or unkown.
#   If Y_t's are unkown, i.e. list_covariates=None B equals the base 
#   transition matrix A at all time-steps.
#
#  @param A, Y_params, covariate_type, M
#  @param series_len List of time series realizations length, initial values 
#   are not included
#  @param list_covariates List of covariates data Y_t where each entry 
#   corresponds to a specific time series. For a time series of length T+order,
#   this is a T x nb_covariates array. 
#  @param list_kappa List of timeseries sampling times, relevant only when 
#   covariate_type=="event.sequences"
#
#  @return List of T_s x M x M 
#
def compute_transition_mat(A, Y_params, covariate_type, M, series_len, \
                           list_covariates, list_kappa):
    # assertion    
    assert(covariate_type in ["real", "event.sequences"])
    
    # number of time series
    N = len(series_len)
    
    if(list_covariates == None):
        list_B = [ -1 * np.ones(dtype=np.float64, shape=(T, M, M)) \
                      for T in series_len ]
        for s in range(N):
            for t in range(series_len[s]):
                list_B[s][t, :, :] = A      
    else:
        # nb covariates
        nb_covariates = list_covariates[0].shape[1]
        
        #assertion
        OK = True
        for s in range(N):
            OK = OK and (series_len[s] == list_covariates[s].shape[0])
        assert(OK)
        
        if(covariate_type == "real"):
            list_B = compute_transition_mat_realCov(A, Y_params, list_covariates, \
                                                    M, nb_covariates, series_len)
        else:
            #assertion
            assert(np.all(list_kappa != None))
            
            list_B = compute_transition_mat_eventSeqCov(A, Y_params["phi"], \
                                        Y_params["delta1"], Y_params["psi"], \
                                        Y_params["delta2"], list_covariates, \
                                        list_kappa, M, nb_covariates, series_len)

    return list_B


#/////////////////////////////////////////////////////////////////////////////
# INFERENCE - GAMMA PROBABILITIES
#/////////////////////////////////////////////////////////////////////////////

## @fn
#  @brief
#
#  @param B T x K x K array
#  @param Pi
#  @param coefficients
#  @param intercepts 
#  @param sigma
#  @param innovation Error term's law
#  @param Obs_seq Sequence to be labelled, Tx1 colomn vector
#  @param weights
#
#  @return 1xT array of inferred states
#
def gamma_probs_based_inference(B, Pi, coefficients, intercepts, sigma, \
                                innovation, Obs_seq):
        
    #compute LL, TxM matrix
    LL = compute_LL(coefficients, intercepts, sigma, innovation, Obs_seq)
    
    #number of states and timeseries length minus AR order
    (T, M) = LL.shape
    
    #compute gamma probabilities
    (_, _, Gamma, _) = modified_FB(M, LL, B, Pi)
    
    #at each time-step the step having the maximum marginal probability is chosen  
    opt_states = np.argmax(Gamma, axis=1)
        
    
    return (Gamma, opt_states.reshape((1,-1)))
    


#/////////////////////////////////////////////////////////////////////////////
# INFERENCE - VITERBI
#/////////////////////////////////////////////////////////////////////////////
## @fun viterbi
#  @brief Maximum A Posterio classification: P(Z,X) = P(Z|X)*P(X)
#   
#  @param B T x K x K array
#  @param Pi
#  @param coefficients
#  @param intercepts 
#  @param sigma
#  @param innovation Error term's law
#  @param Obs_seq Sequence to be labelled, (T+order)x1 colomn vector
#  
#  @return 1xT array of inferred states
#
def viterbi_log(B, Pi, coefficients, intercepts, sigma, innovation, Obs_seq):
    
    #number of states
    M = B.shape[1]     
    #compute LL, TxM matrix
    LL = compute_LL(coefficients, intercepts, sigma, innovation, Obs_seq)
    #effective length of observation sequence
    T = LL.shape[0]
    
    #assertion
    assert(T == B.shape[0])
    
    #------compute log probabilities
    tiny = np.finfo(0.).tiny
    B_log = np.log(B + tiny)
    Pi_log = np.log(Pi + tiny)
    LL_log = np.log(LL + tiny)
    
    #------initialize log probability matrix D_log
    D_log = -np.inf * np.ones(shape=(T,M), dtype=np.float128)
    
    #------initial D probabilities
    for i in range(M):
        D_log[0, i] = Pi_log[0, i] + LL_log[0, i] 
    
    #------compute D for t=1,...,T-1
    for t in range(1, T):       
        for i in range(M):
            temp_sum = B_log[t, :, i] + D_log[t-1, :]
            D_log[t, i] = np.max(temp_sum) + LL_log[t, i] 
    
    #------optimal state computing: backtracking
    opt_states = -1 * np.ones(shape=(1,T), dtype=np.int32)
    opt_states[0, T-1] = np.argmax(D_log[T-1, :])
    
    for t in range(T-2, -1, -1):
        opt_states[0, t] = np.argmax( D_log[t, :] + B_log[t, :, opt_states[0, t+1]] )
     
        
    return opt_states


## @fn 
#  @brief
#
#  @param model_file
#  @param list_Obs_seq List of T_s x dimension array, initial values included
#  @param covariate_type
#  @param list_covariates List of T_s x dimension array
#  @param list_kappa List of timeseries sampling times, relevant only when 
#   covariate_type=="event.sequences"
#  @param method The inference method to be used. Two possible values 
#   'viterbi' and 'gammaProbs'
#  @param set_of_scalers List of scalers to be used to standardize time series.
#   One scaler per time series.
#
#  @return
#
def inference(model_file, list_Obs_seq, covariate_type, list_covariates, \
              list_kappa, method="viterbi", set_of_scalers=None):
    
    if(method != "viterbi" and method != "gammaProbs"):
        print("ERROR: file simulation_inference_forecasting.py: unknown inference method! \n")
        exit(1) 
        
    #model loading
    infile = open(model_file, 'rb')
    model = pickle.load(infile)
    infile.close()
    
    #required model parameters
    innovation = "gaussian"
    A = model[1]
    Pi = model[2]
    ar_coefficients = model[5]
    ar_intercepts = model[7]  
    sigma = model[6]
    Y_params = model[9]
    
    #assertions
    assert(np.sum(A < 0.0) == 0)
    assert(np.sum(Pi < 0.0) == 0) 
    
    #hyper-parameters 
    M = A.shape[0]
    order = len(ar_coefficients[0])
    
    #nb sequence
    N = len(list_Obs_seq)
    # sequences' length
    series_len = [list_Obs_seq[s].shape[0] - order for s in range(N)]
    
    #compute the time dependent transition matrix
    # list of N (T_s x M x M) arrays (the first order initial time-steps are excluded)
    list_B = compute_transition_mat(A, Y_params, covariate_type, M, series_len, \
                                    list_covariates, list_kappa)
    
    #data must be standardized
    standardization = True if (set_of_scalers != None) else False
        
    #-----Inference begins
    #output
    list_states = []
    list_Gamma = []
    
    for s in range(N):
        
        #data standardization
        if(standardization):
            stand_Obs_seq_s = set_of_scalers[s].transform(list_Obs_seq[s])
        else:
            stand_Obs_seq_s = list_Obs_seq[s]
            
        #inference
        if(method == "viterbi"):
            list_states.append( \
                viterbi_log(list_B[s], Pi, ar_coefficients, ar_intercepts, \
                            sigma, innovation, stand_Obs_seq_s) )
        else:
            (Gamma, states) = gamma_probs_based_inference(list_B[s], Pi, \
                                    ar_coefficients, ar_intercepts, sigma, \
                                    innovation, stand_Obs_seq_s) 
            list_states.append(states)
            list_Gamma.append(Gamma)   
    
    #-----outputs
    if(method == "viterbi"):
        return list_states
    else:
        return (list_states, list_Gamma)



#/////////////////////////////////////////////////////////////////////////////
# FORECASTING
#/////////////////////////////////////////////////////////////////////////////
    
## @fn
#
#  @param covariates (T-order) x nb_covariates array of covariates data Y_t 
#   where T stands for the correponding time series length. 
#
def sliding_forecasting(model_file, covariate_type, time_series, H, \
                        covariates, kappa, scaler, use_cov_at_forecast):      
    
    # assertion    
    assert(covariate_type in ["real", "event.sequences"])
    
    if(covariate_type == "event.sequences"):
        forecasts = eventSeqCov_sliding_forecasting(model_file, time_series, \
                                                    H, covariates, kappa, scaler, \
                                                    use_cov_at_forecast)
    else:
        # assertion
        assert(kappa == None)
        
        forecasts = realCov_sliding_forecasting(model_file, time_series, H, \
                                                covariates, scaler, \
                                                use_cov_at_forecast)
        
    return forecasts
        


## @fn
#  @brief Compute several estimates of H-step ahead forecast error. 
#
#  @param begin_t Must be strictly greater than order.
#  @param L Sliding windows size
#  @param set_of_scalers List of scalers to be used to standardize time series.
#   One scaler per time series.
#
#  @return Three arrays where lines correspond to estimations of forecast 
#   error metrics and columns correspond to data dimensions.
#
def sliding_forecasting_error(model_file, covariate_type, set_of_time_series, \
                              H, begin_t, L, set_of_covariates, set_of_kappa, \
                              set_of_scalers, use_cov_at_forecast):  
           
    # assertion    
    assert(covariate_type in ["real", "event.sequences"])
    
    if(covariate_type == "event.sequences"):
        errors = eventSeqCov_sliding_forecasting_error(model_file, \
                        set_of_time_series, H, begin_t, L, set_of_covariates, \
                        set_of_kappa, set_of_scalers, use_cov_at_forecast)
    else:
        # assertion
        assert(set_of_kappa == None)
        
        errors = realCov_sliding_forecasting_error(model_file, \
                        set_of_time_series, H, begin_t, L, set_of_covariates, \
                        set_of_scalers, use_cov_at_forecast)
        
    return errors


#/////////////////////////////////////////////////////////////////////////////
# MODEL RESIDUALS 
#/////////////////////////////////////////////////////////////////////////////
## @fn
#  @brief Compute model residuals. The given data are the one used to trained model
#
def compute_model_residuals(model_file, covariate_type, set_of_time_series, \
                            begin_t, set_of_covariates, set_of_kappa, \
                            set_of_scalers, use_cov_at_forecast):  
           
    # assertion    
    assert(covariate_type in ["real", "event.sequences"])
    
    if(covariate_type == "event.sequences"):
        set_of_residuals = \
            eventSeqCov_compute_model_residuals(model_file, set_of_time_series, \
                                                begin_t, set_of_covariates, \
                                                set_of_kappa, set_of_scalers, \
                                                use_cov_at_forecast)
    else:
        # assertion
        assert(set_of_kappa == None)
        
        set_of_residuals = \
            realCov_compute_model_residuals(model_file, set_of_time_series, \
                                            begin_t, set_of_covariates, \
                                            set_of_scalers, use_cov_at_forecast)
        
    return set_of_residuals



#/////////////////////////////////////////////////////////////////////////////
# SIMULATION: Generate synthetic data from model
#/////////////////////////////////////////////////////////////////////////////



