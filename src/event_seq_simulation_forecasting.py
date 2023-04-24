#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:11:54 2022

@author: dama-f
"""


import numpy as np
import pickle
from scipy.stats import multivariate_normal

import os
import sys
from scaled_modified_forward_backward_recursion import modified_FB
from event_sequence_covariates import compute_transition_mat_eventSeqCov

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                            "../../PHMC-VAR/src/")))
from utils import compute_LL, compute_prediction_probs, compute_means, \
    performance_metrics


#/////////////////////////////////////////////////////////////////////////////

## @fn
#  @brief
#
#  @return TxM matrix of prbabilities Gamma.
#
def compute_gamma_standard_FB_algo(M, LL, B, Pi):
                         
    #the modified FB algorithm for NH-MSVAR
    (_, _, Gamma, _) = modified_FB(M, LL, B, Pi)

    return Gamma


## @fn compute_trans_probs
#
def compute_trans_probs(A, Y_params, M, series_len, covariates, kappa):

    #assertion
    assert(series_len == covariates.shape[0])
    
    # nb covariates
    nb_cov = covariates.shape[1]

    B = compute_transition_mat_eventSeqCov(A, Y_params["phi"], \
                                        Y_params["delta1"], Y_params["psi"], \
                                        Y_params["delta2"], [covariates], \
                                        [kappa], M, nb_cov, [series_len])[0]
    return B


## @fn 
#  @brief Compute state probabilities at forecast horizons
#
#  @param B HxMxM matrix of transition probabilities at forecast horizons
#
def compute_state_probs_at_forecast(Gamma_T, B, H):
    
    #assertion 
    assert(B.shape[0] == H)
    
    #number of states
    M = Gamma_T.shape[0]
    #output
    state_probs = np.zeros(shape=(H, M), dtype=np.float64)
        
    #normalization of Gamma_T
    Gamma_T = Gamma_T / np.sum(Gamma_T)
    
    #--initial case h = 0
    for k in range(M):
        state_probs[0, k] = np.sum(B[0, :, k] * Gamma_T)
        
    state_probs[0, :] = state_probs[0, :] / np.sum(state_probs[0, :])
               
    #--for h = 1, ..., H-1
    for h in range(1, H):
        for k in range(M):
            state_probs[h, k] = np.sum(B[h, :, k] * state_probs[h-1, :])
        
        state_probs[h, :] = state_probs[h, :] / np.sum(state_probs[h, :])
                
    #--assertions
    assert(np.sum(np.isnan(state_probs)) == 0)
    assert(np.sum(state_probs < 0.) == 0)
    assert(np.sum(state_probs > 1.) == 0)
    
    return state_probs



#/////////////////////////////////////////////////////////////////////////////
#     FORECASTING 
#/////////////////////////////////////////////////////////////////////////////

#------------------------------------------BEGIN OPTIMAL POINT FORECAST
    
## @forecasting_one_seq
#  @brief Compute H-step ahead forecasting on the given sequence. 
#   At each forecast horizon h, the expectation of X_{T+h}, knowing its own 
#   past values and those of covariates Y_t, is computed.
#   
#  @param ar_coefficients
#  @param ar_intercepts
#  @param pred_probs H x K array probabilities of states at forecast horizons
#  @param past_values order x dimension array of initial values
#  @param H
#
#  @return A Hxdimension array of predicted values
# 
def forecasting_one_seq(ar_coefficients, ar_intercepts, pred_probs, \
                        past_values, H):
        
    #---hyper-parameters
    nb_regimes = len(ar_coefficients)
    order = len(ar_coefficients[0])
    dimension = ar_intercepts[0].shape[0]
    
    #assertions
    assert(past_values.shape == (order, dimension)) 
    assert(pred_probs.shape == (H, nb_regimes))
 
    #---total X values 
    total_X = np.zeros(shape=(order+H, dimension), dtype=np.float64)    
    total_X[0:order, :] = past_values

    #---forecasting begins
    for t in range(order, H+order, 1):           
        
        #---compute the conditional means of X_t within each regime
        # nb_regimes x dimension array
        means = compute_means(ar_coefficients, ar_intercepts, \
                              total_X[t-order:t, :], nb_regimes, order, \
                              dimension)
        
        #---prediction: weighted sum of conditional means
        for i in range(nb_regimes):
            means[i, :] = means[i, :] * pred_probs[t-order, i]
            
        total_X[t, :] = np.sum(means, axis=0)
                                 
    return total_X[order:, :]

     

#/////////////////////////////////////////////////////////////////////////////
# OPTIMAL H-STEP AHEAD POINT FORECASTING 
#/////////////////////////////////////////////////////////////////////////////
    
## @fn
#
#  @param covariates (T-order) x nb_covariates array of covariates data Y_t 
#   where T stands for the correponding time series length. 
#
def eventSeqCov_sliding_forecasting(model_file, time_series, H, covariates, \
                                    kappa, scaler=None, \
                                    use_cov_at_forecast=False):
    
    print("----------------LOG-INFO: eventSeqCov_sliding_forecasting")
    print("with data scaling = ", True if (scaler != None) else False)
    print("use_cov_at_forecast = ", use_cov_at_forecast)
    print()
    
    #model loading
    infile = open(model_file, 'rb')
    phmc_var = pickle.load(infile)
    infile.close()
    
    #required phmc_var parameters
    innovation = "gaussian"
    A = phmc_var[1]
    Pi = phmc_var[2]
    ar_coefficients = phmc_var[5]
    ar_intercepts = phmc_var[7]     
    sigma = phmc_var[6]
    Y_params = phmc_var[9]
    
    #hyper-parameters 
    order = len(ar_coefficients[0])
    M = A.shape[0]  
    
    #assertions
    assert(np.sum(A < 0.0) == 0)
    assert(np.sum(np.isnan(A)) == 0)
            
    #---compute the time dependent transition probabilities using covariates Y_t
    T = time_series.shape[0]
    # (T-order) x M x M array
    B = compute_trans_probs(A, Y_params, M, (T-order), covariates, kappa)
           
    #---data standardization
    standardization = True if (scaler != None) else False
    
    if(standardization):
        stand_timeseries = scaler.transform(time_series)
    else:
        stand_timeseries = time_series
        
    #---H-step sliding forecasts
    #first projection time-step
    projec_t = order + 1
    nb_projec = 0
    
    #end projection time-step
    end_t = T - H 
    
    #outputs initialization
    total_predictions = stand_timeseries[0:(order+2), :]
        
    while(projec_t < end_t):
        
        #---data splitting
        all_past_values = stand_timeseries[0:(projec_t+1), :]
        order_past_values = stand_timeseries[(projec_t+1-order):(projec_t+1), :]     
        
        #---ll from order to projection time
        LL = compute_LL(ar_coefficients, ar_intercepts, sigma, innovation, \
                        all_past_values)
        #assertion
        assert(LL.shape[0] == (projec_t+1-order))
        
        #---gamma_T
        #index of the last covariable considered in gamma computing 
        last_cov_used = LL.shape[0] - 1
        Gamma_T = compute_gamma_standard_FB_algo(M, LL, B[0:(last_cov_used+1)], \
                                                 Pi)[-1, :]
            
        #---states's probability at forecast horizons
        if(use_cov_at_forecast):
            state_probs_at_forecast = \
                compute_state_probs_at_forecast(Gamma_T, \
                                B[(last_cov_used+1):(last_cov_used+H+1)], H)
        else:               
            state_probs_at_forecast = compute_prediction_probs(A, Gamma_T, H) 

        #---forecasting
        h_step_predictions = forecasting_one_seq(ar_coefficients, \
                                                 ar_intercepts, \
                                                 state_probs_at_forecast, \
                                                 order_past_values, H)
        #assertion
        assert(h_step_predictions.shape[0] == H)
        
        #---forecast error computing 
        total_predictions = np.vstack((total_predictions, h_step_predictions))
        
        #----next projection time, used in prognostic machine health
        projec_t = projec_t + H
        
        nb_projec += 1
             
    #assertion
    assert(total_predictions.shape[0] == (nb_projec*H + order + 2))
    
    #---back to original scale
    if(standardization):
        original_scale_predictions = scaler.inverse_transform(total_predictions)
    else:
        original_scale_predictions = total_predictions
        
    
    return original_scale_predictions



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
def eventSeqCov_sliding_forecasting_error(model_file, set_of_time_series, \
                                          H, begin_t, L, set_of_covariates, \
                                          set_of_kappa, set_of_scalers=None, \
                                          use_cov_at_forecast=False):
    
    print("----------------LOG-INFO: eventSeqCov_sliding_forecasting_error")
    print("H={}, begin_t={}, L={}".format(H, begin_t, L))
    print("with data scaling = ", True if (set_of_scalers != None) else False)
    print("use_cov_at_forecast = ", use_cov_at_forecast)
    print()
    
    #model loading
    infile = open(model_file, 'rb')
    phmc_var = pickle.load(infile)
    infile.close()
    
    #required phmc_var parameters
    innovation = "gaussian"
    A = phmc_var[1]
    Pi = phmc_var[2]
    ar_coefficients = phmc_var[5]
    ar_intercepts = phmc_var[7]     
    sigma = phmc_var[6]
    Y_params = phmc_var[9]
    
    #hyper-parameters 
    order = len(ar_coefficients[0])
    M = A.shape[0]  
       
    #assertions
    assert(np.sum(A < 0.0) == 0)
    assert(np.sum(np.isnan(A)) == 0)
    assert(begin_t > order)
                        
    #---forecasting
    #nb time series
    N = len(set_of_time_series)
    
    #time series's length without initial values
    series_len = [set_of_time_series[s].shape[0] - order for s in range(N)]
    
    #data must be standardized
    standardization = True if (set_of_scalers != None) else False
    
    #outputs
    total_MBias = []
    total_RMSE = [] 
    total_NRMSE = [] 
    total_MAPE = []
                
    for s in range(N):
        
        #time series length
        T_s = set_of_time_series[s].shape[0]
        
        #---first projection time-step
        projec_t = begin_t
                
        #---end projection time-step
        end_t = T_s - H 
        
        #---roling H-step prediction over the s^th time series
        Bias = []
        NBias = []    #normalized bias    

        #---data standardization
        if(standardization):
            stand_timeseries_s = set_of_scalers[s].transform(set_of_time_series[s])
        else:
            stand_timeseries_s = set_of_time_series[s]
            
        #---compute the time dependent transition probabilities
        B = compute_trans_probs(A, Y_params, M, series_len[s], \
                                set_of_covariates[s], set_of_kappa[s])
                            
        while(projec_t < end_t):
            #---data splitting
            all_past_values = stand_timeseries_s[0:(projec_t+1), :]
            order_past_values = stand_timeseries_s[(projec_t+1-order):(projec_t+1), :]   
        
            #---ll from order to projection time
            LL = compute_LL(ar_coefficients, ar_intercepts, sigma, innovation, \
                            all_past_values)
            #assertion
            assert(LL.shape[0] == (projec_t+1-order))
        
            #---gamma_T
            #index of the last covariable considered in prediction 
            last_cov_used = LL.shape[0] - 1
            Gamma_T = compute_gamma_standard_FB_algo(M, LL, \
                                            B[0:(last_cov_used+1)], Pi)[-1, :]
                
            #---states's probability at forecast horizons                   
            if(use_cov_at_forecast):
                state_probs_at_forecast = \
                    compute_state_probs_at_forecast(Gamma_T, \
                                B[(last_cov_used+1):(last_cov_used+H+1)], H)
            else:
                state_probs_at_forecast = compute_prediction_probs(A, Gamma_T, H)
                    
            #---forecasting
            predictions = forecasting_one_seq(ar_coefficients, ar_intercepts, \
                                              state_probs_at_forecast, \
                                              order_past_values, H)
            
            #assertion
            assert(predictions.shape[0] == H)
            
            #---back to original scale
            if(standardization):
                original_scale_predictions = set_of_scalers[s].inverse_transform(predictions)
            else:
                original_scale_predictions = predictions
        
            #---forecast error computing 
            obs_x =  set_of_time_series[s][(projec_t+H), :]
            pred_x = original_scale_predictions[-1, :]
            Bias.append( (obs_x - pred_x) )
            NBias.append( (obs_x - pred_x)/obs_x )         
            
            #----next projection time, used in prognostic machine health
            projec_t = projec_t + L
            
        #NB: from what number of projections it is pertinent to compute RMSE, etc
        #We chose 10 
        if(len(Bias) >= 10):   
            #---forecast error of the s^th time series
            (MBias_, RMSE_, NRMSE_, NMAE_) = performance_metrics(Bias, NBias)
            total_MBias.append(MBias_)
            total_RMSE.append(RMSE_)
            total_NRMSE.append(NRMSE_)
            total_MAPE.append(NMAE_)   
        else:
            print("s={}, nb_projections = {}".format(s, len(Bias)))
                      
    total_MBias = np.vstack(total_MBias)
    total_RMSE = np.vstack(total_RMSE)
    total_NRMSE = np.vstack(total_NRMSE)
    total_MAPE = np.vstack(total_MAPE)   
                       
    return (total_MBias, total_RMSE, total_NRMSE, total_MAPE) 


#/////////////////////////////////////////////////////////////////////////////
#     MODEL RESIDUAL COMPUTING 
#/////////////////////////////////////////////////////////////////////////////

## @fn
#  @brief Compute model residuals 
#
def eventSeqCov_compute_model_residuals(model_file, set_of_time_series, begin_t,\
                                        set_of_covariates, set_of_kappa, \
                                        set_of_scalers=None, \
                                        use_cov_at_forecast=False):
    
    print("----------------LOG-INFO: eventSeqCov_compute_model_residuals")
    print("begin_t={}".format(begin_t))
    print("with data scaling = ", True if (set_of_scalers != None) else False)
    print("use_cov_at_forecast = ", use_cov_at_forecast)
    print()
    
    #model loading
    infile = open(model_file, 'rb')
    phmc_var = pickle.load(infile)
    infile.close()
    
    #required phmc_var parameters
    innovation = "gaussian"
    A = phmc_var[1]
    Pi = phmc_var[2]
    ar_coefficients = phmc_var[5]
    ar_intercepts = phmc_var[7]     
    sigma = phmc_var[6]
    Y_params = phmc_var[9]
    
    #hyper-parameters 
    order = len(ar_coefficients[0])
    M = A.shape[0]  
       
    #assertions
    assert(np.sum(A < 0.0) == 0)
    assert(np.sum(np.isnan(A)) == 0)
                        
    #---forecasting
    #nb time series
    N = len(set_of_time_series)
    
    #time series's length without initial values
    series_len = [set_of_time_series[s].shape[0] - order for s in range(N)]
    
    #data must be standardized
    standardization = True if (set_of_scalers != None) else False
    
    #outputs
    set_of_residuals = []
    H = 1
                
    for s in range(N):
        
        #time series length
        T_s = set_of_time_series[s].shape[0]
        
        #---first projection time-step
        projec_t = begin_t
                
        #---end projection time-step
        end_t = T_s - H
        
        #---roling 1-step prediction over the s^th time series
        Bias = []    

        #---data standardization
        if(standardization):
            stand_timeseries_s = set_of_scalers[s].transform(set_of_time_series[s])
        else:
            stand_timeseries_s = set_of_time_series[s]
            
        #---compute the time dependent transition probabilities
        B = compute_trans_probs(A, Y_params, M, series_len[s], \
                                set_of_covariates[s], set_of_kappa[s])
                            
        while(projec_t < end_t):
            #---data splitting
            all_past_values = stand_timeseries_s[0:(projec_t+1), :]
            order_past_values = stand_timeseries_s[(projec_t+1-order):(projec_t+1), :]   
        
            #---ll from order to projection time
            LL = compute_LL(ar_coefficients, ar_intercepts, sigma, innovation, \
                            all_past_values)
            #assertion
            assert(LL.shape[0] == (projec_t+1-order))
        
            #---gamma_T
            #index of the last covariable considered in prediction 
            last_cov_used = LL.shape[0] - 1
            Gamma_T = compute_gamma_standard_FB_algo(M, LL, \
                                            B[0:(last_cov_used+1)], Pi)[-1, :]
                
            #---states's probability at forecast horizons                   
            if(use_cov_at_forecast):
                state_probs_at_forecast = \
                    compute_state_probs_at_forecast(Gamma_T, \
                                B[(last_cov_used+1):(last_cov_used+H+1)], H)
            else:
                state_probs_at_forecast = compute_prediction_probs(A, Gamma_T, H)
                    
            #---forecasting
            predictions = forecasting_one_seq(ar_coefficients, ar_intercepts, \
                                              state_probs_at_forecast, \
                                              order_past_values, H)
            
            #assertion
            assert(predictions.shape[0] == H)
            
            #---residual at projec_t+1 computed at training data scale
            obs_x =  stand_timeseries_s[(projec_t+H), :]
            pred_x = predictions[-1, :]
            Bias.append( (obs_x - pred_x) )  
            
            #----next projection time, used in prognostic machine health
            projec_t = projec_t + 1
            
        # model residuals for the s^th time series
        set_of_residuals.append(np.array(Bias))
                       
    return set_of_residuals



#/////////////////////////////////////////////////////////////////////////////
# SIMULATION: Generate synthetic data from model
#/////////////////////////////////////////////////////////////////////////////

"""
## @fn event_seq_simulation
#
#  @return 
#   * Simulated event sequence
#   * Associated priors
#
def event_seq_simulation(sampling_times):
    #load hawkes_learner from pickle file, then simulate it 
    pass


## @fn Simulation    
#  @brief
#
#  @param init_values The first initial values. A dxdim matrix where d is 
#   VAR process order and dim is time series dimension
#  @param L Length of the simulated sequence
#  @param coefficients 
#  @param intercept 
#  @param sigma 
#  @param A
#  @param Pi
#  @param innovation 
#  @param sampling_times Time series sampling times
#  
#  @return Simulated couple (timeseries, state_seq)
#
def simulation(init_values, L, coefficients, intercept, sigma, A, Pi, \
               innovation, sampling_times=[]):
        
    #-----hyper-parameters
    #RS-AR order
    order = len(coefficients[0])
    #time series dimension
    dim = init_values.shape[1]      
    #number of classes
    K = A.shape[0]
    assert((init_values.shape[0] == order) and (init_values.shape[1] == dim))
    
    #-----event sequence simulation and weights building
    ####(event_seq, weights) = event_seq_simulation(sampling_times)
    weights = np.ones(dtype=np.float64, shape=(L,K))
    
    #-----output initialization
    total_X = np.ones(shape=(L,dim), dtype=np.float64) * np.nan
    total_X = np.vstack((init_values, total_X))
    selec_states = np.ones(shape=L, dtype=np.int32) * (-1)
    
    #-----initial state probabilities: t=0
    states_probs_t = Pi[0, :] * weights[0, :]
    states_probs_t = states_probs_t / np.sum(states_probs_t)
    
    #-----simulation starts   
    for t in range(order, L+order, 1):  
        
        # conditional mean within each state
        cond_mean = compute_means(coefficients, intercept, \
                                  total_X[t-order:t, :], K, order, dim)
        
        # sample the conditional
        (s, sample) = cond_density_sampling(cond_mean, sigma, innovation, \
                                            states_probs_t)
        selec_states[t-order] = s
        total_X[t, :] = sample
        
        # state probabilities at t+1
        if((t+1-order) < L):
            states_probs_t = A[selec_states[t-order], :] * weights[t+1-order, :]
            states_probs_t = states_probs_t / np.sum(states_probs_t)
        
    return (total_X, selec_states)


## @fn synthetic_data_generation
#  @brief
#
#
def synthetic_data_generation(model_file, innovation, list_L, list_init_values=[]):
    
    #----model loading
    infile = open(model_file, 'rb')
    model = pickle.load(infile)
    infile.close()
    
    #----required model parameters
    A = model[1]
    Pi = model[2]
    ar_coefficients = model[5]
    ar_intercepts = model[7]   
    sigma = model[6]
    psi = model[8]
    
    #assertions
    assert(np.sum(A < 0.0) == 0)  
    assert(np.sum(np.isnan(A)) == 0)
    
    #----hyper-parameters 
    order = len(ar_coefficients[0])
    X_dim = ar_coefficients[0][0].shape[0]
    
    #----output
    syn_data = []
    N = len(list_L)
    
    # if initial values are given
    if(len(list_init_values) != 0):
        given_init_val = True
        assert( len(list_init_values) == N )
    else:
        given_init_val = False
        
    #----Simulation begins
    for n in range(N):
        
        if(given_init_val):
            init_values = list_init_values[n]
        else:
            #initial values simulation
            init_values = np.zeros(shape=(order,X_dim), dtype=np.float64)
            for j in range(1, order+1):
                init_values[order-j, :] = \
                    multivariate_normal.rvs(psi["means"][j-1], psi["covar"], 1) 
        
        #data simulation
        syn_data.append( simulation(init_values, list_L[n], ar_coefficients, \
                                    ar_intercepts, sigma, A, Pi, innovation) )   
        
    return syn_data
"""

