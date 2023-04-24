#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:10:16 2022

@author: dama-f
"""

import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import pickle

from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti, HawkesExpKern


##############################################################################
#       SIMULATION 
##############################################################################

## @fn simulate_Hawkes_expKernel
#  @brief 
#  
#  @param nb_realizations
#
def simulate_Hawkes_expKernel(nb_realizations, index=0, xlabel="", ylabel=""):

    end_time = 100
    
    # parameters
    decays =  1.5 * np.ones(dtype=np.float64, shape=(2, 2)) 
    baseline = [0.2, 0.09] 
    adjacency = [[.2, 0.05], \
                 [0.1, .3]]

    # build process
    hawkes_exp_kernels = SimuHawkesExpKernels(adjacency=adjacency, \
                                              decays=decays, baseline=baseline,
                                              end_time=end_time, \
                                              verbose=False, seed=10)

    #simulation
    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=nb_realizations)

    multi.end_time = [end_time for _ in range(nb_realizations)]
    multi.simulate()

    # fit hawkes learner from simulated data
    hawkes_learner = HawkesExpKern(decays, penalty='l1', C=10)
    hawkes_learner.fit(multi.timestamps)
    
    # plot
    t_max = int(multi.end_time[index])
    intensity_tracked_times = np.array([i for i in range(0, t_max, 5)])
    
    print("index = ", index)
    print("intensity_tracked_times = ", intensity_tracked_times)
    
    plot_intensity_function(hawkes_learner, multi.timestamps[index], \
                            intensity_tracked_times, xlabel, ylabel)
    
    return multi

    
##############################################################################
#       PLOTS 
##############################################################################

## @fn plot_intensity_function
#  @brief For parameter description see function hawkes_estimated_intensity
#
def plot_intensity_function(hawkes_learner, hawkes_realization, \
                            intensity_tracked_times, legends, \
                            xlabel="", ylabel=""):
    
    nb_evnt_types = hawkes_learner.n_nodes
    
    # compute intensity functions at given time stamp
    tracked_intensity = hawkes_estimated_intensity(hawkes_learner, \
                                                   hawkes_realization, \
                                                   intensity_tracked_times)
    # plot 
    (fig, axis) = plt.subplots(figsize=(10,5)) 
    
    # associate a color, a linestyle and a marker to each event type
    list_markers = np.array(['P', 'D', '^', 's', 'v', 'P', 'p', '1', '2', '3', '4', '8'])
    list_colors = ['c', 'orange', 'r', 'b', 'g', 'm', 'y', 'k', 'w', 'deeppink']
    list_linestyles = ['-', '-.', '--', ':', ]
    
    #markersize
    marker_size = 60.
    linewidth = 2.
        
    for cat in range(nb_evnt_types):
        
        axis.plot(intensity_tracked_times, tracked_intensity[:, cat], \
                  color=list_colors[cat], linestyle=list_linestyles[cat], \
                  linewidth=linewidth, label=legends[cat])
        
        #axis.axhline(hawkes_learner.baseline[cat], linestyle=":", \
        #             color=list_colors[cat], lw=1.5, label='_nolegend_')
        
        axis.scatter(hawkes_realization[cat], \
                     np.repeat(hawkes_learner.baseline[cat], hawkes_realization[cat].shape[0]), \
                     color=list_colors[cat], s=marker_size, \
                     marker=list_markers[cat], label="")
    
    plt.xlabel("t", fontsize=20)
    axis.legend(fontsize=18)        
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()   
        
    return 


## @fn max_occurrence_time_distribution
#  @brief
#  @param event_stream List of 2D array where the first column corresponds to
#  occurrence time (expressed in second) and second column corresponds to 
#  event category
#
def max_occurrence_time_distribution(event_stream):
    
    nb_realizations = len(event_stream)
    
    #---maximum occurrence time
    max_occ_time = np.array([ np.max(event_stream[n][:,0]) \
                              for n in range(nb_realizations) ]) / 60
    
    fig, axis = plt.subplots()
    axis.hist(max_occ_time, bins="auto", density=True)
          
    plt.xlabel("max occurrence time (in minute)", fontsize=20)
    plt.ylabel("", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()  
    
    #---total number of events per realization
    event_occ_number = np.array([ event_stream[n].shape[0] \
                                 for n in range(nb_realizations) ])
    
    fig, axis = plt.subplots()
    axis.hist(event_occ_number, bins="auto", density=True)
          
    plt.xlabel("number of events per realization", fontsize=20)
    plt.ylabel("", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()  
    
    
    return (max_occ_time, event_occ_number)
    


## @fn features_visualization_all
#  @brief Note that features are defined as the result of intensity function 
#   evaluated at time series sampling time.
#
#  @param hawkes_realizations Event occurrence times are expressed in timeunit
#  @param list_last_occ_time Expressed in second
#  @param list_serie_sampling_time Expressed in second
#  @param from_s_to_timeunit Conversion of the target timeunit to second
#  @param minus_baseline
#  @param nb_real_to_use 
#
def features_visualization_all(features_file_name, hawkes_realizations, \
                               list_last_occ_time, list_serie_sampling_time, \
                               from_s_to_timeunit,  minus_baseline=False, \
                               nb_real_to_use=1, figsize=(8, 5), xlabel="", \
                               ylabel=""):
    
    with open(features_file_name, 'rb') as f:
        (list_features, baseline, adjacency, categories_name) = pickle.load(f)
    
    nb_events = len(categories_name.keys())
    nb_realizations = len(list_features)
    assert(nb_real_to_use <= nb_realizations)
    
    # print
    print("baseline = ", baseline)
    print("adjacency = ", adjacency)
    
    #----features minus baselines
    if(not minus_baseline):
        new_list_features = list_features
        horizontal_line = baseline
    else:
        new_list_features = [ list_features[n] - baseline \
                              for n in range(nb_realizations) ]
        horizontal_line = np.repeat(0., nb_events)
    
    #----Plot features
    fig, axes = plt.subplots(nb_events, figsize=figsize)
    
    for real in range(nb_real_to_use):
        for cat in range(nb_events):
            axes[cat].plot(list_serie_sampling_time[real][1:, 0] / from_s_to_timeunit, \
                           new_list_features[real][1:, cat])
            
            axes[cat].axhline(horizontal_line[cat], linestyle=":", lw=1.5)
        
            axes[cat].scatter(hawkes_realizations[real][cat], \
                              np.repeat(horizontal_line[cat], \
                                   hawkes_realizations[real][cat].shape[0]))
            
    for cat in range(nb_events):
        axes[cat].set_title(categories_name[cat])
        axes[cat].set_ylim(baseline[cat] - 0.1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    
    #----Scatter plot of features as a function of kappa minus the last 
    # occurrence time time
    fig, axes = plt.subplots(nb_events, figsize=figsize)
    
    for real in range(nb_real_to_use):
        for cat in range(nb_events):
            axes[cat].scatter( \
                (list_serie_sampling_time[real][1:, 0] - \
                 list_last_occ_time[real][1:, cat]) / from_s_to_timeunit, \
                    new_list_features[real][1:, cat])
    
    for cat in range(nb_events):
        axes[cat].set_title(categories_name[cat])
        axes[cat].set_ylim(baseline[cat] - 0.1)
    plt.xlabel("distance between current time and last_occ_time " + xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()  

    #----Plot features distribution
    fig, axes = plt.subplots(nb_events, figsize=figsize)
    Mergerd_features = [new_list_features[n][1:,:] for n in range(nb_real_to_use)]
    Mergerd_features = np.vstack(Mergerd_features)

    for cat in range(nb_events):
        axes[cat].hist(Mergerd_features[:, cat], bins="auto", density=True)
 
    for cat in range(nb_events):
        axes[cat].set_title(categories_name[cat])
    plt.tight_layout()
    plt.show()
    
    
    return new_list_features







##############################################################################
#       UTILS FUNCTIONS
##############################################################################

## @fn hawkes_estimated_intensity
#  @brief Evalutes Hawkes intensity functions at the given times.
#
#  @param hawkes_learner
#  @param hawkes_realization List of nb_evnt_types 1D arrays.
#  @param intensity_tracked_times Numpy 1-D array of times at which intensity 
#   functions have to be evaluated. Expressed in the same unit as 
#   hawkes_realization
#   
#  @return An array of dimension nb_tracked_times x nb_evnt_types. 
#
def hawkes_estimated_intensity(hawkes_learner, hawkes_realization, \
                               intensity_tracked_times):
    
    # output initialization
    nb_evnt_types = hawkes_learner.n_nodes
    nb_tracked_times = intensity_tracked_times.shape[0]
    tracked_intensity = -1 * np.ones(dtype=np.float64, \
                                     shape=(nb_tracked_times, nb_evnt_types))
    
    # local variables
    nb_occ = [hawkes_realization[cat].shape[0] for cat in range(nb_evnt_types)]
    curr_index = [0 for _ in range(nb_evnt_types)]
    event_history = []
    
    for (t, time) in enumerate(intensity_tracked_times):
        
        #---build the history of events that occur before current tracked time
        for cat in range(nb_evnt_types): 
            while( (curr_index[cat] < nb_occ[cat]) and \
                   (hawkes_realization[cat][curr_index[cat]] < time) ):
                
                event_history.append( \
                            (cat, hawkes_realization[cat][curr_index[cat]]) )
                
                curr_index[cat] += 1

        #---compute lambda_t
        for evt_type in range(nb_evnt_types):
            tmp = hawkes_learner.baseline[evt_type]
            
            for (e_j, t_j) in event_history:
                tmp += hawkes_learner.get_kernel_values(evt_type, e_j, \
                                                        np.array([time-t_j]))[0]
            tracked_intensity[t, evt_type] = tmp
            
    #assertion
    assert(np.sum(tracked_intensity < 0.) == 0)
    
    
    return tracked_intensity
    

"""
## @fn
#  @brief Hawkes intensity function evaluation at time series sampling times 
#   for the given realization
#
#  @param hawkes_learner_Exp
#  @param hawkes_realization List of nb_evnt_types 1D arrays.
#  @param time_serie_sampling_time Column vector, expressed in minute.
#
#  @return List of n_nodes 1D numpy array. Where the jth entry correspond to 
#   the jth event type.
#
#  TODO: Evaluation times intensity function are incorrect
#
def lambda_at_timeSeries_sampling_times(hawkes_learner_Exp, hawkes_realization, \
                                        time_serie_sampling_time):
    
    #---intensity function evaluated at both event timestamps and time series 
    #sampling times
    t_max = np.max(time_serie_sampling_time) + 10
    func_values = hawkes_learner_Exp.estimated_intensity(hawkes_realization, \
                                                intensity_track_step=0.125, \
                                                end_time=t_max)
    
    #---intensity function evaluated at time series sampling times 
    
    #local variables and output initialization
    nb_evnt_types = hawkes_learner_Exp.n_nodes
    series_len = time_serie_sampling_time.shape[0]
    intensity_function_eval = [ -1*np.ones(dtype=np.float64, shape=series_len) \
                               for _ in range(nb_evnt_types) ]
    
    #particular case t=0: intensity function is not computed at time-step -1
    t = 0
    for j in range(nb_evnt_types):
        intensity_function_eval[j][t] = 0.
    
    #from t=1
    for t in range(1, series_len):
        tmp = np.argwhere(func_values[1] == time_serie_sampling_time[t,0])[0]
        
        if(len(tmp) == 0):
            print("ERROR: in function evaluate_intensity_function: no evaluation at time {}".format(time_serie_sampling_time[t,0]))
            exit(1)
                
        else:
            t_bis = tmp[0]
            for j in range(nb_evnt_types):
                intensity_function_eval[j][t] = func_values[0][j][t_bis]
                      
    return intensity_function_eval
"""

## @fn lambda_at_timeSeriesSamplingTimes
#  @brief
#
#  @param hawkes_learner
#  @param hawkes_realization
#  @param sampling_times An array of length T. Time series sampling times
#   where the time unit is the one used during Hawkes model training.
#
#  @return series_len x nb_event_category array
#
def lambda_at_timeSeriesSamplingTimes(hawkes_learner, hawkes_realization, \
                                      sampling_time):
    
    ##### My own implementation of function estimated_intensity
    """
    #---lambda_t evaluation from time_1 to time_T
    tracked_intensity = hawkes_estimated_intensity(hawkes_learner, \
                                                   hawkes_realization, \
                                                   sampling_time[1:])
    
    #---manage the special case of time_0 = -1 
    series_len = sampling_time.shape[0]
    nb_evnt_types = hawkes_learner.n_nodes
    intensity_function_eval = -1 * np.ones(dtype=np.float64, \
                                           shape=(series_len, nb_evnt_types))
    
    #particular case t=0: intensity function is not computed at time-step -1
    t = 0
    for j in range(nb_evnt_types):
        intensity_function_eval[t, j] = 0.
    
    #from t=1 to T-1
    intensity_function_eval[1:, :] = tracked_intensity
    
    #assertion
    assert(np.sum(intensity_function_eval < 0.) == 0)
    """
    
    ##### Library Tick implementation of function estimated_intensity, this 
    # function is only available for Hawkes_expKernel
    # The results are equivalent to those obtained with my implementation but
    # the execution is much more fast.
    #
    #---intensity function evaluated at both event timestamps and time series 
    #sampling times
    t_max = np.max(sampling_time) + 10
    func_values = hawkes_learner.estimated_intensity(hawkes_realization, \
                                                intensity_track_step=0.125, \
                                                end_time=t_max)
    
    #---intensity function evaluated at time series sampling times 
    #local variables and output initialization
    nb_evnt_types = hawkes_learner.n_nodes
    series_len = sampling_time.shape[0]
    intensity_function_eval = -1 * np.ones(dtype=np.float64, \
                                           shape=(series_len, nb_evnt_types))
        
    #particular case t=0: intensity function is not computed at time-step -1
    t = 0
    for j in range(nb_evnt_types):
        intensity_function_eval[t,j] = 0.
    
    #from t=1
    for t in range(1, series_len):
        tmp = np.argwhere(func_values[1] == sampling_time[t])[0]
        
        if(len(tmp) == 0):
            print("ERROR: in function evaluate_intensity_function: no evaluation at time {}".format(sampling_time[t,0]))
            exit(1)
                
        else:
            t_bis = tmp[0]
            for j in range(nb_evnt_types):
                intensity_function_eval[t,j] = func_values[0][j][t_bis]
                
    intensity_function_eval = np.array(intensity_function_eval)                  
    
    return intensity_function_eval



##############################################################################
# FROM INTENSITY FUNCTION TO WEIGHTS AND COVARIATES
##############################################################################
    
## @fn sync_eventOcc_with_seriesSamplingTimes
#  @brief Synchronizes event occurrences with time series sampling times.
#   Time series is steadily sampled while the events are asynchroneous.
#   Events are synchronized with the nearest time-step by superior value.
#   Zero, one or several events may be synchronized with the same time-step
#
#  @param event_sequence T_e x 2 array. The first column is events occurrence 
#  times expressed in second.
#  @param sampling_times An array of length T_s. Time series sampling times
#   expressed in second.
#
#  @return A list of T_s entries where each entry corresponds to the list
#   the index of the events synchronized with the corresponding time-step. 
#
def sync_eventOcc_with_seriesSamplingTimes(event_sequence, sampling_times): 
    
    #variables
    T_s = sampling_times.shape[0]
    T_e = event_sequence.shape[0]
    evt_index = 0
    
    #output
    sync_event_occ = [] 
    
    for t in range(T_s):
        
        tmp = []
        while((evt_index < T_e) and \
              (sampling_times[t] >= event_sequence[evt_index,0])):
            tmp.append(evt_index)
            evt_index += 1      
            
        sync_event_occ.append(tmp)        
                    
    assert(T_s == len(sync_event_occ))
    
    return sync_event_occ



## @fn build_weights_from_intensity_fun
#  @brief
#
#  @param hawkes_learner
#  @param hawkes_realization
#  @param event_sequences
#  @param list_time_serie_sampling_times List of column vectors.
#  @param K
#  @param time_scaling Coversion from the timeunit used in the training stage
#   to second 
#
#  @return A list of 2D arrays 
# 
def build_weights_from_intensity_fun(hawkes_learner, hawkes_realizations, \
                                     event_sequences, \
                                     list_time_serie_sampling_times, \
                                     K, time_scaling): 
    
    #number of profiles
    N = len(event_sequences)    
    assert(N == len(hawkes_realizations))
    assert(N == len(list_time_serie_sampling_times))
    
    #output
    Weights = [ {} for _ in range(N) ]
    
    #### BEGIN parallel execution
    futures = [] 
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        # computes Weights for each realization 
        for n in range(N):
            futures.append( executor.submit(__build_weights_from_intensity_fun, \
                                            hawkes_learner,  \
                                            hawkes_realizations[n], \
                                            event_sequences[n], \
                                            list_time_serie_sampling_times[n], \
                                            K, time_scaling, n) ) 
        #collect results as tasks completed
        for f in concurrent.futures.as_completed(futures):
            (w, n) = f.result()
            Weights[n] = w
            
            print("n = ", n)
    #### END parallel execution 
    
    return Weights


def __build_weights_from_intensity_fun(hawkes_learner, hawkes_realization, \
                                       event_sequence, \
                                       time_serie_sampling_times, K, \
                                       time_scaling, n):
        
    #----intensity function evaluation at event occurrence times 
    intensity_func_eval = hawkes_estimated_intensity(hawkes_learner, \
                                        hawkes_realization, \
                                        event_sequence[:,0]/time_scaling)
    
    #----event occurrences synchronization with time series sampling times 
    event_occ_sync = \
        sync_eventOcc_with_seriesSamplingTimes(event_sequence, \
                                        time_serie_sampling_times[:,0]) 
    
    #----compute weights at each time-step
    nb_evnt_types = hawkes_learner.n_nodes
    series_len = time_serie_sampling_times.shape[0]
    weights = np.zeros(dtype=np.float64, shape=(series_len, K))
    
    for t in range(series_len):
            
        #--no events have been synchronized with time-step t
        # therefore equal weights are assigned to all states
        if(len(event_occ_sync[t]) == 0):
            weights[t, :] = np.repeat(1., K) / K
                
        else:
            #--at least one event have been synchronized with time-step t 
            # lambda evaluations at the occurrence times of events 
            # synchronized with time-step t
            lambda_events_sync_t = np.vstack( \
                                        [ intensity_func_eval[index, :] \
                                        for index in event_occ_sync[t] ] )
            
            lambda_t = np.sum(lambda_events_sync_t, axis=0)
            sum_lambda_t = np.sum(lambda_t)
            
            for j in range(nb_evnt_types):
                weights[t, j] = lambda_t[j] / sum_lambda_t
                                   
    return (weights, n)





## @fn build_covariates_from_intensity_fun
#  @brief  
#
#  @param hawkes_learner
#  @param hawkes_realization
#  @param list_time_serie_sampling_times List of column vectors.
#  @param time_scaling Coversion from the timeunit used in the training stage
#   to second 
#
#  @return A list of 2D arrays
#
def build_covariates_from_intensity_fun(hawkes_learner, \
                                        hawkes_realizations, \
                                        list_time_serie_sampling_times, \
                                        time_scaling): 
    
    #number of profiles
    N = len(list_time_serie_sampling_times)
    assert(N == len(hawkes_realizations))
    
    #output
    Covariates = [ {} for _ in range(N) ]
    
    timing = []
    
    #### BEGIN parallel execution
    futures = [] 
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        #compute covariates for each realization 
        for n in range(N):
            futures.append( \
                    executor.submit(__build_covariates_from_intensity_fun, \
                                    hawkes_learner, hawkes_realizations[n], \
                                    list_time_serie_sampling_times[n], \
                                    time_scaling, n) )
        #collect results as tasks completed
        for f in concurrent.futures.as_completed(futures):
            (covar, n, duration) = f.result()
            Covariates[n] = covar
            
            timing.append(duration)            
    #### END parallel execution 
        
    return Covariates, timing

import time 
def __build_covariates_from_intensity_fun(hawkes_learner, \
                                          hawkes_realization, \
                                          time_serie_sampling_times, \
                                          time_scaling, n):
    #start timing
    start_time = time.time()
    
    covariates = lambda_at_timeSeriesSamplingTimes(hawkes_learner, \
                                                   hawkes_realization, \
                                time_serie_sampling_times[:,0]/time_scaling)

    #end timing
    duration = time.time() - start_time
        
    return (covariates, n, duration)
