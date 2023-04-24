#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:53:30 2021

@author: dama-f
"""

import sys
from sys import exit
import numpy as np
import concurrent.futures

from sklearn.metrics import confusion_matrix, recall_score, f1_score, \
    label_ranking_average_precision_score

# Hawkes process 
from tick.hawkes import HawkesExpKern, HawkesSumGaussians

from util_functions import hawkes_estimated_intensity





##############################################################################
#             DATA PREPROCESSING
##############################################################################

## @fn
#  @brief Conversion of the given time unit to second
#
def from_second_to_time_unit(time_unit):
    
    assert(time_unit == "5minutes")
    
    #Time unit proceding. The original time unit used in the raw data is second
    from_s_to_timeunit = -1
       
    if(time_unit == "second"):
        from_s_to_timeunit = 1
    elif(time_unit == "minute"):
        from_s_to_timeunit = 60
    elif(time_unit == "5minutes"):
        from_s_to_timeunit = 300
    elif(time_unit == "15minutes"):
        from_s_to_timeunit = 900
    elif(time_unit == "hour"):
        from_s_to_timeunit = 3600
    else:
        print("ERROR: hawkes_process.py in function ", \
              "from_second_to_time_unit: the given time_unit is wrong!\n")
        sys.exit(1)
        
    return from_s_to_timeunit
    

#  @fn 
#  @brief Build the input data of the Hawkes process learning algorithm.
#
#  @param temporal_events . 
#  @param time_unit The time unit used to expressed event time-step. 
#   Second is used in the raw data.
#
#  @return
#   * hawkes_events_timestamps List of Hawkes processes realizations. 
#     Each realization is a 
#     list of nb_evnt_types 1D numpy arrays. Namely events[n][j] contains the 
#     occurrence timestamps of event type j within realization n. 
#   * fil_temp_events_name, fil_temp_events_ID 
#     'temporal_events' in which non relevant events have been fiterred out. 
#   * event_categories  Dict
#   * categories_name  Dict
#
def hawkes_data_preprocessing(temporal_events: list, events_to_be_ignored, \
                              event_categories, categories_name, time_unit):
    
    #the original time unit used in the raw data is second
    from_s_to_timeunit = from_second_to_time_unit(time_unit)
                                                    
    # Filter out events within 'events_to_be_ignored' from temporal_events.
    # then remplace event names by category ID
    (fil_temp_events_name, fil_temp_events_ID) = filter_out_events(temporal_events, \
                                                events_to_be_ignored, \
                                                event_categories)

    # Hawke process realization building 
    hawkes_events_timestamps = []
    nb_evnt_types = len(categories_name.keys())
    
    N = len(temporal_events)
    for n in range(N):
        timestamps_n = fil_temp_events_ID[n][:, 0]
        IDs_n = fil_temp_events_ID[n][:, 1]
        tmp_list_n = []        
        
        for j in range(nb_evnt_types):
            indices = np.where( IDs_n == j )
            tmp_list_n.append( timestamps_n[indices] / from_s_to_timeunit )
                  
        #add the n^th realization of Hawkes process
        hawkes_events_timestamps.append(tmp_list_n)
    
    
    return (hawkes_events_timestamps, fil_temp_events_name, \
            fil_temp_events_ID, event_categories, categories_name)
    
    

## @fn
#  @brief Filters out events within 'events_to_be_ignored' from temporal_events.
#   Then remplace event names by category ID.
#
#  @param temporal_events List of T_ex2 matrices. Each matrix corresponds
#   to a specific patient data where the first column is "timestamps" and 
#   second column is event occurrences.
#  @param events_to_be_ignored List
#  @param event_categories Dict
#
#  @return 
#   * List of T_ex2 matrices.  
#   * List of T_ex2 matrices.  
#
def filter_out_events(temporal_events, events_to_be_ignored, event_categories):
    
    # outputs initialization
    output1 = []
    output2 = []
    
    #---filter out events within 'events_to_be_ignored'
    N = len(temporal_events)
    for n in range(N):
        tmp = []
        T_n = temporal_events[n].shape[0] 
        for i in range(T_n):
            if(temporal_events[n][i, 1] not in events_to_be_ignored):
                tmp.append(temporal_events[n][i, :])
                
        output1.append( np.array(tmp) )    
    
    #----remplace event names by category ID
    nb_evnt_types = len(event_categories.keys())
    
    for n in range(N):
        tmp = []
        T_n = output1[n].shape[0]
        
        for i in range(T_n):
            timestamp = output1[n][i, 0]
            evnt_name = output1[n][i, 1]
            evnt_ID = -1
            
            #---found the category event belongs to
            for j in range(nb_evnt_types):
                #found
                if(evnt_name in event_categories[j]):
                    evnt_ID = j
                    break
            #not found
            if(evnt_ID == -1):
                print("ERROR: Event {} is associated to no categories ".format(evnt_name))
                sys.exit(1)
            
            tmp.append( [timestamp, evnt_ID] )
            
        # build output
        output2.append( np.array(tmp) )
            
    return (output1, output2)
    
    

##############################################################################
#      HAWKES PROCESS - Exponential Kernel
##############################################################################
    
#### Sources: https://gitter.im/xdata-tick/Lobby?at=5db99011e1c5e91508ffd25d
#    https://x-datainitiative.github.io/tick/
#    https://github.com/X-DataInitiative/tick/issues/133#issuecomment-495999841
#
### Issue about the optimization of Hawkes processes likelihood 
# It is very complex to optimize Hawkes processes likelihood. There are no 
# convergence guarantee as the loss function is not gradient Lipschitz.
#
## Errors obtain when trying likelihood optimization 
# 1) RuntimeError: The sum of the influence on someone cannot be negative. 
#   Maybe did you forget to add a positive constraint to your proximal operator
# 
# - Explication: I mean the vector at the failing iteration. 
#  Shortly, at each step you do w_i <- max(w - g(w), 0)_i where g is the 
#  gradient of the loss and w is the vector of parameters.
#  If at any step w is full of zero then you will have the error message
#  
## Solutions:
#   1) try better initial parameters. For intance initialize with 
#      'least-squares' estimation ----> Used in the sequel
#   2) try smaller learning rate  
#   3) maybe deactivate linesearch. For 'gd' and 'agd' solvers. 
#      This is done as follows hawkes_learner_Exp._solver_obj.linesearch = False.
#      NB: On my data, the drawback of this method is its very slow 
#      convergence speed. In fact, convergence is not reached even after 
#      10000 iterations.
#
## Available solver definition
#  - 'gd'   Proximal gradient descent
#  - 'agd'  Accelerated proximal  gradient descent
#  - 'bfgs' Broyden, Fletcher, Goldfarb, and Shannon (quasi-newton). It only 
#    supports L2 penalization.
#  - 'svrg' Stochastic Variance Reduced Gradient
#
#  These solvers are all gradient descent algorithms. Therefore, they need 
#  a parameter 'step-size' which defines the step of the gradient descent 
#  at each iteration. This parameter can be fixed or tune automatically.
#
#  In 'gd' and 'agd', backtracking linesearch algorithm can be used to 
#  automatically tune the 'step-size'. 
#
#  Similarly, in 'bfgs' linesearch algorithm is used. Besides, the gradient
#  descent direction is carefully chosen in order to accelerate convergence.  
#
#  However, in 'svrg' is be chosen by Barzilai Borwein rule. This choice is 
#  much more adaptive and should be used if optimal step if difficult to obtain.
#  _hawkes_learner_Exp._solver_obj.step_type = 'bb'
#
## In the subsequent, I use likelihood optimization through 'svrg' solver and
#  elasticnet penalization. The 'step-size' is chosen by Barzilai Borwein rule.
#
    
## @fn
#
def expKernel_hawkes_learner(hawkes_events_timestamps, decay, C, \
                             max_iter=10000, verbose=False):
              
    #solver parameters    
    solver_ = 'gd'      
    tol_ = 1e-12
    step_ = 1e-5 
    
    # Least-square estimates with L2 regularization 
    # NB: in 'likelihood' goodness of fit, the same decay value must be 
    # provided for all event categories
    hawkes_expKernel_LS = HawkesExpKern(decays=decay, C=C,  \
                                        gofit='least-squares', \
                                        penalty='l2', \
                                        solver=solver_, \
                                        tol=tol_, max_iter=max_iter, \
                                        step=step_, verbose=verbose)
    hawkes_expKernel_LS.fit(hawkes_events_timestamps) 

    return hawkes_expKernel_LS


## @fn
#
def expKernel_hawkes_score(train_hawkes_events_timestamps, \
                       val_hawkes_events_timestamps, val_event_stream, \
                       decay, mat_decays, C, time_scaling):
    try:
        # model learning 
        hawkes_learner_Exp = expKernel_hawkes_learner(\
                                            train_hawkes_events_timestamps, \
                                            mat_decays, C, verbose=True)
        log_ll = hawkes_learner_Exp.score()
        
        # predictive performance evaluation
        (list_acc, _, _, _, _) = prediction_assessment(hawkes_learner_Exp, \
                                                val_hawkes_events_timestamps, \
                                                val_event_stream, time_scaling)
        prediction_acc = np.mean(list_acc)
        
    except Exception as e:
        log_ll = -1e300
        prediction_acc = 0.
        print("Failed with: " + str(e))
        print("decay = {}, C = {}".format(decay, C))
    
    
    return (log_ll, prediction_acc, decay, C)


## @fn 
#
def expKernel_hawkes_hyperparam_selec(train_hawkes_events_timestamps, \
                                      val_hawkes_events_timestamps, \
                                      val_event_stream, time_unit):
                            
    print("Exponential Kernel Hawkes: hyper-parameters selection begins.....")
    
    #the original time unit used in the raw data is second
    from_s_to_timeunit = from_second_to_time_unit(time_unit)
    
    #number of events
    nb_event = len(train_hawkes_events_timestamps[0])
    
    #---Parametric learning with the exponential decay kernel and regularization level
    # HYPERPARAMETER SELECTION: decay matrix 
    
    # We suppose the same decay parameters for all co-exciting effect
    # This unique value is chosen in decays_candidates
    decays_candidates = []
    decays_candidates.extend([0.001*i for i in range(1, 10)])
    decays_candidates.extend([0.01*i for i in range(1, 10)])
    decays_candidates.extend([0.1*i for i in range(1, 10)])
    decays_candidates.extend([1.*i for i in range(1, 10)])
    decays_candidates.extend([10.*i for i in range(1, 10)])
    decays_candidates.extend([100.*i for i in range(1, 10)])
    decays_candidates.extend([1000.*i for i in range(1, 11)])
    
    #regularization are searched in 
    C_candidates = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    
    # score for each set of hyperparameters
    SCORE = dict()
    
    #### BEGIN parallel execution 
    futures = []  
    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:  
        
        #compute ll score and event prediction accuracy for values of decay matrix
        for decay in decays_candidates:
                        
            mat_decays = decay * np.ones(dtype=np.float64, shape=(nb_event, nb_event))
             
            for C in C_candidates:
                futures.append( executor.submit(expKernel_hawkes_score, \
                                            train_hawkes_events_timestamps, \
                                            val_hawkes_events_timestamps, \
                                            val_event_stream, \
                                            decay, mat_decays, C, \
                                            from_s_to_timeunit) )                                    
        #collect results as tasks completed
        for f in concurrent.futures.as_completed(futures):
            (log_ll, acc, decay, C) = f.result()  
            SCORE[(decay, C)] = [log_ll, acc]
            
            print("decay = {}, C = {}".format(decay, C))
            
    #### END parallel execution 
    
    return (SCORE, decays_candidates, C_candidates) 


##############################################################################
#  HAWKES PROCESS - Kernel parametrized as a sum of Gaussians
############################################################################## 
    
## @fn
#
#  lasso_grouplasso_ratio
#   * For ratio = 0 this is Group-Lasso regularization
#   * For ratio = 1 this is lasso (L1) regularization
#   * For 0 < ratio < 1, the regularization is a linear combination of Lasso 
#     and Group-Lasso.
#  We use Group-Lasso, lasso_grouplasso_ratio is set at 0.0001 because zero
#  is not accepted by the learning algorithm.
#
def sumGaussians_hawkes_learner(hawkes_events_timestamps, max_mean_gaussian_, \
                                n_gaussians_, C_, max_iter_=100000, \
                                em_max_iter_=500, verbose=False):
    tol_ = 1e-12
    step_size_=1e-7
    
    hawkes_learner_SumGaussians = \
                HawkesSumGaussians(max_mean_gaussian=max_mean_gaussian_, \
                                   n_gaussians=n_gaussians_, C=C_, \
                                   lasso_grouplasso_ratio=0.0001, \
                                   step_size=step_size_, max_iter=max_iter_, \
                                   em_max_iter=em_max_iter_, tol=tol_, \
                                   verbose=verbose) 
    hawkes_learner_SumGaussians.fit(hawkes_events_timestamps)

    return hawkes_learner_SumGaussians


## @fn
#
def sumGaussians_hawkes_score(train_hawkes_events_timestamps, \
                       val_hawkes_events_timestamps, val_event_stream, \
                       max_mean_gaussian, n_gaussians, C, time_scaling):
    
    try:
        # model learning 
        hawkes_learner_SumGaussians = \
                sumGaussians_hawkes_learner(train_hawkes_events_timestamps, \
                                            max_mean_gaussian, n_gaussians, C)
        
        # predictive performance evaluation
        (list_acc, _, _, _, _) = \
                prediction_assessment(hawkes_learner_SumGaussians, \
                                      val_hawkes_events_timestamps, \
                                      val_event_stream, time_scaling)
        acc = np.mean(list_acc)
        
    except Exception as e:
        acc = 0.
        print("=========")
        print("Failed with: " + str(e))
        print("max_mean_gaussian={}, n_gaussians={}, penalty={}".format(\
                  max_mean_gaussian, n_gaussians, C))
    
    return (acc, max_mean_gaussian, n_gaussians, C)


## @fn 
#
def sumGaussians_hawkes_hyperparam_selec(train_hawkes_events_timestamps, \
                                         val_hawkes_events_timestamps, \
                                         val_event_stream, time_unit):
                            
    print("Sum of Gaussians Kernel Hawkes: hyper-parameters selection begins...")
    
    #the original time unit used in the raw data is second
    from_s_to_timeunit = from_second_to_time_unit(time_unit)
    
    #---Kernel parametrized as a sum of Gaussians
    # Hyperparameters are:  max_mean_gaussian, n_gaussians and C
    
    list_max_mean_gaussian = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40]
    list_n_gaussians = [1, 2, 3, 4, 5]
    list_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    
    # scores for different set of hyperparameters
    SCORE = dict()
    
    #### BEGIN parallel execution 
    futures = []  
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:  
        #compute event prediction accuracy for tuple 
        # (max_mean_gaussian, n_gaussians, C)
        for max_mean_gaussian in list_max_mean_gaussian:
            for n_gaussians in list_n_gaussians:
                for C in list_C:
                    futures.append( \
                        executor.submit(sumGaussians_hawkes_score, \
                                        train_hawkes_events_timestamps, \
                                        val_hawkes_events_timestamps, \
                                        val_event_stream, max_mean_gaussian, \
                                        n_gaussians, C, from_s_to_timeunit) )                                    
        #collect results as tasks completed
        for f in concurrent.futures.as_completed(futures):
            (acc, max_mean_gaussian, n_gaussians, C) = f.result()  
            SCORE[(max_mean_gaussian, n_gaussians, C)] = acc
            
            print("max_mean_gaussian={}, n_gaussians={}, penalty={}".format(\
                  max_mean_gaussian, n_gaussians, C))
            
    #### END parallel execution 
    
    return (SCORE, list_max_mean_gaussian, list_n_gaussians, list_C)


##############################################################################
#  HAWKES PROCESS - Non-parametric kernel
############################################################################## 



##############################################################################
#               EVENT PREDICTION 
############################################################################## 

## @fn
#  @brief
#
#  @param event_occ_time Numpy 1-D array. Event occurrence times.
#
#  @return probabilities nb_event x nb_category matrix
#
def event_predic_knowing_occ_time(hawkes_learner, hawkes_realization, \
                                  event_occ_time, n=None):
    
    #---intensity function evaluation
    lambda_t = hawkes_estimated_intensity(hawkes_learner, \
                                          hawkes_realization, event_occ_time)

    #---compute event probabilities
    nb_occ = event_occ_time.shape[0]
    nb_evnt_types = hawkes_learner.n_nodes
    
    probabilities = np.zeros(dtype=np.float64, shape=(nb_occ, nb_evnt_types))
    sum_lambda_t = np.sum(lambda_t, axis=1)
    
    for index in range(nb_occ):
        if(sum_lambda_t[index] != 0):
            probabilities[index, :] = lambda_t[index, :] / sum_lambda_t[index] 
    
    #---compute argmax 
    argmax_events = np.zeros(dtype=np.int32, shape=nb_occ)
    
    for index in range(nb_occ):
        sorted_arg = np.argsort(probabilities[index, :])
        argmax_events[index] = sorted_arg[-1]
                            
        
    return (probabilities, argmax_events, n)
    

## @fn
#  @brief For all metrics, the best value is 1 and the worst value is 0
#
#  @param event_seq 1D array of events represented by their ID
#  @param event_categories
#  @param predictions 1D array
#
def event_predic_accurracy(event_seq, predictions):
    
    #accuracy 
    accurr = np.mean(event_seq == predictions)
    
    #confusion matrix
    conf_mat = confusion_matrix(event_seq, predictions, normalize='true')
    
    #recall = tp /(tp + fn)
    # Intuitively it is the ability of the classifier to find all the positive samples
    recall = recall_score(event_seq, predictions, average='macro')
    
    #F1-score = 2 * (precion * recall) / (precison + recall)
    f1Score = f1_score(event_seq, predictions, average='macro')
            
    return (accurr, conf_mat, recall, f1Score)


## @fn
#  @brief Mean Average Rank precision score
#
def mar_precision_score(event_seq, probabilities):
    
    nb_occ = event_seq.shape[0]
    nb_evnt_types = probabilities.shape[1]    
    indicator_vect_format = np.array( [ [0 if event_seq[n] != j  else 1 for \
                        j in range(nb_evnt_types)] for n in  range(nb_occ) ] )
        
    score = label_ranking_average_precision_score(indicator_vect_format, \
                                                  probabilities)
    
    return score


## @fn
#  @brief
#
def prediction_assessment(hawkes_learner, hawkes_realizations, \
                          event_sequences, time_scaling, max_workers=2):
    
    nb_seq = len(event_sequences)
    #outputs
    list_acc = [ {} for _ in range(nb_seq) ]
    list_conf_mat = [ {} for _ in range(nb_seq) ]
    list_recall = [ {} for _ in range(nb_seq) ]
    list_f1_score = [ {} for _ in range(nb_seq) ]
    list_mar = [ {} for _ in range(nb_seq) ]
    
    #### BEGIN parallel execution 
    futures = [] 
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor: 
        for n in range(nb_seq):
            futures.append( executor.submit(event_predic_knowing_occ_time, \
                                            hawkes_learner, \
                                            hawkes_realizations[n], \
                                            event_sequences[n][:,0]/time_scaling, \
                                            n))
        #collect results as tasks completed
        for f in concurrent.futures.as_completed(futures):
            
            try:
                (probabilities, predictions, n) = f.result()
                
                # accuracy metrics
                tmp = event_predic_accurracy(event_sequences[n][:,1], predictions)
                list_acc[n] = tmp[0] 
                list_conf_mat[n] = tmp[1]
                list_recall[n] = tmp[2]
                list_f1_score[n] = tmp[3]
        
                # MAR
                list_mar[n] = mar_precision_score(event_sequences[n][:,1], \
                                                  probabilities)
                
            except Exception as e:
                print("***********While running prediction_assessment function we got ", \
                      "exception: " + str(e))
                sys.exit(1)
    #### END parallel execution 
    
    return (list_acc, list_conf_mat, list_recall, list_f1_score, list_mar)



