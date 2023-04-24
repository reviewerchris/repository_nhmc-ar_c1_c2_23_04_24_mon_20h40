#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 16:03:28 2022

@author: dama-f
"""

import numpy as np
import pickle

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                        "../../model_validation/code")))

from preprocessing import load_temporal_event_seq, build_event_categories_to_be_used                              
from hawkes_process import hawkes_data_preprocessing, \
    expKernel_hawkes_learner, from_second_to_time_unit
from util_functions import build_weights_from_intensity_fun, \
        build_covariates_from_intensity_fun



#------------------------Data loading
train_data_dir = "../../model_validation/data/simulated_data/pegase-patient_150s/"
nb_profiles = 1000
nb_train = 500

#temporal event sequences 
(time_serie_sampling_time, total_events_stream) = \
                        load_temporal_event_seq(train_data_dir, nb_profiles)

#------------------------Data preprocessing: timestamps are expressed in minute

time_unit_to_be_used = '5minutes'
from_s_to_timeunit = from_second_to_time_unit(time_unit_to_be_used)

#----build event categories to be used 
nb_events = 5
(events_to_be_ignored, event_categories, categories_name) = \
                                build_event_categories_to_be_used(nb_events)

#----build data for Hawkes model learning                                                                                
(hawkes_events_timestamps, _, fil_temp_events_ID, _, _) = \
             hawkes_data_preprocessing(total_events_stream, events_to_be_ignored, \
                                       event_categories, categories_name, \
                                       time_unit_to_be_used)

#----split data 
train_hawkes_events_timestamps = hawkes_events_timestamps[0:nb_train]


#------------------------Feature extraction from ExpKernel Hawkes process    
kernel = "expKernel"

# Selected hyper-parameters - Hawkes ExpKernel
decay = 0.3
mat_decays = decay * np.ones(dtype=np.float64, shape=(nb_events, nb_events))
C = 100

# model learning
hawkes_learner = expKernel_hawkes_learner(train_hawkes_events_timestamps, \
                                          mat_decays, C, verbose=True)

# feature extraction 
Covariates_, timing_ = build_covariates_from_intensity_fun(hawkes_learner, \
                                                 hawkes_events_timestamps, \
                                                 time_serie_sampling_time, \
                                                 from_s_to_timeunit)

"""
# save features
kernel_name = "expKernel"
output_file = "model_outputs/" + kernel_name + "/features-extracted_from_" + \
               kernel_name + "-hawkes_nb-events=" + str(nb_events) + ".pkl"

with open(output_file, 'wb') as outfile:
    pickle.dump((Covariates, hawkes_learner.baseline, \
                 hawkes_learner.adjacency, categories_name), outfile)
"""










#------------------------Weigths building
#output_file_base_name = "prior-probs_expKernel_hawkes_event-types=" + str(nb_event)
#
#assert(nb_event in [3, 5])
#
#if(nb_event == 3):
#    for K in [4, 5, 6]:
#        Weights = build_weights_from_intensity_fun(hawkes_learner, \
#                                               hawkes_events_timestamps, \
#                                               fil_temp_events_ID, \
#                                               time_serie_sampling_time, \
#                                               K, from_s_to_timeunit)
#        
#        # save Weights
#        output_file = output_file_base_name + "_K=" + str(K) + ".pkl"
#        with open(output_file, 'wb') as outfile:
#            pickle.dump((Weights, K), outfile)
#        
#        
#else:
#    K = 6
#    Weights = build_weights_from_intensity_fun(hawkes_learner, \
#                                               hawkes_events_timestamps, \
#                                               fil_temp_events_ID, \
#                                               time_serie_sampling_time, \
#                                               K, from_s_to_timeunit)
#    
#    # save Weights
#    output_file = output_file_base_name + "_K=" + str(K) + ".pkl"
#    with open(output_file, 'wb') as outfile:
#        pickle.dump((Weights, K), outfile)
  
    





