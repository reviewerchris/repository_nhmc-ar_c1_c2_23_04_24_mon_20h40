#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:48:29 2022

@author: dama-f
"""

import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                        "../../model_validation/code")))

from preprocessing import load_temporal_event_seq, \
    build_covariates_from_evt_seq, build_event_categories_to_be_used                         
from util_functions import features_visualization_all
from hawkes_process import hawkes_data_preprocessing, from_second_to_time_unit


#----Data loading
train_data_dir = "../../model_validation/data/simulated_data/pegase-patient_150s/"
nb_profiles = 1000

#temporal event sequences 
(time_serie_sampling_time, total_events_stream) = \
                        load_temporal_event_seq(train_data_dir, nb_profiles)
                        
#-----Data preprocessing: timestamps are expressed in minute
time_unit_to_be_used = '5minutes'   
from_s_to_timeunit = from_second_to_time_unit(time_unit_to_be_used)

#build event categories to be used 
# nb_events = 3
# figsize = (8, 5)

# nb_events = 4
# figsize = (8, 7)

nb_events = 5
figsize = (8, 8)

# nb_events = 9
# figsize = (8, 13)

(events_to_be_ignored, event_categories, categories_name) = \
                                 build_event_categories_to_be_used(nb_events)
 
#build data for Hawkes model learning                                         
(hawkes_realizations, _, _, _, _) = \
             hawkes_data_preprocessing(total_events_stream, events_to_be_ignored, \
                                       event_categories, categories_name, \
                                       time_unit_to_be_used)
                 
#----Last occurrence of each event type at all timestep 
(list_last_occ_time, _, _, _) =  \
        build_covariates_from_evt_seq(time_serie_sampling_time, \
                                      total_events_stream, nb_events)
            
# When an event of a specific type never happened in the past, the last
# occurrence time is set at -infty = -1e+300
# For visualization needs, -infty is replaced by -1
#
new_list_last_occ_time = []
nb_realizations = len(list_last_occ_time)

for n in range(nb_realizations):
    new_list_last_occ_time.append( \
        np.where(list_last_occ_time[n] == -1e300, -1, list_last_occ_time[n]))
        
#----Visualization
features_file_name = "./model_outputs/expKernel/5-event-types/" + \
    "features-extracted_from_expKernel-hawkes_nb-events=" + str(nb_events) + ".pkl"

list_features = features_visualization_all(features_file_name, \
                                           hawkes_realizations, \
                                           new_list_last_occ_time, \
                                           time_serie_sampling_time, \
                                           from_s_to_timeunit, \
                                           minus_baseline=True, \
                                           nb_real_to_use=1, \
                                           figsize=figsize, \
                                           xlabel="temps (5 minutes)")