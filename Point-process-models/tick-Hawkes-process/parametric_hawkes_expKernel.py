#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:09:37 2021

@author: dama-f
"""

import numpy as np
import matplotlib.pyplot as plt
from tick.plot import plot_hawkes_kernel_norms, plot_hawkes_kernels

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                        "../../model_validation/code")))

from preprocessing import load_temporal_event_seq, \
    build_event_categories_to_be_used                              
from hawkes_process import expKernel_hawkes_learner, \
    hawkes_data_preprocessing, prediction_assessment, from_second_to_time_unit
from util_functions import plot_intensity_function

#======================================Selected model analysis


#------------Data loading
train_data_dir = "../../model_validation/data/simulated_data/pegase-patient_150s/"
nb_profiles = 1000
nb_train = 500
nb_val = 100

#temporal event sequences 
(time_serie_sampling_time, total_events_stream) = \
                        load_temporal_event_seq(train_data_dir, nb_profiles)


#------------Data preprocessing
time_unit_to_be_used = '5minutes'   
from_s_to_timeunit = from_second_to_time_unit(time_unit_to_be_used)

#----build event categories to be used 
nb_events = 5

(events_to_be_ignored, event_categories, categories_name) = \
                                 build_event_categories_to_be_used(nb_events)
 
#----build data for Hawkes model learning                                         
(_hawkes_events_timestamps, fil_temp_events_name, fil_temp_events_ID, \
 event_categories, categories_name) = \
             hawkes_data_preprocessing(total_events_stream, events_to_be_ignored, \
                                       event_categories, categories_name, \
                                       time_unit_to_be_used)

#----split data 
train_hawkes_events_timestamps = _hawkes_events_timestamps[0:nb_train]   
test_hawkes_events_timestamps = _hawkes_events_timestamps[(nb_train+nb_val):]  

train_event_stream = fil_temp_events_ID[0:nb_train]   
test_event_stream = fil_temp_events_ID[(nb_train+nb_val):]  

del total_events_stream
del fil_temp_events_name


#------------Hawkes process learning for selected hyper-parameters 

#--number of event types
decay = 0.3
mat_decays = decay * np.ones(dtype=np.float64, shape=(nb_events, nb_events))
C = 100

# model learning 
hawkes_learner_Exp = expKernel_hawkes_learner(train_hawkes_events_timestamps, \
                                              mat_decays, C, verbose=True)

#--plot parameters
print()
print("baselines = ", hawkes_learner_Exp.baseline)
print()
print("adjacency = ", hawkes_learner_Exp.adjacency)

#--plot kernel functions
print("Plot kernel functions")
fig, ax = plt.subplots(nb_events, nb_events, figsize=(20,30)) 
plot_hawkes_kernels(hawkes_learner_Exp, ax=ax)

for i in range(nb_events):
    for j in range(nb_events):
        ax[i,j].set_ylabel("", fontsize=2)
        ax[i,j].set_xlabel("", fontsize=2)
#plt.tight_layout(pad=0.1, h_pad=None, w_pad=None, rect=None)
plt.show()

#--plot influence matrix
print("Plot influence matrix")
node_names = categories_name.values()
plot_hawkes_kernel_norms(hawkes_learner_Exp, node_names=node_names)
    
#-Plot intensity function
n = 272
print("Intensity function of event sequence number {}".format(n+1))
hawkes_learner_Exp.plot_estimated_intensity(train_hawkes_events_timestamps[n])

#--Polt score
print("log-likelihood = ", hawkes_learner_Exp.score())


#------------Plot intensity function together
legends = ["douleur faible", "douleur moyenne", "douleur intense", \
                   "hypnotiques", "morphiniques"]
plot_intensity_function(hawkes_learner_Exp, train_hawkes_events_timestamps[n], \
                        time_serie_sampling_time[n][1:,0]/from_s_to_timeunit, \
                        legends)


"""
#------------Model prediction accurracy: event occurrence times are supposed 
# known and we predict their event types

def mean_conf_mat(list_conf_mat):
    N = len(list_conf_mat)
    mean_conf_mat = list_conf_mat[0]
    for i in range(1,N):
        mean_conf_mat += list_conf_mat[i]
    
    return mean_conf_mat/N

(list_acc, list_conf_mat, list_recall, \
     list_f1_score, list_mar) = prediction_assessment(hawkes_learner_Exp, \
                                        train_hawkes_events_timestamps, \
                                        train_event_stream, from_s_to_timeunit, \
                                        max_workers=12)
print("training set prediction ....")
print("  accurracy = ", np.round(np.mean(list_acc), 4), "+/-",\
      np.round(np.std(list_acc), 4))
print("  recall = ", np.round(np.mean(list_recall), 4), "+/-",\
      np.round(np.std(list_recall), 4))
print("  F1-score = ", np.round(np.mean(list_f1_score), 4), "+/-",\
      np.round(np.std(list_f1_score), 4))
print("  conf matrice = \n", np.round(mean_conf_mat(list_conf_mat), 2))
print()

print("  Mean average rank = ", np.round(np.mean(list_mar), 4), "+/-",\
      np.round(np.std(list_mar), 4))
print()
print()


(list_acc, list_conf_mat, list_recall, \
     list_f1_score, list_mar) = prediction_assessment(hawkes_learner_Exp, \
                                        test_hawkes_events_timestamps, \
                                        test_event_stream, from_s_to_timeunit, \
                                        max_workers=12)
print("test set prediction ....")
print("  accurracy = ", np.round(np.mean(list_acc), 4), "+/-",\
      np.round(np.std(list_acc), 4))
print("  recall = ", np.round(np.mean(list_recall), 4), "+/-",\
      np.round(np.std(list_recall), 4))
print("  F1-score = ", np.round(np.mean(list_f1_score), 4), "+/-",\
      np.round(np.std(list_f1_score), 4))
print("  Mean average rank = ", np.round(np.mean(list_mar), 4), "+/-",\
      np.round(np.std(list_mar), 4))

print("  conf matrice = \n", np.round(mean_conf_mat(list_conf_mat), 2))
print()

print("  Mean average rank = ", np.round(np.mean(list_mar), 4), "+/-",\
      np.round(np.std(list_mar), 4))
"""





