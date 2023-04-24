#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 7 2022

@author: dama-f
"""

##############################################################################
#  NH-MSAVAR model learning 
#
#  This script take six input parameters
#   * train_data_dir Name of the directory in which training data must be loaded.
#   * nb_time_series The number of time series that should be considered in train_data_dir.
#   * model_output_dir The name of the directory in which the trained model has to be saved.
#   * D Autoregressive order.
#   * K Number of regimes.
#   * features_file The file that contains the features extracted based on 
#     Hawkes process
#   * minus_baseline If true Hawkes baseline are subtracted from features.
#     Two values allowed, 0 for false and 1 for true
#
##############################################################################

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import time
import pickle
from sklearn.preprocessing import StandardScaler

from EM_learning import hmc_var_parameter_learning
from preprocessing import load_simulated_data
    

#=====================================Begin script

#-----remove warning from numba compiler
import warnings

from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


#----------seven command line arguments are required
if(len(sys.argv) != 7):
    print("ERROR: script hawkes_features_extraction_run_training.py takes 6 arguments !")
    print("Usage: ./hawkes_features_extraction_run_training.py train_data_dir ", \
          "nb_time_series model_output_dir D K features_file")
    sys.exit(1)

train_data_dir = sys.argv[1]
nb_time_series = int(sys.argv[2])
model_output_dir = sys.argv[3]
D = int(sys.argv[4]) 
K = int(sys.argv[5])
features_file = sys.argv[6]
minus_baseline = 0



#---------------------------------------------training data loading 
X_dim = 4
parameters = ["FC", "PAS", "PAM", "PAD"]

#----time series and event sequences loading
dir_name = "../../model_validation/data/simulated_data/pegase-patient_150s/"
(timeseries, _, _) = load_simulated_data(train_data_dir, nb_time_series, \
                                         parameters, sampling_time=False) 
    
#----load features  
with open(features_file, 'rb') as f:
    (list_raw_features, baseline, adjacency, categories_name) = pickle.load(f)   
    
    nb_events = len(categories_name.keys())
    
    if(minus_baseline == 0):
        covariates = [ list_raw_features[n] for n in range(nb_time_series) ]
    else:
        covariates = [ list_raw_features[n] - baseline \
                              for n in range(nb_time_series) ]
            
#---------------------------------------------hyper-parameters setting
innovation = "gaussian"
X_order = D
nb_regimes = K
nb_covariates = nb_events
covariate_type = "real"

#---------------------------------------------data preprocessing

# training time series: list of (T_s-X_oder) x X_dim matrices
data_X = []
# X_{1-X_order} to X_0: list of X_order x X_dim matrices
initial_values = []
# covariates: list of (T_s-X_oder) x nb_evnt_type
data_Y = []

for s in range(nb_time_series):
    
    # data standardization
    scaler = StandardScaler(with_mean=True, with_std=True).fit(timeseries[s]) 
    scaled_series_s = scaler.transform(timeseries[s])   
    
    # initial values 
    initial_values.append( scaled_series_s[0:X_order, :] )
    
    # remaining values
    data_X.append( scaled_series_s[X_order:, :] )

    # Covariates are given features
    data_Y.append( covariates[s][X_order:, :])
                 
  
#----compute the number of EM intializations
if nb_regimes == 1:     # VAR case
    n_init = 1
else:
    n_init = 200

            
#----log info       
print("D={}, K={}, N={}".format(D, K, nb_time_series))
print("features_file = ", features_file)
print("train_data_dir = {}, model_output_dir = {}".format(train_data_dir, \
                                                          model_output_dir))
print("Number of initializations = ", n_init)
print()
print("Hawkes categories_name = ", categories_name)
print("Hawkes baseline = ", baseline)
print("Hawkes adjacency = ", adjacency)
print("minus_baseline = ", minus_baseline)


#---------------------------------------------learning  
#starting time 
start_time = time.time()

model_output = hmc_var_parameter_learning (X_dim, X_order, nb_regimes, \
                                           data_X, initial_values, \
                                           innovation, nb_covariates, \
                                           covariate_type, data_Y, \
                                           epsilon=1e-4, nb_iters=500, \
                                           nb_init=n_init, nb_iters_per_init=5)
       

#execution time estimation ends
duration = time.time() - start_time


#----save model_output
output_file = os.path.join(model_output_dir, "") + \
              "model_minus-baseline=" + str(minus_baseline) + \
              "_nb-covariates=" + str(nb_covariates) +  \
              "_K=" + str(nb_regimes) + "_p=" + str(X_order) + \
              "_N=" + str(nb_time_series)
outfile = open(output_file, 'wb') 
pickle.dump(model_output, outfile)
outfile.close()

print("======================================================================")
print("#learningTime: algorithm lastes {} minutes".format(duration/60))
print("======================================================================")


#----learnt parameters
print("---------------------psi-------------------------")
print(model_output[8])
print("------------------AR process----------------------")
print("#total_log_ll= ", model_output[0])
print("ar_coefficients=", model_output[5])
print("intercept=", model_output[7])
print("sigma=", model_output[6])
print()
print("------------------Markov chain----------------------")
print("Pi=", model_output[2])
print("A=", model_output[1])
print("Y_params=", model_output[9])


