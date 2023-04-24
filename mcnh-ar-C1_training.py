#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 22 2022

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
#   * nb_covariates The number of covariates to be used
#
##############################################################################

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import time
import pickle
from sklearn.preprocessing import StandardScaler

from EM_learning import hmc_var_parameter_learning
from preprocessing import load_simulated_data, build_covariates_from_evt_seq


#=====================================Begin script

#-----remove warning from numba compiler
import warnings

from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


#----------six command line arguments are required
if(len(sys.argv) != 6):
    print("ERROR: script eventSeqCovariate_run_training.py takes 6 arguments !")
    print("Usage: ./eventSeqCovariate_run_training.py train_data_dir ", \
          "nb_time_series model_output_dir D K")
    sys.exit(1)

train_data_dir = sys.argv[1]
nb_time_series = int(sys.argv[2])
model_output_dir = sys.argv[3]
D = int(sys.argv[4]) 
K = int(sys.argv[5])
nb_covariates = 5


#---------------------------------------------training data loading 
X_dim = 4
parameters = ["FC", "PAS", "PAM", "PAD"]


#----time series and event sequences loading
dir_name = "../../model_validation/data/simulated_data/pegase-patient_150s/"
(timeseries, events_set, _, time_serie_sampling_time) = \
                    load_simulated_data(train_data_dir, \
                                        nb_time_series, parameters, \
                                        sampling_time=True) 

#----build covariates    
(covariates, _, _, _) =  build_covariates_from_evt_seq(time_serie_sampling_time, \
                                                       events_set, nb_covariates)

#---------------------------------------------hyper-parameters setting
innovation = "gaussian"
X_order = D
nb_regimes = K
covariate_type = "event.sequences"


#---------------------------------------------data preprocessing

# training time series: list of (T_s-X_oder) x X_dim matrices
data_X = []
# X_{1-X_order} to X_0: list of X_order x X_dim matrices
initial_values = []
# time series sampling time: list of (T_s-X_oder) length arrays
kappa_data = []
# covariates: list of (T_s-X_oder) x nb_evnt_type
data_Y = []
            
# original time unit is second: 1 min (60 s) is used
time_scaling = 300  # 5 min (300 s)   #60

for s in range(nb_time_series):
    
    # data standardization
    scaler = StandardScaler(with_mean=True, with_std=True).fit(timeseries[s]) 
    scaled_series_s = scaler.transform(timeseries[s])   
    
    # initial values 
    initial_values.append( scaled_series_s[0:X_order, :] )
    
    # remaining values
    data_X.append( scaled_series_s[X_order:, :] )

    # times are converted from second to minute
    kappa_data.append( time_serie_sampling_time[s][X_order:] / time_scaling)
    data_Y.append( covariates[s][X_order:, :] / time_scaling)
                           
        
#----compute the number of EM intializations
if nb_regimes == 1:     # VAR case
    n_init = 1
else:
    n_init = 200
    
#----log info
print("D={}, K={}, N={}, nb_covariates={}".format(D, K, nb_time_series, nb_covariates))
print("train_data_dir = {}, model_output_dir = {}".format(train_data_dir, model_output_dir))
print("Number of initializations = ", n_init)


#---------------------------------------------learning
        
#starting time 
start_time = time.time()

model_output = hmc_var_parameter_learning (X_dim, X_order, nb_regimes, \
                                           data_X, initial_values, \
                                           innovation, nb_covariates, \
                                           covariate_type, data_Y, kappa_data, \
                                           epsilon=1e-4, nb_iters=500, \
                                           nb_init=n_init, nb_iters_per_init=5)

#execution time estimation ends
duration = time.time() - start_time


#----save model_output
output_file = os.path.join(model_output_dir, "") + "model_nb-covariates=" + \
                str(nb_covariates) + "_K=" + str(nb_regimes) + \
                "_p=" + str(X_order) + "_N=" + str(nb_time_series)
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






