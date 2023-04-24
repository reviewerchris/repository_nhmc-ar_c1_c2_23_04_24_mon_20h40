#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:03:39 2020

@author: dama-f
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

#------------------------------------------------------------------------------
# FUNCTIONS FOR SIMULATED DATA PREPROCESSING
#------------------------------------------------------------------------------      

## @fn
#  @brief Synchonize the given events with the given time series.
#   Time series is steadily sampled while the events are asynchroneous.
#   Events are synchronized with the nearest time-step by superior value.
#   Note that zero, one or several events may be synchronized with the same 
#   time-step.
#
#  @param timeseries T_s x d array. The first column is sampling time expressed
#   in second.
#  @param events T_e x 2 array. The first column is events occurrence time 
#   expressed in second.
#
#  @return A list of T_s entries where each entry corresponds to the list
#   of events synchronized with the corresponding time-step. 
#
def simulated_data_event_synchronization(timeseries, events):
    
    #output
    sync_events = []   
    #local variables
    T_s = timeseries.shape[0]
    T_e = events.shape[0]
    evt_index = 0
    
    for t in range(T_s):
        
        tmp = []
        while((evt_index < T_e) and (timeseries[t,0] >= events[evt_index,0])):
            tmp.append(events[evt_index,1])
            evt_index = evt_index + 1      
            
        sync_events.append(tmp)        
                    
    assert(T_s == len(sync_events))
    
    return sync_events


#  @fn
#  @brief
#
#  @param path string \n
#   Directory from which data have to be loaded. Each file of this directory
#   corresponds to the anesthesia profile of a specific patient.
#  @nb_profiles 
#  @param parameters 1-D array \n
#   List of parameters to be loaded.
#  @param sampling_time If True time series sampling time is retured.
#
#  @return Three/Four lists of dimension nb_profiles \n
#    * timeseries List of T_sxD arrays of physiological parameters \n
#    * events List of T_ex2 arrays of raw event sequences \n
#    * sync_events List of (list of T_s lists) raw event sequences synchronized 
#      with time series sampling frequency. Each entry is a list of 1D arrays 
#      containing synchornized events
#    * time_serie_sampling_time List of T_s-length arrays. This is returned
#      if and only if sampling_time is True.
#
def load_simulated_data(path, nb_profiles, parameters, sampling_time=False):
        
    #initialization of outputs
    timeseries_set = []
    events_set = []
    sync_events_set = []
    time_serie_sampling_time = []
    #timeseries parameters + sampling times
    parameters_ = []
    parameters_.append("Time")
    parameters_.extend(parameters)
    
    #list all entries iwithin path
    for i in range(1, nb_profiles+1):
        
        #file name
        series_file = os.path.join(path, "parameters-FC-PAS-PAM-PAD_series_"+str(i)+".txt")
        events_file = os.path.join(path, "parameters-FC-PAS-PAM-PAD_events_"+str(i)+".txt")                    

        try:
            #---load time series
            ts = np.array(pd.read_csv(series_file, sep=",",  header=0, \
                                                        usecols=parameters_))
            """
            #around parameter values to the nearest integer np.around(x, 0)
            """
            timeseries_set.append(ts[:, 1:])

            time_serie_sampling_time.append(ts[:, 0])
            
            #---load the corresponding event sequence
            es = np.array(pd.read_csv(events_file, sep=",",  header=0, \
                                na_filter=False, usecols=["Time", "Event"])) 
            events_set.append(es)
            
            sync_es = simulated_data_event_synchronization(ts, es)  
            sync_events_set.append(sync_es)
            
        except (FileNotFoundError):
            print("LOG-INFO: while data loading from in directory {}, files {} and/or {} do not exist".format\
                  (path, series_file, events_file))

    if(sampling_time):
        return (timeseries_set, events_set, sync_events_set, \
                time_serie_sampling_time)
    else:
        return (timeseries_set, events_set, sync_events_set)


## @fn
#  @brief Compute anesthesia events and their occurrence times.
#
#  @return Two lists of dimension nb_profiles where each entry corresponds 
#    to a specific patient.
#    * time_serie_sampling_time List of T_s  x 1 matrix. Each entry corresponds
#      to the sampling time (in second) of a specific patient parameters.
#    * events_set with events_set[s] a T_e x 2 matrix where the first 
#      dimension corresponds to event occurrence times (in second) and 
#      the second dimension contains event occurrences.
#
def load_temporal_event_seq(path, nb_profiles):
        
    #initialization of outputs
    events_set = []
    time_serie_sampling_time = []

    #list all entries iwithin path
    for i in range(1, nb_profiles+1):
        
        #file name
        series_file = os.path.join(path, "parameters-FC-PAS-PAM-PAD_series_"+str(i)+".txt")
        events_file = os.path.join(path, "parameters-FC-PAS-PAM-PAD_events_"+str(i)+".txt")                   

        try:           
            #---load time series sampling time
            time = np.array(pd.read_csv(series_file, sep=",",  header=0, \
                                                        usecols=["Time"]))
            time_serie_sampling_time.append(time)
            
            #---load the corresponding event sequence
            es = np.array(pd.read_csv(events_file, sep=",",  header=0, \
                                      na_filter=False, usecols=["Time", "Event"]))                   
            events_set.append(es)
            
        except (FileNotFoundError):
            print("LOG-INFO: while data loading from in directory {}, files {} and/or {} do not exist".format\
                  (path, series_file, events_file))
                    
            
    return (time_serie_sampling_time, events_set)


## @fn build_event_categories
#  @brief 
#
#  @return
#   * events_to_be_ignored
#   * event_categories Dictonary of event categories where keys are 
#     indices associated with categories and values are lists of event names
#   * categories_name Dictionary of categories' name where keys are categories' 
#     index and values are associated names
#
def build_event_categories():
            
    #---initialization of outputs
    event_categories = dict()
    categories_name = dict()
    
    #----Event categories
    #-surgery stimulus, three different levels: minor, medium and major
    minor_stimulus = ["Mise en place du patient", "pose VVP (cathlon)", \
                      "Pose sonde T°C", "Pansement"]
    medium_stimulus = ["intubation", "Incision", "Fermeture"]
    
    major_stimulus = ["Trocard sous-ombilical", \
                      "Dissection espace prépéritonéal", \
                      "Dissection cordon spermatique", \
                      "Dissection cordon zone herniaire", \
                      "Mise en place prothèse"]
    
    #-anesthetic, morphinic drugs, curare
    hypnotic = ["PROPOFOL_induction", "SEVOFLURANE_continu"]
    morphinic = ["SUFENTANIL_induction", "SUFENTANIL_bolus"]
    curare = ["ATRACURIUM_induction", "ATRACURIUM_bolus", "Décurarusation"]
    
    #-events whose effects are fuzzy
    ventilation_control = ["Ventilation contrôlé intubée"]
    inflation_pneumo = ["Inflation pneumopéritoine"]
    deflation_pneumo = ["Déflation pneumopéritoine"]

    #-events to be ignored: set of events having no impact on parameters
    # FC, PAM, PAS and PAD; monitorage events and others controle events 
    # Note that "stop sevoflurane" is an artificial event created to facilitate
    # data simulation (this event does not appear in PEGASE ???)
    events_to_be_ignored = \
                    ["Monitorage FC", "Monitorage PAM", "Monitorage SPO2", \
                  "Monitorage BIS", "Occlusion oculaire", \
                  "Auscultation pulmonaire", "Surveillance pression ballonnet",\
                  "Surveillance points d'appui", "Contrôle de décurarisation", \
                  "Bair hugger", "Manoeuvre Recrutement Alvéolaire", \
                  "CEFAZOLINE_bolus", "pré-oxygénation", "Oxygène 30%", \
                  "stop sevoflurane"] 
        
    #---associate event categories to unique identifiants 
    event_categories[0] = minor_stimulus
    categories_name[0] = "mi.stimulus"
    
    event_categories[1] = medium_stimulus
    categories_name[1] = "me.stimulus"
    
    event_categories[2] = major_stimulus
    categories_name[2] = "ma.stimulus"
    
    event_categories[3] = hypnotic
    categories_name[3] = "hypnotic"  
    
    event_categories[4] = morphinic
    categories_name[4] = "morphinic"
    
    event_categories[5] = curare
    categories_name[5] = "curare"
    
    event_categories[6] = ventilation_control
    categories_name[6] = "vent.control" 
    
    event_categories[7] = inflation_pneumo
    categories_name[7] = "inf.pneumo"
    
    event_categories[8] = deflation_pneumo
    categories_name[8] = "def.pneumo"
    
    return (events_to_be_ignored, event_categories, categories_name)


## @fn build_event_categories_to_be_used
#  @brief 
#  @param nb_events
#
def build_event_categories_to_be_used(nb_events, delay=False):
    
    if (nb_events not in [3, 4, 5, 9]):
        print("ERROR: file preprocessing.py: in function ", \
              "build_event_categories_to_be_used: nb_events/nb_covariates ", \
              "must be within [3, 4, 5, 9]")
        sys.exit(1)
    
    # build the whole event categories
    (events_to_be_ignored, event_categories, categories_name) = \
                                                    build_event_categories()
                                                    
    total_categories = len(event_categories.keys())
    # assertion
    assert(total_categories == 9)
    
    # delay associated with each event categories : dictionary where keys are
    # event categories' index and values are delays expressed in second 
    categorie_impact_delay = dict()
    
    #--9 event type case: all event categories except "descriptors" category  
    if(nb_events == 9):
        # assertion
        assert(not delay)
        return (events_to_be_ignored, event_categories, categories_name)
                         
    elif(nb_events == 5):
        #--5 event type case: 3 pain levels
        new_event_categories = dict()
        new_categories_name = dict()

        for index in range(5):
            new_event_categories[index] = event_categories[index]
            new_categories_name[index]  = categories_name[index]
    
        new_events_to_be_ignored = events_to_be_ignored
        for index in range(5,9):
            new_events_to_be_ignored.extend(event_categories[index])
            
        #--the delay of event categories's impact
        if(delay):
            for i in range(3):
                categorie_impact_delay[i] = 0.
            categorie_impact_delay[3] = 45.
            categorie_impact_delay[4] = 360.
            
    elif(nb_events == 4):
        #--4 event type case: 2 pain levels
        new_event_categories = dict()
        new_categories_name = dict()
        
        # first stimulus level: minor stimulus + medium stimulus
        new_event_categories[0] = [] 
        for index in range(2):
            new_event_categories[0].extend(event_categories[index])    
        new_categories_name[0] = "mi.me.stimulus"

        # second stimulus level: major stimulus
        new_event_categories[1] = event_categories[2]
        new_categories_name[1]  = categories_name[2]
        
        # hypnotic
        new_event_categories[2] = event_categories[3]
        new_categories_name[2]  = categories_name[3]

        # morphinic
        new_event_categories[3] = event_categories[4]
        new_categories_name[3]  = categories_name[4]
    
        new_events_to_be_ignored = events_to_be_ignored
        for index in range(5,9):
            new_events_to_be_ignored.extend(event_categories[index])
            
        #--the delay of event categories's impact
        if(delay):
            categorie_impact_delay[0] = 0.
            categorie_impact_delay[1] = 0.
            categorie_impact_delay[2] = 45.
            categorie_impact_delay[3] = 360.
            
    else:
        #--3 event type case: 1 pain levels
        new_event_categories = dict()
        new_categories_name = dict()

        # a single stimulus type
        new_event_categories[0] = [] 
        for index in range(3):
            new_event_categories[0].extend(event_categories[index])    
        new_categories_name[0] = "stimulus"

        # hypnotic
        new_event_categories[1] = event_categories[3]
        new_categories_name[1]  = categories_name[3]

        # morphinic
        new_event_categories[2] = event_categories[4]
        new_categories_name[2]  = categories_name[4]
    
        new_events_to_be_ignored = events_to_be_ignored
        for index in range(5,9):
            new_events_to_be_ignored.extend(event_categories[index])
            
        #--the delay of event categories's impact
        if(delay):
            categorie_impact_delay[0] = 0.
            categorie_impact_delay[1] = 45.
            categorie_impact_delay[2] = 360.
        
    # assertion
    assert(len(new_event_categories.keys()) == nb_events)
    
    if delay:
        return (new_events_to_be_ignored, new_event_categories, \
                new_categories_name, categorie_impact_delay)
    else:
        return (new_events_to_be_ignored, new_event_categories, \
                new_categories_name)




    
## @fn
#  @brief Build covaraites Y_t from event sequence such that at time step t
#   Y_t is a vecteur of the last occucurrence time of each event category
#   where t denotes time series time-steps.
#
#  @param time_serie_sampling_time List of S T_s-length arrays where S denotes
#   the number of profiles. Time series X_t sampling time.
#  @param events_set List of S T_e x 2 matrices where first column contains
#   occurrence times (expressed in the same time unity as time_serie_sampling_time)
#   and second column represents event occurrences.
#  @param nb_event_types The number of event categories to be considerered.
#   This is also equal to the number of covariates.
#
#  @return List of S T_s x nb_evnt_type matrices.
#
def build_covariates_from_evt_seq_____(time_serie_sampling_time, events_set, \
                                       nb_event_types):

    # local variables
    S = len(time_serie_sampling_time)
    assert( S == len(events_set) )

    # output
    list_Y_data = []

    # build event categories
    (events_to_be_ignored, event_categories, \
     categories_name, categorie_impact_delay) = \
                build_event_categories_to_be_used(nb_event_types, delay=True)
                                                 
    #------Covariate building begins
    for s in range(S):
        # for each event category its occurrence times till time step t
        H_s_t = [ [] for _ in range(nb_event_types)]

        T_s = time_serie_sampling_time[s].shape[0]
        nb_evnt_s = events_set[s].shape[0]
        Y_s_data = []
        evt_ind = 0

        # sth sequence
        for t in range(T_s):
            # X_t sampling time
            kappa_t = time_serie_sampling_time[s][t]
            
            #----update histories
            while((evt_ind < nb_evnt_s) and (kappa_t >= events_set[s][evt_ind,0])):
                evt_occ_time = events_set[s][evt_ind,0]
                evt_name = events_set[s][evt_ind,1]

                #--if relevant event find its category
                if(evt_name not in events_to_be_ignored):
                    evt_cat = -1
                    for l in range(nb_event_types):
                        #found, update history
                        if(evt_name in event_categories[l]):
                            evt_cat = l
                            H_s_t[l].append(evt_occ_time)
                            break
                    #not found
                    if(evt_cat == -1):
                        print("ERROR: Event {} is associated to no categories ".format(evt_name))
                        sys.exit(1)
                # next event
                evt_ind = evt_ind + 1

            #--compute y_t
            Y_s_data.append([ H_s_t[l][-1]  if(len(H_s_t[l]) > 0) else -1e300 \
                              for l in range(nb_event_types) ])

        Y_s_data = np.array(Y_s_data)
        
        #----add event delay to Y_s_data
        for l in range(nb_event_types):
            Y_s_data[:,l] += categorie_impact_delay[l]
        
        list_Y_data.append( Y_s_data )
    
    
    return (list_Y_data, events_to_be_ignored, event_categories, categories_name)


def build_covariates_from_evt_seq(time_serie_sampling_time, events_set, \
                                  nb_event_types):

    # local variables
    S = len(time_serie_sampling_time)
    assert( S == len(events_set) )

    # output
    list_Y_data = []

    # build event categories
    (events_to_be_ignored, event_categories, categories_name) = \
                            build_event_categories_to_be_used(nb_event_types)
                                                 
    #------Covariate building begins
    for s in range(S):
        # for each event category its occurrence times till time step t
        H_s_t = [ [] for _ in range(nb_event_types)]

        T_s = time_serie_sampling_time[s].shape[0]
        nb_evnt_s = events_set[s].shape[0]
        Y_s_data = []
        evt_ind = 0

        # sth sequence
        for t in range(T_s):
            # X_t sampling time
            kappa_t = time_serie_sampling_time[s][t]
            
            #----update histories
            while((evt_ind < nb_evnt_s) and (kappa_t >= events_set[s][evt_ind,0])):
                evt_occ_time = events_set[s][evt_ind,0]
                evt_name = events_set[s][evt_ind,1]

                #--if relevant event find its category
                if(evt_name not in events_to_be_ignored):
                    evt_cat = -1
                    for l in range(nb_event_types):
                        #found, update history
                        if(evt_name in event_categories[l]):
                            evt_cat = l
                            H_s_t[l].append(evt_occ_time)
                            break
                    #not found
                    if(evt_cat == -1):
                        print("ERROR: Event {} is associated to no categories ".format(evt_name))
                        sys.exit(1)
                # next event
                evt_ind = evt_ind + 1

            #--compute y_t
            Y_s_data.append([ H_s_t[l][-1]  if(len(H_s_t[l]) > 0) else -1e300 \
                              for l in range(nb_event_types) ])
       
        list_Y_data.append( np.array(Y_s_data) )
    
    
    return (list_Y_data, events_to_be_ignored, event_categories, categories_name)


#------------------------------------------------------------------------------
# FUNCTIONS FOR REAL-WORLD DATA PREPROCESSING
#------------------------------------------------------------------------------

#  @fn
#  @brief
#
#  @param path string \n
#   Directory from which data have to be loaded. Each file of this directory
#   corresponds to the anesthesia profile of a specific patient.
#  @nb_profiles 
#  @param parameters 1-D array \n
#   List of parameters to be loaded. 
#  @return  Three lists of same dimension \n
#    * timeseries List of numpy array.
#      Each entry corresponds to blood and respiratory parameters of 
#      a single patient \n
#    * gestureEvents List of dataFrame.
#      The corresponding gesture event sequences 
#    * drugEvents List of dataFrame.
#      The corresponding drug event sequences 
#

def load_real_world_data(path, parameters=["Fc"], sub_cohort_file=""):
    
    print("===================================================================================")
    #----load ID of wanted patients
    if(sub_cohort_file != ""):
        print("SUB-COHORT: ", sub_cohort_file)
        data = pd.read_csv(sub_cohort_file, dtype=np.str_, sep="|", header=0, encoding="utf-8")
        sub_cohort = np.int64(data.LIEN.values)
    
    #----output initialization
    observed_data = []
    gestureEvents = []
    drugEvents = []
    #percentage of missed value in time series
    list_nan_percentage = []
    
    #----list the whole data files
    list_files = os.listdir(path)
    list_files.sort()
    nb_files = len(list_files)

    #for each patient, we have three files sorted as follows: id_drugEvents.csv, 
    #id_gestureEvents.csv and id_parameters.csv; where id is the ID of the patients
    #Here we are only interested by the third file that contained time series data
    print("{} patients collected from PEGASE database\n".format( int(nb_files/3) ))

    #--------for each patient
    for ind in range(0, nb_files, 3):   
    
        if(sub_cohort_file != ""):
            ID = np.int64(list_files[ind].split('_')[0]) 
            if(not (ID in sub_cohort)):
                continue
        
        #--------drugEvents loading
        drug_file_name = path + list_files[ind]
        drugData = pd.read_csv(drug_file_name, sep="|", header=0, encoding="utf-8", \
                                usecols=["DATE_DEBUT_", "INTITULE", "TYPE_1"]) 
        drugEvents.append(drugData)
        
        #--------gestureEvents loading
        gesture_file_name = path + list_files[ind+1]
        gestureData = pd.read_csv(gesture_file_name, sep="|", header=0, encoding="utf-8", \
                                    usecols=["DATE_DEBUT_", "LIBELLE", "TYPE_Evenement"])    
        gestureEvents.append(gestureData)
    
        #--------time series loading
        series_file_name = path + list_files[ind+2]
        data = pd.read_csv(series_file_name, sep="|", header=0, encoding="utf-8", \
                        na_values=0, usecols=parameters)      
        #---profils having more that 20% of missed values are ignored
        na_percentage = (data.isnull().sum().max() / len(data.index)) * 100
        na_percentage = int(np.round_(na_percentage, 0))
            
        if(na_percentage > 20):
            print("LOG_INFO: {}% missing values, file {} has been skipped".format(\
                    na_percentage, list_files[ind+2]))    
        else:       
            #--missing values are imputed by the mean
            #1-steps forward fill follows by 1-steps backward fill,
            #until all missed values have been imputed
            #
            nb_na = data.isnull().sum().max()   
            while(nb_na != 0):
                data.fillna(method='ffill', limit=1, inplace=True)
                data.fillna(method='bfill', limit=1, inplace=True)   
                nb_na = data.isnull().sum().max()
                                         
            observed_data.append(np.array(data))
            
        #---add na_percentage
        list_nan_percentage.append(na_percentage)
        
    """print("LOG_INFO: Percentage of missing values : \n", list_nan_percentage)
    np.savetxt("percentage_of_missed_values.csv", list_nan_percentage, encoding='utf-8')"""
    print("===================================================================================")
    
    return (observed_data, gestureEvents, drugEvents)


#------------------------------------------------------------------------------
# FUNCTIONS FOR PEGASE EXPLORATION, UTILS FUNCTIONS
#------------------------------------------------------------------------------
        
#  @fn
#  @brief Date/hour formatting 
#  @param date In format d/%m/%Y
#  @param hour In format %H:%M:%S
#
#  @return date/hour in format %d/%m/%Y-%H:%M:%S
#  
#  NB: when dateTime objects are written within a file or in standard output,
#  the used format is "%Y-%m-%d %H:%M:%S"
# 
def anesthesia_date_parser (date, hour): 
    
    format = "%d/%m/%Y-%H:%M:%S"
    
    if( (hour is np.nan) or (date is np.nan) ):
        return np.nan 
    else:
        h_mn_s = hour.split("-")[0]
        h_mn_s = h_mn_s.split(".")[0]
        splits = h_mn_s.split(":")
        try:
            h = int(splits[0])
            if(h == 24):  #if h = 24, it is replaced by 0
                h_mn_s = "0:" + splits[1] + ":" + splits[2] 
            if(h > 24):
                h_mn_s = "0:0:0"
            
            return datetime.strptime((date+"-"+ h_mn_s) , format)
        
        except:
            return np.nan    
"""
try:
    datetime.strptime((date+"-"+ h_mn_s) , format)
    return date_time
except ValueError:
    return np.nan 
"""

        
#  @fn
#  @param anesth_inter_path The path of Anesth_Intervention_ANO file 
#  @param output_file_cohort File in which IDs are saved
#
def thesaurus_summary(anesth_inter_path, output_file):
    
    data = pd.read_csv(anesth_inter_path, sep="|", header=0, encoding="cp1252", dtype=np.str_)
    
    tmp = data[["CODE_Thesaurus", "INTITULE", "DATE_DEBUT"]]
    thesaurus = tmp.drop_duplicates(subset=["CODE_Thesaurus", "INTITULE"])
    thesaurus.info()
    print(thesaurus)
    
    thesaurus.to_csv(path_or_buf=output_file, sep=',')
    
    return 0
    

        
#  @fn
#
def date_range_anesthesia(anesth_surveillance_path, anesth_evenement_path, anesth_medicament_path): 
    #
    print()
    print("-----------------------------Anesthesia gestures-----------------------------")    
    gestureEvents = pd.read_csv(anesth_evenement_path, dtype=np.str_, \
                    sep="|",  header=0, encoding="cp1252",\
                    parse_dates={ "DATE_DEBUT_": ["DATE_DEBUT", "HEURE_DEBUT"], \
                                  "DATE_saisie_enreg_": ["DATE_saisie_enreg", \
                                                        "HEURE_saisie_enreg"] }, \
                    date_parser=anesthesia_date_parser, \
                    usecols=["LIEN", "DATE_DEBUT", "HEURE_DEBUT", \
                             "DATE_saisie_enreg", "HEURE_saisie_enreg" ])
    ##gestureEvents.info()
    ##print(gestureEvents.head())    
    years = [ gestureEvents["DATE_DEBUT_"][i].year for i in range(gestureEvents.shape[0]) ]
    print("DATE_DEBUT_ range = ", np.unique(years))  
    years = [ gestureEvents["DATE_saisie_enreg_"][i].year for i in range(gestureEvents.shape[0]) ]
    print("DATE_saisie_enreg_ range = ", np.unique(years))   
    #
    print()
    print("-----------------------------Anesthesia drugs-----------------------------")      
    drugEvents = pd.read_csv(anesth_medicament_path, dtype=np.str_, \
                    sep="|",  header=0, encoding="cp1252", \
                    parse_dates={ "DATE_DEBUT_": ["DATE_DEBUT", "HEURE_DEBUT"], \
                                  "DATE_FIN_": ["DATE_FIN", "HEURE_FIN"], \
                                  "DATE_saisie_enreg_": ["DATE_saisie_enreg", \
                                                        "HEURE_Saisie_Enreg"] }, \
                    date_parser=anesthesia_date_parser, \
                    usecols=["LIEN", "DATE_DEBUT", "HEURE_DEBUT", "DATE_FIN", "HEURE_FIN", \
                              "DATE_saisie_enreg", "HEURE_Saisie_Enreg" ])                       
    ##drugEvents.info()
    ##print(drugEvents.head())
    years = [ drugEvents["DATE_DEBUT_"][i].year for i in range(drugEvents.shape[0]) ]
    print("DATE_DEBUT_ range = ", np.unique(years))
    years = [ drugEvents["DATE_FIN_"][i].year for i in range(drugEvents.shape[0]) ]
    print("DATE_FIN_ range = ", np.unique(years))
    years = [ drugEvents["DATE_saisie_enreg_"][i].year for i in range(drugEvents.shape[0]) ] 
    print("DATE_saisie_enreg_ range = ", np.unique(years))    
    #
    print()
    print("-----------------------------Anesthesia monitorage-----------------------------")  
    parameters = pd.read_csv(anesth_surveillance_path, dtype=np.str_, \
                    sep="|",  header=0, encoding="cp1252", \
                    parse_dates={"DATE_ENREG_": ["DATE_ENREG", "HEURE_ENREG"] }, \
                    date_parser=anesthesia_date_parser,\
                    usecols=["LIEN", "DATE_ENREG", "HEURE_ENREG"])                   
    ##parameters.info()
    ##print(parameters.head())
    years = [ parameters["DATE_ENREG_"][i].year for i in range(parameters.shape[0]) ] 
    print("DATE_ENREG_ range = ", np.unique(years))
    
    
    return 0
    

#------------------------------------------------------------------------------
# FUNCTIONS FOR PEGASE EXPLORATION - COHORT EXTRACTION
#------------------------------------------------------------------------------
# COHORT OF SURGERY MAKER - GROUP DATA PER INTERVENTION
# EACH INTERVENTION IS DEPICTED BY THREE FILES: ONE FOR MONITORED PARAMETERS,
# ONE FOR GESTURE TYPE EVENTS AND ONE FOR DRUG TYPE EVENTS.
#------------------------------------------------------------------------------
        
#  @fn
#  @brief Searches in database the IDs (LIEN within the database) of 
#   interventions having the given thesaurus code which corresponds to a single
#   surgery. 
#
#  @param anesth_inter_path The path of Anesth_Intervention_ANO file 
#  @param code_thesaurus List of thesaurus code of the surgery of interest
#  @param output_file_cohort
#
#  @return 
#
def cohort_of_surgery(anesth_inter_path_1, anesth_inter_path_2, list_code_thesaurus, output_file_cohort):
    
    #-----------LOT1
    data = pd.read_csv(anesth_inter_path_1, dtype=np.str_, sep="|", header=0, encoding="cp1252")
    cohort_1 = data[ data.CODE_Thesaurus.isin(list_code_thesaurus)]   
    print("-----------------------------Cohort within LOT1")
    cohort_1.info()
    print()
    print("Lot1 cohort size ", len(cohort_1.index))
    #
    tmp = cohort_1.copy(deep=True)
    tmp.drop_duplicates(inplace=True)
    print("Lot1 cohort size without duplicated rows = ", len(tmp.index))
    #
    tmp.drop_duplicates(subset=['LIEN'], inplace=True)
    print("Lot1 cohort size without duplicated rows and LIEN = ", len(tmp.index))
    
    #-----------LOT2
    data = pd.read_csv(anesth_inter_path_2, dtype=np.str_, sep="|", header=0, encoding="cp1252")
    cohort_2 = data[ data.CODE_Thesaurus.isin(list_code_thesaurus)]    
    print("-----------------------------Cohort within LOT2")
    cohort_2.info()
    print()
    print("Lot2 cohort size ", len(cohort_2.index))
    #
    tmp = cohort_2.copy(deep=True)
    tmp.drop_duplicates(inplace=True) 
    print("Lot2 cohort size without duplicated rows = ", len(tmp.index))
    #
    tmp.drop_duplicates(subset=['LIEN'], inplace=True)
    print("Lot2 cohort size without duplicated rows and LIEN = ", len(tmp.index))
    
    #-----------TOTAL COHORT: LOT1 + LOT2
    total_cohort = pd.concat([cohort_1, cohort_2], ignore_index=True)
    #
    print("---------------------------------Total cohort")
    print("Total cohort size, lot1 + lot2 = ", len(total_cohort.index))
    #
    #remove duplicated rows based on all columns 
    total_cohort.drop_duplicates(inplace=True)     
    print("Total cohort size without duplicated rows = ", len(total_cohort.index) )
    #
    #remove duplicated rows based on column LIEN
    total_cohort.drop_duplicates(subset=['LIEN'], inplace=True)
    print("Total cohort size without duplicated LIEN = ", len(total_cohort.index))  

    #----save cohort
    #total_cohort.sort_values(by=["DATE_DEBUT_"], ascending=True, inplace=True) #TODO: sorted by date
    total_cohort.to_csv(path_or_buf=output_file_cohort, sep='|')
    
    return 0


#  @fn
#  @brief Use given keywords to make cohort instead of list_code_thesaurus.
#   Keywords are searched within surgery "INTITULE"
#
#  @param anesth_inter_path The path of the file containing interventions
#  @param keywords
#  @param output_file_cohort 
#
def make_cohort_from_keywords(anesth_inter_path, keywords, output_file_cohort):
    
    """ keywords = ["hernie", "aine", "inguinale", "inguinal", "laparoscopie"]
                #, \"coelioscopie", "coe", "coelio"] #to be addded"""
    utf8_keywords = []
    for k in keywords:
        #utf8_keywords.append(k.decode('utf-8'))   
        utf8_keywords.append(k) 
    
    data = pd.read_csv(anesth_inter_path, dtype=np.str_, sep="|", header=0, encoding="utf-8")
    
    nb_nans = 0
    indxs = []
    for i in range(data.shape[0]):
    
        if (data.INTITULE[i] is np.nan):
            indxs.append(False)
            nb_nans = nb_nans + 1 
            
        else:                   
            splits = data.INTITULE[i].split(sep=" ") #add further separators "," ";" ...                      
            contain_keywords = np.isin(keywords, splits)
            #data.INTITULE[i] contains at least one keyword
            if np.sum(contain_keywords) > 0:
                indxs.append(True)
            else:
                indxs.append(False)
            
    print("Nb Nan values = {} !".format(nb_nans))
    
    #create and save cohort 
    cohort = data.loc[indxs]      
    cohort.to_csv(path_or_buf=output_file_cohort, sep='|')
    
    print("Cohort size = ", len(cohort.index))

    
    return 0
    
    
#  @fn
#  @brief Split the cohort of "hernie inguinale avec prothèse" in two sub-cohorts:
#   "hernie inguinale unilatéral avec prothèse" and "hernie inguinale bilatérale avec prothèse" 
#
#  @param cohort_file_path The path of
#
def split_cohort_hernie_inguinal(cohort_file_path):

    #keywords
    unilateral_keywords = ["unilaterale", "unilatérale", "unilate", "HID", "HIG"]
    bilateral_keywords = ["bilaterale", "bilatérale", "bilat"]
    
    #output files
    output_file_uni = "unilaterale.csv"
    output_file_bi = "bilaterale.csv"
    
    make_cohort_from_keywords(cohort_file_path, unilateral_keywords, output_file_uni)
    make_cohort_from_keywords(cohort_file_path, bilateral_keywords, output_file_bi)

    return 0
    
    
#  @fn
#
def interventions_of_year_x(cohort_file, year):
      
    data = pd.read_csv(cohort_file, dtype=np.str_, sep="|", header=0, encoding="utf-8")
    
    indxs = []
    for i in range(data.shape[0]):
    
        if (data.DATE_DEBUT[i] is np.nan):
            indxs.append(False)
        else:
            if (int(data.DATE_DEBUT[i].split("/")[2]) == year):
                indxs.append(True)
            else:
                indxs.append(False)
                
    cohort = data.loc[ indxs ] 
    
    return cohort.LIEN.values


#  @fn
#  @brief For each intervention depicts by an unique IDs 
#   (LIEN within the database):
#       * a file containing the records of the monitored parameters is created.
#         These files are named after IDs prefixed by "parameters_". \n
#       * a file containing the records of gesture type events is created.
#         These files are named after IDs prefixed by "gestureEvents_". \n
#       * a file containing the records of drug type events is created.
#         These files are named after IDs prefixed by "drugEvents_". \n
#   For all files, the records are sorted by increasing date-hour. \n
#
#  @param anesth_surveillance_path The path of Anesth_Surveillance file
#  @param anesth_evenement_path The path of Anesth_Evenement file
#  @param anesth_evenement_path The path of Anesth_Evenement file
#  @param output_dir_path
#  @param cohort_file 
#  @param year
#
#  @return 
#
def anesthesia_data(anesth_surveillance_path, anesth_evenement_path, \
                    anesth_medicament_path, output_dir_path, cohort_file, year):
    
    #IDs of interventions that take place at the given year
    list_IDs = interventions_of_year_x(cohort_file, year) 
    
    print()
    print("==================================================== YEAR ", year)
    print("-----------------------------Gesture event data loading...")
    gestureEvents = pd.read_csv(anesth_evenement_path, dtype=np.str_, \
                        sep="|",  header=0, encoding="cp1252", \
                        parse_dates={ "DATE_DEBUT_": ["DATE_DEBUT", "HEURE_DEBUT"], \
                                      "DATE_saisie_enreg_": ["DATE_saisie_enreg", \
                                                             "HEURE_saisie_enreg"] }, \
                        date_parser=anesthesia_date_parser, \
                        usecols=["LIEN", "DATE_DEBUT", "HEURE_DEBUT", \
                                 "DATE_saisie_enreg", "HEURE_saisie_enreg", \
                                 "LIBELLE", "TYPE_Evenement", "ano"])
    #sorted by IDs, date and hour
    gestureEvents.sort_values(by=["LIEN", "DATE_DEBUT_"], ascending=True, inplace=True)
    ##gestureEvents.info()
    ##print(gestureEvents)
    
    print()
    print("-----------------------------Drug event data loading...")
    drugEvents = pd.read_csv(anesth_medicament_path, dtype=np.str_, \
                    sep="|",  header=0, encoding="cp1252", \
                    parse_dates={ "DATE_DEBUT_": ["DATE_DEBUT", "HEURE_DEBUT"], \
                                  "DATE_FIN_": ["DATE_FIN", "HEURE_FIN"] }, \
                    date_parser=anesthesia_date_parser, \
                    usecols=["LIEN", "DATE_DEBUT", "HEURE_DEBUT", "DATE_FIN", "HEURE_FIN", \
                             "INTITULE", "TYPE_1", "ano"]) #DOSE_QUANTITE UNITE VOIE_ADMINISTRATION, CUMUL
                       
    #sorted by IDs, date and hour
    drugEvents.sort_values(by=["LIEN", "DATE_DEBUT_"], ascending=True, inplace=True)
    ##drugEvents.info()
    ##print(drugEvents)
   
    print()
    print("-----------------------------Monitorage data loading...")
    parameters = pd.read_csv(anesth_surveillance_path, dtype=np.str_, \
                    sep="|",  header=0, encoding="cp1252", \
                    parse_dates={"DATE_ENREG_": ["DATE_ENREG", "HEURE_ENREG"] }, \
                    date_parser=anesthesia_date_parser, \
                    usecols=["LIEN", "NO_ORDRE", "DATE_ENREG", "HEURE_ENREG", \
                             "Fc", "PAS", "PAM", "PAD", "ano"])
    #sorted by IDs, date and hour
    parameters.sort_values(by=["LIEN", "DATE_ENREG_"], ascending=True, inplace=True)
    ##parameters.info()
    ##print(parameters)
      
    #for each ID
    for i in range(len(list_IDs)):
        
        print("----------------------- IDs = {} -----------------------".format(list_IDs[i]))
              
        #output file
        file_base_name = os.path.join(output_dir_path, str(np.int64(list_IDs[i])))
                
        #----load the current patient profile
        #gesture events       
        data_gest = gestureEvents[ gestureEvents.LIEN == list_IDs[i] ]     
        #drug events        
        data_drug = drugEvents[ drugEvents.LIEN == list_IDs[i] ]
        #monitored parameters       
        data_param = parameters[ parameters.LIEN == list_IDs[i] ]  
        
        
        #----save the profile
        if (data_gest.empty or data_drug.empty or data_param.empty):
            print("Incomplete profile found. Available data: (gestureEvents, drugEvents, timeSeries) = ({}, {}, {})".format(\
                   not data_gest.empty, not data_drug.empty, not data_param.empty) )
        else:
            data_gest.to_csv(path_or_buf=(file_base_name+"_gestureEvents.csv"), sep='|')
            data_drug.to_csv(path_or_buf=(file_base_name+"_drugEvents.csv"), sep='|')
            data_param.to_csv(path_or_buf=(file_base_name+"_parameters.csv"), sep='|')
           
    return 0

#------------------------------------------------------------------------------
# REAL-WORLD DATA PREPROCESSING BEFORE LEARNING PHMC-LAR
#------------------------------------------------------------------------------          

#
#  @fn
#  @brief For each intervention, create a new single file containing
#   intervention's data, that is time series and events which have been matched.
#   To this end, each event is associated to the nearest timeseries record (par valeur supérieure). 
#   Thus, very close events can be associated to a same timeseries record.
#   Each line of the new file correponds to a timestep. \n
#   Then, for each timestep, event (eventually list of events) is encoded by 
#   an unique interger value named "state". \n
#   The files are named after intervation IDs prefixed by "anesthesie_".
#
#  @param data_dir_path Directory in which anesthesie data are stored.
#   One file per intervention. Each line of this file corresponds to a timestep
#   and contains the values of monitored parameters and the observed event 
#   (gesture or drug) if there is any.
#   
#  @return Create a new directory within "data_dir_path" and store the new 
#   created files within.
#
def associated_event_to_timestep(data_dir_path):
    pass
  
