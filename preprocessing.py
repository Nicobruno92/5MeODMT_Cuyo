#%%
# This magic command allows interactive plotting in a separate window
%matplotlib qt

# Import necessary libraries for the preprocessing
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mne

# Importing libraries for automatic rejection of bad epochs
from autoreject import AutoReject, get_rejection_threshold

# Import helper functions for preprocessing
import utils.preprocessing_helpers as preprocessing_helpers
from utils.log_preprocessing import LogPreprocessingDetails


"""
The following script performs EEG data preprocessing through several steps:
1. Read raw file
2. Band pass filter 1 to 45hz
3. Crop signal from tmin to tmax
4. Visual inspection of channels. Drop bads
5. Epochs of 2s (non-overlapping)
6. Autoreject Epochs
7. Manual inspection of Epochs
8. ICA 
9. Interpolate bad channels
10. Rereferenced to grand average
"""
#%%
# Participant ID and condition being processed
# participant ID  Ex: S02; S40
id = '022'
week = 1
condition = 'baseline' #name of condition that it should be equal to the folder name

# Filename of the raw EEG data
# filename =  id_week.EDF
filename = f"{id}_{str(week)}.EDF"


##################################
#########    FOLDERS    ##########
##################################

# Defining the paths for saving results and raw data
root_path = 'results'
raw_folder = 'raw'
derivatives_folder = 'derivatives'
save_folder = os.path.join(root_path, derivatives_folder, condition, id)

# Create the directory if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Initialize a report to document the preprocessing steps
report = mne.Report(title=f'Preprocessing Subject {id}, for condition {condition} in week {week}')

# Path to the JSON file where preprocessing details will be stored
json_path = 'logs_preprocessing_details_all_subjects.json'

# Initialize the logging class
log_preprocessing = LogPreprocessingDetails(json_path, id, condition, week)

##################################
########   1.READ RAW   ##########
##################################

# Construct the full file path and read the raw EEG data file
raw_file = os.path.join(root_path,  raw_folder, condition, str(id),filename)
raw = mne.io.read_raw_edf(raw_file, preload=True, verbose=False)

# Set the montage (electrode positions)
raw = preprocessing_helpers.set_chs_montage(raw)
print(raw.info)

# set correct info
raw.info['line_freq'] = 50
# raw.info['highpass'] = 1
# raw.info['lowpass'] = 45 

# Plot sensor location in the scalp
# raw.plot_sensors(show_names=True)
# plt.show()

# Add the raw data info to the report
report.add_raw(raw=raw, title='Raw', psd=True)

# Log the raw data info
log_preprocessing.log_detail('info', str(raw.info))

#%% 
# 2.FILTERING
##################################
########    2.FILTERING   ########
##################################

# Apply a band-pass filter to keep frequencies between 1 and 45 Hz
hpass = 1
lpass = 45
raw_filtered = raw.copy().filter(l_freq=hpass, h_freq=lpass)

# Save the filtered data
raw_filtered.save(os.path.join(save_folder, f'{id}-filtered_eeg.fif'), overwrite=True)

# Log the filter settings
log_preprocessing.log_detail('hpass_filter', hpass)
log_preprocessing.log_detail('lpass_filter', lpass)
log_preprocessing.log_detail('filter_type', 'bandpass')

#%%
# 3. Visual Inspection CHs
##################################
###   3.VISUAL INSPECTION  CHs ###
##################################
# Plots PSD of the filtered data
raw_filtered.compute_psd().plot()

# Plot the filtered data for visual inspection to identify bad channels
raw_filtered.plot(scalings = 'auto')
plt.show(block=True)

# Add the filtered data to the report
report.add_raw(raw=raw_filtered, title='Filtered Raw', psd=True)

# Log the identified bad channels
log_preprocessing.log_detail('bad_channels', raw_filtered.info['bads'])

#%% 
# 4. EPOCHING
##################################
#########    4.EPOCHS   ##########
##################################
# Segment the continuous data into epochs of 2 seconds
duration_epochs = 2.0
epochs = mne.make_fixed_length_epochs(raw_filtered, duration=duration_epochs, preload=True, verbose=None)

# Save the epoched data
epochs.save(os.path.join(save_folder, f'{id}-epoched_eeg.fif'), overwrite=True)

# Add the epochs to the report
report.add_epochs(epochs=epochs, title='Epochs')

# Log the number of epochs and their duration
log_preprocessing.log_detail('n_epochs', len(epochs))
log_preprocessing.log_detail('duration_epochs', duration_epochs)

#%% REJECT EPOCHS
##################################
######    REJECT EPOCHS   ########
##################################

# Automatically reject bad epochs using AutoReject
ar = AutoReject(thresh_method='random_search', random_state=42)
epochs_clean = ar.fit_transform(epochs)
reject = get_rejection_threshold(epochs)

# Log the epochs rejected by AutoReject
ar_reject_epochs = [n_epoch for n_epoch, log in enumerate(epochs_clean.drop_log) if log == ('AUTOREJECT',)] 
log_preprocessing.log_detail('autoreject_epochs', ar_reject_epochs)
log_preprocessing.log_detail('autoreject_threshold', reject)
log_preprocessing.log_detail('len_autoreject_epochs', len(ar_reject_epochs))
#%%
epochs_clean = epochs # to skip autoreject  
ar_reject_epochs = [] # to skip autoreject  

# Manually inspect and reject bad epochs
epochs_clean.plot(scalings = 'auto')
plt.show(block=True)

# Log the epochs rejected manually
manual_reject_epochs = [n_epoch for n_epoch, log in enumerate(epochs_clean.drop_log) if log == ('USER',)]
print(f'Manually rejected epochs: {manual_reject_epochs}')
total_epochs_rejected = (len(ar_reject_epochs) + len(manual_reject_epochs)) / len(epochs) * 100
print(f'Total epochs rejected: {total_epochs_rejected}%')
log_preprocessing.log_detail('manual_reject_epochs', manual_reject_epochs)
log_preprocessing.log_detail('len_manual_reject_epochs', len(manual_reject_epochs))

# Plot the drop log for further inspection
epochs_clean.plot_drop_log()

# Add the cleaned epochs to the report
report.add_epochs(epochs=epochs_clean, title='Epochs clean', psd=False)

# Save the cleaned epochs
epochs_clean.drop_bad()
epochs_clean.save(os.path.join(save_folder, f'{id}-cleaned_epochs_eeg.fif'), overwrite=True)

#%% ICA
##################################
######         ICA        ########
##################################


# Parameters for ICA (Independent Component Analysis) to remove artifacts
n_components = 0.999  # Number of components to keep; typically should be higher, like 0.999
method = 'picard'  # The algorithm to use for ICA
max_iter = 512  # Maximum number of iterations; typically should be higher, like 500 or 1000
random_state = 42  # Seed for random number generator for reproducibility

# Initialize the ICA object with the specified parameters
ica = mne.preprocessing.ICA(n_components=n_components, method=method, max_iter=max_iter, random_state=random_state)

# Fit the ICA model to the cleaned epochs
ica.fit(epochs_clean)

# create epochs based on ECG events, find EOG artifacts in the data via pattern
# matching, and exclude the ECG-related ICA components
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw=raw)
ecg_components, ecg_scores = ica.find_bads_ecg(
    inst=ecg_epochs,
    ch_name='ECG',  # a channel close to the eye
    threshold=1  # lower than the default threshold
)
ica.exclude = ecg_components


# (Optional) Plot the ICA components for visual inspection
# ica.plot_components(inst=epochs_clean, picks=range(15))

# Plot the sources identified by ICA
ica.plot_sources(epochs_clean, block=True, show=True)
plt.show(block=True)

# Add the ICA results to the report
report.add_ica(ica, title='ICA', inst=epochs_clean)

# Apply the ICA solution to the cleaned epochs
epochs_ica = ica.apply(inst=epochs_clean)

# Log the ICA parameters and excluded components
log_preprocessing.log_detail('ica_components', ica.exclude)
log_preprocessing.log_detail('ica_method', method)
log_preprocessing.log_detail('ica_max_iter', max_iter)
log_preprocessing.log_detail('ica_random_state', random_state)

# Manually inspect the epochs after ICA application
epochs_ica.plot(scalings = 'auto')
plt.show(block=True)

# Log manually rejected epochs after ICA
all_manual_epochs = [n_epoch for n_epoch, log in enumerate(epochs_ica.drop_log) if log == ('USER',)]
manual_reject_epochs_after_ica = [n_epoch for n_epoch in all_manual_epochs if n_epoch not in manual_reject_epochs]
print(f'Manually rejected epochs after ICA: {manual_reject_epochs_after_ica}')
total_epochs_rejected = (len(ar_reject_epochs) + len(manual_reject_epochs) + len(manual_reject_epochs_after_ica)) / len(epochs) * 100
print(f'Total epochs rejected: {total_epochs_rejected}%')
log_preprocessing.log_detail('manual_reject_epochs_after_ica', manual_reject_epochs_after_ica)
log_preprocessing.log_detail('len_manual_reject_epochs_after_ica', len(manual_reject_epochs_after_ica))
log_preprocessing.log_detail('total_epochs_rejected', total_epochs_rejected)
log_preprocessing.log_detail('epochs_drop_log', epochs_ica.drop_log)
log_preprocessing.log_detail('epochs_drop_log_description', epochs_ica.drop_log)

# Save the epochs after ICA application and drop epochs
epochs_ica.save(os.path.join(save_folder, f'{id}-ica_eeg.fif'), overwrite=True)


#%% 
# Interpolate and Rereference chs
##################################
######   Interpolate chs  ########
##################################
# Interpolate bad channels in the epochs after ICA application
epochs_interpolate = epochs_ica.copy().interpolate_bads()

# Log the interpolated channels
log_preprocessing.log_detail('interpolated_channels', epochs_ica.info['bads'])

##################################
#######    Rereference   #########
##################################
# Rereference the data to the grand average reference
epochs_rereferenced, ref_data = mne.set_eeg_reference(inst=epochs_interpolate, ref_channels='average', copy=True)

# Save the rereferenced epochs
epochs_rereferenced.save(os.path.join(save_folder, f'{id}-rereferenced_eeg.fif'), overwrite=True)

# Add the final epochs to the report
report.add_epochs(epochs=epochs_rereferenced, title='Epochs interpolated and rereferenced', psd=True)

# Log the rereferencing details
log_preprocessing.log_detail('rereference', 'grand_average')


#%%

####################################################################
########        CROP signal into Baseline and Active        ########
####################################################################
# Define the variables for the baseline and dosis times
t_min_baseline = 60  # Start time of the baseline in seconds
t_max_baseline = t_min_baseline + 5 * 60  # End time of the baseline in seconds
t_0_dosis = 700  # Start time of the dosis in seconds
t_max_dosis = t_0_dosis + 18 * 60  # End time of the dosis in seconds

# Calculate the epoch indices for baseline and dosis
# Since each epoch is 2 seconds, divide the times by 2 to get the epoch indices
idx_start_baseline = int(t_min_baseline / 2)
idx_end_baseline = int(t_max_baseline / 2)
idx_start_dosis = int(t_0_dosis / 2)
idx_end_dosis = int(t_max_dosis / 2)

# Select epochs by indices
epochs_baseline = epochs_rereferenced[idx_start_baseline:idx_end_baseline]
epochs_dosis = epochs_rereferenced[idx_start_dosis:idx_end_dosis]

epochs_baseline.save(os.path.join(save_folder, f'{id}-baseline-prepro_eeg.fif'), overwrite=True)
epochs_dosis.save(os.path.join(save_folder, f'{id}-dosis-prepro_eeg.fif'), overwrite=True)

# Log the preprocessing details
log_preprocessing.log_detail('t_min_baseline', t_min_baseline)
log_preprocessing.log_detail('t_max_baseline', t_max_baseline)
log_preprocessing.log_detail('t_0_dosis', t_0_dosis)
log_preprocessing.log_detail('t_max_dosis', t_max_dosis)


# Save the report as an HTML file
report.save(os.path.join(save_folder, f'{id}-report.html'), overwrite=True)

# Save the preprocessing details to the JSON file
log_preprocessing.save_preprocessing_details()

# %%
###########################################################
#######    Optional: Observe Preprocessed Data    #########
###########################################################
epochs_baseline.plot()
plt.show(block=True)

epochs_dosis.plot()
plt.show(block=True)