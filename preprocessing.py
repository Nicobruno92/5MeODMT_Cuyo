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
import preprocessing_helpers 

"""
The following script performs EEG data preprocessing through several steps:
1. Read raw file
2. Band pass filter 1 to 90hz
3. Notch filter: at 50hz harmonics
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
id = 6
condition = 'baseline'

# Filename of the raw EEG data
# subject = 'S' + str(id)
filename = "051_1.EDF"


##################################
#########    FOLDERS    ##########
##################################

# Defining the paths for saving results and raw data
root_path = 'results'
raw_folder = 'raw'
derivatives_folder = 'derivatives'
save_folder = os.path.join(root_path, derivatives_folder, condition, str(id))

# Create the directory if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Initialize a report to document the preprocessing steps
report = mne.Report(title=f'Preprocessing S{id} {condition}')

# Path to the JSON file where preprocessing details will be stored
json_path = 'logs_preprocessing_details_all_subjects.json'

# Initialize the logging class
log_preprocessing = preprocessing_helpers.LogPreprocessingDetails(json_path, id, condition)

##################################
########   1.READ RAW   ##########
##################################

# Construct the full file path and read the raw EEG data file
file = os.path.join(root_path,  raw_folder, condition, filename)
raw = mne.io.read_raw_edf(file, preload=True, verbose=False)

# Set the montage (electrode positions)
raw = preprocessing_helpers.set_chs_montage(raw)
print(raw.info)

# Plot sensor location in the scalp
# raw.plot_sensors(show_names=True)
# plt.show()

# Add the raw data info to the report
report.add_raw(raw=raw, title='Raw', psd=True)

# Log the raw data info
log_preprocessing.log_detail('info', str(raw.info))

#%% 2.FILTERING
##################################
########    2.FILTERING   ########
##################################

# Apply a band-pass filter to keep frequencies between 1 and 45 Hz
hpass = 1
lpass = 45
raw_filtered = raw.copy().filter(l_freq=hpass, h_freq=lpass)

# Plots PSD of the raw data
# raw_filtered.plot_psd()

# Save the filtered data
raw_filtered.save(os.path.join(save_folder, f'{id}-filtered_eeg.fif'), overwrite=True)

# Log the filter settings
log_preprocessing.log_detail('hpass_filter', hpass)
log_preprocessing.log_detail('lpass_filter', lpass)
log_preprocessing.log_detail('filter_type', 'bandpass')

##################################
########        CROP      ########
##################################
#set the total duration of file to be equivalent
# raw_filtered.crop(tmin=10, tmax=round_length)
# preprocessing_details.append({'id': id, 'condition':condition, 'crop': {'tmin': 10, 'tmax': 'round_length})

#%%
##################################
###   3.VISUAL INSPECTION  CHs ###
##################################
# Plot the filtered data for visual inspection to identify bad channels
raw_filtered.plot()
plt.show(block=True)

# Add the filtered data to the report
report.add_raw(raw=raw_filtered, title='Filtered Raw', psd=True)

# Log the identified bad channels
log_preprocessing.log_detail('bad_channels', raw_filtered.info['bads'])

#%% EPOCHING
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
# Manually inspect and reject bad epochs
epochs_clean.plot()
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

#%%
##################################
######         ICA        ########
##################################


# Parameters for ICA (Independent Component Analysis) to remove artifacts
n_components = 0.999  # Number of components to keep; typically should be higher, like 0.999
method = 'fastica'  # The algorithm to use for ICA
max_iter = 512  # Maximum number of iterations; typically should be higher, like 500 or 1000
fit_params = dict(fastica_it=5)  # Additional parameters for the 'fastica' method
random_state = 42  # Seed for random number generator for reproducibility

# Initialize the ICA object with the specified parameters
ica = mne.preprocessing.ICA(n_components=n_components, method=method, max_iter=max_iter, random_state=random_state)

# Fit the ICA model to the cleaned epochs
ica.fit(epochs_clean)

# (Optional) Create epochs based on EOG (electrooculogram) events to identify and exclude EOG-related ICA components
# eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw_filtered)
# eog_components, eog_scores = ica.find_bads_eog(
#     inst=eog_epochs,
#     ch_name='Fp1',  # A channel close to the eye
#     threshold=1  # Lower than the default threshold
# )
# ica.exclude = eog_components

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
epochs_ica.plot()
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
log_preprocessing.log_detail('rereferenced_channels', ref_data)

# Save the report as an HTML file
report.save(os.path.join(save_folder, f'{id}-report.html'), overwrite=True)

# Save the preprocessing details to the JSON file
log_preprocessing.save_preprocessing_details()

# %%
###########################################################
#######    Optional: Observe Preprocessed Data    #########
###########################################################
epochs_rereferenced.plot()
plt.show(block=True)