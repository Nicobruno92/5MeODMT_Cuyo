import json
import os
import numpy as np
import mne

class LogPreprocessingDetails:
    def __init__(self, json_path, id, condition):
        self.json_path = json_path
        self.id = str(id)
        self.condition = condition
        self.logs = self.load_preprocessing_details()

    def load_preprocessing_details(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                return json.load(f)
        else:
            return {}

    def save_preprocessing_details(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.logs, f, indent=4)

    def initialize_log_structure(self):
        if self.id not in self.logs:
            self.logs[self.id] = {}
        if self.condition not in self.logs[self.id]:
            self.logs[self.id][self.condition] = {}

    def log_detail(self, key, value):
        self.initialize_log_structure()
        if isinstance(value, np.ndarray):
            value = value.tolist()  # Convert numpy arrays to lists
        self.logs[self.id][self.condition][key] = value

    def get_log(self):
        self.initialize_log_structure()
        return self.logs[self.id][self.condition]


def set_chs_montage(raw):
    rename_dict = {
        'EEG Fp1-Ref': 'Fp1',
        'EEG Fp2-Ref': 'Fp2',
        'EEG F3-Ref': 'F3',
        'EEG F4-Ref': 'F4',
        'EEG C3-Ref': 'C3',
        'EEG C4-Ref': 'C4',
        'EEG P3-Ref': 'P3',
        'EEG P4-Ref': 'P4',
        'EEG O1-Ref': 'O1',
        'EEG O2-Ref': 'O2',
        'EEG F7-Ref': 'F7',
        'EEG F8-Ref': 'F8',
        'EEG T3-Ref': 'T7',
        'EEG T4-Ref': 'T8',
        'EEG T5-Ref': 'P7',
        'EEG T6-Ref': 'P8',
        'EEG A1-Ref': 'A1',
        'EEG A2-Ref': 'A2',
        'EEG Fz-Ref': 'Fz',
        'EEG Cz-Ref': 'Cz',
        'EEG Pz-Ref': 'Pz',
        'ECG': 'ECG',
        'Resp oro-nasal': 'Resp',
        'TORAXIC BELT': 'Toracic',
        'ABDOMINAL BELT': 'Abdominal',
        'MICROPHONE': 'Microphone',
        'EMG-0': 'EMG0',
        'EMG-1': 'EMG1',
        'EMG-2': 'EMG2',
        'EMG-3': 'EMG3',
        'EXT1': 'EXT1',
        'EXT2': 'EXT2'
    }
    
    # Rename the channels
    raw.rename_channels(rename_dict)

    channel_types = {
        'Fp1': 'eeg', 'Fp2': 'eeg', 
        'F3': 'eeg', 'F4': 'eeg', 
        'C3': 'eeg', 'C4': 'eeg', 
        'P3': 'eeg', 'P4': 'eeg', 
        'O1': 'eeg', 'O2': 'eeg', 
        'F7': 'eeg', 'F8': 'eeg', 
        'T7': 'eeg', 'T8': 'eeg', 
        'P7': 'eeg', 'P8': 'eeg', 
        'A1': 'eeg', 'A2': 'eeg',
        'Fz': 'eeg', 'Cz': 'eeg', 
        'Pz': 'eeg', 
        'ECG': 'ecg',
        'Resp': 'resp',
        'Toracic': 'misc',
        'Abdominal': 'misc',
        'Microphone': 'misc',
        'EMG0': 'emg', 'EMG1': 'emg', 'EMG2': 'emg', 'EMG3': 'emg',
        'EXT1': 'stim', 'EXT2': 'stim'
    }
    raw.set_channel_types(channel_types)

    # raw.info['bads'] = ['IO', 'L_EYE', 'muerto1', 'muerto2', 'muerto3', 'muerto4', 'muerto5', 'muerto6', 'muerto7', 'muerto8', 'muerto9', 'muerto10']

    # raw.drop_channels(raw.info['bads'])

    raw = raw.set_montage('standard_1020')
    
    return raw
