import numpy as np
import mne
from io import StringIO
class read_raw_akonic:
    def __init__(self, eeg_path):
        self.eeg_path = eeg_path
        self.raw = self._read_raw_akonic()

    def _read_raw_akonic(self):
        # Load the data
        data = self.load_data()
        
        # Check if the number of channels matches the expected number (32 channels)
        if data.shape[0] != 32:  # Assuming the first dimension is the number of channels
            raise ValueError(f"Expected 32 channels, but got {data.shape[0]}")

        # Define the channel types - you'll need to adjust this based on your data
        ch_types = ['eeg'] * 32

        # Define the channel names
        ch_names = [
            'Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'A1',
            'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'A2',
            'Fpz', 'Fz', 'Cz',
            'EKG', 'AF', 'TOR', 'ABD', 'MIC', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'EXT1', 'EXT2'
        ]

        # Create the info structure needed by MNE
        sfreq = 256  # Sampling frequency
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Create the MNE RawArray object
        raw = mne.io.RawArray(data, info)  # Assuming data is already in shape (n_channels, n_times)

        return raw

    def set_chs_montage(self):
        rename_dict = {
            'EEG Fp1-Ref': 'Fp1', 'EEG Fp2-Ref': 'Fp2',
            'EEG F3-Ref': 'F3', 'EEG F4-Ref': 'F4',
            'EEG C3-Ref': 'C3', 'EEG C4-Ref': 'C4',
            'EEG P3-Ref': 'P3', 'EEG P4-Ref': 'P4',
            'EEG O1-Ref': 'O1', 'EEG O2-Ref': 'O2',
            'EEG F7-Ref': 'F7', 'EEG F8-Ref': 'F8',
            'EEG T3-Ref': 'T7', 'EEG T4-Ref': 'T8',
            'EEG T5-Ref': 'P7', 'EEG T6-Ref': 'P8',
            'EEG A1-Ref': 'A1', 'EEG A2-Ref': 'A2',
            'EEG Fz-Ref': 'Fz', 'EEG Cz-Ref': 'Cz',
            'EEG Pz-Ref': 'Pz', 'ECG': 'ECG',
            'Resp oro-nasal': 'Resp', 'TORAXIC BELT': 'Toracic',
            'ABDOMINAL BELT': 'Abdominal', 'MICROPHONE': 'Microphone',
            'EMG-0': 'EMG0', 'EMG-1': 'EMG1', 'EMG-2': 'EMG2', 'EMG-3': 'EMG3',
            'EXT1': 'EXT1', 'EXT2': 'EXT2'
        }

        # Rename the channels
        self.raw.rename_channels(rename_dict)

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
            'Fz': 'eeg', 'Cz': 'eeg', 'Pz': 'eeg',
            'ECG': 'ecg', 'Resp': 'resp',
            'Toracic': 'misc', 'Abdominal': 'misc',
            'Microphone': 'misc',
            'EMG0': 'emg', 'EMG1': 'emg', 'EMG2': 'emg', 'EMG3': 'emg',
            'EXT1': 'stim', 'EXT2': 'stim'
        }

        self.raw.set_channel_types(channel_types)
        self.raw.info['bads'] = [
            'Resp', 'Toracic', 'Abdominal', 'Microphone', 
            'EMG0', 'EMG1', 'EMG2', 'EMG3', 'EXT1', 'EXT2'
        ]
        self.raw.drop_channels(self.raw.info['bads'])
        self.raw = self.raw.set_montage('standard_1020')

        return self.raw
    
    def load_data(self):
        try:
            data = np.loadtxt(self.eeg_path)
            return data
        except ValueError:
            cropped_lines = self.remove_first_n_lines(4)
            data = np.genfromtxt(StringIO(''.join(cropped_lines)))
            return data
    
    def remove_first_n_lines(self, n, write=False):
        with open(self.eeg_path, 'r', encoding='utf-8') as file:
            cropped_lines = file.readlines()[n:]

        if write:
            with open(self.eeg_path, 'w', encoding='utf-8') as file:
                file.writelines(cropped_lines)
        
        return cropped_lines
