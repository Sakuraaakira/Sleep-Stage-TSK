import wfdb
import numpy as np
import os

def load_mit_bih_data(data_dir, record_name):
    path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(path)
    signal = record.p_signal[:, 0]
    fs = record.fs
    ann = wfdb.rdann(path, 'st')
    epoch_len = int(30 * fs)
    epochs, labels = [], []
    label_map = {'W':0, 'w':0, '(W':0, '1':1, '(1':1, '2':2, '(2':2, '3':3, '(3':3, '4':4, '(4':4, 'R':5, 'r':5, '(R':5, 'MT':0}
    for i in range(len(ann.sample)):
        start, end = ann.sample[i], ann.sample[i] + epoch_len
        if end <= len(signal):
            symbol = str(ann.aux_note[i]).strip().replace('\x00', '')
            if symbol in label_map:
                epochs.append(signal[start:end]); labels.append(label_map[symbol])
    return np.array(epochs), np.array(labels), fs