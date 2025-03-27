
import numpy as np
import os
from tqdm import tqdm
import torch
import random
import torch.nn as nn
from sklearn.utils import check_random_state



def get_num_units(
    units_multiplier: int,
    num_basis_functions: int,
    X):

    num_unique_vals = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
    num_units = [min(num_basis_functions, i * units_multiplier) for i in num_unique_vals]

    return num_units

def set_seed(seed_value):
    random.seed(seed_value)  
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        label = data[:, -1]
        data = data[:, :-1]

    return data, label

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))

def imputation(data):

    past_time = 24

    data = np.copy(data)

    delta = np.zeros(data.shape)
    mask = np.ones(data.shape)

    nanIdx = np.where(np.isnan(data))
    # data[nanIdx] = np.take(varmeans,nanIdx[1])
    mask[nanIdx] = 0


    forward = np.copy(data[0, :])
    for t in range(data.shape[0]):
        for i in range(39):
            if mask[t, i] == 1:
                forward[i] = data[t, i]
            else:
                data[t, i] = forward[i]
        if t > 0:
            delta[t, :] = data[t, :] - data[t - 1, :]

    nanIdx = np.where(np.isnan(data))

    data[nanIdx] = np.take([0] * 40, nanIdx[1])

    t = len(data) - 1
    row = list(data[t, :])
    data = np.concatenate((data, mask), axis=1)


    t = len(data) - 1
    row = list(np.expand_dims([data[t, :]],axis=0))
    for j in range(1, past_time):
        if (t - j < 0):
            row.append(np.expand_dims(np.array([0.0] * 80),axis=0))
        else:
            row.append(np.expand_dims(data[t - j, :],axis=0))

    data = np.concatenate(row)

    return np.expand_dims(data,axis=0)


def load_data(input_directory, files):
    label_files, pred_files = [], []
    DATA, LABEL = [], []
    patient_indices = []
    start_idx = 0
    count = 0
    for f in tqdm(files):
    
        # Load data.
        # print(f"{prefix} - File {idx + 1}: {f}")
        input_file = os.path.join(input_directory, f)
        data_, label_ = load_challenge_data(input_file)
        label_files.append(input_file)

        end_idx = start_idx + len(data_)
        patient_indices.append((start_idx, end_idx))
        start_idx = end_idx

        # Make predictions.
        num_rows = len(data_)
        scores = np.zeros(num_rows)
        labels = np.zeros(num_rows)

        for t in range(num_rows):
            data = data_[:t + 1]
            label = np.asarray(label_[t]).reshape(1, )

            data = imputation(data)

            DATA.append(data)
            LABEL.append(label)

    DATA = np.concatenate(DATA)
    LABEL = np.concatenate(LABEL)

    return DATA, LABEL, patient_indices


