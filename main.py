import torch
from sklearn.model_selection import StratifiedKFold
from evaluate_sepsis_score import evaluate_sepsis_score
import torch.optim as optim
import pickle
from model import *
from utils import *
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import time
import datetime

def train_model(model, criterion, optimizer, Xtrain, Ytrain, epochs=10, batch_size=128, criterion_2=None, flag=False):

    train_data = TensorDataset(torch.tensor(Xtrain, dtype=torch.float32), torch.tensor(Ytrain, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
    losses = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs,h1,h2 = model(X_batch)
            outputs = outputs.view(-1)
            loss = criterion(outputs, y_batch)

            if flag:
                loss_2 = criterion_2(h1, h2)
                loss = loss + loss_2 / (loss + loss_2).detach()
                
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss)
def evaluate_model(model, Xtest, Ytest, batch_size=1024):
    test_data = TensorDataset(torch.tensor(Xtest, dtype=torch.float32), torch.tensor(Ytest, dtype=torch.float32))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for X_batch, _ in tqdm(test_loader, desc="Collecting predictions"):
            X_batch = X_batch.to(device)
            outputs, _, _, _ = model(X_batch)
            predictions = outputs.view(-1, 1)
            all_predictions.extend(predictions.cpu().numpy())

    return np.array(all_predictions).reshape(-1)


if __name__ == '__main__':
    gpu_id = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_id)) 
    else:
        device = torch.device("cpu")
    set_seed(42) 
    Mode = "" # dataset name
    input_directory = ""
    output_directory = ""
    input_dim = 80
    n_splits = 5
    num_round = 50
    rank = 0
    ratio = 0.3


    best_auc = -np.inf
    best_model_state = None
    fold_performances = []
    total_auroc, total_auprc, total_accuracy, total_sensitivity, total_specificity, total_f_measure,total_ppv,total_cui = 0, 0, 0, 0, 0,0,0,0
    list_auroc, list_auprc, list_accuracy, list_sensitivity, list_specificity, list_f_measure, list_ppv, list_cui = [], [], [], [], [], [],[],[]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold in range(n_splits):
        train_data_filename = f""
        val_data_filename = f""

        with open(train_data_filename, 'rb') as file:
            Xtrain, train_labels, _ = pickle.load(file)
        with open(val_data_filename, 'rb') as file:
            Xval, val_labels, patient_indices = pickle.load(file)

        sampler = CustomRandomUnderSampler(random_state=42) # if need resample
        Xtrain_resampled, train_labels_resampled = (Xtrain, train_labels)
        Xval, val_labels = (Xval, val_labels)

        input_length = Xtrain_resampled.shape[1]
        input_dim = Xtrain_resampled.shape[-1]

        model = VIGNN_TCL(input_dim, hidden_dim=128, num_layers=1, rank=rank, ratio=ratio, dropout_rate=0.5).to(
            device)

        flag = True
        name = str(model.__class__).split(".")[-1][:-2]

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        C_loss = CLoss()
        criterion = nn.BCELoss()

        train_model(model, criterion, optimizer, Xtrain_resampled, train_labels_resampled,
                    epochs=num_round, criterion_2=C_loss, flag=flag)

        scores = evaluate_model(model, Xval, val_labels)
        labels = (scores >= 0.5)

        auroc, auprc, accuracy, sensitivity, specificity, f_measure, PPV, CUI = evaluate_sepsis_score(val_labels,labels,scores)
        print("\nSet Results:")
        print(
            f"AUROC: {auroc}, AUPRC: {auprc}, F1: {f_measure}, PPV:{PPV}, CUI:{CUI}, Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}")
        
        auroc, auprc, accuracy, sensitivity, specificity, f_measure, PPV, CUI = [x * 100 for x in
                                                                                 [auroc, auprc, accuracy,
                                                                                  sensitivity, specificity,
                                                                                  f_measure, PPV, CUI]]


        total_auroc += auroc
        total_auprc += auprc
        total_accuracy += accuracy
        total_sensitivity += sensitivity
        total_specificity += specificity
        total_f_measure += f_measure
        total_ppv += PPV
        total_cui += CUI


        list_auroc.append(auroc)
        list_auprc.append(auprc)
        list_accuracy.append(accuracy)
        list_sensitivity.append(sensitivity)
        list_specificity.append(specificity)
        list_f_measure.append(f_measure)
        list_ppv.append(PPV)
        list_cui.append(CUI)


    avg_metrics = [total_auroc, total_auprc, total_accuracy, total_sensitivity, total_specificity, total_f_measure,
                   total_ppv, total_cui]
    avg_metrics = [x / n_splits for x in avg_metrics]
    std_metrics = [np.std(x) for x in
                   [list_auroc, list_auprc, list_accuracy, list_sensitivity, list_specificity, list_f_measure,
                    list_ppv, list_cui]]

    metric_names = ["AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "F1", "PPV", "CUI"]
    avg_results_str = ", ".join([f"Average {name}: {avg:.2f} ± {std:.2f}" for name, avg, std in
                                 zip(metric_names, avg_metrics, std_metrics)])

    print(f"\n Average Val Set Results:")
    print(avg_results_str)

    with open("results.txt", "a+") as f:
        f.write(f"\n{name}:")
        f.write(avg_results_str)











