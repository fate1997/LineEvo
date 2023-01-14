import torch
from torch import nn
from torch_geometric.loader import DataLoader

import pandas as pd
import numpy as np
import os
import yaml
import logging
import logging.config
import json
import sys
from easydict import EasyDict
import random

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import Chem
from sklearn import metrics
from random import Random
from collections import defaultdict
from tqdm import tqdm

class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""

    def __init__(self, patience=100, verbose=False,
                 path='./results/trained_models/', model_type=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.model_type = model_type

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model) 
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Saves model when validation score doesn't improve in patience
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, self.model_type + '.pt'))
        self.val_loss_min = val_loss


def evaluate_score(model, data_loader, task):
    if isinstance(model, nn.Module):
        model.eval()
    
    y_true, y_pred, _ = batch_flatten(model, data_loader)

    metric_dict = {}
    if task == 'regression':
        metric_dict['RMSE'] = np.sqrt(((np.array(y_true) - np.array(y_pred)) ** 2).mean(0)).tolist()
        metric_dict['MAE'] = (np.abs((np.array(y_true) - np.array(y_pred))).mean(0)).tolist()
        metric_dict['loss'] = nn.MSELoss()(torch.tensor(y_true, dtype=torch.float32), 
                                        torch.tensor(y_pred, dtype=torch.float32))
    elif task == 'binary':
        y_pred = torch.sigmoid(torch.tensor(y_pred)).tolist()
        metric_dict['ROC-AUC'] = metrics.roc_auc_score(y_true, y_pred)
        metric_dict['Accuracy'] = metrics.average_precision_score(y_true, y_pred)
        metric_dict['loss'] = nn.BCEWithLogitsLoss()(torch.tensor(y_true, dtype=torch.float32), 
                                        torch.tensor(y_pred, dtype=torch.float32))
    return metric_dict


# flatten batch
def batch_flatten(model, data_loader):
    model.eval()
    model.to('cpu')
    y_true = []
    y_predict = []
    IDs = []
    for batch in data_loader:
        batch = batch.to('cpu')
        y_hat = model(batch).detach().reshape((-1, 1)).tolist()
        y_true += batch.y.reshape((-1, 1)).tolist()
        y_predict += y_hat
        IDs += batch.ID
    return y_true, y_predict, IDs


def save_results(model, data_loader, save_folder, model_type):
    y_true, y_pred, smiles = batch_flatten(model, data_loader)

    results = pd.DataFrame(np.array([y_true, y_pred]).squeeze(-1).T, columns=['y_true', 'y_pred'], index=smiles)
    results.to_csv(os.path.join(save_folder, model_type + '.csv'))


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def data2iter(dataset,
              seed: int = None,
              batch_size: int = None,
              train_ratio: float = None):

    # load dataset and split to train and test set
    split_1 = int(train_ratio * len(dataset))
    split_2 = int((train_ratio+(1-train_ratio)/2) * len(dataset))

    # dataset.shuffle(seed)
    r = random.random
    random.seed(seed)
    random.shuffle(dataset.data_list, random=r)
    
    train_loader = DataLoader(dataset[:split_1], batch_size=batch_size)
    val_loader = DataLoader(dataset[split_1:split_2], batch_size=batch_size)
    test_loader = DataLoader(dataset[split_2:], batch_size=batch_size)


    return train_loader, val_loader, test_loader

def scaffold_splitter(
              dataset,
              batch_size: int = None,
              train_ratio: float = None):
    
    split_1 = int(train_ratio * len(dataset))
    split_2 = int((train_ratio+(1-train_ratio)/2) * len(dataset))
    
    if dataset[0].ID[0] == '<':
        smiles_list = [data.ID[1:-1] for data in dataset]
    else:
        smiles_list = [data.ID for data in dataset]

    train_inds = []
    valid_inds = []
    test_inds = []

    scaffolds = {}

    for idx,smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffoldSmiles(mol=mol)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(
        scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > split_1:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > split_2:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    
    train_set = [dataset[i] for i in train_inds]
    val_set = [dataset[i] for i in valid_inds]
    test_set = [dataset[i] for i in test_inds]
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader



def get_logger(name, log_dir, config_dir='./configs/'):

    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger