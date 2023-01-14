from torch import nn
import torch
import numpy as np

import argparse
import os

from utils.train_utils import EarlyStopping, evaluate_score, \
    save_results, load_config, data2iter, get_logger
from dataset.data import MolDataset
from model.GATv2_base import GATv2
from model.GIN_base import GIN


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config, logger):

    # Initialize logger
    desc = config.desc
    seed = config.data.seed
    data_name = config.data.data_name
    save_folder = config.train.save_folder
    model_type = config.model.readout.name + '_' + data_name.split('\\')[-1].split('.')[0] + f'[{str(seed)}]'

    
    logger.info(f'############{desc}############')
    logger.info(vars(config))
    
    # Load train config
    num_epochs = config.train.num_epochs
    lr = config.train.lr 
    patience = config.train.patience
    task = config.train.task
    device = config.train.device
    log_interval = config.train.log_interval

    # Load data
    logger.info("-----------Dataset Loading-----------")
    batch_size = config.data.batch_size
    train_ratio = config.data.train_ratio
    dataset = MolDataset(data_name)
    dataset.transform(depth=config.model.readout.num_layers)

    train_loader, eval_loader, test_loader = data2iter(dataset, seed, batch_size, train_ratio)

    # training init
    seed_all(42)
    early_stopping = EarlyStopping(patience=patience, 
                                   path=os.path.join(save_folder, 'checkpoints/'),
                                   model_type=model_type)
    
    assert task in ['regression', 'binary', 'multiclass']
    # define loss function
    if task == 'regression':
        criterion = nn.MSELoss()
    elif task == 'binary':
        criterion = nn.BCEWithLogitsLoss()
    elif task == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    
    logger.info("------------Model Creating-----------")

    model = GATv2(config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    logger.info("------------Train Running------------")
    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0
        num_examples = 0
        for i, batch in enumerate(train_loader):

            # forward
            model.to(device)
            batch = batch.to(device)
            y = batch.y.reshape((-1, 1))
            outputs = model(batch)

            if task == 'multiclass':
                loss = criterion(outputs, y.flatten())
            else:
                loss = criterion(outputs, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_examples += y.shape[0]
            loss_sum += loss.item() * y.shape[0]

        val_metric = evaluate_score(model, eval_loader, task)

        metric = config.train.metric
        if epoch % log_interval == 0:
            if task == 'regression':
                logger.info(f'epoch:{epoch}, loss = {loss_sum / num_examples: .4f}, '
                    f'val loss = {val_metric["loss"]:.4f}, '
                    f'val {metric} = {np.round(val_metric[metric][0], decimals=4)}')
            else:
                logger.info(f'epoch:{epoch}, loss ={loss_sum / num_examples: .4f}, '
                f'val loss = {val_metric["loss"]:.4f}, '
                f'val {metric} = {np.round(val_metric[metric], decimals=4)}')
    
        # early stopping
        min_metrics = np.array(val_metric[metric]).mean()
        
        if task == 'binary':
            min_metrics = -val_metric[metric]
        early_stopping(min_metrics, model)

        if early_stopping.early_stop:
            logger.info('------------Early stopping------------')
            break

    model.load_state_dict(torch.load(os.path.join(save_folder, 'checkpoints', model_type + '.pt')))
    
    test_metric = evaluate_score(model, test_loader, task)
    val_metric = evaluate_score(model, eval_loader, task)

    if task == 'regression':
        logger.info(f'test {metric} = {np.round(test_metric[metric][0], decimals=4)}, '
            f'val {metric} = {np.round(val_metric[metric][0], decimals=4)}')
    else:
        logger.info(f'test {metric} = {np.round(test_metric[metric], decimals=4)}, '
            f'val {metric} = {np.round(val_metric[metric], decimals=4)}')

    save_results(model, test_loader, os.path.join(save_folder, 'pred_results'), model_type)
    
    return (test_metric[metric][0], val_metric[metric][0]) if task=='regression' else (test_metric[metric], val_metric[metric])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', default='Add')
    parser.add_argument('-desc', default='Testing')
    parser.add_argument('-dataset', default='freesolv')
    parser.add_argument('-config_name', default='train.yml')
    args = parser.parse_args()

    # dataset-task
    dataset_task = {
        'freesolv': 'regression',
        'delaney': 'regression',
        'lipo': 'regression',
        'bbbp': 'binary',
        'bace': 'binary'
    }

    # load config
    config_path = os.path.join('.\configs', args.config_name)
    config = load_config(config_path)
    config.data.data_name = args.dataset
    config.train.task = dataset_task[args.dataset]
    config.model.readout.name = args.model
    config.desc = args.desc
    config.train.metric = 'RMSE' if config.train.task == 'regression' else 'ROC-AUC'

    # logger
    seed = config.data.seed
    save_folder = config.train.save_folder
    model_type = config.model.readout.name + '_' + args.dataset.split('\\')[-1].split('.')[0] + f'[{str(seed)}]'

    logger = get_logger(model_type+f'({config.desc})'+'.log', os.path.join(save_folder, 'logs/'))
    
    train(config, logger)