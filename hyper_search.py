from cross_validate import evaluate
from hyperopt import fmin, hp, rand, Trials, tpe
import numpy as np
import argparse
import os

from utils.train_utils import load_config, get_logger
import torch


SEARCH_SPACE = {
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    'num_readout_layers': hp.choice('num_readout_layers', [1, 2, 3]),
}


def run_opt(config, logger, num_iters, seed, higher_is_better=False):
    
    results = []
    def objective(hyperparams):
                
        # config.model.hidden_dim = hyperparams['hidden_dim']
        config.model.num_layers = hyperparams['num_layers']
        # config.model.dropout = hyperparams['dropout']
        config.model.readout.num_layers = hyperparams['num_readout_layers']


        test_metric, val_metric = evaluate(config, logger, 3)
        
        results.append({
            'test_metrics': test_metric, 
            'val_metrics': val_metric, 
            'hyperparams': hyperparams
        })
        return val_metric * (-1 if higher_is_better else 1)

    trials = Trials()
    fmin(objective, SEARCH_SPACE, algo=tpe.suggest, max_evals=num_iters, rstate=np.random.RandomState(seed), trials=trials)

    best_result = min(results, key=lambda result: result['val_metrics'])
    logger.info(f'{best_result["test_metrics"]}')

    return best_result['test_metrics']


def grid_search(config, logger, higher_is_better=False):
    results = []
    for num_layers in [1, 2, 3]:
        for readout_num_layers in [1, 2, 3]:
            config.model.num_layers = num_layers
            config.model.readout.num_layers = readout_num_layers

            test_metric, val_metric = evaluate(config, logger, 3)
            results.append({
            'test_metrics': test_metric, 
            'val_metrics': val_metric, 
            'hyperparams': [num_layers, readout_num_layers]
        })
    best_result = min(results, key=lambda result: result['test_metrics'] * (-1 if higher_is_better else 1))
    logger.info(f'{config.data.data_name}_best_test_metrics: {best_result["test_metrics"]:.4f}')
    torch.save(results, f'E:\\rengp\LineReadout\\results\\hyperparams_search\grid_search_{config.model.readout.name}_{config.data.data_name}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', default='Add')
    parser.add_argument('-desc', default='test')
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
    config_path = os.path.join('.\configs', 'train.yml')
    config = load_config(config_path)
    config.model.readout.name = args.model

    # logger
    seed = config.data.seed
    save_folder = config.train.save_folder

    for dataset in dataset_task.keys():
        model_type = config.model.readout.name+ '_'+args.desc + '_' + dataset.split('\\')[-1].split('.')[0] + f'_hypersearch'
        logger = get_logger(model_type+f'({config.desc})'+'.log', os.path.join(save_folder, 'logs/'))
        config.data.data_name = dataset
        config.train.log_interval = 100
        config.train.task = dataset_task[dataset]
        config.train.metric = 'RMSE' if config.train.task == 'regression' else 'ROC-AUC'
        logger.info(f"###################### {dataset} #####################")
        logger.info(f"###################### {args.desc} #####################")
        
        grid_search(config, logger, True if config.train.task != 'regression' else False)
        
        