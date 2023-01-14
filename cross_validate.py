import os
import argparse
import numpy as np

from train_main import train
from utils.train_utils import load_config, get_logger


def evaluate(config, logger, num_rounds):
    test_metric_list = []
    val_metric_list = []
    for i in range(num_rounds):
        logger.info(f'###############################')
        logger.info(f'############ Run {i+1} ############')
        logger.info(f'###############################')

        config.data.seed = i
        test_metric, val_metric = train(config, logger)
        test_metric_list.append(test_metric)
        val_metric_list.append(val_metric)
    
    logger.info(f'test metric = {np.mean(test_metric_list): .4f} ± {np.std(test_metric_list):.4f}; '
                f'val metric = {np.mean(val_metric_list): .4f} ± {np.std(val_metric_list):.4f}')
    
    return np.mean(test_metric_list), np.mean(val_metric_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', default='Add')
    parser.add_argument('-num_rounds', default=3)
    parser.add_argument('-dataset', default='freesolv')
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
    config.data.data_name = args.dataset
    config.train.task = dataset_task[args.dataset]
    config.model.readout.name = args.model
    config.train.metric = 'RMSE' if config.train.task == 'regression' else 'ROC-AUC'
    config.desc = args.desc

    # logger
    seed = config.data.seed
    save_folder = config.train.save_folder
    model_type = config.model.readout.name + '_' + config.data.data_name.split('\\')[-1].split('.')[0] + f'_evaluate'
    logger = get_logger(model_type+f'({args.desc})'+'.log', os.path.join(save_folder, 'logs/'))

    evaluate(config, logger, args.num_rounds)