import argparse
import logging
import os
import random
import shutil
from datetime import datetime

import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils as u
import yaml
from feature_classifier import FeatureClassifier
from feature_dataset import CustomNumpyDataset
import pandas as pd
import yaml
import re
import utils as u
import pickle




def crawl_directory(root_path):
    result_dict = dict()
    experiment_dirs = [dir for dir in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, dir))]
    for exp_dir in experiment_dirs:
        print(f'Processing experiment {exp_dir}')
        try:
            exp_path = os.path.join(root_path, exp_dir)

            log_path = os.path.join(exp_path, 'log.log')
            results_path = os.path.join(exp_path, 'results_emnist')


            for res_dir in os.listdir(results_path):
                res_path = os.path.join(results_path, res_dir)
                acc_path = os.path.join(res_path, 'detailed_accuracy.csv')
                df = pd.read_csv(acc_path, skiprows=1)
                result_dict[exp_dir] = df

        except Exception as e:
            print('Error parsing in file {}'.format(exp_dir))
            print(e)

    with open(f'{root_path}/detailed_results', 'wb') as f:
        pickle.dump(result_dict, f)



if __name__ == "__main__":
    root_path = '/cluster/work/users/mateuwa/CBP_EMNIST'  # Specify the root directory here
    crawl_directory(root_path)
    print('Crawling complete')
