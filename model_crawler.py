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


def parse_config_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_log_file(log_path):
    class_labels = list()
    with open(log_path, 'r') as file:
        lines = file.readlines()

    lines = lines[579:627] #CORE50
    #lines = lines[487:531] #EMNIST
    for line in lines:
        if "Replay:" in line:
            match = re.match(r'.*Class: (\d+) Replay: (\d+)', line)
            if match:
                class_labels.append(int(match.groups()[0]))


    last_classes = set(range(50)).difference(set(class_labels))
    last_classes = list(last_classes)
    class_labels = class_labels + last_classes
    return class_labels



def crawl_directory(root_path):
    experiment_dirs = [dir for dir in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, dir))]
    for exp_dir in experiment_dirs:
        print(f'Processing experiment {exp_dir}')
        try:
        #if True: # for debugging
            exp_path = os.path.join(root_path, exp_dir)

            log_path = os.path.join(exp_path, 'log.log')
            results_path = os.path.join(exp_path, 'results_core50')

            class_order = parse_log_file(log_path)

            for res_dir in os.listdir(results_path):
                res_path = os.path.join(results_path, res_dir)
                config_path = os.path.join(res_path, 'config.yaml')
                acc_path = os.path.join(res_path, 'detailed_accuracy.csv')
                dead_path = os.path.join(res_path, 'dead_neurons.csv')
                total_dead_path = os.path.join(res_path, 'total_dead_neurons.csv')
                if os.path.isfile(acc_path) and os.path.isfile(dead_path) and os.path.isfile(total_dead_path):
                    print('Result files exist, skipping')
                    #continue
                    pass

                #with open(acc_path, 'w') as f:
                #    f.write(f'{exp_dir}\n')
                #    f.write(f'Task,Test,Accuracy(%),Class1,Class2\n')

                #with open(dead_path, 'w') as f:
                #    f.write(f'{exp_dir}\n')
                #    f.write(f'Task,Test,Layer,Dead_neurons\n')

                with open(total_dead_path, 'w') as f:
                    f.write(f'{exp_dir}\n')
                    f.write(f'Task,Layer,Dead_neurons\n')


                # read config file
                config = parse_config_file(config_path)
                device = torch.device(config['device'])
                feature_classifier_params = config['feature_classifier']
                cbp_config = config['cbp_config']
                batch_size = config['batch_size']

                train_set, test_set = u.load_dataset(config)

                model_classifier = FeatureClassifier(input_size=feature_classifier_params['input_size'],
                                                     hidden1=feature_classifier_params['hidden1'],
                                                     hidden2=feature_classifier_params['hidden2'],
                                                     num_classes=feature_classifier_params['num_classes'],
                                                     continual_backprop=feature_classifier_params['cbp'],
                                                     cbp_config=cbp_config).to(device)

                for task in range(1, 26):
                    model_classifier.load_state_dict(
                        torch.load(os.path.join(res_path, f'classifier_class_incremental_with_replay_task{task}.pth')))
                    #for test in range(task):
                        #current_classes = [class_order[2 * test], class_order[2 * test + 1]]
                        #test_subset = u.get_subset_of_classes(test_set, current_classes)
                        #test_loader = data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

                        #accuracy = u.evaluate_classifier(model_classifier, test_loader, device)
                        #dead_neurons = u.count_dead_neurons(model_classifier, test_loader, device)
                        # print(f"Task {task} - Test {test} - Accuracy on seen classes: {accuracy:.2f}%")
                        #with open(acc_path, 'a') as f:
                        #    f.write(f'{task},{test + 1},{accuracy:.2f},{current_classes[0]},{current_classes[1]}\n')
                        #with open(dead_path, 'a') as f:
                        #    for ii, (layer, dead_count) in enumerate(dead_neurons.items()):
                        #        f.write(f'{task},{test + 1},{ii},{dead_count}\n')

                    current_classes = [class_order[0], class_order[2 * task + 1]]
                    test_subset = u.get_subset_of_classes(test_set, current_classes)
                    test_loader = data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

                    total_dead_neurons = u.count_dead_neurons(model_classifier, test_loader, device)
                    with open(total_dead_path, 'a') as f:
                        for ii, (layer, dead_count) in enumerate(total_dead_neurons.items()):
                            f.write(f'{task},{ii},{dead_count}\n')
        except:
            print('Error parsing in file {}'.format(exp_dir))



if __name__ == "__main__":
    root_path = '/cluster/work/users/mateuwa/CBP_SGD'  # Specify the root directory here
    crawl_directory(root_path)
    print('Model crawling complete')
