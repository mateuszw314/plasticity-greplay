import os
import pandas as pd
import yaml
import re


def parse_log_file(log_path):
    with open(log_path, 'r') as file:
        lines = file.readlines()

    # Initialize placeholders for the extracted values
    device = None
    generator_param_size = None
    classifier_param_size = None
    replay_info = []

    for line in lines:
        if "Using device" in line:
            device = line.split(": ")[1].strip()
        elif "Generator param size" in line:
            generator_param_size = int(re.search(r'\d+', line.split(": ")[1].strip()).group())
        elif "Classifier param size" in line:
            classifier_param_size = int(re.search(r'\d+', line.split(": ")[1].strip()).group())
        elif "Generating replay for task" in line:

            task_num = int(line.split('task ')[-1])
            replay_info.append((task_num, 0))  # Append task number and initialize count to 0
        elif "Replay:" in line:
            match = re.match(r'.*Class: (\d+) Replay: (\d+)', line)
            if match:
                replay_count = int(match.groups()[1])
                if replay_count > 0:
                    replay_info[-1] = (replay_info[-1][0], replay_info[-1][1] + 1)
    replay_info = [rep[1] for rep in replay_info]
    return device, generator_param_size, classifier_param_size, replay_info


def parse_accuracy_file(accuracy_path):
    accuracies = {}
    with open(accuracy_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            task_num, accuracy = re.search(r'Task (\d+), Accuracy: ([\d.]+)%', line).groups()
            accuracies[int(task_num)] = float(accuracy)
    return accuracies


def parse_config_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def extract_experiment_data(root_path):
    experiment_dirs = [dir for dir in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, dir))]

    data = []
    for exp_dir in experiment_dirs:
        try:
            exp_path = os.path.join(root_path, exp_dir)

            log_path = os.path.join(exp_path, 'log.log')
            device, generator_size, classifier_size, replay_info = parse_log_file(log_path)

            results_path = os.path.join(exp_path, 'results_emnist')
            for res_dir in os.listdir(results_path):
                res_path = os.path.join(results_path, res_dir)
                accuracy_path = os.path.join(res_path, 'accuracy.txt')
                config_path = os.path.join(res_path, 'config.yaml')

                accuracies = parse_accuracy_file(accuracy_path)
                config = parse_config_file(config_path)

                entry = {
                    'Experiment_ID': exp_dir,
                    'Generator_Param_Size': generator_size,
                    'Classifier_Param_Size': classifier_size,
                    #'Replay_Info': replay_info,
                    **{f"Survival task {task}": val for (task, val) in enumerate(replay_info)},
                    'Classifier cbp': int(config['feature_classifier']['cbp']),
                    'Generator cbp': int(config['generator']['cbp']),
                    'Filter': config['softmax_filter'],
                    'Replacement rate': config['cbp_config']['replacement_rate'],
                    'Maturity threshold': config['cbp_config']['maturity_threshold'],

                    **{f'Acc task {task}': val[1] for (task, val) in enumerate(accuracies.items())}
                }
                data.append(entry)

        except:
            print('Error parsing in file {}'.format(exp_path))



    return pd.DataFrame(data)


if __name__ == "__main__":
    root_path = '/cluster/work/users/mateuwa/CBP_EMNIST'  # Specify the root directory here
    df = extract_experiment_data(root_path)
    print(df.head())  # For quick verification
    df.to_csv('/cluster/work/users/mateuwa/CBP_EMNIST/parsed_experiment_results.csv', index=False)  # Save the DataFrame to a CSV file