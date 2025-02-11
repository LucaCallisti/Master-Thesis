import os
import json
import torch
import pandas as pd


class SaveExp():
    def __init__(self, path_folder, n_simulations):        
        existing_folders = [d for d in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, d)) and d.startswith('Experiment_')]
        experiment_numbers = [int(d.split('_')[1]) for d in existing_folders if d.split('_')[1].isdigit()]
        next_experiment_number = max(experiment_numbers, default=0) + 1

        new_folder_name = f'Experiment_{next_experiment_number}'
        self.path_folder = os.path.join(path_folder, new_folder_name)
        os.makedirs(self.path_folder)

        self.data = {}
        self.FinalDict_n_sim = {}
        

    def add_element(self, key, value):
        self.data[key] = value

    def add_multiple_elements(self, list_key, list_value):
        for key, value in zip(list_key, list_value):
            self.data[key] = value

    def save_result_discrete_one_run(self, FinalDict, i_simulation):
        final_dict_serializable = convert_tensors_to_lists(FinalDict)
        with open(os.path.join(self.path_folder, f'FinalDict_{i_simulation}_simulation.json'), 'w') as file:
            json.dump(final_dict_serializable, file, indent=4)

    def save_tensor(self, tensor, filename):
        torch.save(tensor, os.path.join(self.path_folder, filename))

    def save_dict(self):
        for key, value in self.data.items():
            if isinstance(value, list) and all(isinstance(i, tuple) for i in value):
                self.data[key] = str(value)
        df = pd.DataFrame(self.data, index=[0]).T.reset_index()
        df.columns = ['Key', 'Value']
        df.to_csv(os.path.join(self.path_folder, 'data.csv'), index=False)

    def get_folder_path(self):
        return self.path_folder




def convert_tensors_to_lists(d):
    if isinstance(d, dict):
        return {k: convert_tensors_to_lists(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_tensors_to_lists(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.tolist()
    else:
        return d
    
def convert_lists_to_tensors(d):
    if isinstance(d, dict):
        return {k: convert_lists_to_tensors(v) for k, v in d.items()}
    elif isinstance(d, list):
        return torch.tensor(d)
    else:
        return d