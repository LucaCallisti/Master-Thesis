import json
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SaveExp:
    def __init__(self, path_folder, file_path=None):
        self.path_folder = path_folder
        self.file_path = file_path
        self.data = {}
        self.pl = None

    def find_path(self):
        existing_folders = [d for d in os.listdir(self.path_folder) if os.path.isdir(os.path.join(self.path_folder, d)) and d.startswith('Experiment_')]
        experiment_numbers = [int(d.split('_')[1]) for d in existing_folders if d.split('_')[1].isdigit()]
        next_experiment_number = max(experiment_numbers, default=0) + 1

        new_folder_name = f'Experiment_{next_experiment_number}'
        self.path_folder = os.path.join(self.path_folder, new_folder_name)
        os.makedirs(self.path_folder)

        self.file_path = os.path.join(self.path_folder, 'data.csv')

    def add_element(self, key, value):
        self.data[key] = value

    def save_dict(self):
        if self.file_path is None:
            self.find_path()
        df = pd.DataFrame(self.data, index=[0]).T.reset_index()
        df.columns = ['Key', 'Value']
        df.to_csv(self.file_path, index=False)


    def save_result_discrete(self, FinalDict):
        if self.file_path is None:
            self.find_path()
        if self.pl is None:
            self.pl = plot(self.path_folder)

        final_dict_serializable = convert_tensors_to_lists(FinalDict)

        with open(os.path.join(self.path_folder, 'FinalDict.json'), 'w') as file:
            json.dump(final_dict_serializable, file, indent=4)

        self.pl.plot_norm_discrete(FinalDict)

    def save_result_continuous(self, result):
        if self.file_path is None:
            self.find_path()
        if self.pl is None:
            self.pl = plot(self.path_folder)

        result = torch.stack(result).squeeze()
        number_parameters = result.shape[2] // 3
        if len(result.shape) == 2:
            result = result.unsqueeze(0)
        gradient = result[:, :, :number_parameters]
        square_avg = result[:, :, number_parameters:2*number_parameters]
        print(gradient.shape, square_avg.shape)
        gradient_norm = torch.norm(gradient, dim=2)
        square_avg_norm = torch.norm(square_avg, dim=2)
        self.pl.plot_norm_cont(gradient_norm, square_avg_norm)

    def save_loss_sde(self, Loss):
        if self.file_path is None:
            self.find_path()
        if self.pl is None:
            self.pl = plot(self.path_folder)
        self.pl.plot_loss(Loss, self.data['eta'])


def convert_tensors_to_lists(d):
    if isinstance(d, dict):
        return {k: convert_tensors_to_lists(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_tensors_to_lists(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.tolist()
    else:
        return d
    

class plot():
    def __init__(self, path_folder):
        self.path_folder = path_folder
    
    def my_plot(self, list_df, list_legend, title):
        def same_plot(df, legend):
            if legend == '':
                sns.lineplot(data=df, x='step', y='norm', ci=95)
                # sns.lineplot(data=df, x='step', y='norm', hue='run', palette='tab10')
            else:
                sns.lineplot(data=df, x='step', y='norm', ci=95, label=legend)
                # sns.lineplot(data=df, x='step', y='norm', hue='run', palette='tab10')

        plt.figure()
        for df, legend in zip(list_df, list_legend):
            same_plot(df, legend)
        plt.xlabel('Step')
        plt.ylabel('Norm')
        plt.title(f'Norm {title}')
        path = os.path.join(self.path_folder, f'norm_{title}.png')
        plt.savefig(path)

    def plot_norm_discrete(self, FinalDict):
        self.FinalDict = FinalDict
        n_step, _ = FinalDict[1]['Params'].shape
        data_grad = []
        data_sqare_avg = []
        for i_run in FinalDict.keys():
            params = FinalDict[i_run]['Params']
            norm = torch.norm(params, dim=1)
            for i_step in range(n_step): 
                data_grad.append({'step': i_step, 'run': i_run, 'norm': norm[i_step].item()})
            
            square_avg = FinalDict[i_run]['Square_avg']
            norm = torch.norm(square_avg, dim=1) 
            for i_step in range(n_step):
                data_sqare_avg.append({'step': i_step, 'run': i_run, 'norm': norm[i_step].item()})

        self.df_grad = pd.DataFrame(data_grad)
        self.my_plot([self.df_grad], [''], 'Gradient Discrete')
        self.df_square_avg = pd.DataFrame(data_sqare_avg)
        self.my_plot([self.df_square_avg], [''], 'Square_avg Discrete')

    def plot_norm_cont(self, Result_sde_grad, Reuslt_sde_square):
        def aux(Result_sde):
            n_run, n_step = Result_sde.shape
            data = []
            for i_run in range(n_run):
                for t in range(n_step):
                    data.append({'step': t, 'norm': Result_sde[i_run, t].item(), 'run': i_run})
            return data
        data_grad = aux(Result_sde_grad)
        data_square = aux(Reuslt_sde_square)

        df_grad = pd.DataFrame(data_grad)
        df_square = pd.DataFrame(data_square)
        self.my_plot([df_grad, self.df_grad], ['continuous', 'discrete'], 'Parameter Comparison')
        self.my_plot([df_square, self.df_square_avg], ['continuous', 'discrete'], 'Square_avg Comparison')

    def plot_loss(self, Loss, eta = 0.1):
        Loss = torch.stack(Loss)
        Loss = Loss[:, 1:]

        len_loss_cont = Loss.shape[1]
        len_loss_discr = len(self.FinalDict[1]['Loss'])

        temp = len_loss_cont // len_loss_discr
        Loss = Loss[:, ::temp]

        data_loss_cont = []
        for i_run in range(Loss.shape[0]):
            for i_step in range(Loss.shape[1]):
                data_loss_cont.append({'step': i_step * eta, 'loss': Loss[i_run, i_step].item(), 'run': i_run})
        data_loss_disc = []
        for i_run in self.FinalDict.keys():
            loss = self.FinalDict[i_run]['Loss']
            for i_step, l in enumerate(loss):
                data_loss_disc.append({'step': i_step * eta, 'loss': l, 'run': i_run})

        df_loss_cont = pd.DataFrame(data_loss_cont)
        df_loss_disc = pd.DataFrame(data_loss_disc)
        plt.figure()
        sns.lineplot(data=df_loss_cont, x='step', y='loss', ci=95, label='continuous')
        sns.lineplot(data=df_loss_disc, x='step', y='loss', ci=95, label='discrete')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title(f'Loss Comparison')
        plt.savefig('/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/loss_comparison.png')
        plt.show()