import json
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

class SaveExp:
    def __init__(self, path_folder, file_path=None, normalization_plot = True):
        self.path_folder = path_folder
        self.file_path = file_path
        self.data = {}
        self.pl = None
        self.normalization = normalization_plot
    
    def partial_result(self, partial_result, name, Bool = False):
        if self.file_path is None: self.find_path()
        if not Bool:
            num = len(partial_result) 
            path = os.path.join(self.path_folder, f'{name}_{num}.pt')
        else:
            path = os.path.join(self.path_folder, f'{name}.pt')
        torch.save(partial_result, path)
        
    def find_path(self):
        existing_folders = [d for d in os.listdir(self.path_folder) if os.path.isdir(os.path.join(self.path_folder, d)) and d.startswith('Experiment_')]
        experiment_numbers = [int(d.split('_')[1]) for d in existing_folders if d.split('_')[1].isdigit()]
        next_experiment_number = max(experiment_numbers, default=0) + 1

        new_folder_name = f'Experiment_{next_experiment_number}'
        self.path_folder = os.path.join(self.path_folder, new_folder_name)
        os.makedirs(self.path_folder, exist_ok=True)

        self.file_path = os.path.join(self.path_folder, 'data.csv')

    def add_element(self, key, value):
        self.data[key] = value

    def save_dict(self):
        if self.file_path is None: self.find_path()
        for key, value in self.data.items():
            if isinstance(value, list) and all(isinstance(i, tuple) for i in value):
                self.data[key] = str(value)
        df = pd.DataFrame(self.data, index=[0]).T.reset_index()
        df.columns = ['Key', 'Value']
        df.to_csv(self.file_path, index=False)


    def save_result_discrete(self, FinalDict):
        if self.file_path is None:
            self.find_path()
        if self.pl is None:
            self.pl = plot(self.path_folder, self.normalization)
        self.FinalDict = FinalDict
        final_dict_serializable = convert_tensors_to_lists(FinalDict)

        with open(os.path.join(self.path_folder, 'FinalDict.json'), 'w') as file:
            json.dump(final_dict_serializable, file, indent=4)

        self.pl.plot_norm_discrete(FinalDict)

    def save_result_continuous(self, result, order=1):    
        if self.file_path is None:
            self.find_path()
        if self.pl is None:
            self.pl = plot(self.path_folder, self.normalization)
        if order == 1: self.res_1 = result
        if order == 2: self.res_2 = result

        torch.save(result, os.path.join(self.path_folder, 'Result_'+str(order) + '_order.pt'))

        result = torch.stack(result).squeeze()
        if len(result.shape) == 2:
            result = result.unsqueeze(0)
        number_parameters = result.shape[2] // 3
        gradient = result[:, :, :number_parameters]
        square_avg = result[:, :, number_parameters:2*number_parameters]
        print(gradient.shape, square_avg.shape)
        gradient_norm = torch.norm(gradient, dim=2)
        square_avg_norm = torch.norm(square_avg, dim=2)
        self.pl.plot_norm_cont(gradient_norm, square_avg_norm, order)

    def save_loss_sde(self, Loss, order=1):
        if self.file_path is None: self.find_path()
        if self.pl is None: self.pl = plot(self.path_folder, self.normalization)
        if order == 1: self.Loss_1 = Loss
        if order == 2: self.Loss_2 = Loss
        torch.save(Loss, os.path.join(self.path_folder, 'Loss'+str(order)+'_order.pt'))
        self.pl.plot_loss(Loss, order)

    def save_grad_sde(self, Grad, order = 1):
        if self.file_path is None: self.find_path()
        if self.pl is None: self.pl = plot(self.path_folder, self.normalization)
        torch.save(Grad, os.path.join(self.path_folder, 'Grad_'+str(order)+'_order.pt'))
        self.pl.plot_grad(Grad, order)

    def save_comparison_1_and_2_order(self):
        if self.file_path is None: self.find_path()
        if self.pl is None: self.pl = plot(self.path_folder, self.normalization)
        self.pl.plot_comparison_1_and_2_order(self.res_1, self.res_2, self.Loss_1, self.Loss_2, self.FinalDict)



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

def prepare_result_continuous(result):
    result = torch.stack(result).squeeze()
    if len(result.shape) == 2:
        result = result.unsqueeze(0)
    number_parameters = result.shape[2] // 3
    gradient = result[:, :, :number_parameters]
    square_avg = result[:, :, number_parameters:2*number_parameters]
    print(gradient.shape, square_avg.shape)
    gradient_norm = torch.norm(gradient, dim=2)
    square_avg_norm = torch.norm(square_avg, dim=2)
    return gradient_norm, square_avg_norm

class plot():
    def __init__(self, path_folder, normalization = False):
        self.path_folder = path_folder
        self.normalization = normalization
    
    def my_plot(self, list_df, list_legend, title, y='norm'):
        def same_plot(df, legend):
            if legend == '':
                sns.lineplot(data=df, x='step', y=y, errorbar=('ci', 95))
            else:
                sns.lineplot(data=df, x='step', y=y, errorbar=('ci', 95), label=legend)

        sns.set_theme()
        plt.figure()
        for df, legend in zip(list_df, list_legend):
            same_plot(df, legend)
        plt.xlabel('Step')
        plt.ylabel(y)
        plt.title(f'{y} {title}')
        if self.normalization:
            title = f'{title}_normalized'
        path = os.path.join(self.path_folder, f'{y}_{title}.png')
        plt.savefig(path)
        plt.close()

    def plot_norm_discrete(self, FinalDict):
        self.FinalDict = FinalDict
        self.n_step, self.number_parameters = FinalDict[1]['Params'].shape
        data_grad = []
        data_sqare_avg = []
        for i_run in FinalDict.keys():
            params = FinalDict[i_run]['Params']
            norm = torch.norm(params, dim=1)
            for i_step in range(self.n_step): 
                if self.normalization:
                    norm[i_step] /= self.number_parameters
                data_grad.append({'step': i_step, 'run': i_run, 'norm': norm[i_step].item()})
            
            square_avg = FinalDict[i_run]['Square_avg']
            norm = torch.norm(square_avg, dim=1) 
            for i_step in range(self.n_step):
                if self.normalization:
                    norm[i_step] /= self.number_parameters
                data_sqare_avg.append({'step': i_step, 'run': i_run, 'norm': norm[i_step].item()})

        self.df_grad = pd.DataFrame(data_grad)
        self.my_plot([self.df_grad], [''], 'Gradient Discrete')
        self.df_square_avg = pd.DataFrame(data_sqare_avg)
        self.my_plot([self.df_square_avg], [''], 'Square_avg Discrete')

    def plot_norm_cont(self, Result_sde_grad, Reuslt_sde_square, order=1):
        def aux(Result_sde):
            n_run, n_step = Result_sde.shape
            data = []
            for i_run in range(n_run):
                for t in range(n_step):
                    if self.normalization:
                        Result_sde[i_run, t] /= self.number_parameters
                    data.append({'step': t, 'norm': Result_sde[i_run, t].item(), 'run': i_run})
            return data
        
        _, step_c = Result_sde_grad.shape
        Grad_c = Result_sde_grad[:, :: step_c//self.n_step]
        Square_c = Reuslt_sde_square[:, :: step_c//self.n_step]
        data_grad_c = aux(Grad_c)
        data_square_c = aux(Square_c)
        df_grad_c = pd.DataFrame(data_grad_c)
        df_square_c = pd.DataFrame(data_square_c)
        self.my_plot([self.df_grad, df_grad_c], ['discrete', 'continuous'], 'Parameter Comparison'+str(order)+'order', y='norm')
        self.my_plot([self.df_square_avg, df_square_c], ['discrete', 'continuous'], 'Square_avg Comparison'+str(order)+'order', y='norm')

    def plot_loss(self, Loss, order=1):
        df_cont, df_discr = self.prepare_loss_or_grad(Loss, type='Loss')
        self.my_plot([df_cont, df_discr], ['continuous', 'discrete'], 'Loss Comparison_'+str(order)+'_order', y='Loss')

    def plot_grad(self, Grad, order=1):
        df_cont, df_discr = self.prepare_loss_or_grad(Grad, type='Expected_loss_gradient')
        self.my_plot([df_discr, df_cont], ['discrete', 'continuous'], 'Gradient Comparison '+str(order)+' order', y='Expected_loss_gradient')

    def prepare_loss_or_grad(self, data, type ='Loss'):
        data = torch.stack(data)
        if type == 'Expected_loss_gradient':
            data = torch.norm(data, dim=2)

        data_cont = []
        for i_run in range(data.shape[0]):
            for i_step in range(data.shape[1]):
                data_cont.append({'step': i_step, type: data[i_run, i_step].item(), 'run': i_run})

        data_disc = []
        for i_run in self.FinalDict.keys():
            data = self.FinalDict[i_run][type]        
            for i_step, l in enumerate(data):
                if isinstance(l, torch.Tensor):
                    l = l.item()
                data_disc.append({'step': i_step, type: l, 'run': i_run})
                
        return pd.DataFrame(data_cont), pd.DataFrame(data_disc)

    def plot_comparison_1_and_2_order(self, res_1, res_2, Loss_1, Loss_2, FinalDict):
        def aux_res(result, FinalDict):
            result = torch.stack(result).squeeze()
            if len(result.shape) == 2:
                result = result.unsqueeze(0)
            number_parameters = result.shape[2] // 3
            parameter = result[:, :, :number_parameters]
            square_avg = result[:, :, number_parameters:2*number_parameters]
            parameter_norm_c = torch.norm(parameter, dim = 2)
            square_avg_norm_c = torch.norm(square_avg, dim = 2)
            parameter_norm_mean_c =torch.mean(parameter_norm_c, dim = 0)
            square_avg_norm_mean_c = torch.mean(square_avg_norm_c, dim = 0)
            
            parameter_norm_d, square_avg_norm_d = [], []
            for i in range(parameter.shape[0]):
                i_run = i+1
                parameter_norm_d.append(torch.norm(FinalDict[i_run]['Params'], dim = 1))
                square_avg_norm_d.append(torch.norm(FinalDict[i_run]['Square_avg'], dim = 1))
            parameter_norm_mean_d = torch.mean(torch.stack(parameter_norm_d), dim = 0)
            square_avg_norm_mean_d = torch.mean(torch.stack(square_avg_norm_d), dim = 0)

            error_parameter = torch.abs(parameter_norm_mean_d - parameter_norm_mean_c)
            error_square_avg = torch.abs(square_avg_norm_mean_d - square_avg_norm_mean_c)
            data_grad = []
            data_square_avg = []
            for step in range(parameter_norm_mean_c.shape[0]):
                data_grad.append({'step': step, 'error': error_parameter[step].item()})
                data_square_avg.append({'step': step, 'error': error_square_avg[step].item()})
            return pd.DataFrame(data_grad), pd.DataFrame(data_square_avg)
        def aux_loss(loss, FinalDict):
            loss_c = torch.stack(loss)
            loss_c = torch.mean(loss_c, dim = 0)
            loss_d = []
            for i in range(len(loss)):
                i_run = i+1
                loss_d.append(torch.tensor(FinalDict[i_run]['Loss']))
            loss_d = torch.stack(loss_d)
            loss_d = torch.mean(loss_d, dim = 0)
            error = torch.abs(loss_d - loss_c)
            data = []
            for step in range(loss_c.shape[0]):
                data.append({'step': step, 'error': error[step].item()})
            return pd.DataFrame(data)
        gradient_norm_1, square_avg_norm_1 = aux_res(res_1, FinalDict)
        gradient_norm_2, square_avg_norm_2 = aux_res(res_2, FinalDict)
        self.my_plot([gradient_norm_1, gradient_norm_2], ['1 order', '2 order'], 'Parameter Comparison', y = 'error')
        self.my_plot([square_avg_norm_1, square_avg_norm_2], ['1 order', '2 order'], 'Square_avg Comparison', y = 'error')
        loss_error_1 = aux_loss(Loss_1, FinalDict)
        loss_error_2 = aux_loss(Loss_2, FinalDict)
        self.my_plot([loss_error_1, loss_error_2], ['1 order', '2 order'], 'Loss Comparison', y = 'error')


class LoadExp():
    def __init__(self, folder_path, nomralization = True):
        self.folder_path = folder_path
        self.normalization = nomralization
        
    def load_FinalDict(self):
        with open(os.path.join(self.folder_path, 'FinalDict.json'), 'r') as file:
            final_dict = json.load(file)
        aux = convert_lists_to_tensors(final_dict)
        self.FinalDict = {int(k) : v for k, v in aux.items()}
        return self.FinalDict
    
    def _load_file_pt(self, name):
        path = os.path.join(self.folder_path, name)
        return torch.load(path)
    
    def load_result(self, name):
        self.result = self._load_file_pt(name)
        return self.result
    
    def load_loss(self, name):
        self.loss = self._load_file_pt(name)
        return self.loss
    
    def load_grad(self, name):
        self.grad = self._load_file_pt(name)
        return self.grad
    
    def plot(self):
        self.pl = plot(self.folder_path, self.normalization)
        self.pl.plot_norm_discrete(self.FinalDict)
        parameter_norm, square_avg_norm = prepare_result_continuous(self.result)
        self.pl.plot_norm_cont(parameter_norm, square_avg_norm)

    def plot_loss(self):
        self.pl.plot_loss(self.loss)



def MergeExp(path_folder_1, path_folder_2, final_path):
    df1 = pd.read_csv(os.path.join(path_folder_1, 'data.csv'))
    df2 = pd.read_csv(os.path.join(path_folder_2, 'data.csv'))

    checks = ['eta', 'beta', 'steps', 'conv_layers', 'batch_size', 'dataset_size', 'size_img', 'model', 'number_parameters', 't0', 't1']
    for check in checks: assert (df1.loc[df1['Key'] == check].values[0] == df2.loc[df2['Key'] == check].values[0]).all(), f'{check} is different'
    
    key_from_1 = df1['Key'].values
    run_from_1 = [int(re.search(r'time for run (\d+)', key).group(1)) for key in key_from_1 if re.search(r'time for run (\d+)', key)]
    max_number_run = max(run_from_1)

    dic_from_2 = {}
    for key, value in zip(df2['Key'], df2['Value']):
        match = re.match(r'time for run (\d+)', key)
        if match:
            run_number = int(match.group(1))
            new_key = f'time for run {max_number_run + run_number}'
            dic_from_2[new_key] = value

    df1 = pd.concat([df1, pd.DataFrame(dic_from_2.items(), columns=['Key', 'Value'])], ignore_index=True)

    load1 = LoadExp(path_folder_1)
    load2 = LoadExp(path_folder_2)
    FinalDict1 = load1.load_FinalDict()
    FinalDict2 = load2.load_FinalDict()
    result1 = load1.load_result('result.pt')
    result2 = load2.load_result('result.pt')
    loss1 = load1.load_loss('Loss.pt')
    loss2 = load2.load_loss('Loss.pt')
    grad1 = load1.load_grad('grad.pt')
    grad2 = load2.load_grad('grad.pt')

    num_keys_1 = len(FinalDict1)
    dict = {k+num_keys_1: v for k, v in FinalDict2.items()}
    FinalDict = {**FinalDict1, **dict}

    result = result1 + result2
    loss = loss1 + loss2
    grad = grad1 + grad2

    save = SaveExp(final_path)
    save.save_result_discrete(FinalDict)
    save.save_result_continuous(result)
    save.save_loss_sde(loss)
    save.partial_result(grad, 'grad', Bool = True)
    df1.to_csv(os.path.join(save.path_folder, 'data.csv'), index=False)
    

if __name__ == "__main__":
    Path = {}
    for i in range(1, 10):
        Path[i] = f'/home/callisti/Thesis/Master-Thesis/Result3/Experiment_{i}'

    final_path = '/home/callisti/Thesis/Master-Thesis/Result3'
    MergeExp(Path[1], Path[2], final_path)
    MergeExp(Path[6], Path[3], final_path)
    MergeExp(Path[7], Path[4], final_path)
    MergeExp(Path[8], Path[5], final_path)

        