import os
import json
import torch
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt


class SaveExp():
    def __init__(self, path_folder, n_experminet):        
        current_exp_folder_name = f'Experiment_{n_experminet}'
        self.path_folder = os.path.join(path_folder, current_exp_folder_name)

        if os.path.exists(self.path_folder):
            existing_simulations = [d for d in os.listdir(self.path_folder) if os.path.isdir(os.path.join(self.path_folder, d)) and d.startswith('Simulation_')]
            simulation_numbers = [int(d.split('_')[1]) for d in existing_simulations if d.split('_')[1].isdigit()]
            next_simulation_number = max(simulation_numbers, default=0) + 1
            self.path_folder = os.path.join(self.path_folder, f'Simulation_{next_simulation_number}')
        else:
            os.makedirs(self.path_folder)
            self.path_folder = os.path.join(self.path_folder, 'Simulation_0')
        os.makedirs(self.path_folder)

        self.data = {}
        self.FinalDict_n_sim = {}
        

    def add_element(self, key, value):
        self.data[key] = value

    def add_multiple_elements(self, list_key, list_value):
        for key, value in zip(list_key, list_value):
            self.data[key] = value

    def save_result_discrete(self, FinalDict):
        final_dict_serializable = convert_tensors_to_lists(FinalDict)
        with open(os.path.join(self.path_folder, f'FinalDict_simulation.json'), 'w') as file:
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
    

class Merge_simulation():
    def __init__(self, path_folder):
        self.path_folder = path_folder
        existing_simulations = [d for d in os.listdir(self.path_folder) if os.path.isdir(os.path.join(self.path_folder, d)) and d.startswith('Simulation_')]
        self.simulation_numbers = [int(d.split('_')[1]) for d in existing_simulations if d.split('_')[1].isdigit()]

    def merge_data(self):
        merged_data = pd.DataFrame()
        run_counter = 0

        for i in self.simulation_numbers:
            simulation_folder = os.path.join(self.path_folder, f'Simulation_{i}')
            data_file = os.path.join(simulation_folder, 'data.csv')

            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                key_from = df['Key'].values
                run_from = [int(re.search(r'time for run (\d+)', key).group(1)) for key in key_from if re.search(r'time for run (\d+)', key)]
                if run_counter == 0:
                    run_counter = max(run_from)
                else:
                    new_data = {}
                    for key, value in zip(df['Key'], df['Value']):
                        match = re.match(r'time for run (\d+)', key)
                        if match:
                            run_number = int(match.group(1))
                            new_key = f'time for run {run_counter + run_number}'
                            new_data[new_key] = value
                    df = pd.DataFrame(list(new_data.items()), columns=['Key', 'Value'])                    

                merged_data = pd.concat([merged_data, df], ignore_index=True)

        merged_data.to_csv(os.path.join(self.path_folder, 'data_merged.csv'), index=False)
        return merged_data

    def merge_final_dicts(self):
        merged_final_dict = {}
        current_key = 1

        for i in self.simulation_numbers:
            simulation_folder = os.path.join(self.path_folder, f'Simulation_{i}')
            final_dict_file = os.path.join(simulation_folder, 'FinalDict_simulation.json')

            if os.path.exists(final_dict_file):
                with open(final_dict_file, 'r') as file:
                    final_dict = json.load(file)
                    for key in sorted(final_dict.keys(), key=int):
                        merged_final_dict[current_key] = final_dict[key]
                        current_key += 1

        with open(os.path.join(self.path_folder, 'FinalDict_merged.json'), 'w') as file:
            json.dump(merged_final_dict, file, indent=4)
        return merged_final_dict
    
    def merge_list(self, name):
        merged_list = []
        for i in self.simulation_numbers:
            simulation_folder = os.path.join(self.path_folder, f'Simulation_{i}')
            list_file = os.path.join(simulation_folder, name)
            list = torch.load(list_file, weights_only=True)
            merged_list.extend(list)
        torch.save(merged_list, os.path.join(self.path_folder, name[:-3] + '_merged.pt'))
        return merged_list

    def merge_results(self, Plot = True):
        data_csv = self.merge_data()
        grad_1_order = self.merge_list('Grad_1_order.pt')
        loss_1_order = self.merge_list('Loss_1_order.pt')
        result_1_order = self.merge_list('Result_1_order.pt')
        grad_2_order = self.merge_list('Grad_2_order.pt')
        loss_2_order = self.merge_list('Loss_2_order.pt')
        result_2_order = self.merge_list('Result_2_order.pt')
        FinalDict = self.merge_final_dicts()
        self.FinalError(data_csv, FinalDict, loss_1_order, loss_2_order, result_1_order, result_2_order)

        if Plot:
            FinalDict = convert_lists_to_tensors(FinalDict)
            plot = My_plot(self.path_folder)
            plot.plot_loss(loss_1_order, 1, FinalDict)
            plot.plot_loss(loss_2_order, 2, FinalDict)
            plot.plot_grad(grad_1_order, 1)
            plot.plot_grad(grad_2_order, 2)
            plot.plot_comparison_1_and_2_order(result_1_order, result_2_order, loss_1_order, loss_2_order, FinalDict)
            plot.plot_Brownian_motion(result_2_order)

    def FinalError(self, data_csv, FinalDict, Loss_1, Loss_2, Result_1, Result_2, step = -1):
        data = {'eta': float(data_csv.loc[data_csv['Key'] == 'eta', 'Value'].values[0])}
        total_step = float(data_csv.loc[data_csv['Key'] == 'steps', 'Value'].values[0])
        if step == -1:
            step = int(total_step)
        assert step <= total_step, 'Step is bigger than the total number of steps'
        step = step-1
        data['step'] = step
        print('Step:', step)
        def aux(tensors):
            if isinstance(tensors, list): tensors = torch.stack(tensors)
            tensor = tensors[:, step]
            if len(tensor.shape) == 2: 
                tensor = torch.norm(tensor, dim = 1)
            mean_tensor = torch.mean(tensor, dim = 0)
            return mean_tensor

        # final_l1 = aux(Loss_1)
        # final_l2 = aux(Loss_2)
        Result_1 = torch.stack(Result_1).squeeze()
        number_parameters = Result_1[0].shape[1] // 3
        final_p1 = aux(Result_1[:, :, :number_parameters])
        final_v1 = aux(Result_1[:, :, number_parameters:2*number_parameters])
        Result_2 = torch.stack(Result_2).squeeze()
        final_p2 = aux(Result_2[:, :,:number_parameters])
        final_v2 = aux(Result_2[:, :,number_parameters:2*number_parameters])

        loss_d, Params_d, v_d = [], [], []
        for k in FinalDict.keys():
            loss_d.append(FinalDict[k]['Loss'][step])
            Params_d.append(torch.tensor(FinalDict[k]['Params'][step]))
            v_d.append(torch.tensor(FinalDict[k]['Square_avg'][step]))
        final_l_d = torch.mean(torch.tensor(loss_d))
        final_p_d = torch.mean(torch.norm(torch.stack(Params_d, dim = 0), dim = 1))
        final_v_d = torch.mean(torch.norm(torch.stack(v_d, dim = 0), dim = 1))

        # data['Error_Loss_1'] = torch.abs(final_l1 - final_l_d).item()
        # data['Error_Loss_2'] = torch.abs(final_l2 - final_l_d).item()
        data['Error_Params_order_1'] = torch.abs(final_p1 - final_p_d).item()
        data['Error_Params_order_2'] = torch.abs(final_p2 - final_p_d).item()
        data['Error_Square_avg_order_1'] = torch.abs(final_v1 - final_v_d).item()
        data['Error_Square_avg_order_2'] = torch.abs(final_v2 - final_v_d).item()
        with open(os.path.join(self.path_folder, 'FinalError.json'), 'w') as file:
            json.dump(data, file, indent=4)


class My_plot():
    def __init__(self, path_folder):
        self.path_folder = path_folder
    
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
        path = os.path.join(self.path_folder, f'{y}_{title}.png')
        plt.savefig(path)
        plt.close()

    def plot_loss(self, Loss, order=1, FinalDict=None):
        if FinalDict is not None: self.FinalDict = FinalDict
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
            parameter_norm_mean_c =torch.mean(parameter_norm_c, dim = 0).cpu()
            square_avg_norm_mean_c = torch.mean(square_avg_norm_c, dim = 0).cpu()
            
            parameter_norm_d, square_avg_norm_d = [], []
            for i in range(parameter.shape[0]):
                i_run = i+1
                parameter_norm_d.append(torch.norm(FinalDict[i_run]['Params'], dim = 1))
                square_avg_norm_d.append(torch.norm(FinalDict[i_run]['Square_avg'], dim = 1))
            parameter_norm_mean_d = torch.mean(torch.stack(parameter_norm_d), dim = 0).cpu()
            square_avg_norm_mean_d = torch.mean(torch.stack(square_avg_norm_d), dim = 0).cpu()

            error_parameter = torch.abs(parameter_norm_mean_d - parameter_norm_mean_c)
            error_square_avg = torch.abs(square_avg_norm_mean_d - square_avg_norm_mean_c)
            data_grad = []
            data_square_avg = []
            for step in range(parameter_norm_mean_c.shape[0]):
                data_grad.append({'step': step, 'error': error_parameter[step].item()})
                data_square_avg.append({'step': step, 'error': error_square_avg[step].item()})
            return pd.DataFrame(data_grad), pd.DataFrame(data_square_avg)
        def aux_loss(loss, FinalDict):
            warning = ''
            loss_c = torch.stack(loss)
            loss_c = torch.mean(loss_c, dim = 0)
            loss_d = []
            for i in range(len(loss)):
                i_run = i+1
                loss_d.append(FinalDict[i_run]['Loss'].clone().detach())
            loss_d = torch.stack(loss_d)
            loss_d = torch.mean(loss_d, dim = 0)
            if loss_d.shape[0] != loss_c.shape[0]:
                print('error in loss shape', loss_d.shape, loss_c.shape)
                min_shape = min(loss_d.shape[0], loss_c.shape[0])
                loss_d = loss_d[:min_shape]
                loss_c = loss_c[:min_shape]
                warning = 'cutted'
            error = torch.abs(loss_d - loss_c)
            data = []
            for step in range(loss_c.shape[0]):
                data.append({'step': step, 'error': error[step].item()})
            return pd.DataFrame(data), warning
        gradient_norm_1, square_avg_norm_1 = aux_res(res_1, FinalDict)
        gradient_norm_2, square_avg_norm_2 = aux_res(res_2, FinalDict)
        self.my_plot([gradient_norm_1, gradient_norm_2], ['1 order', '2 order'], 'Parameter Comparison', y = 'error')
        self.my_plot([square_avg_norm_1, square_avg_norm_2], ['1 order', '2 order'], 'Square_avg Comparison', y = 'error')
        loss_error_1, w1 = aux_loss(Loss_1, FinalDict)
        loss_error_2, w2 = aux_loss(Loss_2, FinalDict)
        if 'cutted' in w1 or 'cutted' in w2: warning = 'cutted'
        else: warning = ''
        self.my_plot([loss_error_1, loss_error_2], ['1 order', '2 order'], 'Loss Comparison'+warning, y = 'error')

    def plot_Brownian_motion(self, result):
        result = torch.stack(result).squeeze()
        if len(result.shape) == 2:
            result = result.unsqueeze(0)
        number_parameters = result.shape[2] // 3
        Bmotion = result[:, :, 2*number_parameters:]
        BM_norm = torch.norm(Bmotion, dim = 2)
        data = []
        for i in range(BM_norm.shape[0]):
            for step in range(BM_norm.shape[1]):
                data.append({'step': step, 'BM_norm': BM_norm[i, step].item(), 'run': i})
        df = pd.DataFrame(data)
        self.my_plot([df], [''], 'Brownian_motion', y = 'BM_norm')

def Plot_error_different_eta(path_folder, subfolders_number = None):
    if subfolders_number is None:
        subfolders_number = [int(d.split('_')[1]) for d in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, d)) and d.startswith('Experiment_') and d.split('_')[1].isdigit()]
    all_errors = []
    for num in subfolders_number:
        final_error_file = os.path.join(path_folder, f'Experiment_{num}/FinalError.json')
        if os.path.exists(final_error_file):
            with open(final_error_file, 'r') as file:
                final_error_data = json.load(file)
                all_errors.append(final_error_data)

    df_errors = pd.DataFrame(all_errors)
    def aux(df, x, y1, y2, title):
        sns.scatterplot(data=df, x=x, y=y1, label=y1)
        sns.scatterplot(data=df, x=x, y=y2, label=y2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Eta')
        plt.title(title)
        plt.savefig(os.path.join(path_folder, title))
        plt.close()
    print(df_errors)
    aux(df_errors, 'eta', 'Error_Params_order_1', 'Error_Params_order_2', 'Error_Params')
    aux(df_errors, 'eta', 'Error_Square_avg_order_1', 'Error_Square_avg_order_2', 'Error_Square_avg')

if __name__ == "__main__":
    Numbers = [1,2,3, 4, 5]
    for num in Numbers:
        print('Experiment number:', num)
        path_folder = '/home/callisti/Thesis/Master-Thesis/Result_new/Slope_c_0.5/Experiment_'+str(num)
        merge = Merge_simulation(path_folder)
        merge.merge_data()
        merge.merge_final_dicts()
        merge.merge_results()

    path_folder = '/home/callisti/Thesis/Master-Thesis/Result_new/Slope_c_0.5'
    Plot_error_different_eta(path_folder, subfolders_number= Numbers)
