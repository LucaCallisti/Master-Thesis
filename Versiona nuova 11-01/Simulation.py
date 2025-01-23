from CNN import CNN, Train_n_times
from SDE import RMSprop_SDE, RMSprop_SDE_1_order
from Dataset import CIFAR10Dataset
from calculation import function_f_and_Sigma


import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
import torchsde


def Simulation_discrete_dynamics(model, dataset, steps, lr, beta, n_runs, batch_size=1):
    print('model parameters:', model.get_number_parameters())
    print('dimension dataset:', dataset.x_train.shape[0])
    print('number of epoch:', steps * batch_size / dataset.x_train.shape[0], ' final time:', steps * lr)
    Train = Train_n_times(model, dataset, steps=steps, lr=lr, optimizer_name='RMSPROP', beta = beta)
    FinalDict = Train.train_n_times(n = n_runs, batch_size=batch_size)
    return FinalDict

class plot():
    def __init__(self):
        pass

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
        plt.title(f'Norm {title}: {format(time.strftime("%Y-%m-%d %H:%M:%S"))}')
        plt.savefig('/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/norm_'+title+'_discrete.png')
    

    def plot_norm_discrete(self, FinalDict):
        n_step, _ = FinalDict[1]['Params'].shape
        data_grad = []
        data_sqare_avg = []
        for i_run in FinalDict.keys():
            params = FinalDict[i_run]['Params']
            print(params.shape)
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



def simulation():
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    dataset.x_train = dataset.x_train[:512]
    dataset.y_train = dataset.y_train[:512]
    conv_layers = [
        (3, 3, 1, 1),  # filter number, kernel size, stride, padding
        (2, 3, 1, 1),
        (2, 3, 1, 1),
        (1, 3, 1, 1)
    ]
    conv_layers = [
        (1, 3, 1, 1),  
        (1, 3, 1, 1),
        (1, 3, 1, 1),
        (1, 3, 1, 1)
    ]
    # conv_layers = [
    #     (4, 3, 1, 1),  
    #     (8, 3, 1, 1),
    #     (16, 3, 1, 1),
    #     (32, 3, 1, 1)
    # ]
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    steps = 20
    eta, beta = 0.1, 0.99
    FinalDict = Simulation_discrete_dynamics(model, dataset, steps, lr=eta, beta = beta, n_runs=3, batch_size=16)
    torch.save(FinalDict, '/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/Risultati_discrete.pt')
    pl = plot()
    pl.plot_norm_discrete(FinalDict)


    t0, t1 = 0.0, (steps * eta)  
    t = torch.linspace(t0, t1, steps)

    number_parameters = FinalDict[1]['Params'][0].shape[0]
    x0 = torch.zeros(3 * number_parameters)
    result = []
    start = time.time()
    Loss = []  
    for i_run in FinalDict.keys():
        model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
        f = function_f_and_Sigma(model, dataset, dim_dataset = 512, Verbose=False)
        print(f'Run {i_run}')
        x0[: 2*number_parameters] = torch.cat((FinalDict[i_run]['Params'][0], FinalDict[i_run]['Square_avg'][0]), dim=0)
        if i_run != 1:
            print('Error', i_run, torch.norm(last - x0[:2*number_parameters]))
        last = x0[:2*number_parameters]
        if False:
            sde = RMSprop_SDE_1_order(eta, beta, f, Verbose=False)
            res = torchsde.sdeint(sde, x0.unsqueeze(0).to('cuda'), t, method = 'euler', dt =eta)
            loss = sde.get_loss()
            # Loss.append([{'step': i / loss.shape[0] * (eta * steps), 'loss': l.item()} for i, l in enumerate(loss)])
            Loss.append(loss)
            result.append(res)
            torch.save(result, f'/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/Risultati_sde.pt')
            
        else:
            sde = RMSprop_SDE(eta, beta, f, Verbose=False)
            res = torchsde.sdeint(sde, x0.unsqueeze(0).to('cuda'), t, method = 'euler', dt =eta**2)
            loss = sde.get_loss()
            Loss.append(loss)
            result.append(res)
            torch.save(result, f'/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/Risultati_sde.pt')

    breakpoint()
    print(f'Elapsed time for simulating SDE: {time.time() - start:.2f}')
    result = torch.stack(result).squeeze()
    if len(result.shape) == 2:
        result = result.unsqueeze(0)
    gradient = result[:, :, :number_parameters]
    square_avg = result[:, :, number_parameters:2*number_parameters]
    print(gradient.shape, square_avg.shape)
    gradient_norm = torch.norm(gradient, dim=2)
    square_avg_norm = torch.norm(square_avg, dim=2)
    pl.plot_norm_cont(gradient_norm, square_avg_norm)
    
    plot_loss(Loss, FinalDict)






def plot_loss(Loss, FinalDict, eta = 0.1):
    Loss = torch.stack(Loss)
    Loss = Loss[:, 1:]

    len_loss_cont = Loss.shape[1]
    len_loss_discr = len(FinalDict[1]['Loss'])

    temp = len_loss_cont // len_loss_discr
    Loss = Loss[:, ::temp]

    data_loss_cont = []
    for i_run in range(Loss.shape[0]):
        for i_step in range(Loss.shape[1]):
            data_loss_cont.append({'step': i_step * eta, 'loss': Loss[i_run, i_step].item(), 'run': i_run})
    data_loss_disc = []
    for i_run in FinalDict.keys():
        loss = FinalDict[i_run]['Loss']
        for i_step, l in enumerate(loss):
            data_loss_disc.append({'step': i_step * eta, 'loss': l, 'run': i_run})

    df_loss_cont = pd.DataFrame(data_loss_cont)
    df_loss_disc = pd.DataFrame(data_loss_disc)
    plt.figure()
    sns.lineplot(data=df_loss_cont, x='step', y='loss', ci=95, label='continuous')
    sns.lineplot(data=df_loss_disc, x='step', y='loss', ci=95, label='discrete')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss Comparison: {format(time.strftime("%Y-%m-%d %H:%M:%S"))}')
    plt.savefig('/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/loss_comparison.png')
    plt.show()

def plot_load():
    FinalDict = torch.load('/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/Risultati_discrete.pt')
    pl = plot()
    pl.plot_norm_discrete(FinalDict)
    number_parameters = FinalDict[1]['Params'][0].shape[0]

    result = []
    result = torch.load(f'/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/Risultati_sde_{FinalDict.keys()[-1]}.pt')
    breakpoint()
    result = torch.stack(result).squeeze()

    gradient = result[:, :, :number_parameters]
    square_avg = result[:, :, number_parameters:2*number_parameters]

    gradient_norm = torch.norm(gradient, dim=2)
    square_avg_norm = torch.norm(square_avg, dim=2)
    pl.plot_norm_cont(gradient_norm, square_avg_norm)

if __name__ == '__main__':
    simulation()
    # plot_load()