from CNN import CNN, Train_n_times
from SDE import RMSprop_SDE, RMSprop_SDE_1_order
from Dataset import CIFAR10Dataset
from calculation import function_f_and_Sigma
import Save_exp
import sys


import torch
import numpy as np
import time
import torchsde


sys.setrecursionlimit(10000)

def Simulation_discrete_dynamics(model, dataset, steps, lr, beta, n_runs, batch_size=1):
    print('model parameters:', model.get_number_parameters())
    print('dimension dataset:', dataset.x_train.shape[0])
    print('number of epoch:', steps * batch_size / dataset.x_train.shape[0], ' final time:', steps * lr)
    Train = Train_n_times(model, dataset, steps=steps, lr=lr, optimizer_name='RMSPROP', beta = beta)
    FinalDict = Train.train_n_times(n = n_runs, batch_size=batch_size)
    return FinalDict

def simulation(load = False):
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    dataset_size = 128
    dataset.x_train = dataset.x_train[:dataset_size]
    dataset.y_train = dataset.y_train[:dataset_size]
    save_funz = Save_exp.SaveExp('/home/callisti/Thesis/Master-Thesis/Result3')

    conv_layers = [
        (3, 3, 1, 1),  # filter number, kernel size, stride, padding
        (2, 3, 1, 1),
        (2, 3, 1, 1),
        (1, 3, 1, 1)
    ]

    random_seed = torch.seed()
    torch.manual_seed(random_seed)

    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    steps, n_runs = 50, 2
    eta, beta = 0.1, 0.9
    print(eta, beta)
    FinalDict = Simulation_discrete_dynamics(model, dataset, steps, lr=eta, beta = beta, n_runs=n_runs, batch_size=1)
    save_funz.save_result_discrete(FinalDict)
    save_funz.add_element('optimizer', RMSprop_SDE)
    save_funz.add_element('eta', eta)
    save_funz.add_element('beta', beta)
    save_funz.add_element('steps', steps)
    save_funz.add_element('n_runs', n_runs)
    save_funz.add_element('batch_size', 1)
    save_funz.add_element('dataset_size', dataset_size)
    save_funz.add_element('size_img', size_img)
    save_funz.add_element('model', 'CNN')
    save_funz.add_element('conv_layers', conv_layers)

    t0, t1 = 0.0, (steps * eta)  
    t = torch.linspace(t0, t1, steps)
    save_funz.add_element('t0', t0)
    save_funz.add_element('t1', t1)


    number_parameters = FinalDict[1]['Params'][0].shape[0]
    save_funz.add_element('number_parameters', number_parameters)

    x0 = torch.zeros(3 * number_parameters)

    result = []
    start_tot = time.time()
    Loss = []  
    Grad = []
    for i_run in FinalDict.keys():
        model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
        f = function_f_and_Sigma(model, dataset, dim_dataset = 512, Verbose=False)
        if not load:
            x0[: 2*number_parameters] = torch.cat((FinalDict[i_run]['Params'][0], FinalDict[i_run]['Square_avg'][0]), dim=0)
        if load:
            x0 = torch.load(f'/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/Starting_point_{i_run}.pt')
        save_funz.partial_result(x0, f'starting_point_{i_run}', Bool = True)
        print(f'Run {i_run}')

        start = time.time()

        sde = RMSprop_SDE(eta, beta, f, final_time= t1, Verbose=False)
        res = torchsde.sdeint(sde, x0.unsqueeze(0).to('cuda'), t, method = 'euler', dt =eta**2)
        loss = sde.get_loss()
        Loss.append(loss)
        grad = sde.get_loss_grad()
        Grad.append(grad)
        result.append(res)

        save_funz.partial_result(result, 'result', Bool = True)
        save_funz.partial_result(Grad, 'grad', Bool = True)
        save_funz.partial_result(Loss, 'Loss', Bool = True)

        if i_run & 3 == 0 or i_run == 1:
            save_funz.save_result_continuous(result)
            save_funz.save_loss_sde(Loss)

        elapsed_time = time.time() - start
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        save_funz.add_element(f'time for run {i_run}', f'{int(hours):02}:{int(minutes):02}:{seconds:05.2f}')
        save_funz.save_dict()

    elapsed_time_tot = time.time() - start_tot
    hours_tot, rem_tot = divmod(elapsed_time_tot, 3600)
    minutes_tot, seconds_tot = divmod(rem_tot, 60)
    save_funz.add_element('total_time', f'{int(hours_tot):02}:{int(minutes_tot):02}:{seconds_tot:05.2f}')

    print(f'Elapsed time for simulating SDE: {time.time() - start:.2f}')
    save_funz.save_result_continuous(result)
    save_funz.save_loss_sde(Loss)
    save_funz.save_dict()

def Load(num_exp, num_run):
    load = Save_exp.LoadExp(f'/home/callisti/Thesis/Master-Thesis/Result3/Experiment_{num_exp}', nomralization=True)
    FinalDict = load.load_FinalDict()
    if num_run >= 0:
        result = load.load_result(f'result_{num_run}.pt')
        loss = load.load_loss(f'loss_{num_run}.pt')
        grad = load.load_grad(f'grad_{num_run}.pt')
    elif num_run == -1:
        result = load.load_result('result.pt')
        loss = load.load_loss('Loss.pt')
        grad = load.load_grad('grad.pt')
    breakpoint()
    load.plot()
    load.plot_loss()    

if __name__ == "__main__":
    # simulation()
    Load(num_exp = 9, num_run=-1)