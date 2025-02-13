from CNN import CNN, Train_n_times
from SDE import RMSprop_SDE, RMSprop_SDE_1_order, Regularizer_Phi, Regularizer_Phi_cost, Regularizer_arora, Regularizer_identity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Dataset import CIFAR10Dataset
from calculation import function_f_and_Sigma
import Save_exp_sequential as Save_exp
import sys


import torch
import numpy as np
import time
import torchsde


sys.setrecursionlimit(10000)

def return_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds

def Simulation_discrete_dynamics(model, dataset, steps, lr, beta, n_runs, batch_size=1):
    print('model parameters:', model.get_number_parameters())
    print('dimension dataset:', dataset.x_train.shape[0])
    print('number of epoch:', steps * batch_size / dataset.x_train.shape[0], ' final time:', steps * lr)
    Train = Train_n_times(model, dataset, steps=steps, lr=lr, optimizer_name='RMSPROP', beta = beta)
    FinalDict = Train.train_n_times(n = n_runs, batch_size=batch_size)
    return FinalDict

def simulation(eta, beta, n_experiment, folder_name = ''):
    dataset = CIFAR10Dataset()
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    dataset_size = dataset.x_train.shape[0]

    if folder_name != '': folder_name = folder_name + '/'
    save_funz = Save_exp.SaveExp('/home/callisti/Thesis/Master-Thesis/Result_new/'+folder_name, n_experiment)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    conv_layers = [
        (3, 3, 1, 1),  # filter number, kernel size, stride, padding
        (2, 3, 1, 1),
        (2, 3, 1, 1),
        (1, 3, 1, 1)
    ]

    steps, n_runs = 50, 1
    # eta, beta = 0.1
    # beta = 1-0.5*eta
    steps = round(3/eta)
    print(f'eta: {eta}, beta: {beta}, steps: {steps}, n_runs: {n_runs}, total steps per run: {steps / eta}')

    random_seed = torch.seed()
    torch.manual_seed(random_seed)

    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    FinalDict = Simulation_discrete_dynamics(model, dataset, steps, lr=eta, beta = beta, n_runs=n_runs, batch_size=1)
    save_funz.save_result_discrete(FinalDict)
    t0, t1 = eta, (steps * eta)  
    t = torch.linspace(t0, t1, steps)
    number_parameters = FinalDict[1]['Params'][0].shape[0]
    c=(1-eta)/beta
    save_funz.add_multiple_elements(['optimizer', 'eta', 'beta', 'c', 'steps', 'n_runs', 'batch_size', 'dataset_size', 'size_img', 'model', 'conv_layers', 't0', 't1', 'number_parameters'], [RMSprop_SDE, eta, beta, c, steps, n_runs, 1, dataset_size, size_img, 'CNN', conv_layers, t0, t1, number_parameters])

    eps_reg = 1e-8
    Reg_sde_2 = Regularizer_Phi(eps_reg)
    Reg_sde_1 = Regularizer_Phi(eps_reg)
    min_value = 1e-2
    # save_funz.add_multiple_elements(['Regularizer_1_order', 'Regularizer_2_order',  'min value init v_0'], [Reg_sde_1, Reg_sde_2, min_value])
    save_funz.add_multiple_elements(['Regularizer_1_order', 'Regularizer_2_order', 'eps_regularizer', 'min value init v_0'], [Reg_sde_1, Reg_sde_2, eps_reg, min_value])


    x0 = torch.zeros(3 * number_parameters)

    Result_1_order = []
    Result_2_order = []
    start_tot = time.time()
    Loss_1_order = []  
    Loss_2_order = []
    Grad_1_order = []
    Grad_2_order = []
    for i_run in FinalDict.keys():
        start = time.time()
        print(f'Run {i_run}')
        x0[: 2*number_parameters] = torch.cat((FinalDict[i_run]['Params'][0], FinalDict[i_run]['Square_avg'][0]), dim=0)
        x0[number_parameters:2*number_parameters] = torch.clamp(x0[number_parameters:2*number_parameters], min=min_value)        
        save_funz.save_tensor(x0, f'starting_point_{i_run}.pt')

        model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
        f = function_f_and_Sigma(model, dataset, dim_dataset = 512, Verbose=False)
        sde_1_order = RMSprop_SDE_1_order(eta, beta, f, All_time = t, regularizer=Reg_sde_1, Verbose=False)
        t_extended = torch.cat((t, torch.tensor([t[-1] + eta**2 / 10], device=t.device)))
        aux = torchsde.sdeint(sde_1_order, x0.unsqueeze(0).to(device), t_extended, method = 'euler', dt =eta)
        Result_1_order.append(aux[:-1, :, :])
        Loss_1_order.append(sde_1_order.get_loss())
        Grad_1_order.append(sde_1_order.get_loss_grad())

        model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
        f = function_f_and_Sigma(model, dataset, dim_dataset = 512, Verbose=False)
        sde_2_order = RMSprop_SDE(eta, beta, f, All_time = t, regularizer=Reg_sde_2, Verbose=False)
        aux = torchsde.sdeint(sde_2_order, x0.unsqueeze(0).to(device), t_extended, method = 'euler', dt =eta**2)
        Result_2_order.append(aux[:-1, :, :])
        Loss_2_order.append(sde_2_order.get_loss())
        Grad_2_order.append(sde_2_order.get_loss_grad())

        hours, minutes, seconds = return_time(time.time() - start)
        save_funz.add_element(f'time for run {i_run}', f'{int(hours):02}:{int(minutes):02}:{seconds:05.2f}')
    
    save_funz.save_tensor(Result_1_order, 'Result_1_order.pt')
    save_funz.save_tensor(Result_2_order, 'Result_2_order.pt')
    save_funz.save_tensor(Loss_1_order, 'Loss_1_order.pt')
    save_funz.save_tensor(Loss_2_order, 'Loss_2_order.pt')
    save_funz.save_tensor(Grad_1_order, 'Grad_1_order.pt')
    save_funz.save_tensor(Grad_2_order, 'Grad_2_order.pt')
    save_funz.save_dict()

    hours, minutes, seconds = return_time(time.time() - start_tot)
    save_funz.add_element('total_time', f'{int(hours):02}:{int(minutes):02}:{seconds:05.2f}')
    print(f'Elapsed time for simulating SDE: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}')
 

if __name__ == "__main__":
    Exp = [1, 2, 3, 4, 5, 6]
    for exp in Exp:
        print('Experiment:', exp)
        eta = 10**(-exp/3)
        beta = 1-2*eta
        simulation(eta, beta, n_experiment = exp, folder_name = 'Slope_c_2')

    # eta = 0.1
    # beta = 1-eta
    # simulation(0.1, beta, n_experiment = 1, folder_name = 'different_c')

    # eta = 0.1
    # beta = 1-2*eta
    # simulation(0.1, beta, n_experiment = 2, folder_name = 'different_c')

    # eta = 0.1
    # beta = 1-0.5*eta
    # simulation(0.1, beta, n_experiment = 3, folder_name = 'different_c')
    