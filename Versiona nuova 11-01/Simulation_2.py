from CNN import CNN, Train_n_times
from SDE import RMSprop_SDE, RMSprop_SDE_1_order
from Dataset import CIFAR10Dataset
from calculation import function_f_and_Sigma
import Save_exp


import torch
import numpy as np
import time
import torchsde


def Simulation_discrete_dynamics(model, dataset, steps, lr, beta, n_runs, batch_size=1):
    print('model parameters:', model.get_number_parameters())
    print('dimension dataset:', dataset.x_train.shape[0])
    print('number of epoch:', steps * batch_size / dataset.x_train.shape[0], ' final time:', steps * lr)
    Train = Train_n_times(model, dataset, steps=steps, lr=lr, optimizer_name='RMSPROP', beta = beta)
    FinalDict = Train.train_n_times(n = n_runs, batch_size=batch_size)
    return FinalDict

def simulation():
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()
    dataset_size = 256
    dataset.x_train = dataset.x_train[:dataset_size]
    dataset.y_train = dataset.y_train[:dataset_size]
    save_funz = Save_exp.SaveExp('/home/callisti/Thesis/Master-Thesis/Results')

    conv_layers = [
        (3, 3, 1, 1),  # filter number, kernel size, stride, padding
        (2, 3, 1, 1),
        (2, 3, 1, 1),
        (1, 3, 1, 1)
    ]
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)
    steps, n_runs = 5, 2
    eta, beta = 0.1, 0.99
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
        print(f'Run {i_run}')
        x0[: 2*number_parameters] = torch.cat((FinalDict[i_run]['Params'][0], FinalDict[i_run]['Square_avg'][0]), dim=0)
        if i_run != 1:
            print('Error', i_run, torch.norm(last - x0[:2*number_parameters]))
        last = x0[:2*number_parameters]
        if False:
            sde = RMSprop_SDE_1_order(eta, beta, f, Verbose=False)
            res = torchsde.sdeint(sde, x0.unsqueeze(0).to('cuda'), t, method = 'euler', dt =eta)
            loss = sde.get_loss()
            Loss.append(loss)
            result.append(res)
            torch.save(result, f'/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/Risultati_sde.pt')
            
        else:
            start = time.time()

            sde = RMSprop_SDE(eta, beta, f, Verbose=False)
            res = torchsde.sdeint(sde, x0.unsqueeze(0).to('cuda'), t, method = 'euler', dt =eta**2)
            loss = sde.get_loss()
            Loss.append(loss)
            grad = sde.get_loss_grad()
            Grad.append(grad)
            result.append(res)

            save_funz.partial_result(result, 'result')
            save_funz.partial_result(Grad, 'grad')
            save_funz.partial_result(Loss, 'loss')

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

if __name__ == "__main__":
    simulation()