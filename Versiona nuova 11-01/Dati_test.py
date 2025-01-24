import torch
from torch.utils.data import DataLoader, TensorDataset
from Simulation import Simulation_discrete_dynamics, plot, plot_loss
from calculation import function_f_and_Sigma
import time
import torchsde
from SDE import RMSprop_SDE, RMSprop_SDE_1_order
import Save_exp


import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.a = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        a_term = self.a**3 * x
        b_term = self.b**3 * x
        return torch.cat((a_term, b_term), dim=1)
    
    def get_number_parameters(self):
        return 1


class SimpleDataset():
    def __init__(self, n = 100, a = 10):
        self.x_train = torch.rand(n, 1)
        self.x_train += self.x_train.min().clone()
        self.y_train = (a * self.x_train).long()  
        mean_y = torch.mean(self.y_train.float())
        self.y_train = (self.y_train > mean_y).long()
        indices = torch.randperm(n)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

    def dataloader(self, batch_size, steps, seed):
        tensor_dataset = TensorDataset(self.x_train, self.y_train)
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
        return dataloader


def simulation():
    model = SimpleNN()
    dataset = SimpleDataset(n = 1000)
    save_funz = Save_exp.SaveExp('/home/callisti/Thesis/Master-Thesis/Results')

    steps, n_runs = 50, 10
    eta, beta = 0.01, 0.99
    FinalDict = Simulation_discrete_dynamics(model, dataset, steps, lr=eta, beta = beta, n_runs=n_runs, batch_size=1)
    save_funz.save_result_discrete(FinalDict)
    save_funz.add_element('eta', eta)
    save_funz.add_element('beta', beta)
    save_funz.add_element('steps', steps)
    save_funz.add_element('n_runs', n_runs)

    t0, t1 = 0.0, (steps * eta)  
    t = torch.linspace(t0, t1, steps)
    save_funz.add_element('t0', t0)
    save_funz.add_element('t1', t1)


    number_parameters = FinalDict[1]['Params'][0].shape[0]
    save_funz.add_element('number_parameters', number_parameters)

    x0 = torch.zeros(3 * number_parameters)
    result = []
    start = time.time()
    Loss = []  
    Grad = []
    for i_run in FinalDict.keys():
        model = SimpleNN()
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
            sde = RMSprop_SDE(eta, beta, f, Verbose=False)
            res = torchsde.sdeint(sde, x0.unsqueeze(0).to('cuda'), t, method = 'euler', dt =eta**2)
            loss = sde.get_loss()
            Loss.append(loss)
            grad = sde.get_loss_grad()
            Grad.append(grad)
            result.append(res)
            torch.save(result, f'/home/callisti/Thesis/Master-Thesis/Versiona nuova 11-01/Risultati_sde.pt')

    print(f'Elapsed time for simulating SDE: {time.time() - start:.2f}')
    save_funz.save_result_continuous(result)
    save_funz.save_loss_sde(Loss)
    save_funz.save_dict()




if __name__ == "__main__":
    # simulation()
    load_funz = Save_exp.LoadExp('/home/callisti/Thesis/Master-Thesis/Results/Experiment_2')
    load_funz.load_FinalDict()
    load_funz.load_result('Risultati_sde.pt')
    load_funz.plot()