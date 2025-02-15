import sys
sys.setrecursionlimit(10000)

import torch
import torchsde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import math
import json
import os

def Simulazione_dinamica_discreta(eta, step, y0, batch_size=1, seed = 0):
    """
    Simulazione dinamica discreta di un sistema dinamico lineare
    y(t+1) = y(t) + eta * N(0,1)**2
    """
    torch.manual_seed(seed)
    y = torch.zeros(batch_size, step)
    y[:, 0] = y0
    random_values = eta * torch.randn(batch_size, step - 1)**2
    for t in range(1, step):
        y[:, t] = y[:, t-1] + random_values[:, t-1]
    return y

class SDE_opzione_1(torchsde.SDEIto):
    def __init__(self, eta, final_time, batch_size=1):
        super().__init__(noise_type="scalar")
        self.diffusion = torch.sqrt(torch.tensor(2 * eta)).expand(batch_size, 1, 1)
        self.drift = torch.tensor(1.0).expand(batch_size, 1)

        self.checkpoint = [50 * eta * i for i in range(math.ceil(final_time / (50 * eta)))]
        self.i = 0

    def f(self, t, y):
        if self.i < len(self.checkpoint) and t >= self.checkpoint[self.i]:
            if self.i != 0: print('time elapsed =', time.time() - self.start, 's, time simulation', t)
            self.start = time.time()
            self.i += 1
        return self.drift
    
    def g(self, t, y):
        return self.diffusion

class SDE_opzione_2(torchsde.SDEIto):
    def __init__(self, eta, bm, final_time, batch_size=1):
        super().__init__(noise_type="scalar")
        self.eta = eta
        self.bm = bm
        self.drift = torch.tensor(1.0).expand(batch_size, 1)

        self.checkpoint = [50 * eta * i for i in range(math.ceil(final_time / (50 * eta)))]
        self.i = 0

    def f(self, t, y):
        if self.i < len(self.checkpoint) and t >= self.checkpoint[self.i]:
            if self.i != 0: print('time elapsed =', time.time() - self.start, 's, time simulation', t)
            self.start = time.time()
            self.i += 1
        return self.drift

    def g(self, t, y):
        discrete_t = (t // self.eta) * self.eta
        term = self.bm(discrete_t, t)
        return (2*term).unsqueeze(2)

def Simulazione_dinamica_continua(eta, step, y0, final_time, opzione, batch_size=1, seed = 0):
    """
    Simulazione dinamica continua di un sistema dinamico lineare
    y(t+1) = y(t) + eta * N(0,1)**2
    """
    torch.manual_seed(seed)
    ts = torch.linspace(0, final_time, step)
    if opzione == 1:
        sde = SDE_opzione_1(eta, final_time, batch_size)
        y = torchsde.sdeint(sde, y0, ts, method = 'euler', dt = eta**2)
    elif opzione == 2:
        bm = torchsde.BrownianInterval(t0=torch.tensor(0.0), t1 = torch.tensor(final_time), size=(batch_size, 1), device = y0.device, levy_area_approximation="space-time")
        sde = SDE_opzione_2(eta, bm, final_time, batch_size)
        y = torchsde.sdeint(sde, y0, ts, bm = bm, method = 'euler',  dt = eta**2)
    else:
        raise ValueError("Opzione non valida")
    y = y.squeeze()
    y = y.permute(1,0)
    return y

def N_simulazioni(y0 = torch.tensor(0.0), eta = 0.1, final_time = 100, run = 10, batch_size=1):
    step = int(final_time / eta)
    Run_d = []
    Run_c_1 = []
    Run_c_2 = []
    for i in range(run):
        y_discreta = Simulazione_dinamica_discreta(eta, step, y0, batch_size, seed = i)
        y0 = torch.tensor([[0.0]]).expand(batch_size, 1)
        y_continua_1 = Simulazione_dinamica_continua(eta, step, y0, final_time, 1, batch_size, seed= i)
        y_continua_2 = Simulazione_dinamica_continua(eta, step, y0, final_time, 2, batch_size, seed = i)
        Run_d.append(y_discreta)
        Run_c_1.append(y_continua_1)
        Run_c_2.append(y_continua_2)
    Run_d = torch.cat(Run_d)
    Run_c_1 = torch.cat(Run_c_1)
    Run_c_2 = torch.cat(Run_c_2)
    return Run_d, Run_c_1, Run_c_2

def plot_with_confidence_interval(Run_d, Run_c_1, Run_c_2, eta, final_time, i):
    Run_d_mean = torch.mean(Run_d, dim = 0)
    Run_c_1_mean = torch.mean(Run_c_1, dim = 0)
    Run_c_2_mean = torch.mean(Run_c_2, dim = 0)

    Error_c_1 = torch.abs(Run_d_mean - Run_c_1_mean)
    Error_c_2 = torch.abs(Run_d_mean - Run_c_2_mean)
    x = torch.linspace(0, final_time, Run_d.shape[1])
    plt.figure(figsize=(10, 6))
    plt.plot(x, Error_c_1, label = 'Error Option 1')
    plt.plot(x, Error_c_2, label = 'Error Option 2')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title('Errors between Discrete and Continuous Simulations')
    plt.legend()
    path = "/home/callisti/Thesis/Master-Thesis/Francesco"
    path_joined = os.path.join(path, f"Errors_between_simulations_eta_10**{-i/5:.2f}_v3.png")
    plt.savefig(path_joined)
    plt.close()

def Error_per_eta(data):
    # Convert data to DataFrame for plotting
    df_errors = pd.DataFrame(data, index=['Error Option 1', 'Error Option 2', 'Slope 1', 'Slope 2']).T.reset_index()
    df_errors = df_errors.rename(columns={'index': 'eta'})

    # Plot the errors
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_errors, x='eta', y='Error Option 1', label='Error Option 1', marker='o')
    sns.lineplot(data=df_errors, x='eta', y='Error Option 2', label='Error Option 2', marker='o')
    sns.lineplot(data=df_errors, x='eta', y='Slope 1', label='Slope 1', marker='x', linestyle='--')
    sns.lineplot(data=df_errors, x='eta', y='Slope 2', label='Slope 2', marker='x', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eta')
    plt.ylabel('Error')
    plt.title('Errors between Discrete and Continuous Simulations')
    plt.legend()
    path = "/home/callisti/Thesis/Master-Thesis/Francesco"
    path_joined = os.path.join(path, "Errors_between_simulations_v3.png")
    plt.savefig(path_joined)
    plt.close()

def main():
    print("Simulazione dinamica discreta+")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y0 = torch.tensor(0.0, device = device)
    eta = 0.1
    final_time = 50
    run = 1
    batch_size = 1000 
    data = {}
    for i in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9]:
        print(f"Simulazione con eta = 10**(-{i}/5)")
        eta = 10**(-i/5)
        Run_d, Run_c_1, Run_c_2 = N_simulazioni(y0, eta, final_time, run, batch_size)
        plot_with_confidence_interval(Run_d, Run_c_1, Run_c_2, eta, final_time, i)
        path = "/home/callisti/Thesis/Master-Thesis/Francesco"
        torch.save(Run_d, os.path.join(path, f"Run_d_eta_10**{-i/5:.2f}.pt"))
        torch.save(Run_c_1, os.path.join(path, f"Run_c_1_eta_10**{-i/5:.2f}.pt"))
        torch.save(Run_c_2, os.path.join(path, f"Run_c_2_eta_10**{-i/5:.2f}.pt"))
        Run_d_mean = torch.mean(Run_d.squeeze(), dim = 0)
        Run_c_1_mean = torch.mean(Run_c_1.squeeze(), dim = 0)
        Run_c_2_mean = torch.mean(Run_c_2.squeeze(), dim = 0)
        if i == 1:
            Cost_2 = 0.5 * (torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item() + torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item()) / eta**2
            Cost_1 =  0.5 * (torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item() + torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item()) / eta
        slope_2 = Cost_2 * eta**2
        slope_1 = Cost_1 * eta
        data[eta] = [torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item(), torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item(), slope_1, slope_2]
        with open(f"errors_eta.json", "w") as f:
            json.dump(data, f)
        Error_per_eta(data)

main()
