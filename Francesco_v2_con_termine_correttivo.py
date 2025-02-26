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

Path = "/home/callisti/Thesis/Master-Thesis/Prove_a_caso"

def Simulazione_dinamica_discreta(eta, step, y0, batch_size=1, seed = 0):
    """
    Simulazione dinamica discreta di un sistema dinamico lineare
    y(t+1) = y(t) + eta * N(0,1)**2
    """
    torch.manual_seed(seed)
    y = torch.zeros(batch_size, step, device = y0.device)
    y[:, 0] = y0
    random_values = eta * torch.randn(batch_size, step - 1, device = y0.device)**2
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
class SDE_opzione_1_con_correzione(torchsde.SDEIto):
    def __init__(self, eta, final_time, batch_size=1, cost=1):
        super().__init__(noise_type="scalar")
        self.diffusion = torch.sqrt(torch.tensor(2 * eta)).expand(batch_size, 1, 1)
        self.drift = torch.tensor(1.0-cost).expand(batch_size, 1)

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

class SDE_opzione_2_con_correzione(torchsde.SDEIto):
    def __init__(self, eta, bm, final_time, batch_size=1, cost = 1):
        super().__init__(noise_type="scalar")
        self.eta = eta
        self.bm = bm
        self.drift = torch.tensor(1.0-cost).expand(batch_size, 1)

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
    elif opzione == 3:
        sde = SDE_opzione_1_con_correzione(eta, final_time, batch_size, cost = eta/50)
        y = torchsde.sdeint(sde, y0, ts, method = 'euler', dt = eta**2)
    elif opzione == 4:
        bm = torchsde.BrownianInterval(t0=torch.tensor(0.0), t1 = torch.tensor(final_time), size=(batch_size, 1), device = y0.device, levy_area_approximation="space-time")
        sde = SDE_opzione_2_con_correzione(eta, bm, final_time, batch_size, cost = eta/50)
        y = torchsde.sdeint(sde, y0, ts, bm = bm, method = 'euler',  dt = eta**2)
    elif opzione == 5:
        sde = SDE_opzione_1_con_correzione(eta, final_time, batch_size, cost = eta**0.5/50)
        y = torchsde.sdeint(sde, y0, ts, method = 'euler', dt = eta**2)
    elif opzione == 6:
        bm = torchsde.BrownianInterval(t0=torch.tensor(0.0), t1 = torch.tensor(final_time), size=(batch_size, 1), device = y0.device, levy_area_approximation="space-time")
        sde = SDE_opzione_2_con_correzione(eta, bm, final_time, batch_size, cost = eta**0.5/50)
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
    Run_c_1_corr_eta = []
    Run_c_2_corr_eta = []
    for i in range(run):
        y_discreta = Simulazione_dinamica_discreta(eta, step, y0, batch_size, seed = i)
        y0 = torch.tensor([[0.0]]).expand(batch_size, 1)
        y_continua_1 = Simulazione_dinamica_continua(eta, step, y0, final_time, 1, batch_size, seed= i)
        y_continua_2 = Simulazione_dinamica_continua(eta, step, y0, final_time, 2, batch_size, seed = i)
        y_continua_1_corr_eta = Simulazione_dinamica_continua(eta, step, y0, final_time, 3, batch_size, seed= i)
        y_continua_2_corr_eta = Simulazione_dinamica_continua(eta, step, y0, final_time, 4, batch_size, seed = i)
        Run_d.append(y_discreta)
        Run_c_1.append(y_continua_1)
        Run_c_2.append(y_continua_2)
        Run_c_1_corr_eta.append(y_continua_1_corr_eta)
        Run_c_2_corr_eta.append(y_continua_2_corr_eta)
    Run_d = torch.cat(Run_d)
    Run_c_1 = torch.cat(Run_c_1)
    Run_c_2 = torch.cat(Run_c_2)
    Run_c_1_corr = torch.cat(Run_c_1_corr_eta)
    Run_c_2_corr = torch.cat(Run_c_2_corr_eta) 
    return Run_d, Run_c_1, Run_c_2, Run_c_1_corr, Run_c_2_corr

def plot_with_confidence_interval(Run_d, Run_c_1, Run_c_2, Run_c_1_cor, Run_c_2_cor, eta, final_time, i):
    def aux(run, run_d_mean, x, color, label):
        run_mean = torch.mean(run, dim = 0)
        Error = run_d_mean - run_mean
        plt.plot(x, Error, color, label = label)
    Run_d_mean = torch.mean(Run_d, dim = 0)
    x = torch.linspace(0, final_time, Run_d.shape[1])
    plt.figure(figsize=(10, 6))
    aux(Run_c_1, Run_d_mean, x, 'b', 'Error Option 1')
    aux(Run_c_2, Run_d_mean, x, 'r', 'Error Option 2')
    aux(Run_c_1_cor, Run_d_mean, x, 'g', 'Error Option 1 Cor')
    aux(Run_c_2_cor, Run_d_mean, x, 'y', 'Error Option 2 Cor')
    plt.plot(x, -eta * torch.ones_like(x), 'k--', label='eta')
    plt.plot(x, -eta**2 * torch.ones_like(x), 'r--', label='eta^2')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title('Errors between Discrete and Continuous Simulations')
    plt.legend()
    path = Path
    path_joined = os.path.join(path, f"Errors_between_simulations_eta_10**{-i/5:.2f}_v3.png")
    plt.savefig(path_joined)
    plt.close()

def Error_per_eta(data):
    # Convert data to DataFrame for plotting
    df_errors = pd.DataFrame(data, index=['Error Option 1', 'Error Option 2', 'Errore Opt 1 cor', 'Errore Opt 2 cor', 'Slope 1', 'Slope 2']).T.reset_index()
    df_errors = df_errors.rename(columns={'index': 'eta'})

    # Plot the errors
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_errors, x='eta', y='Error Option 1', label='Error Option 1', marker='o', color = '#1f77b4', markersize=8)
    sns.lineplot(data=df_errors, x='eta', y='Error Option 2', label='Error Option 2', marker='s', color = '#1f77b4', linestyle='--', markersize=8)
    sns.lineplot(data=df_errors, x='eta', y='Errore Opt 1 cor', label='Errore Opt 1 cor', marker='o', color = '#ff7f0e', markersize=8)
    sns.lineplot(data=df_errors, x='eta', y='Errore Opt 2 cor', label='Errore Opt 2 cor', marker='s', color = '#ff7f0e', linestyle='--', markersize=8)
    sns.lineplot(data=df_errors, x='eta', y='Slope 1', label='$\eta$', marker='x', linestyle=':', color = 'r')
    sns.lineplot(data=df_errors, x='eta', y='Slope 2', label='$\eta^2$', marker='x', linestyle=':', color = 'g')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eta')
    plt.ylabel('Error')
    plt.title('Errors between Discrete and Continuous Simulations')
    plt.legend()
    path = Path
    path_joined = os.path.join(path, "Errors_between_simulations_v3.png")
    plt.savefig(path_joined)
    plt.close()

def Simulazione_con_sde():
    print("Simulazione dinamica discreta")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y0 = torch.tensor(0.0, device = device)
    eta = 0.1
    final_time = 50
    run = 1
    batch_size = 10000 
    data = {}
    for i in [1, 2, 3, 4, 5, 6]:
        print(f"Simulazione con eta = 10**(-{i}/5)")
        eta = 10**(-i/3)
        Run_d, Run_c_1, Run_c_2, Run_c_1_cor_eta, Run_c_2_cor_eta  = N_simulazioni(y0, eta, final_time, run, batch_size)
        plot_with_confidence_interval(Run_d, Run_c_1, Run_c_2, Run_c_1_cor_eta, Run_c_2_cor_eta, eta, final_time, i)
        path = Path
        torch.save(Run_d, os.path.join(path, f"Run_d_eta_10**{-i/5:.2f}.pt"))
        torch.save(Run_c_1, os.path.join(path, f"Run_c_1_eta_10**{-i/5:.2f}.pt"))
        torch.save(Run_c_2, os.path.join(path, f"Run_c_2_eta_10**{-i/5:.2f}.pt"))
        Run_d_mean = torch.mean(Run_d.squeeze(), dim = 0)
        Run_c_1_mean = torch.mean(Run_c_1.squeeze(), dim = 0)
        Run_c_2_mean = torch.mean(Run_c_2.squeeze(), dim = 0)
        Run_c_1_cor_mean = torch.mean(Run_c_1_cor_eta.squeeze(), dim = 0)
        Run_c_2_cor_mean = torch.mean(Run_c_2_cor_eta.squeeze(), dim = 0)
        if i == 1:
            Cost_2 = 0.5 * (torch.abs(Run_d_mean[-1] - Run_c_1_cor_mean[-1]).item() + torch.abs(Run_d_mean[-1] - Run_c_2_cor_mean[-1]).item()) / eta**2
            Cost_1 =  0.5 * (torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item() + torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item()) / eta
        slope_2 = Cost_2 * eta**2
        slope_1 = Cost_1 * eta
        slope_2 = eta**2
        slope_1 = eta
        data[eta] = [torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item(), torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item(),  torch.abs(Run_d_mean[-1] - Run_c_1_cor_mean[-1]).item(),  torch.abs(Run_d_mean[-1] - Run_c_2_cor_mean[-1]).item(), slope_1, slope_2]
        with open(f"errors_eta.json", "w") as f:
            json.dump(data, f)
        Error_per_eta(data)

def Simulazione_esplicita_eta(eta, final_time, y0, batch_size=1, seed = 0, run=1):
    torch.manual_seed(seed)
    step = int(final_time / eta)
    Run_d = []
    Run_c_1 = []
    Run_c_2 = []
    for i in range(run):
        print('Run:', i, end='\r')
        y_discreta = Simulazione_dinamica_discreta(eta, step, y0, batch_size, seed = i)

        bm = torchsde.BrownianInterval(t0=torch.tensor(0.0), t1 = torch.tensor(final_time), size=(batch_size, 1), device = y0.device)

        W_j_eta = torch.zeros(batch_size, 1, device = y0.device)
        sum_term = torch.zeros(batch_size, 1, device = y0.device)
        result_1 = [y0.expand(batch_size, 1)]
        result_2 = [y0.expand(batch_size, 1)]
        for j in range(step-1):
            delta_W = bm(torch.tensor(j * eta), torch.tensor((j + 1) * eta))
            sum_term = sum_term + W_j_eta * delta_W
            W_j_eta = W_j_eta + delta_W
            result_1.append(y0 + eta*(j+1) + torch.sqrt(torch.tensor(2 * eta)) * W_j_eta)
            result_2.append(y0 + W_j_eta**2 - 2 * sum_term)

        Run_c_1.append(torch.stack(result_1).cpu())
        Run_c_2.append(torch.stack(result_2).cpu())
        Run_d.append(y_discreta.cpu())
    Run_d = torch.cat(Run_d)
    Run_c_1 = torch.cat(Run_c_1, dim = 1)
    Run_c_2 = torch.cat(Run_c_2, dim = 1)
    return Run_d, Run_c_1, Run_c_2

def Simulazione_con_soluzionie_esplicita():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y0 = torch.tensor(0.0, device = device)
    final_time = 50
    run = 100
    batch_size = 300000
    data = {}
    Res = {}
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(f"Simulazione: {i}")
        eta = 10**(-i/3)
        sim = 2*int(final_time*eta**(-4))
        run = max(sim // batch_size, 1)
        print(f"Batch size: {batch_size}, Run: {run}")
        Run_d, Run_c_1, Run_c_2 = Simulazione_esplicita_eta(eta, final_time, y0, batch_size, seed = 0, run = run)

        path = "/home/callisti/Thesis/Master-Thesis/Simulazione_con_soluzione_esplicita"
        torch.save(Res, os.path.join(path, "Result.pt"))

        Run_d_mean = torch.mean(Run_d.squeeze(), dim = 0)
        Run_c_1_mean = torch.mean(Run_c_1.squeeze(), dim = 1).squeeze()
        Run_c_2_mean = torch.mean(Run_c_2.squeeze(), dim = 1).squeeze()
        res = {'discreto' : Run_d_mean, 'opzione_1_non_SDE' : Run_c_1_mean, 'continuo_2_non_SDE' : Run_c_2_mean}
        Res[eta] = res

        if i == 1:
            c1 = 0.5 * (torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item() + torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item()) / eta
            c2 = 0.5 * (torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item() + torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item()) / eta**2
        data[eta] = [torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item(), torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item(), c1*eta, c2*eta**2]
        print(data)
        plot_error_vs_eta(data)
        plot_error(Run_d_mean, Run_c_1_mean, Run_c_2_mean, eta, final_time, i)
def plot_error(Run_d, Run_c_1, Run_c_2, eta, final_time, i):
    error_1 = (Run_d - Run_c_1).detach().cpu().numpy()
    error_2 = (Run_d - Run_c_2).detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    x = torch.arange(len(error_1)) * eta
    plt.plot(x, error_1, label='Error Option 1')
    plt.plot(x, error_2, label='Error Option 2')
    plt.plot(x, eta * torch.ones_like(x), 'r--', label='eta')   
    plt.plot(x, eta**2 * torch.ones_like(x), 'g--', label='eta^2')
    plt.plot(x, -eta * torch.ones_like(x), 'r--')   
    plt.plot(x, -eta**2 * torch.ones_like(x), 'g--')
    plt.legend()
    plt.title(f'Errors between Discrete and Continuous Simulations: eta = 10**{-i/5:.2f}')
    path = "/home/callisti/Thesis/Master-Thesis/Simulazione_con_soluzione_esplicita"
    path_joined = os.path.join(path, f"Errors_between_simulations_explicit_v3_eta_10**{-i/5:.2f}.png")
    plt.savefig(path_joined)
    plt.close()

def plot_error_vs_eta(data):
    # Convert data to DataFrame for plotting
    df_errors = pd.DataFrame(data, index=['Error Option 1', 'Error Option 2', 'eta', 'eta^2']).T.reset_index()

    # Plot the errors
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_errors, x='eta', y='Error Option 1', label='Error Option 1', marker='o', color='#1f77b4', markersize=8)
    sns.lineplot(data=df_errors, x='eta', y='Error Option 2', label='Error Option 2', marker='s', color='#ff7f0e', markersize=8)
    sns.lineplot(data=df_errors, x='eta', y='eta', label='$\eta$', marker='x', linestyle=':', color='r')
    sns.lineplot(data=df_errors, x='eta', y='eta^2', label='$\eta^2$', marker='x', linestyle=':', color='g')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eta')
    plt.ylabel('Error')
    plt.title('Errors between Discrete and Continuous Simulations')
    plt.legend()
    path = "/home/callisti/Thesis/Master-Thesis/Simulazione_con_soluzione_esplicita"
    path_joined = os.path.join(path, "Errors_between_simulations_explicit_v3.png")
    plt.savefig(path_joined)
    plt.close()
Simulazione_con_soluzionie_esplicita()
        

