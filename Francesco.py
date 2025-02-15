import sys
sys.setrecursionlimit(10000)

import torch
import torchsde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time



def Simulazione_dinamica_discreta(eta, step, y0, seed = 0):
    """
    Simulazione dinamica discreta di un sistema dinamico lineare
    y(t+1) = y(t) + eta * N(0,1)**2
    """
    torch.manual_seed(seed)
    y = torch.zeros(step)
    y[0] = y0
    random_values = eta * torch.randn(step - 1)**2
    for t in range(1, step):
        y[t] = y[t-1] + random_values[t-1]
    return y

class SDE_opzione_1(torchsde.SDEIto):
    def __init__(self, eta):
        super().__init__(noise_type="general")
        self.diffusion = torch.sqrt(torch.tensor(2*eta))

    def f(self, t, y):
        return torch.tensor([[1.0]])
    def g(self, t, y):
        return torch.tensor([[[self.diffusion]]])

class SDE_opzione_2(torchsde.SDEIto):
    def __init__(self, eta, bm, final_time):
        super().__init__(noise_type="general")
        self.eta = eta
        self.bm = bm

        self.checkpoint = [10 * eta * i for i in range(int(final_time / (10*eta) ))]
        self.i = 0

    def f(self, t, y):
        if t * self.eta >= self.checkpoint[self.i]:
            if self.i != 0: print('time elapsed =', time.time() - self.start, 's, time simulation', t)
            self.start = time.time()
            self.i += 1

        return torch.tensor([[1.0]])
    def g(self, t, y):
        discrete_t = (t // self.eta) * self.eta
        term = self.bm(discrete_t, t)
        return torch.tensor([[[2 * term ]]])
    

def Simulazione_dinamica_continua(eta, step, y0, final_time, opzione, seed = 0):
    """
    Simulazione dinamica continua di un sistema dinamico lineare
    y(t+1) = y(t) + eta * N(0,1)**2
    """
    torch.manual_seed(seed)
    ts = torch.linspace(0, final_time, step)
    if opzione == 1:
        sde = SDE_opzione_1(eta)
        y = torchsde.sdeint(sde, y0, ts, dt = eta**2)
    elif opzione == 2:
        bm = torchsde.BrownianInterval(t0=torch.tensor(0.0), t1 = torch.tensor(final_time), size=torch.tensor([[0.0]]).shape, device = y0.device)
        sde = SDE_opzione_2(eta, bm, final_time)
        y = torchsde.sdeint(sde, y0, ts, bm = bm, dt = eta**2)
    else:
        raise ValueError("Opzione non valida")
    return y


def N_simulazioni(y0 = torch.tensor(0.0), eta = 0.1, final_time = 100, run = 10):
    step = int(final_time / eta)
    Run_d = []
    Run_c_1 = []
    Run_c_2 = []
    for i in range(run):
        y_discreta = Simulazione_dinamica_discreta(eta, step, y0, seed = i)
        y0 = torch.tensor([[0.0]])
        y_continua_1 = Simulazione_dinamica_continua(eta, step, y0, final_time, 1, seed= i)[:, 0, 0]
        y_continua_2 = Simulazione_dinamica_continua(eta, step, y0, final_time, 2, seed = i)[:, 0, 0]
        Run_d.append(y_discreta)
        Run_c_1.append(y_continua_1)
        Run_c_2.append(y_continua_2)
    Run_d = torch.stack(Run_d)
    Run_c_1 = torch.stack(Run_c_1)
    Run_c_2 = torch.stack(Run_c_2)
    return Run_d, Run_c_1, Run_c_2

def plot_with_confidence_interval(Run_d, Run_c_1, Run_c_2, eta, final_time):
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
    plt.savefig(f"Errors_between_simulations_eta_{eta}.png")
    plt.close()

def Error_per_eta(data):
    # Convert data to DataFrame for plotting
    df_errors = pd.DataFrame(data, index=['Error Option 1', 'Error Option 2']).T.reset_index()
    df_errors = df_errors.rename(columns={'index': 'eta'})

    # Plot the errors
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_errors, x='eta', y='Error Option 1', label='Error Option 1')
    sns.scatterplot(data=df_errors, x='eta', y='Error Option 2', label='Error Option 2')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eta')
    plt.ylabel('Error')
    plt.title('Errors between Discrete and Continuous Simulations')
    plt.legend()
    plt.savefig("Errors_between_simulations.png")
    plt.close()

def main():
    print("Simulazione dinamica discreta")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y0 = torch.tensor(0.0, device = device)
    eta = 0.1
    final_time = 10
    run = 50
    data = {}
    for i in [1, 2, 3, 4, 5, 6]:
        print(f"Simulazione con eta = 10**(-{i}/3)")
        eta = 10**(-i/3)
        Run_d, Run_c_1, Run_c_2 = N_simulazioni(y0, eta, final_time, run)
        plot_with_confidence_interval(Run_d, Run_c_1, Run_c_2, eta, final_time)
        Run_d_mean = torch.mean(Run_d, dim = 0)
        Run_c_1_mean = torch.mean(Run_c_1, dim = 0)
        Run_c_2_mean = torch.mean(Run_c_2, dim = 0)
        data[eta] = [torch.abs(Run_d_mean[-1] - Run_c_1_mean[-1]).item(), torch.abs(Run_d_mean[-1] - Run_c_2_mean[-1]).item()]
        Error_per_eta(data)



main()