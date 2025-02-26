import torchsde
import time
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
import torch

sys.setrecursionlimit(10000)

def create_def_positive_matrix(dim):
    A = torch.randn(dim, dim)
    A = A @ A.T  
    A += torch.eye(dim) * 1e-2
    return A
class funz_quadratica_con_noise():
    # la funzione è f(x, gamma) = 0.5 * (x - gamma)^T A (x - gamma) - 0.5 * tr(A) con gamma che è distribuita come N(0,1)
    def __init__(self, A = None, dim = 10):
        if A != None:
            self.A = A
        else:
            self.A = create_def_positive_matrix(dim)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.A = self.A.to(device)
        self.sigma = torch.matmul(self.A, self.A)
        AA_T_squared = self.sigma**2
        self.M_matrix = 2 * AA_T_squared 

    def function(self, x, gamma):
        return 0.5 * (x-gamma) @ self.A @ (x - gamma) - 0.5 * torch.trace(self.A)
    
    def expected_value(self, x):
        return 0.5 * x @ self.A @ x
    
    def gradient(self, x, gamma):
        return self.A @ (x - gamma)
    
    def expected_value_gradient(self, x):
        return self.A @ x
    
    def hessian(self):
        return self.A
    
    def Sigma(self):
        return self.sigma
    
    def Sqrt_sigma(self):
        return self.A
    
    def M(self):
        return torch.linalg.cholesky(self.M_matrix)
    
    def compute_gradients_expected_grad_f_squared(self, x):
        return (2 * torch.diag(self.expected_value_gradient(x))@self.A).T

class SDE_SGD_1_order(torch.nn.Module):
    def __init__(self, eta, funz):
        self.noise_type = 'additive'
        self.sde_type = 'ito'
        self.eta = eta
        self.funz = funz
        self.A = funz.A
        self.eta_sqrt = torch.sqrt(torch.tensor(eta))

    def f(self, t, x):
        return (-self.A @ x.T).T
    
    def g(self, t, x):
        return self.eta_sqrt*self.A.expand(x.shape[0], x.shape[1], x.shape[1])
    
class SDE_SGD_2_order(torch.nn.Module):
    def __init__(self, eta, funz):
        super(SDE_SGD_2_order, self).__init__()
        self.noise_type = 'additive'
        self.sde_type = 'ito'
        self.eta = eta
        self.funz = funz
        self.A = funz.A
        self.AA = self.A @ self.A
        self.eta_sqrt = torch.sqrt(torch.tensor(eta))

    def f(self, t, x):
        return (-self.A @ x.T).T - 0.5 * self.eta *  (self.AA @ x.T).T
    def g(self, t, x):
        return self.eta_sqrt*self.A.expand(x.shape[0], x.shape[1], x.shape[1])
    
def SGDprop_optimizer_batch(funz, x_0, lr, num_steps = 1000):
    path_x = []  
    batchsize, dim = x_0.shape
    A = funz.A
    for _ in range(num_steps):
        gamma = torch.randn(batchsize, dim, device = x_0.device)
        x_0 = x_0 - lr * (A @ (x_0 - gamma).T).T
        path_x.append(x_0)
    return torch.stack(path_x)


def batch_call(funz, eta, num_steps, x_0):
    method = 'euler'
    method = 'milstein'
    method = 'srk'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = random.randint(0, 10000)
    torch.manual_seed(seed)
    path_x_d = SGDprop_optimizer_batch(funz, x_0, eta, num_steps)
    t1 = eta * num_steps
    t = torch.linspace(eta, t1, num_steps)

    sde_1_order = SDE_SGD_1_order(eta, funz)
    Result_order_1 = torchsde.sdeint(sde_1_order, x_0.to(device), t, method = method, dt = eta)
    sde_2_order = SDE_SGD_2_order(eta, funz)
    Result_order_2 = torchsde.sdeint(sde_2_order, x_0.to(device), t, method = method, dt = eta**2)
   
    return path_x_d, Result_order_1, Result_order_2

def plot_error_vs_eta(FinalError):
    print('Plotting')
    print(FinalError)
    etas = list(FinalError.keys())
    errors_x_1 = [FinalError[eta][0] for eta in etas]
    errors_x_2 = [FinalError[eta][1] for eta in etas]
    eta = [FinalError[eta][2] for eta in etas]
    eta2 = [FinalError[eta][3] for eta in etas]

    plt.figure()
    plt.plot(etas, errors_x_1, label='Error_x_1', marker='o')
    plt.plot(etas, errors_x_2, label='Error_x_2', marker='x')
    plt.plot(etas, eta, label='eta', linestyle='--')
    plt.plot(etas, eta2, label='eta^2', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eta')
    plt.ylabel('Error_x')
    plt.legend()
    plt.title('Error_x vs eta')
    plt.savefig('/home/callisti/Thesis/Master-Thesis/Result_weinan/Error_x_vs_eta.png')
    plt.close()



def different_eta(final_time, dim, batchsize):
    print('Different eta:')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_0 = torch.randn(dim, requires_grad=True).to(device) * 1e2
    funz = funz_quadratica_con_noise(dim = dim)
    FinalError = {}
    Res = {}
    batchsize = 500000
    for i in [2, 3, 4, 5, 6, 7]:
        print('eta', i)
        eta = 10**(-i/3)
        sim = final_time* int(eta**(-4))
        run = max(sim // batchsize, 1)
        x_0 = x_0.expand(batchsize, dim)

        num_steps = int(final_time/eta)

        Path_x_d, Result_order_1, Result_order_2 = [], [], []
        for i in range(run):
            print('run', i, 'of', run, end='\r')
            x_d, Res_order_1, Res_order_2 = batch_call(funz, eta, num_steps, x_0)
            Path_x_d.append(x_d.cpu())
            Result_order_1.append(Res_order_1.cpu())
            Result_order_2.append(Res_order_2.cpu())
            del x_d, Res_order_1, Res_order_2
        Path_x_d = torch.cat(Path_x_d)
        Result_order_1 = torch.cat(Result_order_1)
        Result_order_2 = torch.cat(Result_order_2)

        Path_x_d_mean = torch.mean(Path_x_d, dim = 1)
        Result_order_1_mean = torch.mean(Result_order_1, dim = 1)
        Result_order_2_mean = torch.mean(Result_order_2, dim = 1)

        res = {'discreto' : Path_x_d_mean, 'order_1' : Result_order_1_mean, 'order_2' : Result_order_2_mean}
        Res[eta] = res
        torch.save(Res, f'/home/callisti/Thesis/Master-Thesis/Result_weinan/Res.pt')

        Error_x_1 = torch.norm(Path_x_d_mean[-1]-Result_order_1_mean[-1], p=1)
        Error_x_2 = torch.norm(Path_x_d_mean[-1]-Result_order_2_mean[-1], p=1)
        FinalError[eta] = [Error_x_1.item(), Error_x_2.item(), eta, eta**2]
        plot_error_vs_eta(FinalError)
    
different_eta(50, 2, 10000)