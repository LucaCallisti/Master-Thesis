import torchsde
import time
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from diffeqpy import de
import torch

sys.setrecursionlimit(10000)

def create_def_positive_matrix(dim):
    A = torch.randn(dim, dim)
    A = A @ A.T  
    A += torch.eye(dim) * 1e-6
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
    



class RMSprop_SDE(torchsde.SDEIto):
    def __init__(self, eta, beta, function_f_and_Sigma, All_time, regularizer, eps = 1e-8, Verbose = True):
        super().__init__(noise_type="general")

        self.eta = torch.tensor(eta)
        self.c = (1-beta)/eta
        self.eps = eps
        self.function_f_and_Sigma = function_f_and_Sigma

        self.theta_old = None
        self.diffusion = None
        self.drift = None
        self.i = 1
        self.verbose = Verbose
        self.final_time = All_time[-1]
        self.All_time = All_time

        self.regularizer = regularizer

    def f_julia(self, *input):
        breakpoint()
        input = torch.cat(input)
        return self.f(input[-1], input[:-1]).detach().cpu().numpy()
    def g_julia(self, *input):
        breakpoint()
        input = torch.cat(input)
        return self.g(input[-1], input[:-1]).detach().cpu().numpy()
    
    def f(self, t, x):
        if self.i == 1: self.start_new_f = time.time()
        if self.i > 1:   
            if self.i % 100**2 == 0: 
                print(f'time between f {self.i} calls {(time.time() - self.start_new_f):.2f}s, time: {t:.4f}')
                self.start_new_f = time.time()
        self.i += 1
        
        theta, v, w = self.divide_input(x)
        
        if (v<0).sum() > 0: 
            if v.min() < -self.eta**2: print('Warning: negative values in v', (v<0).sum().item(), v.min().item(), 'at time ', t)
            v[v<0] = 0
    
        if self.diffusion is None and self.drift is None: self.regularizer.set_costant(v, self.c, self.eta, self.final_time)
        if self.theta_old is None or (self.theta_old != theta).any(): self.update_quantities(theta, t)
        if self.verbose: print(f'v: {torch.norm(v)/v.shape[0]}, grad: {torch.norm(self.f_grad)/self.f_grad.shape[0]}')        


        v_reg = self.regularizer.regulariz_function(v)
        v_reg_grad = self.regularizer.derivative_regulariz_function(v)

        f_grad_square = torch.pow(self.f_grad, 2)    
        # Theta coefficient
        denom = 1/(torch.sqrt(v_reg) + self.eps)

        coef_theta = (self.f_hessian * (torch.ger(denom, denom))) @ self.f_grad + self.c* (f_grad_square + self.diag_Sigma - v_reg) * (denom**3) * (self.f_grad  * v_reg_grad)
        # coef_theta = (self.f_hessian * (torch.ger(denom, denom))) @ self.f_grad + self.c*(f_grad_square + self.diag_Sigma - v_reg) * (self.f_grad * torch.pow(denom, 3) * v_reg_grad)
        coef_theta = - self.f_grad  * denom - self.eta/2 * coef_theta

        # V coefficient
        coef_v = (self.c + self.c**2 * self.eta/2) * (f_grad_square + self.diag_Sigma - v_reg)
        coef_v = coef_v + 0.5 * self.eta * self.c * (self.grad_expected_frad_f_squared @ self.f_grad) * denom

        # W coefficient
        coef_w = torch.zeros_like(w)
        self.drift = torch.concat((coef_theta, coef_v, coef_w), dim = 0).unsqueeze(0)
        
        if torch.isnan(self.drift).any():
            print("Warning: NaN values detected in drift")
            breakpoint()
        return self.drift
    

    def g(self, t, x):
        theta, v, w = self.divide_input(x)

        if self.diffusion is None and self.drift is None: self.regularizer.set_costant(v, self.c, self.eta, self.final_time)
        if self.theta_old is None or (self.theta_old != theta).any(): self.update_quantities(theta, t)
        if (v<0).sum() > 0:
            print('Warning: negative values in v', (v<0).sum().item(), v.min().item(), 'at time ', t)
            v[v<0] = 0
        v_reg = self.regularizer.regulariz_function(v)

        denom = 1/(torch.sqrt(v_reg) + self.eps)

        M_theta = torch.sqrt(self.eta) * torch.diag(denom) @ self.Sigma_sqrt
        M_v = -2 * torch.sqrt(self.eta) * self.c * torch.diag(self.f_grad) @  self.Sigma_sqrt + torch.sqrt(torch.tensor(2)) * self.c * self.square_root_var_z_squared @ torch.diag(w)
        # M_v = -2 * torch.sqrt(self.eta) * self.c * torch.diag(self.f_grad) @  self.Sigma_sqrt
        M_w = torch.eye(M_theta.shape[0], device = M_theta.device)
        self.diffusion = torch.concat((M_theta, M_v, M_w), dim = 0).unsqueeze(0)

        if torch.isnan(self.diffusion).any():
            self.found_Nan = True
            print("Warning: NaN values detected in diffusion")
            breakpoint()
        # if torch.any(torch.isclose(self.All_time, t, atol=1e-4)):
        #     breakpoint()
        return self.diffusion
    
    def update_quantities(self, theta, t):
        self.diffusion = None
        self.drift = None

        self.theta_old = theta

        self.f_grad = self.function_f_and_Sigma.expected_value_gradient(theta) 
        self.f_hessian = self.function_f_and_Sigma.hessian()
        self.Sigma_sqrt = self.function_f_and_Sigma.Sqrt_sigma()
        self.diag_Sigma = torch.diag(self.function_f_and_Sigma.Sigma())
        self.grad_expected_frad_f_squared = self.function_f_and_Sigma.compute_gradients_expected_grad_f_squared(theta)
        self.square_root_var_z_squared = self.function_f_and_Sigma.M()

    def divide_input(self, x):
        x = x.squeeze()
        aux = x.shape[0] // 3
        theta = x[:aux]
        v = x[aux:2*aux]
        w = x[2*aux:3*aux]
        return theta, v, w

class Regularizer_Phi():
    def __init__(self, eps):
        self.eps = eps
    def set_costant(self, v, c, eta, final_time):
        self.C_reg = (1-c*eta) **final_time * (v) 
        self.C_reg = 9/10 * self.C_reg
        self.C_reg[self.C_reg < self.eps] = self.eps
    def regulariz_function(self, x):
        return torch.where(x > self.C_reg, x, self.C_reg * torch.exp(x / self.C_reg - 1))
    def derivative_regulariz_function(self, x):
        return torch.where(x > self.C_reg, torch.ones_like(x), torch.exp(x / self.C_reg - 1))

class Regularizer_Phi_cost(Regularizer_Phi):
    def __init__(self, cost):
        self.cost = cost       
    def set_costant(self, v, c, eta, final_time):
        self.C_reg = self.cost * torch.ones_like(v) 

class Regularizer_arora():
    def set_costant(self, v, c, eta, final_time):
        self.u_min = v
    
    def tau(self, z):
        return torch.where(z >= 1, torch.ones_like(z),
                           torch.where((z > 0) & (z < 1),
                                       torch.exp(-1/z) / (torch.exp(-1/z) + torch.exp(-1/(1-z))),
                                       torch.zeros_like(z)))
    
    def regulariz_function(self, u):
        z = 2 * u / self.u_min - 1
        return 0.5 * self.u_min + self.tau(z) * (u - 0.5 * self.u_min)
    
    def derivative_regulariz_function(self, u):
        z = 2 * u / self.u_min - 1
        exp_neg1_z = torch.exp(-1/z)
        exp_neg1_1z = torch.exp(-1/(1-z))
        tau_prime = torch.where((z > 0) & (z < 1),
                                (exp_neg1_z * (1/z**2) * exp_neg1_z - exp_neg1_z * exp_neg1_1z * (-1/(1-z)**2))
                                / (exp_neg1_z + exp_neg1_1z)**2,
                                torch.zeros_like(z))
        return torch.where((z > 0) & (z < 1), tau_prime * (2 / self.u_min) * (u - 0.5 * self.u_min) + self.tau(z), torch.zeros_like(z))

class Regularizer_identity():
    def set_costant(self, v, c, eta, final_time):
        pass
    
    def regulariz_function(self, u):
        return u
    
    def derivative_regulariz_function(self, u):
        return torch.ones_like(u)

class RMSprop_SDE_1_order(torchsde.SDEIto):
    def __init__(self, eta, beta, function_f_and_Sigma, All_time, regularizer, eps = 1e-8, Verbose = True):
        super().__init__(noise_type="general")
        self.sde_type = "ito"

        self.eta = torch.tensor(eta)
        self.c = (1-beta)/eta
        self.eps = eps
        self.function_f_and_Sigma = function_f_and_Sigma

        self.theta_old = None
        self.diffusion = None
        self.drift = None
        self.i = 1
        self.verbose = Verbose
        self.final_time = All_time[-1]
        self.All_time = All_time

        self.regularizer = regularizer

    def f_julia(self, *input):
        input = torch.cat(input)
        return self.f(input[-1], input[:-1]).detach().cpu().numpy()
    def g_julia(self, *input):
        input = torch.cat(input)
        return self.g(input[-1], input[:-1]).detach().cpu().numpy()
    
    def f(self, t, x):
        if self.i == 1: self.start_new_f = time.time()
        if self.i > 1:   
            if self.i % 1000 == 0: 
                print(f'time between f {self.i} calls {(time.time() - self.start_new_f):.2f}s, time: {t:.4f}')
                self.start_new_f = time.time()
        self.i += 1
        
        theta, v, w = self.divide_input(x)

        if (v<0).sum() > 0: 
            if v.min() < -self.eta**2: print('Warning: negative values in v', (v<0).sum().item(), v.min().item(), 'at time ', t)
            v[v<0] = 0
        
        if self.diffusion is None and self.drift is None: self.regularizer.set_costant(v, self.c, self.eta, self.final_time)
        if self.theta_old is None or (self.theta_old != theta).any(): self.update_quantities(theta, t)
        if self.verbose: print(f'v: {torch.norm(v)/v.shape[0]}, grad: {torch.norm(self.f_grad)/self.f_grad.shape[0]}')        

        v_reg = self.regularizer.regulariz_function(v)

        # Theta coefficient
        denom = 1/(torch.sqrt(v_reg) + self.eps)
        coef_theta = - self.f_grad  * denom

        # V coefficient
        f_grad_square = torch.pow(self.f_grad, 2)
        coef_v = self.c  * (f_grad_square + self.diag_Sigma - v_reg)

        # W coefficient
        coef_w = torch.zeros_like(w)
        self.drift = torch.concat((coef_theta, coef_v, coef_w), dim = 0).unsqueeze(0)

        if torch.isnan(self.drift).any():
            print("Warning: NaN values detected in drift")
            breakpoint()
        return self.drift

    def g(self, t, x):
        theta, v, w = self.divide_input(x)

        if self.diffusion is None and self.drift is None: self.regularizer.set_costant(v, self.c, self.eta, self.final_time)       
        if self.theta_old is None or (self.theta_old != theta).any(): self.update_quantities(theta, t)
        if (v<0).sum() > 0:
            print('Warning: negative values in v', (v<0).sum().item(), v.min().item(), 'at time ', t)
            v[v<0] = 0
        v_reg = self.regularizer.regulariz_function(v)

        denom = 1/(torch.sqrt(v_reg) + self.eps)

        M_theta = torch.sqrt(self.eta) * torch.diag(denom) @ self.Sigma_sqrt
        M_v = torch.zeros_like(M_theta)
        M_w = torch.eye(M_theta.shape[0], device = M_theta.device)
        self.diffusion = torch.concat((M_theta, M_v, M_w), dim = 0).unsqueeze(0)

        if torch.isnan(self.diffusion).any():
            self.found_Nan = True
            print("Warning: NaN values detected in diffusion")
            breakpoint()
        return self.diffusion

    def update_quantities(self, theta, t):
        # torch.cuda.empty_cache()
        self.diffusion = None
        self.drift = None

        self.theta_old = theta

        self.f_grad = self.function_f_and_Sigma.expected_value_gradient(theta) 
        self.Sigma_sqrt = self.function_f_and_Sigma.Sqrt_sigma()
        self.diag_Sigma = torch.diag(self.function_f_and_Sigma.Sigma())

    def divide_input(self, x):
        x = x.squeeze()
        aux = x.shape[0] // 3
        theta = x[:aux]
        v = x[aux:2*aux]
        w = x[2*aux:3*aux]
        return theta, v, w
    
class Sde_2(torchsde.SDEIto):
    def __init__(self, eta, beta, function_f_and_Sigma, All_time, regularizer, eps = 1e-8, Verbose = True):
        super().__init__(noise_type="general")
        self.sde_type = "ito"
        self.eta = torch.tensor(eta)
        self.c = (1-beta)/eta
        self.eps = eps
        self.function_f_and_Sigma = function_f_and_Sigma
        self.H = function_f_and_Sigma.A
    
    def f(self, t, x):
        theta, v, w = self.divide_input(x)
        denom = 1/(torch.sqrt(v) + self.eps)

        term = (self.H@theta)**2 + torch.diag(self.H@self.H) - v
        drift_theta = denom*self.H @ theta -self.eta/2 *( (self.H * torch.ger(denom, denom)) @ self.H @ theta + self.c*torch.diag(denom**2)@torch.diag(term)@((self.H@theta)*denom) )
        drift_v = self.c*term +self.eta/2 * (self.c*2*torch.diag(self.H@theta)@self.H@ self.H @theta * denom + self.c**2*term)
        drift_w = torch.zeros_like(w)
        drift = torch.concat((drift_theta, drift_v, drift_w), dim = 0).unsqueeze(0)
        return drift
    
    def g(self, t, x):
        theta, v, w = self.divide_input(x)
        denom = 1/(torch.sqrt(v) + self.eps)

        M_theta = torch.sqrt(self.eta) * torch.diag(denom) @ self.H
        M_v = -2*self.c*torch.diag(self.H@theta)@self.H + torch.sqrt(torch.tensor(2))*self.c*self.function_f_and_Sigma.M()@torch.diag(w)
        M_w = torch.eye(M_theta.shape[0], device = M_theta.device)
        diffusion = torch.concat((M_theta, M_v, M_w), dim = 0).unsqueeze(0)
        return diffusion

    def divide_input(self, x):
        x = x.squeeze()
        aux = x.shape[0] // 3
        theta = x[:aux]
        v = x[aux:2*aux]
        w = x[2*aux:3*aux]
        return theta, v, w

def RMSprop_optimizer(funz, x_0, lr, beta, num_steps = 1000):
    optimizer = torch.optim.RMSprop([x_0.detach().clone().requires_grad_(True)], lr=lr, alpha=beta)
    path_x = []  
    path_v = [] 
    dim = x_0.shape[0]

    for step in range(num_steps):
        optimizer.zero_grad()
        gamma = torch.randn(dim, device=x_0.device)
        x = optimizer.param_groups[0]['params'][0]
        path_x.append(x.detach().clone())
        if step == 0:
            path_v.append(torch.zeros_like(x))
        else:
            path_v.append(optimizer.state[x]['square_avg'])

        loss = funz.function(x, gamma)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'Step {step}, Loss: {loss.item()}')
    return torch.stack(path_x), torch.stack(path_v)

def divide_Result(Res):
    dim = Res.shape[2] // 3
    return Res[:, 0, :dim], Res[:, 0, dim:2*dim], Res[:, 0, 2*dim:3*dim]

def plot_error(error_list, label_list, x, title, path):
    x = x.cpu().detach().numpy()
    lenght = error_list[0].shape[0]
    x = x[-lenght:]
    for error, label in zip(error_list, label_list):
        error = error.cpu().detach().numpy()
        plt.plot(x, error, label=label)
    plt.legend()
    plt.title(title)
    final_path = path+'/'+title+'_.png'
    plt.savefig(final_path)
    plt.close()
def aux(path_x_d, path_v_d, Res):
    path_x, path_v, _= divide_Result(Res)
    Error_x = torch.norm(path_x_d[1:] - path_x, p =1, dim = 1)
    Error_v = torch.norm(path_v_d[1:] - path_v, p=1, dim = 1)
    return Error_x, Error_v[-300:]

def simulation():
    print('Simulation')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 2
    funz = funz_quadratica_con_noise(dim = dim)

    x_0 = torch.randn(dim, requires_grad=True).to(device) * 1e4
     
    eta = 0.1 
    beta = 0.9
    num_steps = 1000
    path_x_d, path_v_d = RMSprop_optimizer(funz, x_0, eta, beta, num_steps)

    eps_reg = 1e-8
    Reg_sde_2 = Regularizer_Phi(eps_reg)
    Reg_sde_1 = Regularizer_Phi(eps_reg)

    y_0 = torch.zeros(3*dim, requires_grad=True).to(device)
    y_0[:dim] = path_x_d[1, :]
    y_0[dim:2*dim] = path_v_d[1, :]
    t1 = eta * num_steps
    t = torch.linspace(eta, t1, num_steps-1)
    print('SDE eta - 1st order')
    sde_1_order = RMSprop_SDE_1_order(eta, beta, funz, All_time = t, regularizer=Reg_sde_1, Verbose=False)
    Result_eta_1_order_1 = torchsde.sdeint(sde_1_order, y_0.unsqueeze(0).to(device), t, method = 'euler', dt =eta)
    print('SDE eta**2 - 1st order')
    sde_1_order = RMSprop_SDE_1_order(eta, beta, funz, All_time = t, regularizer=Reg_sde_1, Verbose=False)
    Result_eta_2_order_1 = torchsde.sdeint(sde_1_order, y_0.unsqueeze(0).to(device), t, method = 'euler', dt =eta**2)
    print('SDE adattivo - 1st order')
    sde_1_order = RMSprop_SDE_1_order(eta, beta, funz, All_time = t, regularizer=Reg_sde_1, Verbose=False)
    Result_eta_adattivo_order_1 = torchsde.sdeint(sde_1_order, y_0.unsqueeze(0).to(device), t, method = 'euler', adaptive=True)

    print('SDE eta - 2nd order')
    sde_2_order = RMSprop_SDE(eta, beta, funz, All_time = t, regularizer=Reg_sde_2, Verbose=False)
    Result_eta_1_order_2 = torchsde.sdeint(sde_2_order, y_0.unsqueeze(0).to(device), t, method = 'euler', dt =eta)
    print('SDE eta**2 - 2nd order')
    sde_2_order = RMSprop_SDE(eta, beta, funz, All_time = t, regularizer=Reg_sde_2, Verbose=False)
    Result_eta_2_order_2 = torchsde.sdeint(sde_2_order, y_0.unsqueeze(0).to(device), t, method = 'euler', dt =eta**2)
    print('SDE adattivo - 2nd order')
    sde_2_order = RMSprop_SDE(eta, beta, funz, All_time = t, regularizer=Reg_sde_2, Verbose=False)
    Result_eta_adattivo_order_2 = torchsde.sdeint(sde_2_order, y_0.unsqueeze(0).to(device), t, method = 'euler', adaptive=True)

    print('Errors')
    Error_label = ['dt=\eta ord=1', 'dt=\eta^2 ord=1', 'Adaptive ord=1', 'dt=\eta ord=2', 'dt=\eta^2 ord=2', 'Adaptive ord=2']

    Error_x_eta_1_ord_1, Error_v_eta_1_ord_1 = aux(path_x_d, path_v_d, Result_eta_1_order_1)
    Error_x_eta_2_ord_1, Error_v_eta_2_ord_1 = aux(path_x_d, path_v_d, Result_eta_2_order_1)
    Error_x_adattivo_ord_1, Error_v_adattivo_ord_1 = aux(path_x_d, path_v_d, Result_eta_adattivo_order_1)

    Error_x_eta_1_ord_2, Error_v_eta_1_ord_2 = aux(path_x_d, path_v_d, Result_eta_1_order_2)
    Error_x_eta_2_ord_2, Error_v_eta_2_ord_2 = aux(path_x_d, path_v_d, Result_eta_2_order_2)
    Error_x_adattivo_ord_2, Error_v_adattivo_ord_2 = aux(path_x_d, path_v_d, Result_eta_adattivo_order_2)

    Error_x = [Error_x_eta_1_ord_1, Error_x_eta_2_ord_1, Error_x_adattivo_ord_1, Error_x_eta_1_ord_2, Error_x_eta_2_ord_2, Error_x_adattivo_ord_2]
    Error_v = [Error_v_eta_1_ord_1, Error_v_eta_2_ord_1, Error_v_adattivo_ord_1, Error_v_eta_1_ord_2, Error_v_eta_2_ord_2, Error_v_adattivo_ord_2]
    path = '/home/callisti/Thesis/Master-Thesis/Result_new/funz_semplice'
    # plot_error(Error_x, Error_label, t, 'Error x', path)
    # plot_error(Error_v, Error_label, t, 'Error v', path)

    plot_error([Error_x[i] for i in [0, 3]], [Error_label[i] for i in [0, 3]], t, 'Error x eta=1', path)
    plot_error([Error_v[i] for i in [0, 3]], [Error_label[i] for i in [0, 3]], t, 'Error v eta=1', path)
    plot_error([Error_x[i] for i in [1, 4]], [Error_label[i] for i in [1, 4]], t, 'Error x eta=2', path)
    plot_error([Error_v[i] for i in [1, 4]], [Error_label[i] for i in [1, 4]], t, 'Error v eta=2', path)
    plot_error([Error_x[i] for i in [2, 5]], [Error_label[i] for i in [2, 5]], t, 'Error x adattivo', path)
    plot_error([Error_v[i] for i in [2, 5]], [Error_label[i] for i in [2, 5]], t, 'Error v adattivo', path)

def one_call(funz, eta, beta, num_steps, x_0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = random.randint(0, 10000)
    torch.manual_seed(seed)
    path_x_d, path_v_d = RMSprop_optimizer(funz, x_0, eta, beta, num_steps)
    eps_reg = 1e-8
    Reg_sde_2 = Regularizer_Phi(eps_reg)
    Reg_sde_1 = Regularizer_Phi(eps_reg)
    dim = x_0.shape[0]

    y_0 = torch.zeros(3*dim, requires_grad=True).to(device)
    y_0[:dim] = path_x_d[1, :]
    y_0[dim:2*dim] = path_v_d[1, :]
    t1 = eta * num_steps
    t = torch.linspace(eta, t1, num_steps-1)

    sde_1_order = RMSprop_SDE_1_order(eta, beta, funz, All_time = t, regularizer=Reg_sde_1, Verbose=False)
    Result_eta_1_order_1 = torchsde.sdeint(sde_1_order, y_0.unsqueeze(0).to(device), t, method = 'euler', adaptive = True, atol = eta)
    sde_2_order = RMSprop_SDE(eta, beta, funz, All_time = t, regularizer=Reg_sde_2, Verbose=False)
    Result_eta_1_order_2 = torchsde.sdeint(sde_2_order, y_0.unsqueeze(0).to(device), t, method = 'euler', adaptive = True, atol = eta**2)

    return path_x_d, path_v_d, Result_eta_1_order_1, Result_eta_1_order_2

def different_eta(final_time, dim):
    print('Different eta+')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_0 = torch.randn(dim, requires_grad=True).to(device) * 1e2
    funz = funz_quadratica_con_noise(dim = x_0.shape[0])
    FinalError = {}
    for i in [1, 2, 3, 4]:
        print('run', i)
        eta = 10**(-i/4)
        beta = 1-0.5*eta
        num_steps = int(final_time/eta)
        All_Path_x_d, All_Path_v_d = [], []
        All_Result_order_1, All_Result_order_2 = [], []
        for _ in range(3):
            path_x_d, path_v_d, Result_order_1, Result_order_2 = one_call(funz, eta, beta, num_steps, x_0)
            All_Path_x_d.append(path_x_d)
            All_Path_v_d.append(path_v_d)
            All_Result_order_1.append(Result_order_1)
            All_Result_order_2.append(Result_order_2)
        All_Path_v_d = torch.stack(All_Path_v_d)
        All_Path_x_d = torch.stack(All_Path_x_d)
        All_Result_order_1 = torch.stack(All_Result_order_1)
        All_Result_order_2 = torch.stack(All_Result_order_2)

        Path_v_d_mean = torch.mean(All_Path_v_d, dim = 0)
        Path_x_d_mean = torch.mean(All_Path_x_d, dim = 0)
        Result_order_1_mean = torch.mean(All_Result_order_1, dim = 0)
        Result_order_2_mean = torch.mean(All_Result_order_2, dim = 0)

        Error_x_1 = torch.norm(Path_x_d_mean[-1]-Result_order_1_mean[-1, 0, :dim], p=1)
        Error_v_1 = torch.norm(Path_v_d_mean[-1]-Result_order_1_mean[-1, 0, dim:2*dim], p=1)
        Error_x_2 = torch.norm(Path_x_d_mean[-1]-Result_order_2_mean[-1, 0, :dim], p=1)
        Error_v_2 = torch.norm(Path_v_d_mean[-1]-Result_order_2_mean[-1, 0, dim:2*dim], p=1)
        FinalError[eta] = [Error_x_1, Error_v_1, Error_x_2, Error_v_2]
    print('Plotting')
    etas = list(FinalError.keys())
    errors_x_1 = [FinalError[eta][0].item() for eta in etas]
    errors_v_1 = [FinalError[eta][1].item() for eta in etas]
    errors_x_2 = [FinalError[eta][2].item() for eta in etas]
    errors_v_2 = [FinalError[eta][3].item() for eta in etas]

    plt.figure()
    plt.plot(etas, errors_x_1, label='Error_x_1')
    plt.plot(etas, errors_x_2, label='Error_x_2')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eta')
    plt.ylabel('Error_x')
    plt.legend()
    plt.title('Error_x vs eta')
    plt.savefig('/home/callisti/Thesis/Master-Thesis/Result_new/funz_semplice/Error_x_vs_eta.png')
    plt.close()

    plt.figure()
    plt.plot(etas, errors_v_1, label='Error_v_1')
    plt.plot(etas, errors_v_2, label='Error_v_2')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('eta')
    plt.ylabel('Error_v')
    plt.legend()
    plt.title('Error_v vs eta')
    plt.savefig('/home/callisti/Thesis/Master-Thesis/Result_new/funz_semplice/Error_v_vs_eta.png')
    plt.close()

def Julia():
    print('Julia+')
    dim = 2
    funz = funz_quadratica_con_noise(dim = dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_0 = torch.randn(dim, requires_grad=True).to(device) * 1e2
    eta = 0.1
    beta = 0.9
    num_steps = 300
    path_x_d, path_v_d = RMSprop_optimizer(funz, x_0, eta, beta, num_steps)

    dim_stato = 3*dim
    dim_rumore = 2
    noise_prototype = np.zeros((dim_stato, dim_rumore))

    y_0 = torch.zeros(3*dim, requires_grad=True).to(device)
    y_0[:dim] = path_x_d[1, :]
    y_0[dim:2*dim] = path_v_d[1, :]
    t1 = eta * num_steps
    t = torch.linspace(eta, t1, num_steps-1)

    eps_reg = 1e-8
    Reg_sde_2 = Regularizer_Phi(eps_reg)
    Reg_sde_1 = Regularizer_Phi(eps_reg)

    sde_2_order = RMSprop_SDE(eta, beta, funz, All_time = t, regularizer=Reg_sde_2, Verbose=False)


    # Definizione del problema SDE
    prob = de.SDEProblem(sde_2_order.f_julia, sde_2_order.g_julia, y_0.detach().cpu().numpy(), (eta, t1), noise_rate_prototype=noise_prototype)

    # Risoluzione con metodo di Milstein (SRIW1 è di ordine 1.5, simile a Milstein)
    sol = de.solve(prob, de.LambaEM())

    # Estrazione dei risultati
    t = sol.t
    X = np.array(sol)
    breakpoint()

different_eta(150, 2)