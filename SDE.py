import torch
import torchsde
import time



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

        self.Loss_grad = []
        self.Loss = []


    def f(self, t, x):
        if self.i == 1: self.start_new_f = time.time()
        if self.i > 1:   
            if self.i % 100 == 0: 
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

        f_grad_square = torch.pow(self.f_grad, 2)    

        v_reg = self.regularizer.regulariz_function(v)
        v_reg_grad = self.regularizer.derivative_regulariz_function(v)

        # Theta coefficient
        denom = 1/(torch.sqrt(v_reg) + self.eps)

        coef_theta = (self.f_hessian * (torch.ger(denom, denom))) @ self.f_grad + self.c*(f_grad_square + self.diag_Sigma - v_reg) * (self.f_grad * torch.pow(denom, 3) * v_reg_grad)
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
        # M_v = -2 * torch.sqrt(self.eta) * self.c * torch.diag(self.f_grad) @  self.Sigma_sqrt + torch.sqrt(torch.tensor(2)) * self.c * self.square_root_var_z_squared @ torch.diag(w)
        M_v = -2 * torch.sqrt(self.eta) * self.c * torch.diag(self.f_grad) @  self.Sigma_sqrt
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

        self.function_f_and_Sigma.update_parameters(theta)
        self.f_grad = self.function_f_and_Sigma.compute_gradients_f() 
        self.f_hessian = self.function_f_and_Sigma.compute_hessian()
        self.Sigma_sqrt, self.diag_Sigma = self.function_f_and_Sigma.compute_sigma()
        self.grad_expected_frad_f_squared = self.function_f_and_Sigma.compute_gradients_expected_grad_f_squared()
        self.square_root_var_z_squared = self.function_f_and_Sigma.compute_var_z_squared()

        if torch.any(torch.isclose(self.All_time, t, atol=self.eta**2 / 3)):
            # print('salvo loss', t)
            self.Loss_grad.append(self.f_grad.cpu())
            self.Loss.append(self.function_f_and_Sigma.compute_f().cpu())

    def get_loss_grad(self):
        return torch.stack(self.Loss_grad).cpu()
    def get_loss(self):
        return torch.stack(self.Loss).cpu()

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

        self.Loss_grad = []
        self.Loss = []

    
    def f(self, t, x):
        if self.i == 1: self.start_new_f = time.time()
        if self.i > 1:   
            if self.i % 100 == 0: 
                print(f'time between f {self.i} calls {(time.time() - self.start_new_f):.2f}s, time: {t:.4f}')
                self.start_new_f = time.time()
        self.i += 1
        
        theta, v, w = self.divide_input(x)
        
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

        if self.diffusion is None and self.drift is None:
           self.regularizer.set_costant(v, self.c, self.eta, self.final_time)       

        if self.theta_old is None or (self.theta_old != theta).any():
            self.update_quantities(theta, t)
        
        v_reg = self.regularizer.regulariz_function(v)

        denom = 1/(torch.sqrt(v_reg) + self.eps)

        M_theta = torch.sqrt(self.eta) * torch.diag(denom) @ self.Sigma_sqrt
        M_v = torch.zeros_like(M_theta)
        M_w = torch.ones_like(M_theta)
        self.diffusion = torch.concat((M_theta, M_v, M_w), dim = 0).unsqueeze(0)

        if torch.isnan(self.diffusion).any():
            self.found_Nan = True
            print("Warning: NaN values detected in diffusion")
            breakpoint()
        return self.diffusion
    
    def get_loss_grad(self):
        return torch.stack(self.Loss_grad).cpu()
    def get_loss(self):
        return torch.stack(self.Loss).cpu()

    def update_quantities(self, theta, t):
        # torch.cuda.empty_cache()
        self.diffusion = None
        self.drift = None

        self.theta_old = theta

        self.function_f_and_Sigma.update_parameters(theta)
        self.f_grad = self.function_f_and_Sigma.compute_gradients_f() 
        self.Sigma_sqrt, self.diag_Sigma = self.function_f_and_Sigma.compute_sigma()

        if torch.any(torch.isclose(self.All_time, t, atol=self.eta/3)):
            # print('salvo loss', t)
            self.Loss_grad.append(self.f_grad.cpu())
            self.Loss.append(self.function_f_and_Sigma.compute_f().cpu())

    def get_loss_grad(self):
        return torch.stack(self.Loss_grad).cpu()
    def get_loss(self):
        return torch.stack(self.Loss).cpu()

    def divide_input(self, x):
        x = x.squeeze()
        aux = x.shape[0] // 3
        theta = x[:aux]
        v = x[aux:2*aux]
        w = x[2*aux:3*aux]
        return theta, v, w