import torch
import numpy as np
from diffeqpy import de
import time

def create_def_positive_matrix(dim):
    A = torch.randn(dim, dim)
    A = A @ A.T  
    A += torch.eye(dim) * 1e-6
    return A

class FunzQuadraticaConNoise:
    def __init__(self, A=None, dim=10):
        if A is not None:
            self.A = A
        else:
            self.A = create_def_positive_matrix(dim)
        self.A = self.A.cpu().numpy()
        self.sigma = self.A @ self.A
        AA_T_squared = self.sigma**2
        self.M_matrix = 2 * AA_T_squared 

    def function(self, x, gamma):
        return 0.5 * (x-gamma).T @ self.A @ (x - gamma) - 0.5 * np.trace(self.A)
    
    def expected_value(self, x):
        return 0.5 * x.T @ self.A @ x
    
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
        return np.linalg.cholesky(self.M_matrix)
    
    def compute_gradients_expected_grad_f_squared(self, x):
        return (2 * np.diag(self.expected_value_gradient(x)) @ self.A).T

def sde_drift_1(u, p, t):
    theta, v, w = np.split(u, 3)
    f_grad = p['funz'].expected_value_gradient(theta)
    diag_Sigma = np.diag(p['funz'].Sigma())
    v_reg = np.maximum(v, p['C_reg'])
    denom = 1 / (np.sqrt(v_reg) + 1e-8)
    coef_theta = - f_grad * denom
    coef_v = p['c'] * (f_grad**2 + diag_Sigma - v_reg)
    coef_w = np.zeros_like(w)
    return np.concatenate((coef_theta, coef_v, coef_w))

def sde_diffusion_1(u, p, t):
    theta, v, w = np.split(u, 3)
    v_reg = np.maximum(v, p['C_reg'])
    denom = 1 / (np.sqrt(v_reg) + 1e-8)
    M_theta = np.sqrt(p['eta']) * np.diag(denom) @ p['funz'].Sqrt_sigma()
    M_v = np.zeros_like(M_theta)
    M_w = np.eye(M_theta.shape[0])
    return np.concatenate((M_theta, M_v, M_w), axis=0)

def sde_drift_2(u, p, t):
    theta, v, w = np.split(u, 3)
    v_reg = np.maximum(v, p['C_reg'])
    denom = 1 / (np.sqrt(v_reg) + 1e-8)
    H = p['funz'].A
    diag_Sigma = np.diag(p['funz'].Sigma())
    aux = H @ theta
    term = (aux)**2 + diag_Sigma - v_reg
    drift_theta = denom * aux - p['eta']/2 * ((H * np.outer(denom, denom)) @ aux + p['c'] * np.diag(denom**2) @ np.diag(term) @ (aux * denom))
    drift_v = p['c'] * term + p['eta']/2 * (p['c'] * 2 * H @ np.diag(aux) @ aux * denom + p['c']**2 * term)
    drift_w = np.zeros_like(w)
    return np.concatenate((drift_theta, drift_v, drift_w))

def sde_diffusion_2(u, p, t):
    theta, v, w = np.split(u, 3)
    v_reg = np.maximum(v, p['C_reg'])
    denom = 1 / (np.sqrt(v_reg) + 1e-8)
    H = p['funz'].A
    M_theta = np.sqrt(p['eta']) * np.diag(denom) @ H
    M_v = -2 * p['c'] * np.diag(H @ theta) @ H + np.sqrt(2) * p['c'] * p['funz'].M() @ np.diag(w)
    M_w = np.eye(M_theta.shape[0])
    return np.concatenate((M_theta, M_v, M_w), axis=0)

print('Start+')
dim = 5
funz = FunzQuadraticaConNoise(dim=dim)
u0 = np.random.randn(3 * dim)
u0[dim:2*dim] = 0.01
u0[2*dim:] = 0
tspan = (0.0, 1.0)
p = {'funz': funz, 'eta': 0.01, 'c': 0.1, 'C_reg': 1e-8}

start = time.time()
prob1 = de.SDEProblem(sde_drift_1, sde_diffusion_1, u0, tspan, p, noise_rate_prototype=np.zeros((3 * dim, dim)))
sol1 = de.solve(prob1, de.EM(), dt = 0.01)
end = time.time()
print('Time elapsed: ', end - start)

start = time.time()
prob2 = de.SDEProblem(sde_drift_2, sde_diffusion_2, u0, tspan, p,  noise_rate_prototype=np.zeros((3 * dim, dim)))
sol2 = de.solve(prob2, de.EM(), dt = 0.01)
end = time.time()
print('Time elapsed: ', end - start)

breakpoint()
