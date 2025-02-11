import torch.multiprocessing as mp
from CNN import CNN, Train_n_times
from SDE import RMSprop_SDE, RMSprop_SDE_1_order
from Dataset import CIFAR10Dataset
from calculation import function_f_and_Sigma
import Save_exp_multiprocess as Save_exp
import torch
import numpy as np
import time
import torchsde
import sys

sys.setrecursionlimit(10000)

def create_dataset(dataset_size=128):
    """
    Crea e ritorna un dataset preprocessato.
    """
    dataset = CIFAR10Dataset()
    dataset.to_grayscale()
    dataset.downscale(50)
    dataset.x_train = dataset.x_train[:dataset_size]
    dataset.y_train = dataset.y_train[:dataset_size]
    return dataset

def Simulation_discrete_dynamics(model, dataset, steps, lr, beta, n_runs, batch_size=1):
    """
    Simula la dinamica discreta.
    """
    print('dimension dataset:', dataset.x_train.shape[0])
    print('number of epoch:', steps * batch_size / dataset.x_train.shape[0], ' final time:', steps * lr)
    Train = Train_n_times(model, dataset, steps=steps, lr=lr, optimizer_name='RMSPROP', beta=beta)
    FinalDict = Train.train_n_times(n=n_runs, batch_size=batch_size)
    return FinalDict

def simulate_continuous_dynamics(FinalDict, args, dataset, t, eta, beta, device):
    """
    Simula le dinamiche continue con torchsde (1° e 2° ordine).
    """
    input_channels, num_classes, conv_layers, size_img = args 
    number_parameters = FinalDict[1]['Params'][0].shape[0]
    x0 = torch.zeros(3 * number_parameters, device=device)

    Result_1_order = []
    Result_2_order = []
    Loss_1_order = []
    Loss_2_order = []
    Grad_1_order = []
    Grad_2_order = []

    for i_run in FinalDict.keys():
        print(f'Run {i_run}')
        model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)        
        f = function_f_and_Sigma(model, dataset, dim_dataset=512, Verbose=False)
        x0[: 2 * number_parameters] = torch.cat((FinalDict[i_run]['Params'][0], FinalDict[i_run]['Square_avg'][0]), dim=0)

        # Simulazione con RMSprop_SDE_1_order
        sde_1_order = RMSprop_SDE_1_order(eta, beta, f, All_time=t, Verbose=False)
        aux = torchsde.sdeint(sde_1_order, x0.unsqueeze(0).to(device), t, method='euler', dt=eta)
        Result_1_order.append(aux)
        Loss_1_order.append(sde_1_order.get_loss())
        Grad_1_order.append(sde_1_order.get_loss_grad())

        # Simulazione con RMSprop_SDE (2° ordine)
        sde_2_order = RMSprop_SDE(eta, beta, f, All_time=t, Verbose=False)
        aux = torchsde.sdeint(sde_2_order, x0.unsqueeze(0).to(device), t, method='euler', dt=eta**2)
        Result_2_order.append(aux)
        Loss_2_order.append(sde_2_order.get_loss())
        Grad_2_order.append(sde_2_order.get_loss_grad())

    return Result_1_order, Result_2_order, Loss_1_order, Loss_2_order, Grad_1_order, Grad_2_order

def run_simulation(i_sim, random_seed, dataset_size, steps, eta, beta, n_runs, device):
    """
    Funzione eseguita da ogni processo.
    """
    print(f"Process {i_sim} avviato con seed {random_seed}")

    # Creazione del dataset
    dataset = create_dataset(dataset_size)

    # Parametri del dataset
    num_classes = np.unique(dataset.y_train).shape[0]
    input_channels, size_img, _ = dataset.get_image_size()

    # Setup della directory per salvare i risultati
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.4, device)

    # device = torch.device(f'cuda:{i_sim % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.device_count())
    print(f"Process {i_sim} usa dispositivo {device}")

    # Configurazione del modello
    conv_layers = [
        (3, 3, 1, 1),  # filter number, kernel size, stride, padding
        (2, 3, 1, 1),
        (2, 3, 1, 1),
        (1, 3, 1, 1)
    ]

    # if i_sim == 0:
        # save_exp.add_multiple_elements(['input channels', 'size img', 'conv layers'], [input_channels, size_img, conv_layers])

    # Imposta il seed
    torch.manual_seed(random_seed)

    # Modello
    model = CNN(input_channels=input_channels, num_classes=num_classes, conv_layers=conv_layers, size_img=size_img)

    print(f'Process {i_sim}: eta={eta}, beta={beta}, steps={steps}, n_runs={n_runs}, total steps per run={steps / eta}')

    # Simula la dinamica discreta
    FinalDict = Simulation_discrete_dynamics(model, dataset, steps, lr=eta, beta=beta, n_runs=n_runs, batch_size=1)
    # save_exp.save_result_discrete_one_run(FinalDict, i_sim)

    # Configura il tempo per torchsde
    t0, t1 = eta, (steps * eta)
    t = torch.linspace(t0, t1, steps, device=device)

    # Simula la dinamica continua
    args = (input_channels, num_classes, conv_layers, size_img)
    Result_1_order, Result_2_order, Loss_1_order, Loss_2_order, Grad_1_order, Grad_2_order = simulate_continuous_dynamics(
        FinalDict, args, dataset, t, eta, beta, device
    )

    # Salva i risultati finali
    # save_exp.save_tensor(Result_1_order, f'Result_1_order_{i_sim}_sim.pt')
    # save_exp.save_tensor(Result_2_order, f'Result_2_order_{i_sim}_sim.pt')
    # save_exp.save_tensor(Loss_1_order, f'Loss_1_order_{i_sim}_sim.pt')
    # save_exp.save_tensor(Loss_2_order, f'Loss_2_order_{i_sim}_sim.pt')
    # save_exp.save_tensor(Grad_1_order, f'Grad_1_order_{i_sim}_sim.pt')
    # save_exp.save_tensor(Grad_2_order, f'Grad_2_order_{i_sim}_sim.pt')

    print(f"Process {i_sim}: Simulazione completata.")

def parallel_simulation_torch(num_sim, dataset_size, steps, eta, beta, n_runs, output_dir):
    """
    Esegue le simulazioni in parallelo utilizzando torch.multiprocessing.
    """
    random_seeds = np.random.randint(0, 10000, size=num_sim)  # Crea una lista di seed casuali
    # save_exp = Save_exp.SaveExp(output_dir, num_sim)  # Crea un oggetto per salvare i risultati
    # save_exp.add_multiple_elements(['eta', 'beta', 'steps', 'n_simulations', 'n_runs', 'dataset_size'], [eta, beta, steps, num_sim, n_runs, dataset_size])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Avvia i processi
    processes = []
    for i_sim in range(num_sim):
        p = mp.Process(
            target=run_simulation,
            args=(i_sim, random_seeds[i_sim], dataset_size, steps, eta, beta, n_runs, device)
        )
        p.start()
        processes.append(p)

    # Attendi che tutti i processi siano completati
    for p in processes:
        p.join()
    
    # save_exp.save_dict()

if __name__ == "__main__":
    mp.set_start_method("spawn", force = True)  # Metodo per avviare i processi (necessario per CUDA)

    # Parametri principali
    num_sim = 2  # Numero di simulazioni da eseguire
    dataset_size = 128  # Dimensione del dataset
    steps = 25
    eta, beta = 0.1, 0.9
    n_runs = 1  # Numero di run per simulazione
    output_dir = "/home/callisti/Thesis/Master-Thesis/Result"

    num_threads = mp.cpu_count()
    print(f"Numero di thread CPU disponibili (multiprocessing): {num_threads}")

    # Avvio simulazioni in parallelo
    parallel_simulation_torch(num_sim, dataset_size, steps, eta, beta, n_runs, output_dir)