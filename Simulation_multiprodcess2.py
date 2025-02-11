import os

def run_simulation_on_gpu(script_path, gpu_id, num_simulations):
    """Esegue più istanze dello script in parallelo usando il comando di shell."""
    for _ in range(num_simulations):
        command = f"python3 {script_path} &"
        os.system(command)  # Esegue il comando come se fosse da terminale

if __name__ == "__main__":
    print('nuovo1')
    script_path = "/home/callisti/Thesis/Master-Thesis/Simulation_2.py"
    gpu_id = 0
    num_simulations = 2

    run_simulation_on_gpu(script_path, gpu_id, num_simulations)
