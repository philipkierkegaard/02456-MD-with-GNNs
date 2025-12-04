import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import linregress
from ase import units
from src.painn import PaiNN
from src.MD17Dataset import MD17

# =====================================================
#  Configuration
# =====================================================
TIMESTEPS_TO_TEST = [0.1]
REPETITIONS = 1          # Runs per time step

# TIME SETTINGS
TOTAL_TIME_PS = 15.0      # Total duration of simulation
EQUILIBRATION_TIME_PS = 10.0 # Time to discard from the beginning

TEMPERATURE = 300.0      # Kelvin
MOLECULE = "paracetamol_dft"
WEIGHT_PATH = "models/best_model_paracetamol_dft.pth"
DATASET_PATH = "data/md17"
MAX_FORCE_CLIP = 30.0    # Cap forces to prevent immediate explosions

# Robust Device Selection
device = torch.device("cpu")
print(f"Using device: {device}")

# =====================================================
#  Helpers
# =====================================================
MASS_TABLE = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998}

def compute_energy_and_forces(model, z, R):
    R_model = R.detach().clone().float().requires_grad_(True)
    z = z.to(device)
    graph_indexes = torch.zeros(len(z), dtype=torch.long, device=device)

    # Forward
    atomic_contrib = model(atoms=z, atom_positions=R_model, graph_indexes=graph_indexes)
    energy = atomic_contrib.sum()
    
    # Backward
    forces = -torch.autograd.grad(energy, R_model, create_graph=False)[0]
    
    if torch.isnan(forces).any():
        return energy.detach().double(), torch.zeros_like(R).double(), True
        
    forces = forces.double()
    forces = torch.clamp(forces, min=-MAX_FORCE_CLIP, max=MAX_FORCE_CLIP)
    
    return energy.detach().double(), forces, False

def init_velocities(z, T, seed):
    torch.manual_seed(seed) 
    masses = torch.tensor([MASS_TABLE[int(a)] for a in z], device=device).view(-1, 1)
    std = torch.sqrt((units.kB * T) / masses)
    v = torch.randn((len(z), 3), device=device) * std
    v_cm = (v * masses).sum(dim=0) / masses.sum()
    v -= v_cm
    return v

# =====================================================
#  Single Simulation Run
# =====================================================
def run_single_simulation(dt_fs, n_steps, seed, model, data):
    # Setup
    z = data.x.to(device)
    R = data.pos.to(device).double()
    v = init_velocities(z, TEMPERATURE, seed).double()
    masses = torch.tensor([MASS_TABLE[int(a)] for a in z], device=device).view(-1, 1).double()
    dt = dt_fs * units.fs
    
    # Initial State
    curr_E_pot, curr_F, err = compute_energy_and_forces(model, z, R)
    if err: return None
    
    times_ps = []
    energies = []
    
    # Run Loop
    for step in range(n_steps):
        v += 0.5 * dt * (curr_F / masses)
        R += dt * v
        new_E_pot, new_F, err = compute_energy_and_forces(model, z, R)
        
        if err: return None
            
        v += 0.5 * dt * (new_F / masses)
        curr_E_pot = new_E_pot
        curr_F = new_F
        
        # Energy Calculation
        E_kin = 0.5 * (masses * v**2).sum()
        E_tot = curr_E_pot + E_kin
        
        # Record time and energy
        current_time_ps = step * dt_fs / 1000.0
        times_ps.append(current_time_ps)
        energies.append(E_tot.item())

    # --- EQUILIBRATION FILTERING ---
    times_np = np.array(times_ps)
    energies_np = np.array(energies)
    
    # Create a mask for times greater than equilibration time
    mask = times_np >= EQUILIBRATION_TIME_PS
    
    prod_times = times_np[mask]
    prod_energies = energies_np[mask]
    
    # Safety Check: If simulation crashed early or time < equilib, return None
    if len(prod_times) < 2:
        return None

    # Calculate Drift Slope on the Production Phase Only
    slope, _, _, _, _ = linregress(prod_times, prod_energies)
    
    n_atoms = len(z)
    drift = abs(slope * 1000) / n_atoms
    return drift

# =====================================================
#  Main Execution
# =====================================================
def main():
    print(f"Loading {MOLECULE}...")
    dataset = MD17(DATASET_PATH, MOLECULE)
    data = dataset[0]
    
    print(f"Loading Model {WEIGHT_PATH}...")
    model = PaiNN(num_message_passing_layers=3).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()

    results_mean = []
    results_std = []
    valid_timesteps = []

    print(f"\n--- Starting Analysis (Total: {TOTAL_TIME_PS} ps, Equil: {EQUILIBRATION_TIME_PS} ps) ---")
    print(f"Calculated drift will be based on the last {TOTAL_TIME_PS - EQUILIBRATION_TIME_PS} ps.")
    print(f"{'dt (fs)':<10} | {'Steps':<10} | {'Mean Drift':<15} | {'Std Dev':<15}")
    print("-" * 60)

    for dt in TIMESTEPS_TO_TEST:
        # Calculate steps needed for FULL duration
        n_steps_needed = int((TOTAL_TIME_PS * 1000) / dt)
        
        drifts = []
        for i in range(REPETITIONS):
            seed = i + 100 
            drift = run_single_simulation(dt, n_steps_needed, seed, model, data)
            if drift is not None:
                drifts.append(drift)
        
        if len(drifts) > 0:
            mean_d = np.mean(drifts)
            std_d = np.std(drifts)
            
            results_mean.append(mean_d)
            results_std.append(std_d)
            valid_timesteps.append(dt)
            print(f"{dt:<10} | {n_steps_needed:<10} | {mean_d:<15.5f} | {std_d:<15.5f}")
        else:
            print(f"{dt:<10} | {n_steps_needed:<10} | CRASHED")

    # =====================================================
    #  Plotting
    # =====================================================
    if not valid_timesteps:
        print("No valid simulations to plot.")
        return

    arr_timesteps = np.array(valid_timesteps)
    arr_mean = np.array(results_mean)
    arr_std = np.array(results_std)

    plt.figure(figsize=(10, 6))
    
    label_text = f'Mean Drift\n(Total={TOTAL_TIME_PS}ps, Equil={EQUILIBRATION_TIME_PS}ps)'
    plt.plot(arr_timesteps, arr_mean, linewidth=2, 
             color='tab:green', label=label_text)

    lower_bound = arr_mean - arr_std
    upper_bound = arr_mean + arr_std
    lower_bound = np.maximum(lower_bound, 1e-5) # Clip for log scale

    plt.xlabel("Time Step $\Delta t$ [fs]")
    plt.ylabel("Energy Drift [meV / atom / ps]")
    plt.title(f"Integration Stability Analysis\n{MOLECULE}")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xticks(TIMESTEPS_TO_TEST, [str(t) for t in TIMESTEPS_TO_TEST])
    
    filename = "drift_vs_timestep_equilibrated.png"
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")
    plt.show()

if __name__ == "__main__":
    main()
