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
TIMESTEPS_TO_TEST = [0.01, 0.05, 0.1, 0.2, 0.5]
REPETITIONS = 3         # Runs per time step

# FIXED TIME APPROACH
TARGET_TIME_PS = 0.5     # We simulate exactly 0.5 ps for every setting

TEMPERATURE = 300.0      # Kelvin
MOLECULE = "benzene2018_dft"
WEIGHT_PATH = "models/best_model_benzene2018_dft_400epochs.pth"
DATASET_PATH = "data/md17"
MAX_FORCE_CLIP = 100.0

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
        
        times_ps.append(step * dt_fs / 1000.0)
        energies.append(E_tot.item())

    # Calculate Drift Slope
    slope, _, _, _, _ = linregress(times_ps, energies)
    
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

    print(f"\n--- Starting Fixed Time Analysis (Target: {TARGET_TIME_PS} ps) ---")
    print(f"{'dt (fs)':<10} | {'Steps':<10} | {'Mean Drift':<15} | {'Std Dev':<15}")
    print("-" * 60)

    for dt in TIMESTEPS_TO_TEST:
        # Dynamic Steps calculation
        n_steps_needed = int((TARGET_TIME_PS * 1000) / dt)
        
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
    #  Plotting with Shaded Area (Fixed for Log Scale)
    # =====================================================
    arr_timesteps = np.array(valid_timesteps)
    arr_mean = np.array(results_mean)
    arr_std = np.array(results_std)

    plt.figure(figsize=(10, 6))
    
    # 1. Plot the Mean Line
    plt.plot(arr_timesteps, arr_mean, linewidth=2, 
             color='tab:green', label=f'Mean Drift (Time={TARGET_TIME_PS}ps)')

    # 2. Calculate Bounds and Fix for Log Scale
    lower_bound = arr_mean - arr_std
    upper_bound = arr_mean + arr_std
    
    # CRITICAL FIX: Clip the lower bound to be positive
    # We set any value <= 0 to a tiny fraction of the mean, or a hard floor like 1e-5
    lower_bound = np.maximum(lower_bound, 1e-5)

    
    plt.xlabel("Time Step $\Delta t$ [fs]")
    plt.ylabel("Energy Drift [meV / atom / ps]")
    plt.title(f"Integration Stability Analysis (Fixed Duration)\n{MOLECULE}")
    plt.legend()
    plt.grid(True, alpha=0.3, which="both") # 'which="both"' adds grid lines for log sub-ticks
    
    plt.xscale('log')
    plt.yscale('log')
    
    # Fix X-axis ticks
    plt.xticks(TIMESTEPS_TO_TEST, [str(t) for t in TIMESTEPS_TO_TEST])
    
    filename = "drift_vs_timestep_shaded_fixed.png"
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")
    plt.show()

if __name__ == "__main__":
    main()