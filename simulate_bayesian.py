import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from copy import deepcopy
import math # Import math for velocity initialization

# --------------------------------------------------------
# IMPORT YOUR Bayesian PaiNN MODEL HERE
# --------------------------------------------------------
from src.painn_baysian import PaiNN       # <--- Modified import
from src.MD17Dataset import MD17        


# ========================================================
# UTILITY: FORCE COMPUTATION FROM MODEL
# ========================================================
def compute_forces(model, atoms, positions, graph_idx):
    """
    Computes forces as the negative gradient of predicted energy.
    
    NOTE for Bayesian Model: The forward pass samples weights from the
    approximate posterior q(w|theta). This calculation represents a
    SINGLE Monte Carlo sample of the energy from the model. 
    The KL divergence loss is calculated internally but is ignored for force
    computation, as it is only for training regularization.
    """
    positions = positions.clone().detach().requires_grad_(True)

    # Model outputs atomic energy contributions (E_i) → sum them (E_total)
    atomic_energy = model(atoms, positions, graph_idx)  # [N, 1]
    total_energy = atomic_energy.sum()
    
    # We do NOT include model.kl_loss() in the total energy for force calculation,
    # as KL divergence is a regularization term for the loss function, not part
    # of the potential energy surface.

    # Compute gradients
    forces = -torch.autograd.grad(
        total_energy,
        positions,
        grad_outputs=torch.ones_like(total_energy),
        create_graph=False,
        retain_graph=False
    )[0]

    return forces, positions


# ========================================================
# VELOCITY VERLET
# ========================================================
def velocity_verlet_step(model, atoms, graph_idx,
                         positions, velocities, masses, dt):
    # --- Forces at time t ---
    forces_t, positions = compute_forces(model, atoms, positions, graph_idx)
    accel_t = forces_t / masses[:, None]

    # --- Position update ---
    new_positions = positions + velocities * dt + 0.5 * accel_t * (dt ** 2)

    # --- Forces at t + dt (using a new MC sample from the Bayesian model) ---
    forces_tdt, new_positions = compute_forces(model, atoms, new_positions, graph_idx)
    accel_tdt = forces_tdt / masses[:, None]

    # --- Velocity update ---
    new_velocities = velocities + 0.5 * (accel_t + accel_tdt) * dt

    return new_positions.detach(), new_velocities.detach(), forces_t.detach()


# ========================================================
# MAIN SIMULATION LOOP
# ========================================================
def simulate(model, atoms, initial_positions, graph_idx,
             num_steps=200, dt=1e-3, temperature_c=300):
    N = atoms.shape[0]

    # Mass table (atomic numbers → approximate masses in amu)
    periodic_table_masses = torch.zeros(101)
    periodic_table_masses[1] = 1.008   # H
    periodic_table_masses[6] = 12.01   # C
    periodic_table_masses[7] = 14.01   # N
    periodic_table_masses[8] = 16.00   # O

    masses_amu = periodic_table_masses[atoms]          # [N]
    masses_kg = masses_amu * 1.66054e-27               # convert amu → kg

    # Temperature in Kelvin
    T = temperature_c + 273.15

    # Boltzmann constant in J/K
    k_B = 1.380649e-23

    # Standard deviation of velocities
    std = torch.sqrt(k_B * T / masses_kg)  # [N], SI units m/s
    # Convert to Å/fs (standard units for MD with amu/Å)
    # 1 m/s = 1e10 Å/s = 1e-5 Å/fs
    std_ang_fs = std * 1e-5

    # Initialize velocities from Maxwell-Boltzmann
    velocities = torch.normal(mean=0.0, std=std_ang_fs[:, None])
    
    # Remove center-of-mass motion (important for stable NVT/NVE)
    total_mass = masses_amu.sum()
    com_velocity = (masses_amu[:, None] * velocities).sum(dim=0) / total_mass
    velocities -= com_velocity


    positions = initial_positions.clone()
    trajectory = [positions.cpu().numpy()]

    for step in range(num_steps):
        # NOTE: Each call to velocity_verlet_step will use two different
        # Monte Carlo samples of the model's weights (at t and t+dt).
        # This introduces 'noise' but is computationally cheaper than averaging
        # over many samples for the force calculation.
        positions, velocities, forces = velocity_verlet_step(
            model=model,
            atoms=atoms,
            graph_idx=graph_idx,
            positions=positions,
            velocities=velocities,
            masses=masses_amu,
            dt=dt
        )

        trajectory.append(positions.cpu().numpy())

        if step % 20 == 0:
            # Kinetic energy calculation should use amu, Å/fs, resulting in energy
            # in (amu * Å^2) / fs^2. This is often an arbitrary unit in MD, but
            # is consistent for monitoring.
            kinetic_energy = 0.5 * (masses_amu[:, None] * velocities**2).sum()
            print(f"Step {step:4d} | KE={kinetic_energy:.6f}")

    return trajectory


# ========================================================
# MAIN FUNCTION
# ========================================================
def main():
    # -------------------------
    # Load trained model
    # -------------------------
    model = PaiNN(
        num_message_passing_layers=3,
        num_features=128,
        num_outputs=1,
        num_rbf_features=20,
        num_unique_atoms=100,
        cutoff_dist=5.0,
    )
    # Ensure you have a 'best_model.pth' trained with the Bayesian Readout
    # to avoid errors.
    try:
        model.load_state_dict(torch.load("benzene_bayesian.pth", map_location="cpu"))
    except FileNotFoundError:
        print("WARNING: best_model.pth not found. Using randomly initialized weights.")
        print("Please train your Bayesian PaiNN model first!")

    model.eval()

    # -------------------------
    # Load dataset and select one datapoint
    # NOTE: You will need to make sure your MD17Dataset is correctly imported
    # or available, as 'src.MD17Dataset' is a placeholder in your original code.
    # -------------------------
    # This requires an actual MD17 implementation or dummy data generation.
    # For now, I'll comment out the MD17 import and use dummy data for testing 
    # if you cannot provide the MD17 class. Assuming you have it for now:
    try:
        from src.MD17Dataset import MD17
        dataset = MD17(root="data", molecule_name="benzene2018_dft", cutoff=5.0)
        data = dataset[0]
        num_atoms = data.pos.shape[0]

        # For benzene: first 6 are C (Z=6), next 6 are H (Z=1)
        atoms = torch.tensor([6]*6 + [1]*6, dtype=torch.long)
        positions = data.pos.float()
        graph_idx = torch.zeros(num_atoms, dtype=torch.long)  # single molecule
        print(f"Loaded molecule with {num_atoms} atoms.")
        
    except ImportError:
        print("ERROR: MD17Dataset not available. Cannot load molecule data.")
        print("Creating dummy data for demonstration.")
        num_atoms = 3
        atoms = torch.tensor([6, 8, 1], dtype=torch.long) # C, O, H
        positions = torch.randn(num_atoms, 3, dtype=torch.float)
        graph_idx = torch.zeros(num_atoms, dtype=torch.long)

    # -------------------------
    # Run simulation
    # -------------------------
    # dt=1e-3 (1 fs) is a standard and safe timestep for MD.
    # Your original dt=1e-2 (10 fs) is likely too large for stability.
    trajectory = simulate(
        model=model,
        atoms=atoms,
        initial_positions=positions,
        graph_idx=graph_idx,
        num_steps=1000,
        dt=1*10**-3 # Corrected to a more stable timestep (1 fs)
    )

    print("Simulation finished. Trajectory length:", len(trajectory))

    # -------------------------
    # Save trajectory as numpy file
    # -------------------------
    import numpy as np
    np.save("trajectory.npy", np.array(trajectory))
    print("Trajectory saved to trajectory.npy")


if __name__ == "__main__":
    main()