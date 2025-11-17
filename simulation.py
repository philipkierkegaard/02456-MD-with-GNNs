import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from copy import deepcopy

# --------------------------------------------------------
# IMPORT YOUR PaiNN MODEL HERE
# --------------------------------------------------------
from src.painn import PaiNN       # <-- replace with your actual model file
from src.MD17Dataset import MD17        # <-- replace with your MD17 dataset file


# ========================================================
# UTILITY: FORCE COMPUTATION FROM MODEL
# ========================================================
def compute_forces(model, atoms, positions, graph_idx):
    """
    Computes forces as the negative gradient of predicted energy.
    """
    positions = positions.clone().detach().requires_grad_(True)

    # Model outputs atomic energy contributions → sum them
    atomic_energy = model(atoms, positions, graph_idx)  # [N, 1]
    total_energy = atomic_energy.sum()

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

    # --- Forces at t + dt ---
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
    # Convert to Å/fs for typical MD timestep (~1 fs)
    # 1 m/s = 1e10 Å/s = 1e-5 Å/fs
    std_ang_fs = std * 1e-5

    # Initialize velocities from Maxwell-Boltzmann
    velocities = torch.normal(mean=0.0, std=std_ang_fs[:, None])

    positions = initial_positions.clone()
    trajectory = [positions.cpu().numpy()]

    for step in range(num_steps):
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
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()

    # -------------------------
    # Load dataset and select one datapoint
    # -------------------------
    dataset = MD17(root="data", molecule_name="benzene2018_dft", cutoff=5.0)
    data = dataset[0]

    num_atoms = data.pos.shape[0]

    # For benzene: first 6 are C, next 6 are H
    atoms = torch.tensor([6]*6 + [1]*6, dtype=torch.long)
    positions = data.pos.float()
    graph_idx = torch.zeros(num_atoms, dtype=torch.long)  # single molecule

    print(f"Loaded molecule with {num_atoms} atoms.")

    # -------------------------
    # Run simulation
    # -------------------------
    trajectory = simulate(
        model=model,
        atoms=atoms,
        initial_positions=positions,
        graph_idx=graph_idx,
        num_steps=300,
        dt=1*10**-2
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
