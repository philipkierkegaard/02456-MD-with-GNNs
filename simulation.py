import os
import numpy as np
import torch
from src.painn import PaiNN
from src.MD17Dataset import MD17

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
#  Energy + Force computation for YOUR PaiNN model
# =====================================================

def compute_energy_and_forces(model, z, R):
    """
    Computes energy and forces for your PaiNN model.
    R may be float64 for integration, but the model MUST get float32.
    """
    # R_model is float32 for PaiNN
    R_model = R.detach().clone().float().requires_grad_(True)

    z = z.to(device)

    # Input graph_indexes = zeros for single molecule
    graph_indexes = torch.zeros(len(z), dtype=torch.long, device=device)

    atomic_contrib = model(
        atoms=z,
        atom_positions=R_model,
        graph_indexes=graph_indexes
    )   # (n_atoms, 1)

    energy = atomic_contrib.sum()

    # Force = -dE/dR_model
    forces_32 = -torch.autograd.grad(
        energy,
        R_model,
        grad_outputs=torch.ones_like(energy),
        create_graph=False
    )[0]

    # Convert model output forces back to float64 for integration
    forces_64 = forces_32.double()

    return energy.detach(), forces_64



# =====================================================
#   Maxwell–Boltzmann initial velocities (simple)
# =====================================================

MASS_TABLE = {1: 1.008, 6: 12.011}  # H, C for benzene


def init_velocities(z, T=300.0):
    """
    Returns velocities with shape (N, 3).
    Units are arbitrary but consistent.
    """
    masses = torch.tensor([MASS_TABLE[int(a)] for a in z], device=device).view(-1, 1)
    std = torch.sqrt(T / masses)
    v = torch.randn((len(z), 3), device=device) * std

    v_cm = (v * masses).sum(dim=0) / masses.sum()
    v -= v_cm
    return v


# =====================================================
#   Velocity Verlet integration
# =====================================================

def velocity_verlet(R, v, m, model, z, dt):
    """
    Performs one velocity Verlet step.
    """
    # Compute forces at current position
    _, F = compute_energy_and_forces(model, z, R)

    # Half-step velocity
    v_half = v + 0.5 * dt * (F / m)

    # Position update
    R_new = R + dt * v_half

    # Compute forces at new position
    _, F_new = compute_energy_and_forces(model, z, R_new)

    # Full velocity step
    v_new = v_half + 0.5 * dt * (F_new / m)

    return R_new, v_new, F_new


# =====================================================
#                MAIN SIMULATION
# =====================================================

def run_simulation(
    dataset_path="data/md17",
    molecule="benzene2018_dft",
    weight_path="best_model_benzene.pth",
    n_steps=1000,
    dt=0.0001,
    temperature=200.0,
    save_path="trajectory_benzene_200k.npz"
):
    print("Loading dataset...")
    dataset = MD17(dataset_path, molecule)
    data = dataset[0]

    R = data.pos.to(device).double()
    z = data.x.to(device)
    n_atoms = len(z)

    print("Loading model...")
    model = PaiNN(num_message_passing_layers=3).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    print("Initializing velocities...")
    v = init_velocities(z, T=temperature).double()

    masses = torch.tensor([MASS_TABLE[int(a)] for a in z], device=device).view(-1, 1).double()

    # Storage
    traj_R = torch.zeros((n_steps, n_atoms, 3), dtype=torch.float64)
    traj_v = torch.zeros((n_steps, n_atoms, 3), dtype=torch.float64)

    traj_R[0] = R.cpu()
    traj_v[0] = v.cpu()

    print("Starting simulation...")
    for step in range(1, n_steps):
        R, v, _ = velocity_verlet(R, v, masses, model, z, dt)

        traj_R[step] = R.cpu()
        traj_v[step] = v.cpu()

        if step % 100 == 0:
            print(f"Step {step}/{n_steps}")

    print(f"Saving trajectory to {save_path}")
    np.savez(
        save_path,
        R=traj_R.numpy(),
        v=traj_v.numpy(),
        z=z.cpu().numpy(),
        dt=dt,
        T=temperature
    )


if __name__ == "__main__":
    run_simulation()
