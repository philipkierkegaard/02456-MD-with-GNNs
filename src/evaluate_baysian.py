import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_sum
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Import your specific model and dataset
from painn_baysian import PaiNN
from MD17Dataset import MD17

# ============================================================
# Configuration
# ============================================================

MOLECULE = "benzene2018_dft"
DATA_ROOT = "data"
CHECKPOINT_PATH = "best_model.pth" # or "final_model.pth"
CUTOFF = 5.0
BATCH_SIZE = 1  # Keep 1 for MD17 usually
NUM_MC_SAMPLES = 20  # How many stochastic passes to run per input
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_energy_and_forces(model, batch):
    """
    Runs a single forward pass.
    Note: Since the model has Bayesian layers, calling this multiple times
    will result in different outputs due to weight sampling.
    """
    batch = batch.to(DEVICE)
    pos = batch.pos.clone().detach().requires_grad_(True)

    # Forward pass
    atomic_E = model(batch.x.long(), pos, batch.batch)
    
    # Molecular energies
    total_E = scatter_sum(atomic_E, batch.batch, dim=0)

    # Forces = -grad(E)
    # We need create_graph=True if we were training, but for eval 
    # we just need the gradients to exist.
    total_F = -torch.autograd.grad(total_E.sum(), pos, create_graph=False)[0]

    return total_E, total_F

def evaluate_uncertainty():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Dataset
    # We use the same split logic as train_baysian.py to ensure we test on unseen data
    dataset = MD17(root=DATA_ROOT, molecule_name=MOLECULE, cutoff=CUTOFF)
    num_train = int(0.9 * len(dataset))
    val_dataset = dataset[num_train:]
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Loaded validation set: {len(val_dataset)} samples")

    # 2. Load Model
    model = PaiNN(
        num_message_passing_layers=3,
        num_features=128,
        num_outputs=1,
        num_rbf_features=20,
        cutoff_dist=CUTOFF,
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Loaded weights from {CHECKPOINT_PATH}")
    else:
        print(f"Error: Checkpoint {CHECKPOINT_PATH} not found!")
        return

    # Ensure model is in a mode where it can sample
    # (Your BayesianLinear implementation samples regardless of .eval() or .train(), 
    # but strictly speaking, BNNs often use .train() logic for dropout. 
    # Your code samples in forward() explicitly, so .eval() is fine/safer for BatchNorm if you had it.)
    model.eval()

    # 3. MC Sampling Loop
    print(f"Starting MC Inference with {NUM_MC_SAMPLES} samples per molecule...")
    
    results = {
        "E_true": [], "E_pred_mean": [], "E_pred_std": [],
        "F_true": [], "F_pred_mean": [], "F_pred_std": []
    }

    for batch in tqdm(val_loader):
        batch = batch.to(DEVICE)
        
        # Lists to store the N stochastic forward passes
        mc_energies = []
        mc_forces = []

        for _ in range(NUM_MC_SAMPLES):
            # Enable grad is required for force calculation even in eval mode
            with torch.enable_grad():
                E, F = compute_energy_and_forces(model, batch)
            
            mc_energies.append(E.detach().cpu())
            mc_forces.append(F.detach().cpu())

        # Stack results: Shape (NUM_SAMPLES, Batch_Size, ...)
        E_stack = torch.stack(mc_energies) 
        F_stack = torch.stack(mc_forces)

        # Compute Mean and Standard Deviation (Uncertainty)
        # Energy
        E_mean = E_stack.mean(dim=0)
        E_std  = E_stack.std(dim=0)
        
        # Forces
        F_mean = F_stack.mean(dim=0)
        F_std  = F_stack.std(dim=0)

        # Store results
        results["E_true"].append(batch.y.cpu())
        results["E_pred_mean"].append(E_mean)
        results["E_pred_std"].append(E_std)
        
        results["F_true"].append(batch.force.cpu())
        results["F_pred_mean"].append(F_mean)
        results["F_pred_std"].append(F_std)

    # Concatenate all batches
    E_true = torch.cat(results["E_true"]).numpy().flatten()
    E_pred = torch.cat(results["E_pred_mean"]).numpy().flatten()
    E_std  = torch.cat(results["E_pred_std"]).numpy().flatten()
    
    F_true = torch.cat(results["F_true"]).numpy().flatten()
    F_pred = torch.cat(results["F_pred_mean"]).numpy().flatten()
    F_std  = torch.cat(results["F_pred_std"]).numpy().flatten()

    # 4. Metrics & Analysis
    E_mae = np.mean(np.abs(E_pred - E_true))
    F_mae = np.mean(np.abs(F_pred - F_true))
    
    print("\n=== Results ===")
    print(f"Energy MAE: {E_mae:.5f}")
    print(f"Force MAE:  {F_mae:.5f}")
    print(f"Avg Energy Uncertainty (Std): {np.mean(E_std):.5f}")
    print(f"Avg Force Uncertainty (Std):  {np.mean(F_std):.5f}")

    # 5. Visualization
    plot_results(E_true, E_pred, E_std, F_true, F_pred, F_std)


def plot_results(E_true, E_pred, E_std, F_true, F_pred, F_std):
    """
    Creates a 2x2 plot:
    Row 1: Parity Plots (True vs Pred)
    Row 2: Uncertainty Analysis (Absolute Error vs Uncertainty)
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # --- Energy Parity ---
    axs[0, 0].scatter(E_true, E_pred, alpha=0.5, s=10)
    # Plot ideal line
    lims = [min(E_true.min(), E_pred.min()), max(E_true.max(), E_pred.max())]
    axs[0, 0].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    axs[0, 0].set_title("Energy: True vs Predicted")
    axs[0, 0].set_xlabel("True Energy")
    axs[0, 0].set_ylabel("Predicted Energy (Mean)")

    # --- Force Parity ---
    # Downsample forces for plotting if too many atoms
    indices = np.random.choice(len(F_true), size=min(2000, len(F_true)), replace=False)
    axs[0, 1].scatter(F_true[indices], F_pred[indices], alpha=0.5, s=10)
    lims_f = [min(F_true.min(), F_pred.min()), max(F_true.max(), F_pred.max())]
    axs[0, 1].plot(lims_f, lims_f, 'k--', alpha=0.75, zorder=0)
    axs[0, 1].set_title("Forces: True vs Predicted (Subsampled)")
    axs[0, 1].set_xlabel("True Force")
    axs[0, 1].set_ylabel("Predicted Force (Mean)")

    # --- Energy Error vs Uncertainty ---
    # We expect higher error to correlate with higher uncertainty
    E_error = np.abs(E_pred - E_true)
    axs[1, 0].scatter(E_std, E_error, alpha=0.6, c='purple', s=15)
    axs[1, 0].set_title("Energy: Uncertainty vs Absolute Error")
    axs[1, 0].set_xlabel("Predicted Uncertainty (Std Dev)")
    axs[1, 0].set_ylabel("Absolute Error")
    
    # Add correlation coefficient
    if len(E_std) > 1:
        corr_e = np.corrcoef(E_std, E_error)[0, 1]
        axs[1, 0].annotate(f"Corr: {corr_e:.2f}", xy=(0.05, 0.9), xycoords='axes fraction')

    # --- Force Error vs Uncertainty ---
    F_error = np.abs(F_pred - F_true)
    # Downsample for clarity
    axs[1, 1].scatter(F_std[indices], F_error[indices], alpha=0.6, c='purple', s=15)
    axs[1, 1].set_title("Forces: Uncertainty vs Absolute Error")
    axs[1, 1].set_xlabel("Predicted Uncertainty (Std Dev)")
    axs[1, 1].set_ylabel("Absolute Error")

    if len(F_std) > 1:
        corr_f = np.corrcoef(F_std, F_error)[0, 1]
        axs[1, 1].annotate(f"Corr: {corr_f:.2f}", xy=(0.05, 0.9), xycoords='axes fraction')

    plt.tight_layout()
    plt.savefig("uncertainty_analysis.png", dpi=150)
    print("Graph saved to uncertainty_analysis.png")
    # plt.show() # Uncomment if running locally with display

if __name__ == "__main__":
    evaluate_uncertainty()