import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_sum

import wandb   # ⭐ NEW

from painn import PaiNN
from MD17Dataset import MD17


import sys, torch, torch_cluster
print("Python:", sys.executable)
print("Torch CUDA:", torch.version.cuda)
print("torch-cluster:", torch_cluster.__file__)
print("torch-cluster version:", torch_cluster.__version__)



# -------------------------------
# 1️⃣  Hyperparameters
# -------------------------------

MOLECULE = "benzene2018_dft"
DATA_ROOT = "data"
CUTOFF = 5.0
LR = 3e-4
EPOCHS = 50
RHO = 0.98
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# ⭐ 2️⃣ Initialize W&B
# -------------------------------
wandb.init(
    entity = "lyngsberg-danmarks-tekniske-universitet-dtu",
    project="DeepLearningProject",
    name=f"painn-{MOLECULE}",
    config={
        "molecule": MOLECULE,
        "cutoff": CUTOFF,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "rho": RHO,
    }
)

# -------------------------------
# ataset & loaders
# -------------------------------
dataset = MD17(root=DATA_ROOT, molecule_name=MOLECULE, cutoff=CUTOFF)
num_train = int(0.9 * len(dataset))
train_dataset = dataset[:num_train]
val_dataset   = dataset[num_train:]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# -------------------------------
# Model & optimizer
# -------------------------------
model = PaiNN(
    num_message_passing_layers=3,
    num_features=128,
    num_outputs=1,
    num_rbf_features=20,
    cutoff_dist=CUTOFF,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# -------------------------------
# Helpers
# -------------------------------
def compute_energy_and_forces(model, batch):
    batch = batch.to(device)

    pos = batch.pos.clone().detach().requires_grad_(True)

    atomic_E = model(batch.x.long(), pos, batch.batch)
    total_E = scatter_sum(atomic_E, batch.batch, dim=0)

    total_F = -torch.autograd.grad(
        outputs=total_E.sum(),
        inputs=pos,
        create_graph=True
    )[0]

    return total_E, total_F



def energy_force_loss(E_pred, E_true, F_pred, F_true, rho=RHO):
    e_loss = F.mse_loss(E_pred, E_true)
    f_loss = F.mse_loss(F_pred, F_true)
    return rho * f_loss + (1 - rho) * e_loss


# -------------------------------
# 3️⃣ Training loop
# -------------------------------
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        E_pred, F_pred = compute_energy_and_forces(model, batch)
        loss = energy_force_loss(E_pred, batch.y, F_pred, batch.force)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # ⭐ Log batch loss every 100 steps
        if i % 100 == 0:
            wandb.log({"train/batch_loss": loss.item()})

    avg_train_loss = total_loss / len(train_loader)

    # -------------------------------
    # Validation
    # -------------------------------
    model.eval()
    val_loss = 0.0
    mae_E, mae_F = 0.0, 0.0

    for batch in val_loader:
        # we MUST allow gradients here because we use autograd.grad
        with torch.enable_grad():
            E_pred, F_pred = compute_energy_and_forces(model, batch)
            loss = energy_force_loss(E_pred, batch.y, F_pred, batch.force)

        val_loss += loss.item()
        mae_E += torch.mean(torch.abs(E_pred - batch.y)).item()
        mae_F += torch.mean(torch.abs(F_pred - batch.force)).item()

    val_loss /= len(val_loader)
    mae_E   /= len(val_loader)
    mae_F   /= len(val_loader)

    print(
        f"Epoch {epoch:03d} | "
        f"TrainLoss {avg_train_loss:.6f} | "
        f"ValLoss {val_loss:.6f} | "
        f"MAE_E {mae_E:.6f} | MAE_F {mae_F:.6f}"
    )

    # ⭐ Log all metrics to W&B
    wandb.log({
        "epoch": epoch,
        "train/epoch_loss": avg_train_loss,
        "val/loss": val_loss,
        "val/mae_E": mae_E,
        "val/mae_F": mae_F,
        "lr": optimizer.param_groups[0]["lr"],
    })

    # ⭐ Save best model to W&B
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")
        print("💾 Saved new BEST model!")

# -------------------------------
# Final save
# -------------------------------
torch.save(model.state_dict(), "final_model.pth")
wandb.save("final_model.pth")

print("🎉 Training complete!")
wandb.finish()
