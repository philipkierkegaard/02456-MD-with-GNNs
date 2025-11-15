# train_painn_md17.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_sum


from painn import PaiNN                     # your PAINN model file
from MD17Dataset import MD17        # the class you just made

# -------------------------------
# 1️⃣  Hyperparameters
# -------------------------------
MOLECULE = "benzene2018_dft"
DATA_ROOT = "data"
CUTOFF = 5.0
LR = 1e-3
EPOCHS = 200
RHO = 0.95             # weight for force loss
BATCH_SIZE = 1         # variable-size molecules ⇒ batch = 1 is common


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 2️⃣  Dataset and DataLoaders
# -------------------------------
dataset = MD17(root=DATA_ROOT, molecule_name=MOLECULE, cutoff=CUTOFF)
num_train = int(0.9 * len(dataset))
train_dataset = dataset[:num_train]
val_dataset   = dataset[num_train:]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# -------------------------------
# 3️⃣  Model and optimizer
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
# 4️⃣  Helper: compute energy & forces
# -------------------------------
def compute_energy_and_forces(model, batch):
    batch = batch.to(device)
    batch.pos.requires_grad_(True)

    # Predict per-atom energies
    atomic_E = model(batch.x.long(), batch.pos, batch.batch)
    total_E = scatter_sum(atomic_E, batch.batch, dim=0)  # per molecule

    # Forces: negative gradient of energy wrt positions
    total_F = -torch.autograd.grad(
        outputs=total_E.sum(),
        inputs=batch.pos,
        create_graph=True
    )[0]

    return total_E, total_F

# -------------------------------
# 5️⃣  Loss function
# -------------------------------
def energy_force_loss(E_pred, E_true, F_pred, F_true, rho=RHO):
    e_loss = F.mse_loss(E_pred, E_true)
    f_loss = F.mse_loss(F_pred, F_true)
    return rho * f_loss + (1 - rho) * e_loss

# -------------------------------
# 6️⃣  Training loop
# -------------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        E_pred, F_pred = compute_energy_and_forces(model, batch)
        loss = energy_force_loss(E_pred, batch.y, F_pred, batch.force)

        if i % 100 == 0:
            print(f"batch {i}, loss: {loss}")

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    

    # -------------------------------
    # Validation
    # -------------------------------
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        mae_E, mae_F = 0.0, 0.0
        for batch in val_loader:
            E_pred, F_pred = compute_energy_and_forces(model, batch)
            loss = energy_force_loss(E_pred, batch.y, F_pred, batch.force)
            val_loss += loss.item()
            mae_E += torch.mean(torch.abs(E_pred - batch.y)).item()
            mae_F += torch.mean(torch.abs(F_pred - batch.force)).item()
        val_loss /= len(val_loader)
        mae_E /= len(val_loader)
        mae_F /= len(val_loader)

    print(
        f"Epoch {epoch:03d} | "
        f"TrainLoss {avg_train_loss:.6f} | "
        f"ValLoss {val_loss:.6f} | "
        f"MAE_E {mae_E:.6f} | MAE_F {mae_F:.6f}"
    )

print("✅ Training finished!")
torch.save(model.state_dict(), f"painn_{MOLECULE}.pth")
print(f"Model saved to painn_{MOLECULE}.pth")
