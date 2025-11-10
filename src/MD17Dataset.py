import torch
import os
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

class MD17(Dataset):
    """
    Simple MD17 dataset loader for PaiNN or other atomistic GNNs.
    Loads .npz files containing R (positions), F (forces), E (energies), z (atomic numbers).
    """
    def __init__(self, root, molecule_name, cutoff=4.0, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.molecule_name = molecule_name
        self.cutoff = cutoff
        self.file_path = os.path.join(root, f"{molecule_name}.npz")

        # --- Load data from the npz file ---
        data_npz = np.load(self.file_path)
        self.R = data_npz["R"]     # [n_frames, n_atoms, 3]
        self.F = data_npz["F"]     # [n_frames, n_atoms, 3]
        self.E = data_npz["E"]     # [n_frames]
        self.z = data_npz["z"]     # [n_atoms]
        self.n_atoms = len(self.z)

    def len(self):
        """Number of MD trajectory frames"""
        return self.R.shape[0]
    
    def get(self, idx):
        """Return one molecular frame as a PyG Data object"""
        pos = self.R[idx]
        energy = self.E[idx]
        forces = self.F[idx]

        # --- Construct edges based on cutoff ---
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        mask = (dist < self.cutoff) & (dist > 1e-6)
        edge_index = np.array(np.nonzero(mask))
        edge_attr = dist[edge_index[0], edge_index[1]][:, None]

        # --- Node features: atomic numbers (long type for embedding) ---
        x = torch.tensor(self.z, dtype=torch.long)

        data = Data(
            x=x,
            pos=torch.tensor(pos, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor([energy], dtype=torch.float),
            force=torch.tensor(forces, dtype=torch.float)
        )
        return data



if __name__ == "__main__":
    dataset = MD17(root="data/md17", molecule_name="azobenzene_dft")

    # Split into train / val sets
    num_train = int(0.9 * len(dataset))
    train_dataset = dataset[:num_train]
    val_dataset   = dataset[num_train:]

    # Create PyTorch Geometric DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1)

    # Inspect one batch
    for batch in train_loader:
        print(batch)
        print("Atoms:", batch.x.shape[0],
              "| Edges:", batch.edge_index.shape[1],
              "| Energy:", batch.y.item(),
              "| Forces:", batch.force.shape)
        break
