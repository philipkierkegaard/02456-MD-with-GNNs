import torch
import os
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import radius_graph


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

        # Load npz file
        data_npz = np.load(self.file_path)
        self.R = torch.tensor(data_npz["R"], dtype=torch.float)    # [frames, atoms, 3]
        self.F = torch.tensor(data_npz["F"], dtype=torch.float)    # [frames, atoms, 3]
        self.E = torch.tensor(data_npz["E"], dtype=torch.float)    # [frames]
        self.z = torch.tensor(data_npz["z"], dtype=torch.long)     # [atoms]

        self.n_atoms = self.z.shape[0]

    def len(self):
        """Number of MD trajectory frames"""
        return self.R.shape[0]
    
    def get(self, idx):
        """Return one molecular frame as a PyG Data object"""

        pos = self.R[idx]              # [n_atoms, 3]
        energy = self.E[idx].unsqueeze(0)
        forces = self.F[idx]           # [n_atoms, 3]
        z = self.z                     # [n_atoms]

        # --- Compute edges via efficient radius_graph ---
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            loop=False
        )   # shape [2, num_edges]

        row, col = edge_index
        edge_attr = torch.norm(pos[row] - pos[col], dim=-1).unsqueeze(-1)

        return Data(
            x=z,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=energy,        # shape [1]
            force=forces     # shape [n_atoms, 3]
        )


# Optional test
if __name__ == "__main__":
    dataset = MD17(root="data", molecule_name="benzene2018_dft", cutoff=5.0)
    print("Dataset size:", len(dataset))

    data = dataset[0]
    print(data)
    print("Atoms:", data.x.shape[0],
          "| Edges:", data.edge_index.shape[1],
          "| Energy:", data.y.item(),
          "| Forces:", data.force.shape)
