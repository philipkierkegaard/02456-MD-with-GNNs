import math
import torch
import torch.nn as nn
from typing import Union, Tuple
from torch_geometric.nn import radius_graph


# ============================================================
#  BAYESIAN LAST LAYER (VBLL)
# ============================================================

class BayesianLinear(nn.Module):
    """
    Variational Bayesian linear layer for approximate Bayesian inference.
    Uses the reparameterization trick on weights and biases.
    """
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters (mu, rho -> sigma)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        # Prior
        self.prior_std = prior_std
        self.log_prior_std = math.log(prior_std)

        self.kl = 0.0  # store KL for training

    def sample_weights(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        eps_w = torch.randn_like(weight_sigma)
        eps_b = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * eps_w
        bias = self.bias_mu + bias_sigma * eps_b
        return weight, bias

    def kl_divergence(self, w, b):
        """
        KL divergence between q(w|θ) and N(0, prior_std^2).
        """
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        kld_weight = (
            torch.log(self.prior_std / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2 * self.prior_std**2)
            - 0.5
        ).sum()

        kld_bias = (
            torch.log(self.prior_std / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2 * self.prior_std**2)
            - 0.5
        ).sum()

        return kld_weight + kld_bias

    def forward(self, x):
        weight, bias = self.sample_weights()
        self.kl = self.kl_divergence(weight, bias)
        return x @ weight.t() + bias


# ============================================================
#  READOUT NETWORK WITH BAYESIAN LAST LAYER
# ============================================================

class BayesianReadoutNetwork(nn.Module):
    """
    Same architecture as build_readout_network, but with a Bayesian last layer.
    """
    def __init__(
        self,
        num_in_features: int,
        num_out_features: int = 1,
        num_layers: int = 2,
        activation: nn.Module = nn.SiLU
    ):
        super().__init__()

        hidden_dims = [
            num_in_features,
            *[
                max(num_out_features, num_in_features // 2**(i + 1))
                for i in range(num_layers - 1)
            ]
        ]

        layers = []
        for i, (n_in, n_out) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(activation())

        self.det_layers = nn.Sequential(*layers)

        # Variational last layer
        self.bayesian_out = BayesianLinear(hidden_dims[-1], num_out_features)

    def forward(self, x):
        x = self.det_layers(x)
        return self.bayesian_out(x)

    def kl_loss(self):
        """Return KL divergence from the final Bayesian layer."""
        return self.bayesian_out.kl


# ============================================================
#  ALL YOUR ORIGINAL PAI NN CODE (UNCHANGED)
# ============================================================

class SinusoidalRBFLayer(nn.Module):
    def __init__(self, num_basis: int = 20, cutoff_dist: float = 5.0) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.cutoff_dist = cutoff_dist     
        self.register_buffer(
            'freqs',
            math.pi * torch.arange(1, self.num_basis + 1) / self.cutoff_dist
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = distances.unsqueeze(-1)
        return torch.sin(self.freqs * distances) / distances


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_dist: float = 5.0) -> None:
        super().__init__()
        self.cutoff_dist = cutoff_dist

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.where(
            distances < self.cutoff_dist,
            0.5 * (torch.cos(distances * math.pi / self.cutoff_dist) + 1),
            0
        )


class PaiNNMessageBlock(nn.Module):
    def __init__(self, num_features: int = 128, num_rbf_features: int = 20) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features

        self.scalar_network = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, 3*num_features)
        )
        self.rbf_network = nn.Linear(num_rbf_features, 3*num_features)

    def forward(
        self, idx_i, idx_j, rel_dir, rel_dist_cut, rbf_features,
        scalar_features, vector_features
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        phi = self.scalar_network(scalar_features)
        W = self.rbf_network(rbf_features) * rel_dist_cut.unsqueeze(-1)
        phi_W = phi[idx_j] * W
        phi_W_vv, phi_W_ss, phi_W_vs = torch.split(
            phi_W, self.num_features, dim=-1
        )

        # scalar residuals
        scalar_residuals = torch.zeros_like(scalar_features)
        scalar_residuals.index_add_(0, idx_i, phi_W_ss)

        # vector residuals
        vector_residuals = torch.zeros_like(vector_features)
        vector_residuals_per_edge = (
            vector_features[idx_j] * phi_W_vv.unsqueeze(-1)
            + phi_W_vs.unsqueeze(-1) * rel_dir.unsqueeze(-2)
        )
        vector_residuals.index_add_(0, idx_i, vector_residuals_per_edge)

        return scalar_features + scalar_residuals, vector_features + vector_residuals


class PaiNNUpdateBlock(nn.Module):
    def __init__(self, num_features: int = 128) -> None:
        super().__init__()
        self.num_features = num_features

        self.U = nn.Linear(num_features, num_features, bias=False)
        self.V = nn.Linear(num_features, num_features, bias=False)
        self.scalar_vector_network = nn.Sequential(
            nn.Linear(2*num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, 3*num_features)
        )

    def forward(self, scalar_features, vector_features):

        U_vec = self.U(vector_features.movedim(-2, -1)).movedim(-2, -1)
        V_vec = self.V(vector_features.movedim(-2, -1)).movedim(-2, -1)

        a = self.scalar_vector_network(
            torch.cat([torch.linalg.vector_norm(V_vec, dim=-1), scalar_features], dim=-1)
        )
        a_vv, a_sv, a_ss = torch.split(a, self.num_features, dim=-1)

        vector_residuals = U_vec * a_vv.unsqueeze(-1)
        scalar_residuals = a_ss + a_sv * torch.sum(U_vec * V_vec, dim=-1)

        return scalar_features + scalar_residuals, vector_features + vector_residuals


# ============================================================
#  PAI NN WITH BAYESIAN LAST LAYER
# ============================================================

class PaiNN(nn.Module):
    """
    Same PaiNN architecture, but the readout is now Bayesian.
    """
    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist

        self.atom_embedding = nn.Embedding(
            num_embeddings=num_unique_atoms + 1,
            embedding_dim=num_features,
            padding_idx=0
        )
        self.cosine_cut = CosineCutoff(cutoff_dist)
        self.radial_basis = SinusoidalRBFLayer(num_rbf_features, cutoff_dist)

        self.message_blocks = nn.ModuleList([
            PaiNNMessageBlock(num_features, num_rbf_features)
            for _ in range(num_message_passing_layers)
        ])
        self.update_blocks = nn.ModuleList([
            PaiNNUpdateBlock(num_features)
            for _ in range(num_message_passing_layers)
        ])

        # ---- changed to Bayesian readout ----
        self.readout_network = BayesianReadoutNetwork(
            num_in_features=num_features,
            num_out_features=num_outputs,
            num_layers=2,
            activation=nn.SiLU
        )

    def kl_loss(self):
        """KL divergence from the Bayesian last layer."""
        return self.readout_network.kl_loss()

    def forward(self, atoms, atom_positions, graph_indexes):
        scalar_features = self.atom_embedding(atoms)
        vector_features = torch.zeros(
            scalar_features.size() + (3,),
            dtype=scalar_features.dtype,
            device=scalar_features.device,
        )

        _, num_nodes_per_graph = torch.unique(graph_indexes, return_counts=True)

        idx_i, idx_j = radius_graph(
            x=atom_positions,
            r=self.cutoff_dist,
            batch=graph_indexes,
            loop=False,
            max_num_neighbors=torch.max(num_nodes_per_graph),
            flow='target_to_source',
            batch_size=len(num_nodes_per_graph),
        )

        rel_pos = atom_positions[idx_j] - atom_positions[idx_i]
        rel_dist = torch.linalg.vector_norm(rel_pos, dim=1)
        rel_dir = rel_pos / rel_dist.unsqueeze(-1)
        rel_dist_cut = self.cosine_cut(rel_dist)
        rbf_features = self.radial_basis(rel_dist)

        for message, update in zip(self.message_blocks, self.update_blocks):
            scalar_features, vector_features = message(
                idx_i, idx_j, rel_dir, rel_dist_cut,
                rbf_features, scalar_features, vector_features
            )
            scalar_features, vector_features = update(scalar_features, vector_features)

        atomic_contributions = self.readout_network(scalar_features)
        return atomic_contributions
