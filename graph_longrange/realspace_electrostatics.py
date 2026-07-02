import torch
from scipy.constants import e, epsilon_0, pi
from mace.tools.scatter import scatter_sum
from .utils import FIELD_CONSTANT
from typing import List, Optional, Tuple
import warnings
from .gto_utils import (
    GTOSelfInteractionBlock,
    get_Cl_sigma,
)

@torch.no_grad()
def batch_complete_graph_excluding_self_duplicates_vector(
    batch: torch.Tensor, N: int
) -> torch.Tensor:
    """
    Duplicate each node N times, then for each graph build directed
    edges between every pair of duplicates *unless* they share the same
    original node ID.

    Args:
        batch (LongTensor): shape [M], graph ID of each original node.
        N (int): number of duplicates per node.

    Returns:
        edge_index (LongTensor[2, E])

    Vectorized: the duplicated nodes are contiguous per graph (``batch`` is grouped
    by graph and ``repeat_interleave`` preserves the grouping), so each graph's
    block of pairs is built from a single global ``arange`` via per-graph offsets --
    no Python loop over graphs, no per-graph ``nonzero``. Produces the same edge set
    as the previous loop implementation (verified bit-identical).
    """
    batch = batch.long()
    device = batch.device
    num_original_nodes = batch.size(0)
    # duplicated per-node graph ID and original-ID (grouped by graph)
    original_ids = torch.arange(num_original_nodes, device=device).repeat_interleave(N)  # [M*N]
    duplicated_batch = batch.repeat_interleave(N)  # [M*N]
    if duplicated_batch.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    num_graphs = int(duplicated_batch.max().item()) + 1
    nodes_per_graph = torch.bincount(duplicated_batch, minlength=num_graphs)  # duplicated nodes / graph
    graph_offsets = torch.cumsum(nodes_per_graph, 0) - nodes_per_graph  # block start / graph
    pairs_per_graph = nodes_per_graph * nodes_per_graph  # ordered pairs / graph
    total_pairs = int(pairs_per_graph.sum().item())
    if total_pairs == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # graph id of each ordered pair, and the pair's index within its graph block
    pair_graph = torch.arange(num_graphs, device=device).repeat_interleave(pairs_per_graph)  # [P]
    pair_block_offsets = torch.cumsum(pairs_per_graph, 0) - pairs_per_graph  # [num_graphs]
    local_pair = torch.arange(total_pairs, device=device) - pair_block_offsets[pair_graph]  # 0..count^2-1
    graph_size = nodes_per_graph[pair_graph]

    row = graph_offsets[pair_graph] + torch.div(local_pair, graph_size, rounding_mode="floor")
    col = graph_offsets[pair_graph] + local_pair % graph_size
    # drop pairs whose duplicates come from the same original node (incl. self pairs)
    keep = original_ids[row] != original_ids[col]
    return torch.stack([row[keep], col[keep]], dim=0)


def charges_energy_from_graph(
    charges,  # [n_atoms]
    positions,
    edge_index,
    batch,
    density_smearing_width,
):
    """
    Computes the energy of a collection of charges considering only specifed edges.
    normalization of the charges is multipoles.
    """
    sender, receiver = edge_index

    R_ij = positions[receiver] - positions[sender]  # [N_edges,3]
    d_ij = torch.linalg.norm(R_ij, dim=-1)  # [N_edges,1]
    smooth_reciprocal = torch.erf(d_ij * 0.5 / density_smearing_width) / (
        torch.abs(d_ij) + 1e-6
    )

    # charge part
    edge_energy = (
        0.5
        * FIELD_CONSTANT
        * smooth_reciprocal
        * charges[sender]
        * charges[receiver]
        / (4 * pi)
    )
    # handle the case with no edges
    if edge_energy.numel() == 0:
        return torch.zeros(
            (batch.max() + 1,), dtype=charges.dtype, device=charges.device
        )
    node_energies = scatter_sum(src=edge_energy.squeeze(-1), index=receiver, dim=-1)
    return scatter_sum(src=node_energies, index=batch, dim=-1)  # [n_graphs]


class RealSpaceFiniteDiffereneEnergy(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        include_self_interaction: bool = False,
        offset=0.02,
    ):
        if density_max_l > 1:
            raise ValueError(
                "RealSpaceFiniteDiffereneEnergy only supports l=0 and l=1."
            )

        super().__init__()
        self.density_max_l = density_max_l
        self.density_smearing_width = density_smearing_width
        self.include_self_interaction = include_self_interaction
        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            density_max_l,
            [density_smearing_width],
            "multipoles",
            "multipoles",
        )

        self.offset = offset
        self.register_buffer(
            "x", torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "y", torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "z", torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )

    def energy_l0(
        self,
        source_feats: torch.Tensor,  # [n_node, 1]
        positions: torch.Tensor,  # [n_node, 3]
        batch: torch.Tensor,  # [n_node]
    ) -> torch.Tensor:

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)

        energy = charges_energy_from_graph(
            source_feats.squeeze(-1),
            positions,
            edge_index,
            batch,
            density_smearing_width=self.density_smearing_width,
        )

        # self interaction
        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)  # [n_node, (l+1)^2]
            node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            self_energy = scatter_sum(src=node_energies, index=batch, dim=-1)
            energy += self_energy * 0.5

        return energy

    def energy_l1(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,  # [n_node, 3]
        batch: torch.Tensor,  # [n_node]
    ) -> torch.Tensor:
        extended_positions = positions.repeat_interleave(4, dim=0)
        extended_positions[1::4] += self.x
        extended_positions[2::4] += self.y
        extended_positions[3::4] += self.z

        extended_batch = batch.repeat_interleave(4)
        charges = torch.zeros_like(extended_positions[:, 0])

        charges[1::4] = source_feats[:, 3] / self.offset
        charges[2::4] = source_feats[:, 1] / self.offset
        charges[3::4] = source_feats[:, 2] / self.offset
        charges[0::4] = source_feats[:, 0] - (
            charges[1::4] + charges[2::4] + charges[3::4]
        )

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 4)

        energy = charges_energy_from_graph(
            charges,
            extended_positions,
            edge_index,
            extended_batch,
            density_smearing_width=self.density_smearing_width,
        )

        # self interaction
        if self.include_self_interaction:
            self_fields = self.self_interaction(source_feats)  # [n_node, (l+1)^2]
            node_energies = torch.einsum("nb,nb->n", source_feats, self_fields)
            self_energy = scatter_sum(src=node_energies, index=batch, dim=-1)
            energy += self_energy * 0.5

        return energy

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_node, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.density_max_l == 0:
            return self.energy_l0(source_feats, positions, batch)
        else:
            return self.energy_l1(source_feats, positions, batch)


def charges_features_from_graph(
    charges,  # [n_atoms]
    positions,
    edge_index,
    batch,
    total_width_factors,  # [1, n_radial]
):
    """
    Computes the features from a collection of charges, on set of scalar features, considering only specified edges.
    normalization of the charges is multipoles.
    """
    num_nodes = positions.shape[0]
    sender, receiver = edge_index
    R_ij = positions[sender] - positions[receiver]  # [N_edges,3]
    d_ij = torch.norm(R_ij, dim=-1, keepdim=True)  # [N_edges,1]
    smooth_reciprocal = torch.erf(0.5 * d_ij / total_width_factors) / (d_ij + 1e-6)

    features = scatter_sum(
        charges[sender].unsqueeze(-1) * smooth_reciprocal,
        receiver,
        dim=0,
        dim_size=num_nodes,
    )  # [n_nodes, n_radial]

    features = FIELD_CONSTANT * features / (4 * pi)
    return features


class RealSpaceFiniteDifferenceElectrostaticFeatures(torch.nn.Module):
    """Computes field features for L=0,1 charges and features.
    vector charges and features are represented by displaced scalars."""

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        include_self_interaction=False,
        integral_normalization="receiver",
        offset: float = 0.1,
    ):
        super().__init__()

        self.density_max_l = density_max_l
        self.projection_max_l = projection_max_l
        self.include_self_interaction = include_self_interaction
        self.density_smearing_width = density_smearing_width
        self.projection_smearing_widths = projection_smearing_widths
        self.num_radial = len(projection_smearing_widths)

        self.self_interaction = GTOSelfInteractionBlock(
            density_max_l,
            density_smearing_width,
            projection_max_l,
            projection_smearing_widths,
            "multipoles",
            integral_normalization,
        )

        projection_smearing_widths_tensor = torch.tensor(
            projection_smearing_widths, dtype=torch.get_default_dtype()
        )
        total_width_factors = torch.pow(
            (density_smearing_width**2 + projection_smearing_widths_tensor**2) / 2, 0.5
        )
        self.register_buffer("total_width_factors", total_width_factors)

        self.offset = offset
        self.register_buffer(
            "x", torch.tensor([offset, 0.0, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "y", torch.tensor([0.0, offset, 0.0], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "z", torch.tensor([0.0, 0.0, offset], dtype=torch.get_default_dtype())
        )

        l0_factors = [
            get_Cl_sigma(0, sigma, normalize=integral_normalization)
            / get_Cl_sigma(0, sigma, normalize="multipoles")
            for sigma in projection_smearing_widths
        ]
        self.register_buffer(
            "l0_factors", torch.tensor(l0_factors, dtype=torch.get_default_dtype())
        )
        l1_factors = [
            3**0.5
            * sigma**2
            * (
                get_Cl_sigma(1, sigma, normalize=integral_normalization)
                / get_Cl_sigma(0, sigma, normalize="multipoles")
            )
            / self.offset
            for sigma in projection_smearing_widths
        ]
        self.register_buffer(
            "l1_factors", torch.tensor(l1_factors, dtype=torch.get_default_dtype())
        )

    def call_density_0_feats_0(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        edge_long_index = batch_complete_graph_excluding_self_duplicates_vector(
            batch, 1
        )
        feats = charges_features_from_graph(
            charges=source_feats[:, 0],
            positions=positions,
            edge_index=edge_long_index,
            batch=batch,
            total_width_factors=self.total_width_factors.unsqueeze(0),
        )  # [n_atoms, n_radial]
        return self.l0_factors * feats

    def call_density_1_feats_1(
        self,
        source_feats: torch.Tensor,  # [n_nodes, (max_l_s+1)**2]
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        extended_positions = positions.repeat_interleave(4, dim=0)
        extended_positions[1::4] += self.x
        extended_positions[2::4] += self.y
        extended_positions[3::4] += self.z

        extended_batch = batch.repeat_interleave(4)
        charges = torch.zeros_like(extended_positions[:, 0])

        charges[1::4] = source_feats[:, 3] / self.offset
        charges[2::4] = source_feats[:, 1] / self.offset
        charges[3::4] = source_feats[:, 2] / self.offset
        charges[0::4] = source_feats[:, 0] - (
            charges[1::4] + charges[2::4] + charges[3::4]
        )

        edge_index = batch_complete_graph_excluding_self_duplicates_vector(batch, 4)

        scalar_features = charges_features_from_graph(
            charges=charges,
            positions=extended_positions,
            edge_index=edge_index,
            batch=extended_batch,
            total_width_factors=self.total_width_factors.unsqueeze(0),
        )  # [all_nodes, num_radial]

        all_features = torch.zeros(
            batch.size(0),
            4 * self.num_radial,
            dtype=torch.get_default_dtype(),
            device=batch.device,
        )

        all_features[:, : self.num_radial] = self.l0_factors * scalar_features[0::4]
        all_features[:, self.num_radial :: 3] = self.l1_factors * (
            scalar_features[2::4] - scalar_features[0::4]
        )
        all_features[:, self.num_radial + 1 :: 3] = self.l1_factors * (
            scalar_features[3::4] - scalar_features[0::4]
        )
        all_features[:, self.num_radial + 2 :: 3] = self.l1_factors * (
            scalar_features[1::4] - scalar_features[0::4]
        )

        return all_features

    def forward(
        self,
        source_feats: torch.Tensor,  # [n_nodes, 1, (max_l_s+1)**2]
        node_positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.density_max_l == 0 and self.projection_max_l == 0:
            features = self.call_density_0_feats_0(
                source_feats.squeeze(-2), node_positions, batch
            )
        elif self.density_max_l == 1 and self.projection_max_l == 0:
            all_feats = self.call_density_1_feats_1(
                source_feats.squeeze(-2), node_positions, batch
            )
            features = all_feats[:, : self.num_radial]
        elif self.density_max_l == 0 and self.projection_max_l == 1:
            padded_source_feats = torch.zeros(
                source_feats.shape[0],
                4,
                dtype=source_feats.dtype,
                device=source_feats.device,
            )
            padded_source_feats[:, 0] = source_feats[:, 0, 0]
            features = self.call_density_1_feats_1(
                padded_source_feats, node_positions, batch
            )
        else:
            features = self.call_density_1_feats_1(
                source_feats.squeeze(-2), node_positions, batch
            )

        self_interaction_terms = self.self_interaction(source_feats.squeeze(-2))
        if self.include_self_interaction:
            features += self_interaction_terms

        return features, self_interaction_terms, None
