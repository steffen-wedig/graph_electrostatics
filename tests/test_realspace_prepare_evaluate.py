"""Real-space geometry as a value: prepare once, evaluate many times.

The analytical real-space modules split into prepare (pair list, separations and the
smeared-Coulomb kernels, from the positions only) and evaluate (sources on a prepared
geometry). These tests pin that the split changes no number: against a frozen copy of
the previous single-call evaluation, for values, forces and the backward through the
forces; that one geometry serves modules with different widths; that several source
sets evaluate as one; and that the blocks pass a prepared geometry through.
"""

import math

import pytest
import torch
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.realspace_electrostatics import (
    CARTESIAN_TO_E3NN_COMPONENTS,
    E3NN_TO_CARTESIAN_COLUMNS,
    RealSpaceAnalyticalElectrostaticFeatures,
    RealSpaceAnalyticalEnergy,
    merge_kernel_widths,
    prepare_real_space_geometry,
    real_space_pair_index,
    smeared_coulomb_kernels,
)
from graph_longrange.utils import FIELD_CONSTANT
from mace.tools.scatter import scatter_sum
from scipy.constants import pi

DENSITY_SMEARING_WIDTH = 1.5
FEATURE_SMEARING_WIDTHS = [1.5, 3.0]


@pytest.fixture(autouse=True)
def float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def random_clusters(sizes=(5, 1, 7, 3), seed=0):
    """Open clusters with overlapping and well-separated pairs (both kernel branches)."""
    generator = torch.Generator().manual_seed(seed)
    positions = torch.cat(
        [2.5 * torch.rand(size, 3, generator=generator) for size in sizes]
    )
    batch = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    source_feats = 0.5 * torch.randn(sum(sizes), 4, generator=generator)
    return source_feats, positions, batch


def features_module(include_self_interaction=False):
    return RealSpaceAnalyticalElectrostaticFeatures(
        density_max_l=1,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        projection_max_l=1,
        projection_smearing_widths=FEATURE_SMEARING_WIDTHS,
        include_self_interaction=include_self_interaction,
        integral_normalization="receiver",
    )


def energy_module():
    return RealSpaceAnalyticalEnergy(
        density_max_l=1, density_smearing_width=DENSITY_SMEARING_WIDTH
    )


def reference_features(module, source_feats, positions, batch):
    """Frozen copy of the single-call feature evaluation before prepare/evaluate."""
    num_nodes = positions.shape[0]
    sender, receiver = real_space_pair_index(batch)
    separation = positions[receiver] - positions[sender]
    distance_squared = torch.sum(separation * separation, dim=-1, keepdim=True)
    widths = torch.tensor(
        [
            math.sqrt(DENSITY_SMEARING_WIDTH**2 + width**2)
            for width in FEATURE_SMEARING_WIDTHS
        ]
    )
    kernels = smeared_coulomb_kernels(distance_squared, widths, 2)
    sender_charges = source_feats[:, 0][sender].unsqueeze(-1)
    sender_dipoles = source_feats[:, E3NN_TO_CARTESIAN_COLUMNS][sender]
    along = torch.sum(sender_dipoles * separation, dim=-1, keepdim=True)
    scalar = scatter_sum(
        sender_charges * kernels.b0 + along * kernels.b1,
        receiver,
        dim=0,
        dim_size=num_nodes,
    )
    scalar = FIELD_CONSTANT / (4 * pi) * module.l0_factors * scalar
    gradient = (
        -sender_charges.unsqueeze(-1) * separation.unsqueeze(1) * kernels.b1.unsqueeze(-1)
        + sender_dipoles.unsqueeze(1) * kernels.b1.unsqueeze(-1)
        - along.unsqueeze(-1) * separation.unsqueeze(1) * kernels.b2.unsqueeze(-1)
    )
    gradient = scatter_sum(gradient, receiver, dim=0, dim_size=num_nodes)
    gradient = FIELD_CONSTANT / (4 * pi) * module.l1_factors.unsqueeze(-1) * gradient
    vectors = gradient[..., CARTESIAN_TO_E3NN_COMPONENTS].reshape(num_nodes, -1)
    return torch.cat([scalar, vectors], dim=-1)


def reference_energy(source_feats, positions, batch, num_graphs):
    """Frozen copy of the single-call energy evaluation before prepare/evaluate."""
    sender, receiver = real_space_pair_index(batch)
    separation = positions[receiver] - positions[sender]
    kernels = smeared_coulomb_kernels(
        torch.sum(separation * separation, dim=-1),
        math.sqrt(2.0) * DENSITY_SMEARING_WIDTH,
        2,
    )
    charges = source_feats[:, 0]
    dipoles = source_feats[:, E3NN_TO_CARTESIAN_COLUMNS]
    sender_along = torch.sum(dipoles[sender] * separation, dim=-1)
    receiver_along = torch.sum(dipoles[receiver] * separation, dim=-1)
    pair_energy = (
        charges[sender] * charges[receiver] * kernels.b0
        - (charges[sender] * receiver_along - charges[receiver] * sender_along)
        * kernels.b1
        + torch.sum(dipoles[sender] * dipoles[receiver], dim=-1) * kernels.b1
        - sender_along * receiver_along * kernels.b2
    )
    node_energies = scatter_sum(
        0.5 * FIELD_CONSTANT / (4 * pi) * pair_energy,
        receiver,
        dim=-1,
        dim_size=positions.shape[0],
    )
    return scatter_sum(node_energies, batch, dim=-1, dim_size=num_graphs)


def force_training_quantities(total_energy, positions, source_feats):
    """Energy, forces and the gradients of a force loss (the double backward)."""
    (forces,) = torch.autograd.grad(-total_energy, positions, create_graph=True)
    force_loss = (forces**2).sum()
    loss_gradients = torch.autograd.grad(force_loss, (positions, source_feats))
    return total_energy.detach(), forces.detach(), loss_gradients


def test_prepare_evaluate_matches_single_call_reference():
    """Values, forces and force-loss gradients equal the previous evaluation."""
    module = features_module()
    energy = energy_module()
    source_feats, positions, batch = random_clusters()

    def new(source, position):
        geometry = prepare_real_space_geometry(
            position,
            merge_kernel_widths(module.kernel_widths, energy.kernel_widths),
            2,
            batch=batch,
        )
        features, _ = module.evaluate(source, geometry)
        return (features**2).sum() + energy.evaluate(source, geometry, batch, 4).sum()

    def old(source, position):
        features = reference_features(module, source, position, batch)
        return (features**2).sum() + reference_energy(source, position, batch, 4).sum()

    results = []
    for evaluation in (new, old):
        position = positions.clone().requires_grad_(True)
        source = source_feats.clone().requires_grad_(True)
        results.append(
            force_training_quantities(evaluation(source, position), position, source)
        )
    (new_energy, new_forces, new_gradients), (old_energy, old_forces, old_gradients) = (
        results
    )
    torch.testing.assert_close(new_energy, old_energy, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(new_forces, old_forces, rtol=1e-12, atol=1e-12)
    for new_gradient, old_gradient in zip(new_gradients, old_gradients):
        torch.testing.assert_close(new_gradient, old_gradient, rtol=1e-11, atol=1e-11)


def test_merged_widths_serve_features_and_energy():
    """One geometry with both modules' widths equals each module's own geometry."""
    module = features_module()
    energy = energy_module()
    source_feats, positions, batch = random_clusters()
    merged = merge_kernel_widths(module.kernel_widths, energy.kernel_widths)
    # The energy width sqrt(2) * 1.5 is the first feature width: one column serves both.
    assert len(merged) == 2

    shared = prepare_real_space_geometry(positions, merged, 2, batch=batch)
    shared_features, _ = module.evaluate(source_feats, shared)
    own_features, _ = module.evaluate(source_feats, module.prepare(positions, batch))
    torch.testing.assert_close(shared_features, own_features, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(
        energy.evaluate(source_feats, shared, batch, 4),
        energy(source_feats, positions, batch),
        rtol=1e-14,
        atol=1e-14,
    )


def test_reordered_widths_select_the_right_columns():
    module = features_module()
    source_feats, positions, batch = random_clusters()
    reordered = (5.0, *reversed(module.kernel_widths))
    geometry = prepare_real_space_geometry(positions, reordered, 2, batch=batch)
    torch.testing.assert_close(
        module.evaluate(source_feats, geometry)[0],
        module(source_feats, positions, batch)[0],
        rtol=1e-14,
        atol=1e-14,
    )


def test_geometry_without_needed_width_or_order_is_refused():
    module = features_module()
    _, positions, batch = random_clusters()
    source_feats = torch.zeros(positions.shape[0], 4)
    missing_width = prepare_real_space_geometry(positions, (0.7,), 2, batch=batch)
    with pytest.raises(ValueError, match="no kernel of combined width"):
        module.evaluate(source_feats, missing_width)
    low_order = prepare_real_space_geometry(
        positions, module.kernel_widths, 1, batch=batch
    )
    with pytest.raises(ValueError, match="order 2 is needed"):
        module.evaluate(source_feats, low_order)


@pytest.mark.parametrize("include_self_interaction", [False, True])
def test_source_sets_evaluate_as_one(include_self_interaction):
    """[n_nodes, n_sets, C] sources equal one evaluation per set (the spin channels)."""
    module = features_module(include_self_interaction)
    _, positions, batch = random_clusters()
    generator = torch.Generator().manual_seed(3)
    source_sets = torch.randn(positions.shape[0], 2, 4, generator=generator)
    geometry = module.prepare(positions, batch)

    features, self_interaction_terms = module.evaluate(source_sets, geometry)
    for index in range(2):
        single_features, single_terms = module.evaluate(source_sets[:, index], geometry)
        torch.testing.assert_close(features[:, index], single_features)
        torch.testing.assert_close(self_interaction_terms[:, index], single_terms)


def test_given_pair_index_equals_built_pairs():
    module = features_module()
    source_feats, positions, batch = random_clusters()
    pairs = real_space_pair_index(batch)
    torch.testing.assert_close(
        module.evaluate(source_feats, module.prepare(positions, pair_index=pairs))[0],
        module.evaluate(source_feats, module.prepare(positions, batch))[0],
    )


def test_double_backward_through_a_shared_geometry_is_exact():
    """gradgradcheck through one geometry evaluated by features and energy."""
    module = features_module()
    energy = energy_module()
    source_feats, positions, batch = random_clusters(sizes=(3, 2))
    widths = merge_kernel_widths(module.kernel_widths, energy.kernel_widths)

    def total(position, source):
        geometry = prepare_real_space_geometry(position, widths, 2, batch=batch)
        alpha, _ = module.evaluate(source, geometry)
        beta, _ = module.evaluate(0.5 * source, geometry)
        return (alpha * beta).sum() + energy.evaluate(source, geometry, batch, 2).sum()

    inputs = (positions.requires_grad_(True), source_feats.requires_grad_(True))
    assert torch.autograd.gradcheck(total, inputs)
    assert torch.autograd.gradgradcheck(total, inputs)


def make_blocks(pbc_handling):
    features = GTOElectrostaticFeatures(
        density_max_l=1,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        feature_max_l=1,
        feature_smearing_widths=FEATURE_SMEARING_WIDTHS,
        include_self_interaction=False,
        kspace_cutoff=1.5,
        pbc_handling=pbc_handling,
    )
    energy = GTOElectrostaticEnergy(
        density_max_l=1,
        density_smearing_width=DENSITY_SMEARING_WIDTH,
        kspace_cutoff=1.5,
        pbc_handling=pbc_handling,
    )
    return features, energy


def block_geometry_arguments(positions, batch, num_graphs):
    return dict(
        k_vectors=positions.new_empty((0, 3)),
        k_norm2=positions.new_empty((0,)),
        k_vector_batch=batch.new_empty((0,)),
        k0_mask=positions.new_empty((0,)),
        node_positions=positions,
        batch=batch,
        volume=torch.ones(num_graphs),
        pbc=torch.zeros(num_graphs, 3, dtype=torch.bool),
    )


@pytest.mark.parametrize("pbc_handling", ["realspace", "auto"])
def test_blocks_share_a_caller_prepared_geometry(pbc_handling):
    """The features block keeps the geometry in its returned value, never on itself;
    the energy block evaluates a passed geometry like its own."""
    features_block, energy_block = make_blocks(pbc_handling)
    source_feats, positions, batch = random_clusters()
    arguments = block_geometry_arguments(positions, batch, 4)
    widths = merge_kernel_widths(
        features_block.realspace_features.kernel_widths,
        energy_block.realspace_energy.kernel_widths,
    )
    shared = prepare_real_space_geometry(positions, widths, 2, batch=batch)

    cache = features_block.precompute_geometry(**arguments, realspace_geometry=shared)
    assert cache["realspace_geometry"] is shared
    assert features_block.static_quantities is None
    own_cache = features_block.precompute_geometry(**arguments)
    torch.testing.assert_close(
        features_block.forward_dynamic(cache, source_feats),
        features_block.forward_dynamic(own_cache, source_feats),
        rtol=1e-14,
        atol=1e-14,
    )

    source_sets = torch.stack([source_feats, -0.3 * source_feats], dim=1)
    stacked = features_block.forward_dynamic(cache, source_sets)
    torch.testing.assert_close(
        stacked[:, 1], features_block.forward_dynamic(cache, source_sets[:, 1])
    )

    energy_arguments = {
        key: value for key, value in arguments.items() if key != "node_positions"
    }
    torch.testing.assert_close(
        energy_block(
            **energy_arguments,
            node_positions=positions,
            source_feats=source_feats,
            realspace_geometry=shared,
        ),
        energy_block(
            **energy_arguments, node_positions=positions, source_feats=source_feats
        ),
        rtol=1e-14,
        atol=1e-14,
    )


def test_energy_block_refuses_geometry_in_periodic_mode():
    _, energy_block = make_blocks("pbc")
    source_feats, positions, batch = random_clusters()
    arguments = block_geometry_arguments(positions, batch, 4)
    geometry = energy_module().prepare(positions, batch)
    with pytest.raises(ValueError, match="pbc_handling='pbc'"):
        energy_block(**arguments, source_feats=source_feats, realspace_geometry=geometry)


def test_module_apply_still_reaches_submodules():
    """evaluate, not apply: torch.nn.Module.apply(fn) must keep recursing."""
    visited = []
    features_block, energy_block = make_blocks("realspace")
    for block in (features_block, energy_block):
        block.apply(lambda module: visited.append(type(module).__name__))
    assert "RealSpaceAnalyticalElectrostaticFeatures" in visited
    assert "RealSpaceAnalyticalEnergy" in visited
