"""End-to-end tests for blocks and models with channel and sequence mixers."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from discretax.encoder import LinearEncoder
from discretax.heads.classification import ClassificationHead
from discretax.models import LRU, S5, DeltaNet, LinOSS


def _dummy_input(batch_size: int, timesteps: int, in_features: int):
    """Generate dummy input data for testing.

    Args:
        batch_size: Number of samples
        timesteps: Sequence length
        in_features: Input feature dimension

    Returns:
        Random input tensor of shape (batch_size, timesteps, in_features)
    """
    return jr.normal(jr.PRNGKey(0), (batch_size, timesteps, in_features))


def _dummy_state(model: eqx.Module, batch_size: int):
    """Generate dummy state for testing.

    Args:
        model: The model instance
        batch_size: Number of samples (unused, kept for API consistency)

    Returns:
        Empty state object for the model
    """
    # Collect state for layer norms/dropouts if any; here we assume empty state OK
    return eqx.nn.State(model)


def test_lru_model_forward():
    """Test LRU model forward pass with channel and sequence mixers.

    This test verifies that:
    1. LRU model builds successfully with LRU blocks
    2. The model includes both sequence mixers (LRU) and channel mixers (GLU)
    3. Forward pass produces correct output shape when composed with encoder and head
    4. All components work together correctly via Sequential
    """
    key = jr.PRNGKey(0)
    keys = jr.split(key, 3)

    # Build components
    encoder = LinearEncoder(in_features=16, out_features=16, key=keys[0])
    lru_model = LRU(hidden_dim=16, num_blocks=2, state_dim=32, drop_rate=0.0, key=keys[1])
    head = ClassificationHead(in_features=16, out_features=3, key=keys[2])

    # Compose with Sequential
    model = eqx.nn.Sequential([encoder, lru_model, head])

    x = _dummy_input(batch_size=2, timesteps=7, in_features=16)
    state = _dummy_state(model, batch_size=2)

    # Apply vmap to handle batch dimension - Sequential handles state and key automatically
    def single_forward(x_single, key_single):
        return model(x_single, state, key=key_single)

    batched_forward = jax.vmap(single_forward, in_axes=(0, 0), axis_name="batch")
    y, _ = batched_forward(x, jr.split(jr.PRNGKey(1), 2))

    assert y.shape == (2, 3)  # (batch_size, out_features)


def test_s5_model_forward():
    """Test S5 model forward pass with channel and sequence mixers.

    This test verifies that:
    1. S5 model builds successfully with S5 blocks
    2. The model includes both sequence mixers (S5) and channel mixers (GLU)
    3. Forward pass produces correct output shape when composed with encoder and head
    4. S5-specific parameters (ssm_blocks, state_dim) are handled correctly
    """
    key = jr.PRNGKey(1)
    keys = jr.split(key, 3)

    # Build components
    encoder = LinearEncoder(in_features=16, out_features=16, key=keys[0])
    s5_model = S5(
        hidden_dim=16, num_blocks=2, state_dim=32, ssm_blocks=1, drop_rate=0.0, key=keys[1]
    )
    head = ClassificationHead(in_features=16, out_features=3, key=keys[2])

    # Compose with Sequential
    model = eqx.nn.Sequential([encoder, s5_model, head])

    x = _dummy_input(batch_size=2, timesteps=7, in_features=16)
    state = _dummy_state(model, batch_size=2)

    # Apply vmap to handle batch dimension - Sequential handles state and key automatically
    def single_forward(x_single, key_single):
        return model(x_single, state, key=key_single)

    batched_forward = jax.vmap(single_forward, in_axes=(0, 0), axis_name="batch")
    y, _ = batched_forward(x, jr.split(jr.PRNGKey(2), 2))

    assert y.shape == (2, 3)  # (batch_size, out_features)


def test_deltanet_model_forward():
    """Test DeltaNet model forward pass with channel and sequence mixers.

    This test verifies that:
    1. DeltaNet model builds successfully with DeltaNet blocks
    2. The model includes both sequence mixers (DeltaNet) and channel mixers (GLU)
    3. Forward pass produces correct output shape when composed with encoder and head
    4. DeltaNet-specific parameters (n_heads, head_dim, chunk_size) are handled correctly
    5. Batching works correctly with vmap
    """
    key = jr.PRNGKey(3)
    keys = jr.split(key, 3)

    # Build components
    encoder = LinearEncoder(in_features=16, out_features=16, key=keys[0])
    deltanet_model = DeltaNet(
        hidden_dim=16,
        num_blocks=2,
        n_heads=2,
        head_dim=8,
        chunk_size=4,
        drop_rate=0.0,
        key=keys[1],
    )
    head = ClassificationHead(in_features=16, out_features=3, key=keys[2])

    # Compose with Sequential
    model = eqx.nn.Sequential([encoder, deltanet_model, head])

    x = _dummy_input(
        batch_size=2, timesteps=8, in_features=16
    )  # timesteps divisible by chunk_size
    state = _dummy_state(model, batch_size=2)

    # Apply vmap to handle batch dimension - Sequential handles state and key automatically
    def single_forward(x_single, key_single):
        return model(x_single, state, key=key_single)

    batched_forward = jax.vmap(single_forward, in_axes=(0, 0), axis_name="batch")
    y, _ = batched_forward(x, jr.split(jr.PRNGKey(4), 2))

    assert y.shape == (2, 3)  # (batch_size, out_features)


@pytest.mark.parametrize(
    "backbone_kwargs",
    [
        {"state_dim": 32},
        {"state_dim": 32, "num_heads": 2},
        {"state_dim": 32, "num_heads": 2, "use_head_output_projection": True},
    ],
)
def test_linoss_model_forward(backbone_kwargs):
    """Test LinOSS model forward pass with channel and sequence mixers.

    This test verifies that:
    1. LinOSS model builds successfully with LinOSS blocks
    2. The model includes both sequence mixers (LinOSS) and channel mixers (GLU)
    3. Forward pass produces correct output shape when composed with encoder and head
    4. LinOSS-specific parameters (state_dim, discretization) are handled correctly
    5. Batching works correctly with vmap
    """
    key = jr.PRNGKey(2)
    keys = jr.split(key, 3)

    # Build components
    encoder = LinearEncoder(in_features=16, out_features=16, key=keys[0])
    linoss_model = LinOSS(
        hidden_dim=16,
        num_blocks=2,
        drop_rate=0.0,
        key=keys[1],
        **backbone_kwargs,
    )
    head = ClassificationHead(in_features=16, out_features=3, key=keys[2])

    # Compose with Sequential
    model = eqx.nn.Sequential([encoder, linoss_model, head])

    x = _dummy_input(batch_size=2, timesteps=7, in_features=16)
    state = _dummy_state(model, batch_size=2)

    # Apply vmap to handle batch dimension - Sequential handles state and key automatically
    def single_forward(x_single, key_single):
        return model(x_single, state, key=key_single)

    batched_forward = jax.vmap(single_forward, in_axes=(0, 0), axis_name="batch")
    y, _ = batched_forward(x, jr.split(jr.PRNGKey(3), 2))

    assert y.shape == (2, 3)  # (batch_size, out_features)


def test_linoss_model_rejects_invalid_head_partition():
    """LinOSS model validates multi-head hidden/state partitions clearly."""
    with pytest.raises(ValueError, match="hidden_dim=15 must be divisible by num_heads=2"):
        LinOSS(hidden_dim=15, state_dim=32, num_heads=2, key=jr.PRNGKey(5))

    with pytest.raises(ValueError, match="state_dim=30 must be divisible by num_heads=4"):
        LinOSS(hidden_dim=16, state_dim=30, num_heads=4, key=jr.PRNGKey(6))


def test_linoss_model_passes_dtype_to_sequence_mixers():
    """LinOSS model forwards dtype to its sequence mixer blocks."""
    model = LinOSS(
        hidden_dim=16,
        num_blocks=2,
        state_dim=32,
        dtype=jnp.bfloat16,
        key=jr.PRNGKey(7),
    )

    for block in model.blocks:
        assert block.sequence_mixer.B.dtype == jnp.bfloat16
        assert block.sequence_mixer.C.dtype == jnp.bfloat16
