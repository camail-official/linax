"""LinOSS model."""

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from discretax.blocks.standard import StandardBlock
from discretax.channel_mixers.glu import GLU
from discretax.sequence_mixers.linoss import LinOSSSequenceMixer
from discretax.utils.config_mixin import PartialModule


class LinOSS(eqx.nn.StatefulLayer, PartialModule):
    """LinOSS model.

    This model implements stacked blocks with LinOSS sequence mixers and GLU channel mixers.
    Use with eqx.nn.Sequential to compose with encoder and head.

    Attributes:
        blocks: List of standard blocks with LinOSS sequence mixers.

    Example:
        ```python
        import equinox as eqx
        import jax.random as jr
        from discretax.encoder import LinearEncoder
        from discretax.heads import ClassificationHead
        from discretax.models import LinOSS

        key = jr.PRNGKey(0)
        keys = jr.split(key, 3)

        encoder = LinearEncoder(in_features=784, out_features=64, key=keys[0])
        model = LinOSS(hidden_dim=64, num_blocks=4, key=keys[1])
        head = ClassificationHead(in_features=64, out_features=10, key=keys[2])

        # Compose with Sequential
        full_model = eqx.nn.Sequential([encoder, model, head])
        ```

    Reference:
        LinOSS: https://openreview.net/pdf?id=GRMfXcAAFh
    """

    blocks: list[StandardBlock]

    def __init__(
        self,
        key: PRNGKeyArray,
        *args,
        hidden_dim: int,
        num_blocks: int = 4,
        state_dim: int = 64,
        num_heads: int = 1,
        use_head_output_projection: bool = False,
        discretization: Literal["IM", "IMEX", "IMEX2", "IMEX3", "EX"] = "IMEX",
        initialization: Literal["RT", "AG"] = "AG",
        damping: bool = True,
        stability: Literal["oscillatory", "stable"] = "oscillatory",
        projection_eps: float = 0.0,
        input_normalization: bool = False,
        r_min: float = 0.9,
        theta_max: float = jnp.pi / 4,
        A_max: float = 1.0,
        G_max: float = 1.0,
        drop_rate: float = 0.1,
        prenorm: bool = True,
        use_bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the LinOSS model.

        Args:
            key: JAX random key for initialization.
            hidden_dim: hidden dimension for the model.
            num_blocks: number of LinOSS blocks to stack.
            state_dim: state space dimension for LinOSS sequence mixers.
            num_heads: number of independent LinOSS heads.
            use_head_output_projection: whether to apply a dense projection after
                concatenating multi-head outputs.
            discretization: discretization method ("IM", "IMEX", "IMEX2", "IMEX3", or "EX").
            initialization: initialization strategy for damped variants ("AG" or "RT").
            damping: whether to use damping in LinOSS.
            stability: "oscillatory" (complex conjugate eigenvalues)
                       or "stable" (full Jury region).
            projection_eps: epsilon buffer inset from eigenvalue stability boundaries.
                A_high is scaled by (1 - eps) and A_low (where non-negative) by (1 + eps).
                0.0 disables the buffer.
            input_normalization: LRU-style per-mode input gain init. Damped only.
            r_min: minimum value for the radius in LinOSS.
            theta_max: maximum value for theta parameter in LinOSS.
            A_max: upper bound for A in AG initialization.
            G_max: upper bound for G in AG initialization.
            drop_rate: dropout rate for blocks.
            prenorm: whether to apply prenorm in blocks.
            use_bias: whether to use bias in GLU channel mixers.
            dtype: dtype for LinOSS sequence mixer parameters and computation.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        if state_dim % num_heads != 0:
            raise ValueError(f"state_dim={state_dim} must be divisible by num_heads={num_heads}")

        keys = jr.split(key, 3 * num_blocks)

        # Build blocks with sequence mixers and channel mixers
        self.blocks = []
        for i in range(num_blocks):
            # Build sequence mixer
            seq_mixer = LinOSSSequenceMixer(
                in_features=hidden_dim,
                key=keys[i],
                state_dim=state_dim,
                num_heads=num_heads,
                use_head_output_projection=use_head_output_projection,
                discretization=discretization,
                initialization=initialization,
                damping=damping,
                stability=stability,
                projection_eps=projection_eps,
                input_normalization=input_normalization,
                r_min=r_min,
                theta_max=theta_max,
                A_max=A_max,
                G_max=G_max,
                dtype=dtype,
            )

            # Build channel mixer
            chan_mixer = GLU(
                in_features=hidden_dim,
                key=keys[num_blocks + i],
                out_features=None,
                use_bias=use_bias,
            )

            # Build block
            block = StandardBlock(
                in_features=hidden_dim,
                sequence_mixer=seq_mixer,
                channel_mixer=chan_mixer,
                key=keys[2 * num_blocks + i],
                drop_rate=drop_rate,
                prenorm=prenorm,
            )
            self.blocks.append(block)

    def __call__(
        self, x: Array, state: eqx.nn.State, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        """Forward pass through the LinOSS blocks.

        Args:
            x: Input tensor.
            state: Current state for stateful layers.
            key: JAX random key for operations.

        Returns:
            Tuple containing the output tensor and updated state.
        """
        # Prepare the keys
        block_keys = jr.split(key, len(self.blocks))

        # Apply the blocks
        for block, block_key in zip(self.blocks, block_keys):
            x, state = block(x, state, key=block_key)

        return x, state


if __name__ == "__main__":
    import jax.random as jr

    from discretax.encoder import LinearEncoder
    from discretax.heads import ClassificationHead

    key = jr.PRNGKey(0)
    keys = jr.split(key, 3)

    encoder = LinearEncoder(in_features=784, out_features=64, key=keys[0])
    model = LinOSS(hidden_dim=64, num_blocks=4, key=keys[1])
    head = ClassificationHead(in_features=64, out_features=10, key=keys[2])

    full_model = eqx.nn.Sequential([encoder, model, head])
    print(full_model)
