"""Sequence mixer for LinOSS (IM, IMEX, IMEX2, IMEX3, EX) and Damped LinOSS variants.

See: https://openreview.net/pdf?id=GRMfXcAAFh
"""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn, random
from jax.nn.initializers import normal
from jaxtyping import Array, PRNGKeyArray

from discretax.sequence_mixers.base import AbstractSequenceMixer

# --- Matrix Registry ----------------------------------


def _mat_im(A, G, step):
    S = 1 + step * G + step**2 * A
    return 1 / S, -step * A / S, step / S, (1 + step * G) / S, step / S, step**2 / S


def _mat_imex(A, G, step):
    S = 1 + step * G
    return 1 / S, -step * A / S, step / S, 1 - step**2 * A / S, step / S, step**2 / S


def _mat_imex2(A, G, step):
    return 1 - step * G, -step * A, step * (1 - step * G), 1 - step**2 * A, step, step**2


def _mat_imex3(A, G, step):
    S = 1 + step**2 * A
    return (
        (1 - step * G) / S,
        -step * A / S,
        step * (1 - step * G) / S,
        1 / S,
        step / S,
        step**2 / S,
    )


def _mat_ex(A, G, step):
    return 1 - step * G, -step * A, step, 1 + A * 0, step, A * 0


MATRIX_FNS = {
    "IM": _mat_im,
    "IMEX": _mat_imex,
    "IMEX2": _mat_imex2,
    "IMEX3": _mat_imex3,
    "EX": _mat_ex,
}


def _lambda_sq(discretization: str, A, G, step):
    """|λ|² = det(M); product of eigenvalues for the 2×2 block."""
    m11, m12, m21, m22, _, _ = MATRIX_FNS[discretization](A, G, step)
    return m11 * m22 - m12 * m21


def _init_gamma_log(discretization, stability, A_flat, G_flat, steps_flat, projection_eps, dtype):
    """Initialize per-mode log-gain to sqrt(1 - |λ|²) using post-projection (A, G)."""
    h_init = nn.sigmoid(steps_flat)
    proj = _project_ag_oscillatory if stability == "oscillatory" else _project_ag_stability
    A_proj, G_proj = proj(discretization, A_flat, G_flat, h_init, projection_eps)
    lam_sq = _lambda_sq(discretization, A_proj, G_proj, h_init)
    return (0.5 * jnp.log(jnp.clip(1.0 - lam_sq, 1e-6, None))).astype(dtype)


# --- RT Initialization Helpers ----------------------------------


def _rt_to_ag(
    discretization: str, trace: Array, determinant: Array, step: Array
) -> tuple[Array, Array]:
    """Map recurrence trace/determinant targets to continuous-time A/G parameters."""
    h = step
    h2 = jnp.maximum(h**2, 1e-6)
    det = jnp.maximum(determinant, 1e-6)

    if discretization == "IM":
        A = (1.0 - trace + det) / (det * h2)
        G = (trace / det - 2.0) / h
    elif discretization == "IMEX":
        A = (1.0 + det - trace) / (det * h2)
        G = (1.0 / det - 1.0) / h
    elif discretization == "IMEX2":
        A = (1.0 + det - trace) / h2
        G = (1.0 - det) / h
    elif discretization == "IMEX3":
        denom = jnp.maximum(trace - det, 1e-6)
        A = (1.0 - trace + det) / (denom * h2)
        G = (trace - 2.0 * det) / (denom * h)
    elif discretization == "EX":
        A = (1.0 + det - trace) / h2
        G = (2.0 - trace) / h
    else:
        raise NotImplementedError(f"Discretization {discretization} not implemented")
    return A, G


# --- LinOSSSequenceMixer ----------------------------------


class LinOSSSequenceMixer(AbstractSequenceMixer):
    """LinOSS sequence mixer supporting IM, IMEX, IMEX2, IMEX3, and EX discretizations.

    Parameters are projected into a stable region at each forward pass via
    `_project_ag_oscillatory` ("oscillatory") or `_project_ag_stability` ("stable").

    If damping=false, G_diag always set to zero with no gradient flow.
    """

    A_diag: jax.Array
    G_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    gamma_log: jax.Array
    head_output_projection: eqx.nn.Linear | None

    # non learnable static fields
    discretization: Literal["IM", "IMEX", "IMEX2", "IMEX3", "EX"] = eqx.field(static=True)
    initialization: Literal["RT", "AG"] = eqx.field(static=True)
    damping: bool = eqx.field(static=True)
    stability: Literal["oscillatory", "stable"] = eqx.field(static=True)
    projection_eps: float = eqx.field(static=True)
    input_normalization: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_hidden_dim: int = eqx.field(static=True)
    head_state_dim: int = eqx.field(static=True)
    use_head_output_projection: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        key: PRNGKeyArray,
        *args,
        state_dim: int = 64,
        discretization: Literal["IM", "IMEX", "IMEX2", "IMEX3", "EX"] = "IMEX",
        initialization: Literal["RT", "AG"] = "AG",
        damping: bool = True,
        stability: Literal["oscillatory", "stable"] = "oscillatory",
        projection_eps: float = 0.0,
        input_normalization: bool = False,
        r_min: float = 0.9,
        theta_max: float = jnp.pi / 4,
        num_heads: int = 1,
        use_head_output_projection: bool = False,
        A_max: float = 1.0,
        G_max: float = 1.0,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        """Initialize the LinOSS sequence mixer layer.

        Args:
            in_features: dimension of the input features.
            key: JAX random key for initialization.
            state_dim: dimension of the state space.
            discretization: discretization method to use.
            initialization: initialization strategy for damped variants.
            damping: whether to use damping.
            stability: "oscillatory" (complex conjugate eigenvalues)
                       or "stable" (full Jury region).
            projection_eps: epsilon buffer inset from eigenvalue stability boundaries.
                A_high is scaled by (1 - eps) and A_low (where non-negative) by (1 + eps).
                0.0 disables the buffer.
            input_normalization: LRU-style per-mode input gain initialized to
                sqrt(1 - |λ|²) for unit steady-state variance at init. Damped only.
            r_min: minimum value for the radius.
            theta_max: maximum value for the theta parameter.
            num_heads: number of independent LinOSS heads.
            use_head_output_projection: whether to apply a learned output projection after
                concatenating multi-head outputs.
            A_max: upper bound for A in AG initialization.
            G_max: upper bound for G in AG initialization.
            dtype: dtype for sequence mixer parameters and computation.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).
        """
        dtype = jnp.dtype(dtype)
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if in_features % num_heads != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by num_heads={num_heads}"
            )
        if state_dim % num_heads != 0:
            raise ValueError(f"state_dim={state_dim} must be divisible by num_heads={num_heads}")
        if input_normalization and not damping:
            raise ValueError("input_normalization requires damping=True (|λ|=1 → gamma=0).")

        self.num_heads = num_heads
        self.head_hidden_dim = in_features // num_heads
        self.head_state_dim = state_dim // num_heads
        self.use_head_output_projection = use_head_output_projection and num_heads > 1

        # Key generator
        def key_gen(key: PRNGKeyArray):
            while True:
                key, subkey = jr.split(key)
                yield subkey

        gen = key_gen(key)
        nxt = lambda: next(gen)  # noqa

        if not damping:
            A_flat, G_flat, steps_flat = _init_linoss(nxt(), state_dim, 0.0, A_max, dtype=dtype)
        else:
            if initialization == "RT":
                A_flat, G_flat, steps_flat = _init_damped_linoss_rt(
                    nxt(), state_dim, discretization, r_min, 1.0, 0.0, theta_max, dtype=dtype
                )
            elif initialization == "AG":
                A_flat, G_flat, steps_flat = _init_damped_linoss_ag(
                    nxt(), state_dim, 0.0, A_max, 0.0, G_max, dtype=dtype
                )
            else:
                raise NotImplementedError(f"Initialization {initialization} not implemented")

        self.A_diag = A_flat.reshape(num_heads, self.head_state_dim)
        self.G_diag = G_flat.reshape(num_heads, self.head_state_dim)
        self.steps = steps_flat.reshape(num_heads, self.head_state_dim)

        self.B = _simple_uniform_init(
            nxt(),
            shape=(num_heads, self.head_state_dim, self.head_hidden_dim, 2),
            half_width=1.0 / math.sqrt(self.head_hidden_dim),
            dtype=dtype,
        )
        self.C = _simple_uniform_init(
            nxt(),
            shape=(num_heads, self.head_hidden_dim, self.head_state_dim, 2),
            half_width=1.0 / math.sqrt(self.head_state_dim),
            dtype=dtype,
        )
        self.D = normal(stddev=1.0)(nxt(), (num_heads, self.head_hidden_dim), dtype=dtype)

        gamma_log_flat = (
            _init_gamma_log(
                discretization, stability, A_flat, G_flat, steps_flat, projection_eps, dtype
            )
            if input_normalization
            else jnp.zeros((state_dim,), dtype=dtype)
        )
        self.gamma_log = gamma_log_flat.reshape(num_heads, self.head_state_dim)

        self.head_output_projection = (
            eqx.nn.Linear(in_features, in_features, key=nxt(), dtype=dtype)
            if self.use_head_output_projection
            else None
        )

        self.discretization = discretization
        self.initialization = initialization
        self.damping = damping
        self.stability = stability
        self.projection_eps = projection_eps
        self.input_normalization = input_normalization

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        """Forward pass of the LinOSS sequence mixer layer.

        Args:
            x: Input sequence of features.
            key: JAX random key (unused; present for interface compatibility).

        Returns:
            The output of the LinOSS sequence mixer.
        """
        steps = nn.sigmoid(self.steps)
        return self._apply_multi_head(x, steps)

    def _apply_multi_head(self, x: Array, steps: Array) -> Array:
        """Apply independent LinOSS heads and merge them back into the hidden stream."""
        G_diag = self.G_diag if self.damping else jnp.zeros_like(self.G_diag)
        x_heads = x.reshape(x.shape[0], self.num_heads, self.head_hidden_dim)
        scan_inputs = jnp.swapaxes(x_heads, 0, 1)
        ys = jax.vmap(self._apply_recurrence)(
            self.A_diag, G_diag, self.B, scan_inputs, steps, self.gamma_log
        )
        ys = jnp.swapaxes(ys, 0, 1)  # (L, num_heads, head_state_dim) complex

        C_complex = self.C[..., 0] + 1j * self.C[..., 1]
        head_outputs = jnp.real(jnp.einsum("hfs,lhs->lhf", C_complex, ys))
        head_outputs = head_outputs + x_heads * self.D[None, ...]

        xs = head_outputs.reshape(x.shape[0], x.shape[1])
        if self.head_output_projection is not None:
            xs = jax.vmap(self.head_output_projection)(xs)
        return xs

    def _apply_recurrence(
        self,
        A_diag: Array,
        G_diag: Array,
        B: Array,
        x: Array,
        steps: Array,
        gamma_log: Array,
    ) -> Array:
        """Apply one head's LinOSS recurrence. Returns (L, head_state_dim) complex."""
        if self.stability == "oscillatory":
            A, G = _project_ag_oscillatory(
                self.discretization, A_diag, G_diag, steps, self.projection_eps
            )
        else:
            A, G = _project_ag_stability(
                self.discretization, A_diag, G_diag, steps, self.projection_eps
            )
        mat_fn = MATRIX_FNS[self.discretization]
        B_complex = B[..., 0] + 1j * B[..., 1]
        return _apply_linoss(mat_fn, A, G, B_complex, x, steps, gamma_log)


# --- Initialization Helpers ----------------------------------


def _simple_uniform_init(
    rng: PRNGKeyArray,
    shape: tuple[int],
    half_width: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
):
    """Simple uniform initialization over [-half_width, half_width].

    Args:
        rng: JAX random key for initialization.
        shape: Shape of the weights.
        half_width: Half-width of the uniform distribution (weights sampled from
            [-half_width, half_width]).
        dtype: dtype of the initialized weights.

    Returns:
        Weights initialized using a simple uniform distribution.
    """
    weights = random.uniform(rng, shape, dtype=dtype) * 2.0 * half_width - half_width
    return weights


def _init_linoss(
    rng: PRNGKeyArray,
    state_dim: int,
    A_min: float,
    A_max: float,
    dtype: jnp.dtype = jnp.float32,
):
    """Initialize recurrence parameters for undamped LinOSS.

    Samples A_diag uniformly in [A_min, A_max]. G_diag is set to zeros.

    Args:
        rng: JAX random key for initialization.
        state_dim: Size of the state dimension.
        A_min: Lower bound for uniform A sampling.
        A_max: Upper bound for uniform A sampling.
        dtype: dtype for the returned arrays.

    Returns:
        Initialized (A_diag, G_diag, steps)
    """
    A_key, step_key = jr.split(rng, 2)
    A_diag = (A_min + random.uniform(A_key, shape=(state_dim,)) * (A_max - A_min)).astype(dtype)
    steps = normal(stddev=0.5)(step_key, (state_dim,), dtype=dtype)
    return A_diag, jnp.zeros_like(A_diag), steps


def _init_damped_linoss_ag(
    rng: PRNGKeyArray,
    state_dim: int,
    A_min: float,
    A_max: float,
    G_min: float,
    G_max: float,
    dtype: jnp.dtype = jnp.float32,
):
    """Initialize recurrence parameters for Damped LinOSS (AG strategy).

    Samples A and G uniformly in their respective ranges.

    Args:
        rng: JAX random key for initialization.
        state_dim: Size of the state dimension.
        A_min: Lower bound for uniform A sampling.
        A_max: Upper bound for uniform A sampling.
        G_min: Lower bound for uniform G sampling.
        G_max: Upper bound for uniform G sampling.
        dtype: dtype for the returned arrays.

    Returns:
        Initialized (A_diag, G_diag, steps)
    """
    A_key, G_key, step_key = jr.split(rng, 3)
    A_diag = (A_min + random.uniform(A_key, shape=(state_dim,)) * (A_max - A_min)).astype(dtype)
    G_diag = (G_min + random.uniform(G_key, shape=(state_dim,)) * (G_max - G_min)).astype(dtype)
    steps = normal(stddev=0.5)(step_key, (state_dim,), dtype=dtype)
    return A_diag, G_diag, steps


def _init_damped_linoss_rt(
    rng: PRNGKeyArray,
    state_dim: int,
    discretization: Literal["IM", "IMEX", "IMEX2", "IMEX3", "EX"],
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
    dtype: jnp.dtype = jnp.float32,
):
    """Initialize recurrence parameters for Damped LinOSS (RT strategy).

    Samples uniformly in the 2D annulus specified by radius, theta bounds.
    Uses the recurrence trace and determinant to initialize the continuous-time
    A and G parameters for the chosen discretization.

    Args:
        rng: JAX random key for initialization.
        state_dim: Size of the state dimension.
        discretization: discretization method to use.
        r_min: Lower bound for the radius.
        r_max: Upper bound for the radius.
        theta_min: Lower bound for the theta parameter.
        theta_max: Upper bound for the theta parameter.
        dtype: dtype for the returned arrays.

    Returns:
        Initialized (A_diag, G_diag, steps)
    """
    # Sample timesteps
    mag_key, arg_key, step_key = jr.split(rng, 3)
    step_vals = normal(stddev=0.5)(step_key, (state_dim,))
    step_sigmoid = nn.sigmoid(step_vals)

    # Sample eigenvalues in ring
    mag = jnp.sqrt(jr.uniform(mag_key, shape=(state_dim,)) * (r_max**2 - r_min**2) + r_min**2)
    arg = jr.uniform(arg_key, shape=(state_dim,)) * (theta_max - theta_min) + theta_min
    tr_vals = 2 * mag * jnp.cos(arg)
    det_vals = mag**2

    # Convert to (A, G) representation
    a_vals, g_vals = _rt_to_ag(discretization, tr_vals, det_vals, step_sigmoid)

    return (
        jnp.array(a_vals, dtype=dtype),
        jnp.array(g_vals, dtype=dtype),
        step_vals.astype(dtype),
    )


# --- Projection Operations ----------------------------------


def _project_ag_oscillatory(discretization, A_diag, G_diag, steps, eps=0.0):
    """Project A, G into the oscillator parameter space given the discretization.

    Args:
        discretization: discretization method to use.
        A_diag: un-projected A_diag parameters.
        G_diag: un-projected G_diag parameters.
        steps: pre-activated (e.g. sigmoid) timesteps.
        eps: epsilon buffer inset from stability boundaries. A_high is scaled by
            (1 - eps); A_low (where non-negative) is scaled by (1 + eps).

    Returns:
        Projected (A_diag, G_diag)
    """
    h = steps
    h2 = jnp.maximum(steps**2, 1e-6)

    if discretization == "IM":
        A_low_1 = -G_diag / h
        A_low_2 = G_diag**2 / 4
        A_diag = jnp.maximum(jnp.maximum(A_diag, A_low_1), A_low_2)
    elif discretization == "IMEX":
        G_diag = nn.relu(G_diag)
        A_low = (2 + h * G_diag - 2 * jnp.sqrt(1 + h * G_diag)) / h2
        A_high = (2 + h * G_diag + 2 * jnp.sqrt(1 + h * G_diag)) / h2
        A_diag = jnp.clip(A_diag, A_low * (1 + eps), A_high * (1 - eps))
    elif discretization == "IMEX2":
        G_diag = jnp.clip(G_diag, 0.0, (1 / h) * (1 - eps))
        A_low = (2 - h * G_diag - 2 * jnp.sqrt(1 - h * G_diag)) / h2
        A_high = (2 - h * G_diag + 2 * jnp.sqrt(1 - h * G_diag)) / h2
        A_diag = jnp.clip(A_diag, A_low * (1 + eps), A_high * (1 - eps))
    elif discretization == "IMEX3":
        G_diag = jnp.clip(G_diag, 0.0, (1 / h) * (1 - eps))
        A_low = G_diag**2 / jnp.maximum(4 * (1 - h * G_diag), 1e-6)
        A_diag = A_low * (1 + eps) + nn.relu(A_diag - A_low * (1 + eps))
    elif discretization == "EX":
        G_diag = jnp.clip(G_diag, 0.0, (4 / h) * (1 - eps))
        A_low = 1 / 4 * G_diag**2
        A_high = G_diag / h
        A_diag = jnp.clip(A_diag, A_low * (1 + eps), A_high * (1 - eps))

    return A_diag, G_diag


def _project_ag_stability(discretization, A_diag, G_diag, steps, eps=0.0):
    """Project A, G into the stable parameter space given the discretization.

    Args:
        discretization: discretization method to use.
        A_diag: un-projected A_diag parameters.
        G_diag: un-projected G_diag parameters.
        steps: pre-activated (e.g. sigmoid) timesteps.
        eps: epsilon buffer inset from stability boundaries. A_high is scaled by
            (1 - eps); A_low (where non-negative) is scaled by (1 + eps).

    Returns:
        Projected (A_diag, G_diag)
    """
    h = steps
    h2 = jnp.maximum(steps**2, 1e-6)

    if discretization == "IM":
        A_low_1 = -G_diag / h
        A_low_2 = -(2 * h * G_diag + 4) / h2
        A_diag = jnp.maximum(jnp.maximum(jnp.maximum(A_diag, A_low_1), A_low_2), 0.0)
    elif discretization == "IMEX":
        G_diag = nn.relu(G_diag)
        A_high = (4 + 2 * h * G_diag) / h2
        A_diag = jnp.clip(A_diag, 0.0, A_high * (1 - eps))
    elif discretization == "IMEX2":
        G_diag = jnp.clip(G_diag, 0.0, (2 / h) * (1 - eps))
        A_high = (4 - 2 * h * G_diag) / h2
        A_diag = jnp.clip(A_diag, 0.0, A_high * (1 - eps))
    elif discretization == "IMEX3":
        A_low_1 = (2 * h * G_diag - 4) / h2
        A_low_2 = -G_diag / h
        A_diag = jnp.maximum(jnp.maximum(jnp.maximum(A_diag, A_low_1), A_low_2), 0.0)
    elif discretization == "EX":
        G_diag = jnp.clip(G_diag, 0.0, (4 / h) * (1 - eps))
        A_low = nn.relu((2 * h * G_diag - 4) / h2)
        A_high = G_diag / h
        A_diag = jnp.clip(A_diag, A_low * (1 + eps), A_high * (1 - eps))

    return A_diag, G_diag


# --- Scan Operations ----------------------------------


@jax.vmap
def _binary_operator(q_i, q_j):  # noqa: N802
    """Binary operator for parallel scan of the 2×2 block linear recurrence."""
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N : 1 * N]
    iB_ = A_i[1 * N : 2 * N]
    iC_ = A_i[2 * N : 3 * N]
    iD_ = A_i[3 * N : 4 * N]
    jA_ = A_j[0 * N : 1 * N]
    jB_ = A_j[1 * N : 2 * N]
    jC_ = A_j[2 * N : 3 * N]
    jD_ = A_j[3 * N : 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j


def _apply_linoss_scan(
    M_11: Array,
    M_12: Array,
    M_21: Array,
    M_22: Array,
    F1: Array,
    F2: Array,
) -> Array:
    """Run the shared LinOSS scan."""
    state_dim = M_11.shape[0]

    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = jnp.broadcast_to(M, (F1.shape[0], M.shape[0]))
    F = jnp.hstack([F1, F2])

    _, xs = jax.lax.associative_scan(_binary_operator, (M_elements, F))

    return xs[:, state_dim:]


def _apply_linoss(mat_fn, A, G, B_complex, x, step, gamma_log):
    """Unified LinOSS apply using a matrix function from MATRIX_FNS.

    Args:
        mat_fn: A function from MATRIX_FNS returning (M_11, M_12, M_21, M_22, f1, f2).
        A: Diagonal state matrix.
        G: Diagonal damping matrix (zeros for undamped).
        B_complex: Input matrix.
        x: Input sequence of features.
        step: Pre-activated discretization time-steps.
        gamma_log: Per-mode log input gain, shape (state_dim,). Zeros = no-op.

    Returns:
        Hidden state sequence of shape (L, state_dim), complex.
    """
    Bu = jax.vmap(lambda u: B_complex @ u)(x)
    Bu = Bu * jnp.exp(gamma_log)[None, :]

    M_11, M_12, M_21, M_22, f1, f2 = mat_fn(A, G, step)
    F1 = Bu * f1[None, :]
    F2 = Bu * f2[None, :]

    return _apply_linoss_scan(M_11, M_12, M_21, M_22, F1, F2)
