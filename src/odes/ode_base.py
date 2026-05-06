from .utils import rk4, rk4_positive_fixed_dt
import torch


def _check_no_nan(x):
    if torch.isnan(x).any():
        raise ValueError("NaN detected in the input tensor")
    return x


def _broadcast_to_shape(x, shape):
    # if shape = (..., d), x=(d), then x --> ((|...| - 1)*[1], d)
    if x.shape == shape:
        return x
    else:
        return x.reshape([1] * (len(shape) - 1) + [-1])


class ODEBase:
    METHOD = "difference"
    mu = 0.0
    std = 1.0
    time_scale = 1.0
    NUM_NODES = None
    MIN_VALUE = 1e-5
    MAX_VALUE = 1e8
    debug = False
    max_halves = 5
    log_mode = False
    # (idx_x, idx_y, label_x, label_y) — best 2D projection for phase portraits
    PHASE_PORTRAIT_DIMS = None
    # Biological function category
    CATEGORY = None

    def denormalize(self, x):
        if isinstance(self.mu, list):
            mu = torch.tensor(self.mu, device=x.device)
            mu = _broadcast_to_shape(mu, x.shape)
        else:
            mu = self.mu
        if isinstance(self.std, list):
            std = torch.tensor(self.std, device=x.device)
            std = _broadcast_to_shape(std, x.shape)
        else:
            std = self.std
        return x * std + mu

    def normalize(self, x):
        if isinstance(self.mu, list):
            mu = torch.tensor(self.mu, device=x.device)
            mu = _broadcast_to_shape(mu, x.shape)
        else:
            mu = self.mu

        if isinstance(self.std, list):
            std = torch.tensor(self.std, device=x.device)
            std = _broadcast_to_shape(std, x.shape)
        else:
            std = self.std
        return (x - mu) / std

    def random_initial_state(self, batch_size=1, device="cpu"):
        """Generate random initial state in normalized space."""
        # Gaussian sample at normalized space
        x = self._random_initial_state(batch_size, device)

        # Clip at normalized space
        if self.MIN_VALUE is not None and self.MAX_VALUE is not None:
            x = torch.clamp(x, self.MIN_VALUE, self.MAX_VALUE)
        x = self.normalize(x)

        return x

    def _random_initial_state(self, batch_size=1, device="cpu"):
        """Generate random initial state in original space."""
        x = torch.randn(batch_size, self.NUM_NODES, device=device)
        return x

    def project_to_manifold(self, x):
        """Project state to model-specific manifold constraints."""
        return x

    def project_normalized_state(self, x):
        """Project a normalized state to manifold and return normalized state."""
        x_denorm = self.denormalize(x)
        x_proj = self.project_to_manifold(x_denorm)
        return self.normalize(x_proj)

    def ode(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.ode(self.project_to_manifold(x))

    def simulate(self, x, dt=0.001, T=1.5):
        dt = dt * self.time_scale
        T = T * self.time_scale

        x = self.denormalize(x)
        trace = []
        lower = self.MIN_VALUE
        for i in range(int(T / dt)):
            x = rk4_positive_fixed_dt(
                self, x, dt, lower=lower, eps=1e-12, max_halves=self.max_halves
            )
            trace.append(_check_no_nan(x))

        traj = torch.stack(trace, dim=0)
        if self.log_mode:
            return traj.log()
        else:
            return self.normalize(traj)
