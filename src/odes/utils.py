import torch


def rk4(func, x, h):
    k1 = func(x)
    k2 = func(x + 0.5 * h * k1)
    k3 = func(x + 0.5 * h * k2)
    k4 = func(x + h * k3)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_positive_fixed_dt(func, x_t, dt, *, lower=0.0, eps=1e-12, max_halves=5):
    lb = torch.as_tensor(lower, dtype=x_t.dtype, device=x_t.device)
    if lb.ndim == 0:
        lb = torch.full_like(x_t, float(lb))

    def ok(z):
        return torch.isfinite(z).all() and (z > lb).all()

    def advance(x, h, depth):
        if depth > max_halves:
            # Ensure the result is positive
            x1 = x + h * func(x)
            return torch.clamp(x1, min=lb + eps)

        x_try = rk4(func, x, h)
        if ok(x_try):
            return torch.clamp(x_try, min=lb + eps)

        # Backtrack by half steps
        x_mid = advance(x, h * 0.5, depth + 1)
        x_end = advance(x_mid, h * 0.5, depth + 1)
        return x_end

    x_t = torch.clamp(x_t, min=lb + eps)
    return advance(x_t, dt, 0)
