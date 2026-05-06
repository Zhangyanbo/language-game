from torch.utils.data import Dataset, ConcatDataset
import torch
import random


class TrajIntervalDataset(Dataset):
    def __init__(self, traj, dt: float):
        super().__init__()
        self.traj = traj
        self.dt = torch.tensor([dt])
        self.n, self.dim = traj.shape

    def __len__(self):
        return self.n // 2 - 1

    def __getitem__(self, idx):
        x0 = self.traj[idx]
        di = random.randint(1, self.n // 2)
        xt = self.traj[idx + di]
        delta_t = di * self.dt
        return x0, xt, delta_t


class DynamicalSystemDataset(Dataset):
    def __init__(self, trajs, dt: float, batch_dim: int = 1):
        """
        Args:
            trajs: torch.Tensor
                Shape: [time_steps, batch, dim]
                Note: The batch-dimension can be changed by the batch_dim argument.
            dt: float
            batch_dim: int, the dimension of the batch. Default is 1.
        """
        super().__init__()
        trajs = trajs.transpose(batch_dim, 0)
        datasets = [TrajIntervalDataset(traj, dt) for traj in trajs]
        self.dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_dataset(ode_model, dt, T, batch_size=4096):
    x = ode_model.random_initial_state(batch_size, device="cpu")
    trace = ode_model.simulate(x, dt, T)
    dataset = DynamicalSystemDataset(trace, dt)
    return dataset
