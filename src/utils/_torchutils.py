import torch
import torch.nn.functional


# Class below is taking from pyrofit-utils:
# https://github.com/kosiokarchev/pyrofit-utils/blob/master/pyrofit/utils/torchutils.py
# It is used instead of scipy.interpolate.griddata
class TorchInterpNd:
    """Curently only works in 2D and 3D because of limitations in torch's grid_sample"""
    def __init__(self, data, *ranges):
        self.ranges = torch.tensor(ranges, dtype=torch.get_default_dtype(), device=data.device)
        self.extents = self.ranges[:, 1] - self.ranges[:, 0]
        self.ndim = len(ranges)

        self.data = data.unsqueeze(0) if data.ndim == self.ndim else data
        assert self.data.ndim == self.ndim + 1
        self.channels = self.data.shape[0]

    def __call__(self, *p_or_args):
        p = p_or_args if len(p_or_args) == 1 else torch.stack(torch.broadcast_tensors(*p_or_args), -1)
        assert p.shape[-1] == self.ndim

        p = 2 * (p - self.ranges[:, 0]) / self.extents - 1

        p_flat = p.reshape(*((1,) * self.ndim), -1, self.ndim)
        data_flat = self.data.unsqueeze(0)

        res = torch.nn.functional.grid_sample(data_flat, p_flat, align_corners=True)
        return torch.movedim(res.reshape(self.channels, *p.shape[:-1]), 0, -1)
