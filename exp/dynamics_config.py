from __future__ import annotations
import math
from typing import Callable, Dict, Optional
import torch

# ================================================================
# In GeoPT, the three dimensions correspond to x z y respectively
# ================================================================

def _direction_craft(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # x: (b, n, 3)
    # cond: (b, 1, 3) -> [mach, aoa, beta]
    b, n, _ = x.shape
    device = x.device
    dtype = x.dtype

    aoa = cond[:, :, 1:2]  # (b,1,1) degrees
    beta = cond[:, :, 2:3]  # (b,1,1) degrees
    mach = cond[:, :, 0:1]  # (b,1,1)

    vx = torch.cos(torch.pi * aoa / 180.0).repeat(1, n, 1) * torch.cos(torch.pi * beta / 180.0).repeat(1, n, 1)
    vy = torch.sin(torch.pi * aoa / 180.0).repeat(1, n, 1)
    vz = torch.cos(torch.pi * aoa / 180.0).repeat(1, n, 1) * torch.sin(torch.pi * beta / 180.0).repeat(1, n, 1)

    v = torch.cat([vx, vy, vz], dim=-1).to(device=device, dtype=dtype)
    extra = (mach.repeat(1, n, 1) / 3.0).to(device=device, dtype=dtype)
    return torch.cat([v, extra], dim=-1)


def _direction_nasa(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # cond: (b,1,2) -> [mach, aoa]  or (b,1,>=3) if you append others
    b, n, _ = x.shape
    device, dtype = x.device, x.dtype

    aoa = cond[:, :, 1:2]
    mach = cond[:, :, 0:1]

    vx = torch.cos(torch.pi * aoa / 180.0).repeat(1, n, 1)
    vy = torch.sin(torch.pi * aoa / 180.0).repeat(1, n, 1)
    vz = torch.zeros(b, n, 1, device=device, dtype=dtype)

    v = torch.cat([vx, vy, vz], dim=-1)
    extra = (mach.repeat(1, n, 1) * 1.6).to(dtype)
    return torch.cat([v, extra], dim=-1)


def _direction_crash(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # cond: (b,1,1) or (b,1,*) -> here assume first dim is angle in radians
    b, n, _ = x.shape
    device, dtype = x.device, x.dtype

    angle = cond[..., :1]  # (b,1,1)
    vx = torch.cos(torch.pi * angle / 180.0).repeat(1, n, 1)
    vy = torch.zeros(b, n, 1, device=device, dtype=dtype)
    vz = torch.sin(torch.pi * angle / 180.0).repeat(1, n, 1)
    v = torch.cat([vx, vy, vz], dim=-1).to(dtype)

    x_max = torch.max(x[:, :, 0:1], dim=1, keepdim=True)[0]
    x_min = torch.min(x[:, :, 0:1], dim=1, keepdim=True)[0]
    speed = (x[:, :, 0:1] - x_min) / (x_max - x_min + 1e-8)
    extra = (speed * 0.5).to(dtype)
    return torch.cat([v, extra], dim=-1)


def _direction_hull(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # cond: (b,1,1) -> [angle degrees]
    b, n, _ = x.shape
    device, dtype = x.device, x.dtype

    angle = cond  # degrees
    vx = torch.cos(torch.pi * angle / 180.0).repeat(1, n, 1)
    vy = torch.zeros(b, n, 1, device=device, dtype=dtype)
    vz = torch.sin(torch.pi * angle / 180.0).repeat(1, n, 1)
    v = torch.cat([vx, vy, vz], dim=-1).to(dtype)

    thr = 0.17428
    mask = (x[:, :, 1] > thr).unsqueeze(-1)  # True means set extra=0
    extra = (0.3 * (~mask).to(dtype)).to(device=device)  # (b,n,1)
    return torch.cat([v, extra], dim=-1)


def _direction_drivAerML(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    # cond: (b,1,2) -> [weight, angle_degrees]
    b, n, _ = x.shape
    device, dtype = x.device, x.dtype

    speed = 0.3 # default
    vx = torch.ones(b, n, 1, device=device, dtype=dtype)
    vy = torch.zeros(b, n, 1, device=device, dtype=dtype)
    vz = torch.zeros(b, n, 1, device=device, dtype=dtype)
    w = torch.ones(b, n, 1, device=device, dtype=dtype) * speed

    return torch.cat([vx, vy, vz, w], dim=-1).to(dtype)


# -------- registry --------
_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "craft": _direction_craft,
    "nasa": _direction_nasa,
    "crash": _direction_crash,
    "hull": _direction_hull,
    "drivAerML": _direction_drivAerML,
}

_ALIASES: Dict[str, str] = {
    "Craft": "craft",
    "NASA": "nasa",
    "Hull": "hull",
    "Car": "drivAerML",
    "drivAerml": "drivAerML",
}


def get_direction(dynamics_config: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    dynamics_config -> direction(x, cond)
    """
    if dynamics_config in _ALIASES:
        dynamics_config = _ALIASES[dynamics_config]

    if dynamics_config not in _REGISTRY:
        raise ValueError(f"Unknown dynamics_config='{dynamics_config}'. Supported: {list(_REGISTRY.keys())}")

    return _REGISTRY[dynamics_config]
