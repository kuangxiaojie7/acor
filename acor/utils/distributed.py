from __future__ import annotations

import datetime as dt
import os
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist


def init_distributed_mode(backend: str = "nccl") -> None:
    if dist.is_available() and _dist_env_ready():
        dist.init_process_group(backend=backend, timeout=dt.timedelta(seconds=1800))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())


def _dist_env_ready() -> bool:
    return all(var in os.environ for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def global_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return global_rank() == 0


@contextmanager
def distributed_zero_first() -> Iterator[None]:
    if not is_distributed():
        yield
        return
    if is_main_process():
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield
