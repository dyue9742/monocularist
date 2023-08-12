import platform
import torch


def d(name: str):
    return torch.device(name)


def acceleration():
    if platform.system() == "Darwin":
        if torch.backends.mps.is_available():
            return d("mps")
    elif platform.system() == "Windows":
        if torch.cuda.is_available():
            return d("cuda")
    return d("cpu")
