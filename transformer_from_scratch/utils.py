import torch, os, gc
import torch.nn as nn
from datetime import datetime
from colorama import Fore


NUM_PROC = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # type: ignore


def free_memory(*args: str):
    """Free memory by deleting variables and calling garbage collector.

    Parameters
    ----------
    args : str
        Names of variables to delete.

        Examples
        --------
        >>> a = torch.randn(1000, 1000).to("cuda")
        >>> free_memory("a")
        >>> print(torch.cuda.memory_summary())
    """
    for name in args:
        try:
            arg = globals()[name]
            arg.to("cpu")
            del arg
        except KeyError:
            pass
    gc.collect()
    torch.cuda.empty_cache()


def log(message: str, file: str | None = None):
    """Log a message to a file, and print it to stdout.

    Parameters
    ----------
    msg : str
        Message to log.
    file : str, optional
        File to log to. If None, only print to stdout. Default: None.
    """
    print(message)
    if file:
        with open(file, "a") as f:
            f.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')}: {message}\n")


def sec2hms(seconds: float) -> str:
    """Convert seconds to hours, minutes, and seconds.

    Parameters
    ----------
    seconds : float
        Seconds to convert.

    Returns
    -------
    str
        String representation of hours, minutes, and seconds.

        Examples
        --------
        >>> sec2hms(3661)
        '1:01:01'
    """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}:{minutes:02}:{seconds:02}"


def analyze_params(module: nn.Module):
    """Print the parameters in the module and it's submodules."""
    print(f"Total number of parameters: {sum(p.numel() for p in module.parameters())}")
    for name, param in module.named_parameters():
        if param.requires_grad:
            print(f"{Fore.GREEN}{name}{Fore.RESET}")
            print(f"\t{str(tuple(param.shape)):<20} {str(param.dtype):<13}", end="\t")
            # get the mean and std of the parameters
            value = param.detach().cpu().numpy()
            print(f"param = {value.mean():>12.7f} +/- {value.std():>11.7f}", end="\t")
            # get the mean and std of the gradients
            if param.grad is None:
                print("grad = None")
            else:
                grad = param.grad.detach().cpu().numpy()
                print(f"grad = {grad.mean():>12.7f} +/- {grad.std():>11.7f}")


def compare_params(module1: nn.Module, module2: nn.Module):
    """Compare the parameters in two modules. Look for the root mean squared difference."""
    assert type(module1) == type(module2), "Modules must be of the same type."
    for (n1, p1), (n2, p2) in zip(module1.named_parameters(), module2.named_parameters()):
        assert n1 == n2, "Modules must have the same parameters."
        assert p1.shape == p2.shape, "Parameters must have the same shape."
        assert p1.dtype == p2.dtype, "Parameters must have the same dtype."
        value1 = p1.detach().cpu().numpy()
        value2 = p2.detach().cpu().numpy()
        print(f"{Fore.GREEN}{n1}{Fore.RESET}")
        print(f"{str(tuple(p1.shape)):<20}", end="\t")
        print(f"param1 = {value1.mean():>12.7f} +/- {value1.std():>11.7f}", end="\t")
        print(f"param2 = {value2.mean():>12.7f} +/- {value2.std():>11.7f}", end="\t")
        print(f"diff(rms) = {((value1 - value2) ** 2).mean() ** 0.5:>11.7f}", end="\t")
        print(f"diff(max) = {abs(value1 - value2).max():>11.7f}")


def truncate_sequence(seq: torch.Tensor, seq_key_padding_mask: torch.Tensor):
    """Truncate the sequence length to the nearest exponent of 2.

    Parameters
    ----------
    seq : torch.Tensor
        (batch_size, seq_len)
    seq_key_padding_mask : torch.Tensor
        (batch_size, seq_len)  0 for padding, 1 for non-padding

    Returns
    -------
    seq, seq_key_padding_mask
        Truncated sequence.
    """
    for i in range(seq.size(1)):
        if seq_key_padding_mask[:, i].sum() == 0:
            # find the nearest exponent of 2
            n = 2 ** (i - 1).bit_length()
            seq = seq[:, :n]
            seq_key_padding_mask = seq_key_padding_mask[:, :n]
            break
    return seq, seq_key_padding_mask
