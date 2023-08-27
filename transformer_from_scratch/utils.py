import torch, os, gc
from datetime import datetime


NUM_PROC = os.cpu_count()
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()  # type: ignore
    else "cpu"
)


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
            f.write(
                f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')}: {message}\n"
            )


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
