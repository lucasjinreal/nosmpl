import time
from typing import Callable, Iterable, Optional, Sequence, TypeVar, Union
try:
    from rich.progress import track
except ImportError as e:
    print(f'rich not installed: {e}')


ProgressType = TypeVar("ProgressType")
StyleType = Union[str, "Style"]


def pbar(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    description: str = "Working...",
    total: Optional[float] = None,
    auto_refresh: bool = True,
    transient: bool = False,
    get_time: Optional[Callable[[], float]] = None,
    refresh_per_second: float = 10,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    update_period: float = 0.1,
    disable: bool = False,
    show_speed: bool = True,
) -> Iterable[ProgressType]:
    return track(
        sequence=sequence,
        description=description,
        total=total,
        auto_refresh=auto_refresh,
        transient=transient,
        get_time=get_time,
        refresh_per_second=refresh_per_second,
        style=style,
        complete_style=complete_style,
        finished_style=finished_style,
        pulse_style=pulse_style,
        update_period=update_period,
        disable=disable,
        show_speed=show_speed,
    )


def prange(start=None, end=None, step=None, desc='solving'):
    if start is None and end is None and step is None:
        raise ValueError("At least one argument must be provided")

    if start is not None and end is not None and step is not None:
        raise ValueError("Only one of start, end, or length can be provided")

    if step is not None:
        end = start + step
        start = 0 if step > 0 else step
    else:
        if end is None:
            end = start
            start = 0
        step = 1 if start < end else -1
    return pbar(range(start, end, step), description=desc)
