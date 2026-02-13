'''
Implementing data loading pipeline:

load_preferences() -> merge_datasets() -> shuffled_stream() -> batched_iterator()

Trying to use itertools and lazy evaluation heavily.

TODO:
linter
tests
'''

from __future__ import annotations

import itertools as itools
from dataclasses import dataclass, field
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, TypeVar, Callable
import functools
import random

from pathlib import Path
from PIL import Image

T = TypeVar('T') #using python 3.10 -> 3.12 can use [T]

@dataclass(slots=True, frozen=True)
class PreferencePair:
    '''
    Expect data for DPO in the form:
    (x, y_w, y_l) where y_w is preferred over y_l
    
    slots=True should improve memory utilization but confirm with profiling. TODO check this
    '''

    prompt: str
    chosen: str
    rejected: str
    image_path: Path | None = None
    image: Image.Image | None = field(default=None, repr=False, hash=False)
    metadata: dict[str, Any] = field(default_factory=dict, hash=False)

    def load_image(self) -> Image.Image | None:
        if self.image is not None:
            return self.image
        
        if self.image_path is None:
            return Image.open(self.image_path).convert('RGB')
        
        return None

#function overloading for load_preferences using singledispatch
@functools.singledispatch
def load_preferences(source: Any) -> Iterator[PreferencePair]:
    raise TypeError(f"Cannot load preference dataset from {type(source).__name__}")

@load_preferences.register(str)
def _load_from_path(source: str) -> Iterator[PreferencePair]:
    import json
    from pathlib import Path

    path = Path(source)
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            yield PreferencePair(
                prompt=d['prompt'],
                chosen=d['chosen'],
                rejected=d['rejected'],
                image_path=Path(d['image_path']) if 'image_path' in d else None,
                metadata=d['metadata'],
            )

@load_preferences.register(list)
def _load_from_list(source: list[PreferencePair]) -> Iterator[PreferencePair]:
    #already loaded
    yield from source

def batched_iterator(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    '''
    Memory-efficient batching using generators and islice (using islice)
    Note: the assumption here is that iterable is read lazily. See utils.py
    '''

    it = iter(iterable) #get iterator since islice called multiple times in loop
    while True:
        batch = list(itools.islice(it, batch_size))
        if not batch: return
        yield batch

#why shuffle
#https://bgerofi.github.io/papers/tnguyen-IPDPS22.pdf - TODO read carefully
#https://link.springer.com/article/10.1007/s00778-024-00845-0 - TODO read
#https://arxiv.org/pdf/2109.14119 - TODO read carefully (counter-example?)
def shuffled_buffer(data: Iterable[T], buffer_size: int = 10000, seed: int = 42) -> Iterator[T]:
    rng = random.Random(seed)
    it = iter(data)

    buf = list(itools.islice(it, buffer_size))
    
    #yield randomly selected item from buffer
    #replace element by next element in iterator
    for item in it:
        idx = rng.randrange(len(buf))
        yield buf[idx]
        buf[idx] = item

    #have only |buf| elements left. yield them
    rng.shuffle(buf)
    yield from buf #pep 380

def shuffled_stream(data: Sequence[T], seed: int | None = None) -> Iterator[T]:
    '''
    Create infinite stream (shuffled each epoch) of data points
    '''
    
    rng = random.Random(seed)
    indices = list(range(len(data)))
    epoch = 0

    while True:
        rng.shuffle(indices)
        epoch += 1
        for idx in indices:
            yield data[idx]

def merge_datasets(*sources: Iterable[PreferencePair]) -> Iterator[PreferencePair]:
    '''
    In case want to combine multiple datasets (lazily)
    '''
    return itools.chain.from_iterable(sources)

def running_mean(values: Iterable[float]) -> Iterator[float]:
    '''
    Can also use a coroutine here
    '''
    def _custom_add(total: tuple[float,int], val:float) -> tuple[float, int]:
        total_val, total_count = total
        return (total_val+val, total_count+1)

    gen = itools.accumulate(values,
                               _custom_add,
                               initial=(0.,0))
    
    next(gen) #skipping initial value (0/0)
    for total, count in gen:
        yield total/count

def exponential_moving_average(values: Iterable[float], alpha: float = 0.1) -> Iterator[float]:

    def _ema_add(current_mean: float, val: float) -> float:
        return (1-alpha)*current_mean + alpha*val

    return itools.accumulate(values, _ema_add)

#typing less useful here. want flexibility in composition chain
def compose(*functions: Callable[..., Any]) -> Callable[..., Any]:
    '''
    Build pipeline by composing functions

    return lambda function from reduce() call
    for lambda: left arg = accumulated, right arg = iterable element
    See: https://docs.python.org/3/library/functools.html
    Note: itertools.accumulate returns intermediate values too
    '''
    return functools.reduce(
        lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)),
        functions, #iterable
    )

def build_training_pipeline(
        sources: list[Iterable[PreferencePair]],
        seed: int = 42,
        buffer_size: int | None = 10000,
        batch_size: int = 8
) -> Iterator[list[PreferencePair]]:
    
    ds: Iterator[PreferencePair] = merge_datasets(*sources)
    stream: Iterator[PreferencePair]    

    if buffer_size is None:
        all_pairs: Sequence[PreferencePair] = list(ds)
        stream = shuffled_stream(all_pairs, seed)
    else:
        stream = shuffled_buffer(ds, buffer_size, seed)

    return batched_iterator(stream, batch_size)