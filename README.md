# Souden MVDR beamformer in CuPy

This package is modified from the core parts of [pb_bss](https://github.com/fgnt/pb_bss) 
and modifies it to use [CuPy](https://github.com/cupy/cupy) for accelerated GPU-based inference.

At the moment, it is meant to be used with the [GSS](https://github.com/desh2608/gss) toolkit, 
but it can also be used as a general beamformer.

## Installation

```bash
> pip install cupy-cuda102  # modify according to your CUDA version (https://docs.cupy.dev/en/stable/install.html#installing-cupy)
> pip install beamformer-gpu
```

## Usage

```python
from beamformer import beamform_mvdr

import cupy as cp

X = cp.random.rand(4, 1000, 513) # D, T, F
X_mask = cp.random.rand(1000, 513)  # T, F
N_mask = cp.random.rand(1000, 513)  # T, F

X_hat = beamform_mvdr(X, X_mask, N_mask, ban=True)
```
