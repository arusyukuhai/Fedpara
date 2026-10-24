# Apple MPS float64 crash: Fedpara correction

## Symptom

`Cannot convert/compile MPS Tensor to float64` during a `train-live` /
`--mode synchronous` training run with a Mac MPS accelerator.

## Root cause

`patchlib/elastic.py` and `patchlib/livefed.py` previously evaluated update
RMS by using `delta.double()` **on the accelerator**. Metal MPS cannot
allocate a `torch.float64` tensor, even for checks that do not backpropagate.

## Correction

These validations now call `patchlib.numerics.squared_l2`. It normalizes each
finite reconstructed update block by the largest absolute element, computes
its squared norm entirely in float32 on the current accelerator, and performs
only the final scaling/accumulation with ordinary Python `float`s. The
existing `torch.isfinite` preflight checks remain in place.

The arithmetic is approximate (floating point as before) but avoids overflow
of the raw FP32 sum of squares, MPS float64 allocation, and large GPU-to-CPU
block copies.

## Run

In your existing project, replace `patchlib/elastic.py` and
`patchlib/livefed.py`, and add `patchlib/numerics.py`. Or replace the project
with the enclosed distribution, preserving your user-specific journal, cached
`.fstate`, room details and other local data. Do not overwrite your own
settings/data with example files.

```sh
python -m patchlib.cli gui
# Or resume your previous `train-live` invocation.
```

Verification from CPU-only test environment: `87 passed, 3 skipped`.
The MPS-only test is skipped; a real Mac GPU run remains necessary.

If an independent model, checkpoint, or third-party custom module creates a
float64 parameter and tries to move it onto MPS, it must be converted to
float32 **before** going to MPS, or the double-precision computation must stay
on CPU. This patch specifically corrects the Fedpara RMS validation sites.
